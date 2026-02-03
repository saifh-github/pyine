# Multi-Node Training Guide

Run distributed RL training across multiple compute nodes using local `/scratch` filesystems for maximum performance.

## Overview

**Architecture:**

- DeepSpeed Zero Stage 3 for distributed training
- vLLM colocated on training GPUs
- Local `/scratch` on each node (fast ext4)
- Manual code sync between nodes

**Why local /scratch?**

- Your home (`/home`) and NAS (`/nas`) use NFS with `sync` mount = very slow writes
- `/scratch` is local ext4 SSD = much faster I/O
- Trade-off: Must manually sync code before each run

______________________________________________________________________

## Quick Start

### 1. Sync Code to All Nodes

```bash
cd /nas/users/a.palmas/code-interp-benchmark

# Sync code from NAS to all nodes' /scratch
./scripts/sync_code_to_nodes.sh
```

**Run this:**

- Before EVERY training run
- After ANY code changes
- After pulling new commits

### 2. Launch Training

```bash
# Set workspace to local scratch
export LOCAL_WORKSPACE="/scratch/a.palmas/code-interp-benchmark"

# Set training arguments
export TRAIN_ARGS="+experiment=original/v0_rl.yaml"

# Set accelerate config (must match number of nodes!)
# For 2 nodes (16 GPUs total):
export ACCELERATE_CONFIG="pyine/configs/accelerate/deepspeed_zero3_multinode_2x8gpu.yaml"
# For 3 nodes (24 GPUs total):
# export ACCELERATE_CONFIG="pyine/configs/accelerate/deepspeed_zero3_multinode_3x8gpu.yaml"
# For 4 nodes (32 GPUs total):
# export ACCELERATE_CONFIG="pyine/configs/accelerate/deepspeed_zero3_multinode_4x8gpu.yaml"

# Launch with krenew for Kerberos ticket renewal (specify nodes as arguments)
krenew -K 60 -- ./scripts/launch_multinode.sh gpu05 gpu06
```

### 3. Monitor

```bash
# View logs (on NAS - accessible from login node)
tail -f /nas/users/a.palmas/logs/multinode_*/train_gpu05.log

# Check GPUs
ssh gpu05 nvidia-smi
ssh gpu06 nvidia-smi
```

______________________________________________________________________

## What Gets Stored Where

### Local /scratch (fast, per-node):

- ✅ **Code**: `/scratch/a.palmas/code-interp-benchmark/` (synced manually)
- ✅ **Checkpoints**: Saved with Hydra output or custom `CHECKPOINT_DIR` (collect after training)
- ✅ **Caches**: HuggingFace models, Triton kernels, torch extensions

### NAS (slow, shared):

- 📁 **Source code**: Keep master copy on `/nas` for easy access
- 📁 **Training logs**: `/nas/users/a.palmas/logs/multinode_*/` (accessible from login node)
- 📁 **Collected checkpoints**: Copy from scratch after training (optional)

______________________________________________________________________

## Configuration

### Environment Variables

```bash
# Workspace location (required)
export LOCAL_WORKSPACE="/scratch/a.palmas/code-interp-benchmark"

# Training arguments (required)
export TRAIN_ARGS="+experiment=original/v0_rl_profiling_eval.yaml"

# Accelerate config - MUST match the number of nodes you're using!
# Available configs in pyine/configs/accelerate/:
#   - deepspeed_zero3_multinode_2x8gpu.yaml  (2 nodes, 16 GPUs)
#   - deepspeed_zero3_multinode_3x8gpu.yaml  (3 nodes, 24 GPUs)
#   - deepspeed_zero3_multinode_4x8gpu.yaml  (4 nodes, 32 GPUs)
export ACCELERATE_CONFIG="pyine/configs/accelerate/deepspeed_zero3_multinode_2x8gpu.yaml"

# Optional overrides
export CHECKPOINT_DIR="/scratch/a.palmas/checkpoints/my_run"
export LOG_DIR="/scratch/a.palmas/logs/my_run"

# Resume from checkpoint (optional)
export RESUME_FROM_RUN_DIR="/scratch/a.palmas/code-interp-benchmark/logs/runs/rl_trainer/original/RL_HT_38/original/RL_HT_38"
export RESUME_CHECKPOINT="checkpoint-500"  # Optional: specific checkpoint name
```

### Example: Full Training Run

```bash
# 1. Sync code
./scripts/sync_code_to_nodes.sh

# 2. Configure
export LOCAL_WORKSPACE="/scratch/a.palmas/code-interp-benchmark"
export TRAIN_ARGS="+experiment=rl_experiment \
    config.grpo_config.num_train_epochs=3 \
    config.grpo_config.per_device_train_batch_size=1 \
    config.grpo_config.gradient_accumulation_steps=4 \
    config.grpo_config.use_vllm=true"

# 3. Set accelerate config matching your node count
export ACCELERATE_CONFIG="pyine/configs/accelerate/deepspeed_zero3_multinode_2x8gpu.yaml"

# 4. Launch with krenew for Kerberos ticket renewal (specify nodes as arguments)
krenew -K 60 -- ./scripts/launch_multinode.sh gpu05 gpu06

# Or with 3 nodes (remember to change ACCELERATE_CONFIG!):
# export ACCELERATE_CONFIG="pyine/configs/accelerate/deepspeed_zero3_multinode_3x8gpu.yaml"
# krenew -K 60 -- ./scripts/launch_multinode.sh gpu01 gpu02 gpu03

# Or with 4 nodes (32 GPUs):
# export ACCELERATE_CONFIG="pyine/configs/accelerate/deepspeed_zero3_multinode_4x8gpu.yaml"
# krenew -K 60 -- ./scripts/launch_multinode.sh gpu01 gpu02 gpu03 gpu04

# 5. Monitor (logs are on NAS)
tail -f /nas/users/a.palmas/logs/multinode_*/train_gpu05.log
```

______________________________________________________________________

## Resuming from Checkpoint

Resume training from a previous checkpoint using environment variables:

### Resume on Same Nodes (Recommended)

When resuming on the **same nodes** that created the checkpoint, each node already has its own shards locally. No file copying needed:

```bash
# Resume from latest checkpoint in run directory (auto-detects latest checkpoint)
RESUME_FROM_RUN_DIR=/scratch/a.palmas/checkpoints/run_20250202_054800 \
  ./scripts/launch_multinode.sh gpu05 gpu06

# Resume from a specific checkpoint
RESUME_FROM_RUN_DIR=/scratch/a.palmas/checkpoints/run_20250202_054800 \
RESUME_CHECKPOINT=checkpoint-500 \
  ./scripts/launch_multinode.sh gpu05 gpu06
```

**Important:** Use the same node order as the original training (gpu05, gpu06) to ensure ranks match their shards.

### Resume on Different Nodes

If resuming on **different nodes**, you must first consolidate all checkpoint shards to a shared location:

```bash
# 1. Collect shards from original nodes to NAS
mkdir -p /nas/users/a.palmas/checkpoints/run_20250202_054800/checkpoint-500/global_step500
for node in gpu05 gpu06; do
  scp -r $node:/scratch/a.palmas/checkpoints/run_20250202_054800/checkpoint-500/* \
     /nas/users/a.palmas/checkpoints/run_20250202_054800/checkpoint-500/
done

# 2. Resume from NAS (all nodes can access it)
RESUME_FROM_RUN_DIR=/nas/users/a.palmas/checkpoints/run_20250202_054800 \
RESUME_CHECKPOINT=checkpoint-500 \
  ./scripts/launch_multinode.sh gpu07 gpu08
```

### Checkpoint Structure

DeepSpeed Zero3 checkpoints have this structure:

```
checkpoint-500/
├── config.json                    # Model config
├── model-00001-of-00002.safetensors  # Consolidated model (rank 0 only)
├── model-00002-of-00002.safetensors  # Consolidated model (rank 0 only)
├── trainer_state.json             # Training state (step, epoch, etc.)
├── scheduler.pt                   # LR scheduler state
├── rng_state_{0-15}.pth          # RNG states (split across nodes)
├── global_step500/               # DeepSpeed shards
│   ├── bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt   # Optimizer shard
│   ├── zero_pp_rank_0_mp_rank_00_model_states.pt       # Model shard
│   └── ... (one pair per rank)
└── zero_to_fp32.py               # Conversion script
```

- **gpu05** has: `rng_state_{0-7}.pth` + `global_step500/*rank_{0-7}*` + consolidated model
- **gpu06** has: `rng_state_{8-15}.pth` + `global_step500/*rank_{8-15}*` only

### How Resume Works

1. All ranks load the **base model** from HuggingFace (not from checkpoint)
2. GRPO Trainer is created with the fresh model
3. `trainer.train(resume_from_checkpoint=...)` is called
4. DeepSpeed loads each rank's shard from `global_step500/`
5. Training resumes from the saved step

______________________________________________________________________

## Collecting Checkpoints

After training completes, checkpoints are split across nodes (DeepSpeed Zero3 shards them). Collect them:

```bash
# Create destination
mkdir -p /nas/users/a.palmas/final_checkpoints/my_run

# Copy from all nodes
for node in gpu05 gpu06; do
  scp -r $node:/scratch/a.palmas/checkpoints/run_YYYYMMDD_HHMMSS/* \
     /nas/users/a.palmas/final_checkpoints/my_run/
done
```

### Using Checkpoints for Inference

The consolidated model on rank 0's node (`model*.safetensors`) can be used directly for inference without DeepSpeed:

```python
from transformers import AutoModelForCausalLM

# Load consolidated model (from rank 0's checkpoint or after collecting)
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/checkpoint-500",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

______________________________________________________________________

## Validation

Before first use, validate your setup:

```bash
# Validate specific nodes (required - same nodes you'll use for training)
./scripts/validate_multinode.sh gpu05 gpu06

# With custom workspace
LOCAL_WORKSPACE=/scratch/user/project ./scripts/validate_multinode.sh gpu05 gpu06

# Also check profiling prerequisites
ENABLE_PROFILING=true ./scripts/validate_multinode.sh gpu05 gpu06
```

This checks:

01. SSH connectivity
02. Node-to-node communication
03. Workspace existence and code sync
04. Git access (for `git pull` during launch)
05. UV and Python environment
06. Accelerate CLI availability
07. GPU availability
08. Required packages (torch, transformers, accelerate, trl, vllm, deepspeed)
09. CUDA functionality
10. Accelerate configuration file
11. Scratch space availability
12. Cache directory write access
13. NAS log directory access
14. Kerberos ticket renewal (krenew)
15. Nsight Systems (optional, when `ENABLE_PROFILING=true`)

______________________________________________________________________

## Troubleshooting

### "Cannot connect to node"

```bash
# Test SSH
ssh gpu05 echo "OK"
ssh gpu06 echo "OK"

# Set up passwordless SSH if needed
ssh-copy-id gpu05
ssh-copy-id gpu06
```

### "Workspace missing on node"

```bash
# Sync code
./scripts/sync_code_to_nodes.sh
```

### "Code mismatch on node"

```bash
# Resync code
./scripts/sync_code_to_nodes.sh
```

### "Module not found" during training

Code not synced properly:

```bash
./scripts/sync_code_to_nodes.sh
```

### "No space left on device"

Clean old caches:

```bash
for node in gpu05 gpu06; do
  ssh $node "rm -rf /scratch/a.palmas/tmp/*cache_*"
done
```

### Training hangs at initialization

Check node-to-node connectivity:

```bash
ssh gpu05 "ping -c 3 gpu06"
ssh gpu06 "ping -c 3 gpu05"

# Get IPs
ssh gpu05 "hostname -I"
ssh gpu06 "hostname -I"
```

### Training hangs during collective operations (NCCL timeout)

If you see errors like:

```
[Rank N] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=..., OpType=_ALLGATHER_BASE, ...)
```

Enable debug mode to get detailed stack traces and NCCL logs:

```bash
ENABLE_DEBUG=true ./scripts/launch_multinode.sh gpu05 gpu06
```

See the [Debug Mode](#debug-mode-ncclpytorch-distributed) section below for details.

______________________________________________________________________

## Manual Sync Commands

If you prefer manual control over syncing:

```bash
# Sync from NAS to nodes
rsync -avz --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='logs' \
  /nas/users/a.palmas/code-interp-benchmark/ \
  gpu05:/scratch/a.palmas/code-interp-benchmark/

rsync -avz --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='logs' \
  /nas/users/a.palmas/code-interp-benchmark/ \
  gpu06:/scratch/a.palmas/code-interp-benchmark/

# Verify sync
for node in gpu05 gpu06; do
  ssh $node "find /scratch/a.palmas/code-interp-benchmark -name '*.py' -type f -exec md5sum {} \; | sort | md5sum"
done
```

______________________________________________________________________

## Performance Tips

### Reduce Checkpoint Frequency

Checkpoints are saved to local scratch (fast), but still add overhead:

```bash
export TRAIN_ARGS="+experiment=my_exp \
    config.grpo_config.save_steps=500 \
    config.grpo_config.save_total_limit=2"
```

### Memory Optimization

If OOM:

```bash
export TRAIN_ARGS="+experiment=my_exp \
    config.grpo_config.per_device_train_batch_size=1 \
    config.grpo_config.gradient_accumulation_steps=8 \
    config.grpo_config.num_generations=2"
```

### Check Disk Space

```bash
for node in gpu05 gpu06; do
  echo "=== $node ==="
  ssh $node "df -h /scratch/a.palmas/"
done
```

______________________________________________________________________

## GPU Profiling with Nsight Systems

Profile your training runs to identify performance bottlenecks using NVIDIA Nsight Systems.

### Enabling Profiling

```bash
# Enable profiling with minimal overhead
export ENABLE_PROFILING=true
export PROFILE_LEVEL=minimal

# Then launch as usual
./scripts/launch_multinode.sh
```

### Profiling Levels

| Level      | Traces                      | CPU Sampling | Overhead | Use Case                    |
| ---------- | --------------------------- | ------------ | -------- | --------------------------- |
| `minimal`  | CUDA kernels, NVTX          | No           | ~1-5%    | Quick GPU performance check |
| `moderate` | + OS runtime calls          | No           | ~5-15%   | I/O and system bottlenecks  |
| `full`     | + cuDNN, cuBLAS, backtraces | Yes          | >15%     | Deep debugging              |

```bash
# Minimal - just CUDA and NVTX (recommended for most cases)
export PROFILE_LEVEL=minimal

# Moderate - includes OS runtime and syscalls
export PROFILE_LEVEL=moderate

# Full - comprehensive profiling (significant overhead)
export PROFILE_LEVEL=full
```

**Note for `full` level:** CPU sampling requires `perf_event_paranoid <= 2`. Check with:

```bash
ssh gpu01 "nsys status -e"
```

If sampling is not available, ask your sysadmin to run:

```bash
sudo sh -c 'echo 2 > /proc/sys/kernel/perf_event_paranoid'
```

### Output Files

Profile files are saved to the log directory on NAS:

```
${LOG_DIR}/profile_<node>_rank<N>.nsys-rep
```

For example:

```
/nas/users/a.palmas/logs/multinode_20250116_143022/
├── train_gpu01.log
├── train_gpu07.log
├── profile_gpu01_rank0.nsys-rep
└── profile_gpu07_rank1.nsys-rep
```

### Collecting Profile Files

Profile files are automatically saved to the NAS log directory, so no manual collection is needed. They're accessible from the login node immediately after training completes.

### Viewing Profiles

**Interactive GUI (recommended):**

```bash
# Open profile in Nsight Systems UI
nsys-ui /nas/users/a.palmas/logs/multinode_*/profile_gpu01_rank0.nsys-rep
```

**Generate summary statistics:**

```bash
# Quick summary of GPU activity
nsys stats /nas/users/a.palmas/logs/multinode_*/profile_gpu01_rank0.nsys-rep

# Export to SQLite for custom analysis
nsys export --type=sqlite --output=profile.sqlite profile_gpu01_rank0.nsys-rep
```

**Compare profiles across nodes:**

```bash
# Generate stats for all profiles
for f in /nas/users/a.palmas/logs/multinode_*/profile_*.nsys-rep; do
  echo "=== $(basename $f) ==="
  nsys stats "$f" --report gputrace 2>/dev/null | head -30
done
```

### Common Analysis Tasks

**Find slowest CUDA kernels:**

```bash
nsys stats profile.nsys-rep --report cudaapisum
```

**Analyze memory transfers:**

```bash
nsys stats profile.nsys-rep --report cudamemcpysum
```

**Check GPU utilization timeline:**

```bash
nsys stats profile.nsys-rep --report gpumemsizesum
```

### Tips

- Start with `minimal` level to avoid impacting training performance
- Profile short runs first (a few hundred steps) to validate setup
- Use `full` level only when debugging specific memory or CPU issues
- Compare profiles between nodes to identify load imbalances

______________________________________________________________________

## Debug Mode (NCCL/PyTorch Distributed)

Enable verbose debugging for diagnosing hangs, deadlocks, or collective operation failures.

### Enabling Debug Mode

```bash
# Enable debug mode
export ENABLE_DEBUG=true

# Then launch as usual
./scripts/launch_multinode.sh gpu05 gpu06
```

Or inline:

```bash
ENABLE_DEBUG=true ./scripts/launch_multinode.sh gpu05 gpu06
```

### What Debug Mode Enables

| Environment Variable           | Value    | Purpose                                             |
| ------------------------------ | -------- | --------------------------------------------------- |
| `NCCL_DEBUG`                   | `INFO`   | Verbose NCCL logging for all operations             |
| `NCCL_DEBUG_SUBSYS`            | `ALL`    | Log all NCCL subsystems (COLL, NET, etc.)           |
| `TORCH_NCCL_TRACE_BUFFER_SIZE` | `1000`   | Enable flight recorder for stack traces on failures |
| `TORCH_DISTRIBUTED_DEBUG`      | `DETAIL` | Add synchronization barriers and validation checks  |
| `TORCH_SHOW_CPP_STACKTRACES`   | `1`      | Show C++ stack traces on errors                     |

### Performance Impact

⚠️ **WARNING:** Debug mode has significant performance overhead (10-30%+). Only use for debugging, not production training.

| Flag                             | Overhead         | Notes                                |
| -------------------------------- | ---------------- | ------------------------------------ |
| `NCCL_DEBUG=INFO`                | ~5-15%           | Logging overhead for every NCCL op   |
| `NCCL_DEBUG_SUBSYS=ALL`          | High (with INFO) | Logs all subsystems                  |
| `TORCH_NCCL_TRACE_BUFFER_SIZE`   | ~1-3%            | Minimal - just ring buffer in memory |
| `TORCH_DISTRIBUTED_DEBUG=DETAIL` | ~10-30%          | Adds barriers and validation         |

### When to Use Debug Mode

- Training hangs with no error output
- NCCL timeout errors (`Watchdog caught collective operation timeout`)
- Suspected deadlocks between ranks
- Debugging inter-node communication issues

### Reading Debug Output

With debug enabled, you'll see detailed NCCL logs like:

```
gpu05:2851859:2851859 [0] NCCL INFO AllReduce: opCount 1234 sendbuff 0x... recvbuff 0x... count 655360 datatype 6 op 0 root 0 comm ...
```

On failures, you'll get stack traces showing exactly where the hang occurred:

```
[Rank 7] Watchdog caught collective operation timeout...
Stack trace:
  frame #0: ...
  frame #1: c10d::ProcessGroupNCCL::...
```

### Combining with Profiling

You can enable both debug and profiling:

```bash
ENABLE_DEBUG=true ENABLE_PROFILING=true PROFILE_LEVEL=minimal ./scripts/launch_multinode.sh gpu05 gpu06
```

This gives you both detailed NCCL logs and GPU performance profiles.

______________________________________________________________________

## Filesystem Comparison

Based on your testing:

| Location               | Type         | Write Mode | Performance  |
| ---------------------- | ------------ | ---------- | ------------ |
| `/home/a.palmas/`      | NFS4         | `sync`     | ❌ Very slow |
| `/nas/users/a.palmas/` | NFS4         | `sync`     | ❌ Very slow |
| `/scratch/a.palmas/`   | ext4 (local) | `relatime` | ✅ Fast      |

**Mount details:**

- Home: `login:/home on /home type nfs4 (rw,relatime,sync,...)`
- Scratch: `/dev/mapper/crypted_scratch on /scratch type ext4 (rw,relatime,stripe=768)`

______________________________________________________________________

## Files Overview

```
scripts/
├── launch_multinode.sh          # Main launcher (local scratch mode)
├── sync_code_to_nodes.sh        # Sync code from NAS to all nodes
├── validate_multinode.sh        # Validate setup before running
└── MULTINODE_GUIDE.md          # This guide
```

______________________________________________________________________

## Quick Reference

```bash
# Full workflow
./scripts/sync_code_to_nodes.sh
export LOCAL_WORKSPACE="/scratch/a.palmas/code-interp-benchmark"
export TRAIN_ARGS="+experiment=my_exp"
export ACCELERATE_CONFIG="pyine/configs/accelerate/deepspeed_zero3_multinode_2x8gpu.yaml"
krenew -K 60 -- ./scripts/launch_multinode.sh gpu05 gpu06

# Resume from checkpoint (same nodes)
RESUME_FROM_RUN_DIR=/scratch/a.palmas/checkpoints/run_20250202_054800 \
  ./scripts/launch_multinode.sh gpu05 gpu06

# Resume from specific checkpoint
RESUME_FROM_RUN_DIR=/scratch/a.palmas/checkpoints/run_20250202_054800 \
RESUME_CHECKPOINT=checkpoint-500 \
  ./scripts/launch_multinode.sh gpu05 gpu06

# With debug mode (for troubleshooting hangs)
ENABLE_DEBUG=true ./scripts/launch_multinode.sh gpu05 gpu06

# With profiling
ENABLE_PROFILING=true PROFILE_LEVEL=minimal ./scripts/launch_multinode.sh gpu05 gpu06

# Show help
./scripts/launch_multinode.sh --help

# Monitor (logs on NAS)
tail -f /nas/users/a.palmas/logs/multinode_*/train_*.log
ssh gpu05 nvidia-smi

# Clean caches
for node in gpu05 gpu06; do
  ssh $node "rm -rf /scratch/a.palmas/tmp/*cache_*"
done

# Collect checkpoints
mkdir -p /nas/users/a.palmas/final_checkpoints/my_run
for node in gpu05 gpu06; do
  scp -r $node:/scratch/a.palmas/checkpoints/run_*/* /nas/users/a.palmas/final_checkpoints/my_run/
done
```

______________________________________________________________________

## Additional Resources

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Accelerate Multi-Node Guide](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
- [TRL GRPO Documentation](https://huggingface.co/docs/trl/grpo)
- [vLLM Documentation](https://docs.vllm.ai/)
