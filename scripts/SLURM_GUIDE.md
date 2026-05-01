# SLURM Multi-Node Training Guide

Run distributed RL training across multiple compute nodes on a SLURM-managed cluster using DeepSpeed Zero3 and colocated vLLM.

______________________________________________________________________

## Architecture Overview

```
                     SLURM Controller
                      (sbatch / srun)
                     /                \
              Node 0 (8 GPUs)    Node 1 (8 GPUs)
              ┌────────────┐    ┌────────────┐
              │ accelerate  │    │ accelerate  │
              │  launch     │◄──►│  launch     │
              │  (rank 0)   │    │  (rank 1)   │
              │             │    │             │
              │ 8 GPU procs │    │ 8 GPU procs │
              │ + vLLM colo │    │ + vLLM colo │
              └────────────┘    └────────────┘
                   │  │              │  │
                   ▼  ▼              ▼  ▼
             /raid (fast)       /raid (fast)
            caches, ckpts      caches, ckpts

                     /lambdafs (shared)
                   code, logs, configs
```

- **DeepSpeed Zero Stage 3** for distributed training
- **vLLM colocated** on training GPUs (no separate vLLM service needed)
- **SLURM** handles node allocation, process placement, and cleanup
- **`srun`** launches one `accelerate launch` per node; accelerate spawns 8 GPU processes locally

______________________________________________________________________

## Filesystem Layout

| Path         | Type                 | Speed  | Used for                                 |
| ------------ | -------------------- | ------ | ---------------------------------------- |
| `/lambdafs/` | Shared (NFS/similar) | Slower | Code repo, datasets, logs, Hydra outputs |
| `/raid/`     | Node-local SSD       | Fast   | HF cache, compiled kernels, temp files   |

**Key principle:** Everything that multiple nodes need to read/write goes on `/lambdafs/` (shared). Only per-process caches and temp files go on `/raid/` (fast, node-local).

### What lives where

```
/lambdafs/users/user/
├── new_tests/
│   └── code-interp-benchmark/             # Git repo (shared, all nodes see it)
│       ├── pyine/                         # Training code
│       ├── scripts/launch_slurm.sh        # This launcher
│       ├── full_checkpoints/              # Pre-trained model weights (in repo)
│       │   └── RL_HT_49-600/             # Base model for RL training
│       ├── data/                          # Datasets (LMDB traces) — PYINE_DATA_ROOT
│       │   └── traces/TACO/              # TACO trace datasets
│       │       ├── *.v1.2025-01-15.lmdb/ # LMDB dataset directories
│       │       └── ...
│       └── logs/                          # Hydra run outputs — PYINE_LOGS_ROOT
│           └── runs/hf_trainer/           # Trainer outputs
│               └── KW_3/KW_3/            # {exp_name}/{run_name}
│                   ├── .hydra/            # Hydra configs (reproducibility)
│                   ├── checkpoint-500/    # DeepSpeed checkpoint shards
│                   │   ├── global_step500/# All nodes write rank-specific files here
│                   │   ├── trainer_state.json
│                   │   └── ...
│                   └── benchmark_export/  # Eval results
└── logs/slurm/                            # SLURM job logs
    ├── job_12345.out                      # SLURM stdout
    ├── job_12345.err                      # SLURM stderr
    └── run_12345_20250401/                # Per-run launcher logs
        ├── train_node0.log
        ├── train_node1.log
        └── job_env.txt                    # Saved env vars for reproducibility

/raid/  (on each node independently)
├── tmp/cache/                             # All caches (HF, torch, triton, wandb, etc.)
│   ├── huggingface/hub/                   # Downloaded model weights (cached)
│   ├── torch/                             # Compiled kernels
│   ├── triton/                            # Triton cache
│   └── tmp/                               # Temp files
```

### Datasets

Datasets live under `data/` in the repo (controlled by `PYINE_DATA_ROOT`). Since the repo is on `/lambdafs/`, datasets are automatically shared across all nodes. The training app loads LMDB datasets into memory at startup — this is a one-time sequential read, not a continuous I/O bottleneck. **No special handling needed.**

### Hydra outputs (configs, checkpoints, trainer state)

Hydra writes to `logs/` in the repo (controlled by `PYINE_LOGS_ROOT`). On `/lambdafs/`, this means:

- All nodes can write checkpoints to the same directory (DeepSpeed shards have rank-specific filenames, no conflicts)
- Outputs survive job completion and are accessible from the login node
- Resume works from any nodes (no need to request the same nodes)

**Trade-off:** Checkpoint writes are slower on shared FS vs `/raid/`. If checkpoint saving becomes a bottleneck (visible as long `save_model` times in logs), override with `CHECKPOINT_DIR=/raid/...` — but then you must collect checkpoints manually and request the same nodes for resume.

______________________________________________________________________

## Quick Start

### 1. Clone/update the repo on shared filesystem

```bash
# First time
cd /lambdafs
git clone <repo-url> code-interp-benchmark

# Updates
cd /lambdafs/users/user/new_tests/code-interp-benchmark
git pull
```

Since `/lambdafs/` is shared, all nodes see the same code. No sync step needed.

### 2. Submit the job

```bash
cd /lambdafs/users/user/new_tests/code-interp-benchmark

# Default: 2 nodes, keywords/v0_rl experiment
sbatch scripts/launch_slurm.sh

# Override training args
TRAIN_ARGS="+experiment=keywords/v0_rl.yaml" sbatch scripts/launch_slurm.sh

# Override multiple settings
TRAIN_ARGS="+experiment=keywords/v0_rl.yaml config.grpo_config.num_train_epochs=3" \
NETWORK_INTERFACE=bond0 \
  sbatch scripts/launch_slurm.sh

# Different node count (also change ACCELERATE_CONFIG!)
ACCELERATE_CONFIG="pyine/configs/accelerate/deepspeed_zero3_multinode_4x8gpu.yaml" \
  sbatch --nodes=4 scripts/launch_slurm.sh
```

### 3. Monitor

```bash
# Job status
squeue -u $USER

# Live logs (SLURM output captures everything)
tail -f /lambdafs/users/user/logs/slurm/job_<JOBID>.out

# Per-node logs
tail -f /lambdafs/users/user/logs/slurm/run_<JOBID>_*/train_*.log

# GPU usage (from login node, if SSH to compute nodes is allowed)
srun --jobid=<JOBID> --nodelist=<node> nvidia-smi
```

### 4. Cancel

```bash
scancel <JOBID>
```

SLURM automatically cleans up all processes on all nodes.

______________________________________________________________________

## Configuration Reference

All settings are environment variables with sensible defaults. Override by exporting before `sbatch` or editing the script.

### Required

| Variable            | Default                           | Description                                 |
| ------------------- | --------------------------------- | ------------------------------------------- |
| `TRAIN_ARGS`        | `+experiment=keywords/v0_rl.yaml` | Hydra training arguments                    |
| `NETWORK_INTERFACE` | `bond0`                           | Network interface for NCCL and IP detection |

### Workspace & Paths

| Variable         | Default                                                   | Description                   |
| ---------------- | --------------------------------------------------------- | ----------------------------- |
| `WORKSPACE`      | `/lambdafs/users/user/new_tests/code-interp-benchmark`    | Repo path (shared filesystem) |
| `RAID_BASE`      | `/raid`                                                   | Node-local fast storage root  |
| `CACHE_BASE`     | `/raid/tmp/cache`                                         | Cache directory (node-local)  |
| `CHECKPOINT_DIR` | _(Hydra output_dir)_                                      | Custom checkpoint path        |
| `LOG_DIR`        | `/lambdafs/users/user/logs/slurm/run_<JOBID>_<timestamp>` | Log directory (shared)        |

### Training

| Variable            | Default                                                          | Description                               |
| ------------------- | ---------------------------------------------------------------- | ----------------------------------------- |
| `ACCELERATE_CONFIG` | `pyine/configs/accelerate/deepspeed_zero3_multinode_2x8gpu.yaml` | Accelerate config (must match node count) |
| `TRAIN_SCRIPT`      | `pyine/apps/trainers/hf_trainer.py`                              | Training entry point                      |
| `GPUS_PER_NODE`     | `8`                                                              | GPUs per node                             |
| `MODULE_LOADS`      | _(empty)_                                                        | Space-separated modules to load           |

### Resume

| Variable              | Default   | Description                                       |
| --------------------- | --------- | ------------------------------------------------- |
| `RESUME_FROM_RUN_DIR` | _(empty)_ | Path to resume from (checkpoint parent dir)       |
| `RESUME_CHECKPOINT`   | _(empty)_ | Specific checkpoint name (e.g., `checkpoint-500`) |

### Debug

| Variable         | Default | Description                                          |
| ---------------- | ------- | ---------------------------------------------------- |
| `ENABLE_DEBUG`   | `false` | Enable NCCL/PyTorch debug logging (10-30%+ overhead) |
| `RECREATE_CACHE` | `false` | Delete and recreate cache before training            |

______________________________________________________________________

## SBATCH Directives

The `#SBATCH` lines at the top of `launch_slurm.sh` control the SLURM resource request. Edit these directly or override with `sbatch` flags:

```bash
#SBATCH --job-name=rl-train          # Job name (visible in squeue)
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --ntasks-per-node=1          # One srun task per node (accelerate handles GPUs)
#SBATCH --gpus-per-node=8            # Request all 8 GPUs per node
#SBATCH --cpus-per-task=32           # CPUs for data loading workers
#SBATCH --exclusive                  # No sharing the node with other jobs
#SBATCH --time=48:00:00              # Wall time limit (HH:MM:SS)
#SBATCH --output=...                 # Stdout file (%j = job ID)
#SBATCH --error=...                  # Stderr file
# #SBATCH --partition=gpu             # Uncomment if partition required
# #SBATCH --account=your_account      # Uncomment if account required
```

Override at submission time:

```bash
# 4 nodes, 24-hour limit, specific partition
sbatch --nodes=4 --time=24:00:00 --partition=a100 scripts/launch_slurm.sh
```

______________________________________________________________________

## Accelerate Config & Node Count

The accelerate config **must match** the number of SLURM nodes. Available configs:

| Nodes | GPUs | Config file                             |
| ----- | ---- | --------------------------------------- |
| 2     | 16   | `deepspeed_zero3_multinode_2x8gpu.yaml` |
| 3     | 24   | `deepspeed_zero3_multinode_3x8gpu.yaml` |
| 4     | 32   | `deepspeed_zero3_multinode_4x8gpu.yaml` |
| 5     | 40   | `deepspeed_zero3_multinode_5x8gpu.yaml` |

The CLI flags `--num_machines`, `--num_processes`, and `--machine_rank` override the config file values at runtime, so the config is mainly needed for DeepSpeed-specific settings.

______________________________________________________________________

## How It Works

### SLURM + Accelerate integration

1. **`sbatch`** submits the job. SLURM allocates nodes and starts the batch script on the first node.
2. The batch script resolves the main node IP and prepares environment.
3. **`srun --ntasks-per-node=1`** launches one task per allocated node.
4. Each task runs **`accelerate launch`** with:
   - `--machine_rank $SLURM_PROCID` (0 for first node, 1 for second, etc.)
   - `--main_process_ip` pointing to node 0
   - `--num_machines` and `--num_processes` matching the allocation
5. **Accelerate** spawns 8 GPU processes per node via `torch.distributed`.
6. **DeepSpeed Zero3** handles model sharding, optimizer states, and gradient aggregation across all 16 processes.

### Key SLURM variables used

| Variable             | Value               | Usage                     |
| -------------------- | ------------------- | ------------------------- |
| `SLURM_JOB_ID`       | Job number          | Log paths, identification |
| `SLURM_JOB_NODELIST` | Compact node list   | Resolve hostnames         |
| `SLURM_NNODES`       | Number of nodes     | `--num_machines`          |
| `SLURM_PROCID`       | Task ID (0, 1, ...) | `--machine_rank`          |

______________________________________________________________________

## Differences from launch_multinode.sh

| Aspect          | `launch_multinode.sh` (old cluster)   | `launch_slurm.sh` (SLURM cluster)    |
| --------------- | ------------------------------------- | ------------------------------------ |
| Node allocation | Manual (pass node names as args)      | SLURM allocates automatically        |
| Process launch  | SSH + nohup on each node              | `srun` (SLURM-managed)               |
| Code location   | `/raid` (node-local, manual git pull) | `/lambdafs` (shared, no sync needed) |
| Code sync       | `sync_code_to_nodes.sh` + git pull    | Not needed (shared FS)               |
| Cleanup         | trap + SSH pkill                      | SLURM handles it (`scancel`)         |
| Kerberos        | `krenew` wrapper needed               | Managed by cluster (if applicable)   |
| Log access      | SSH to nodes or NAS                   | SLURM output files + shared FS       |
| Job monitoring  | Manual `ps` / `nvidia-smi` via SSH    | `squeue`, `sacct`, SLURM output      |

______________________________________________________________________

## Resuming from Checkpoint

### Same nodes (not guaranteed with SLURM)

SLURM may allocate different nodes each time. If checkpoints are on `/raid/` (node-local), you need to ensure the same nodes are used:

```bash
# Request specific nodes
RESUME_FROM_RUN_DIR=/raid/checkpoints/run_20250401 \
RESUME_CHECKPOINT=checkpoint-500 \
  sbatch --nodelist=node01,node02 scripts/launch_slurm.sh
```

### Shared filesystem (recommended for resumability)

Save checkpoints to `/lambdafs/` so any nodes can resume:

```bash
# Save checkpoints to shared FS (slower writes but always resumable)
CHECKPOINT_DIR=/lambdafs/users/user/checkpoints/my_run \
  sbatch scripts/launch_slurm.sh

# Resume from shared FS (any nodes work)
RESUME_FROM_RUN_DIR=/lambdafs/users/user/checkpoints/my_run \
RESUME_CHECKPOINT=checkpoint-500 \
  sbatch scripts/launch_slurm.sh
```

### Collecting checkpoints from /raid after training

If checkpoints are on `/raid/`, collect them while the job is still running or immediately after (before nodes are reassigned):

```bash
# Check which nodes were used
sacct -j <JOBID> --format=NodeList

# Collect from those nodes (if SSH access is available)
for node in node01 node02; do
    scp -r $node:/raid/checkpoints/my_run/* /lambdafs/users/user/checkpoints/my_run/
done
```

______________________________________________________________________

## Model Weights

The experiment config (`keywords/v0_rl.yaml`) references model weights inside the repo:

```yaml
base_model: /lambdafs/users/user/new_tests/code-interp-benchmark/full_checkpoints/RL_HT_49-600
```

Since the repo is on `/lambdafs/` (shared), all nodes can read these weights directly. No copying needed.

Alternatively, use a HuggingFace Hub model (downloaded once per node to `/raid/` cache):

```yaml
base_model: Qwen/Qwen3-4B-Instruct-2507
```

______________________________________________________________________

## Troubleshooting

### Finding the right network interface

```bash
# On a compute node, list interfaces
srun --nodes=1 --ntasks=1 ip link show

# Common choices:
#   eth0 / ens5        - Ethernet
#   bond0              - Bonded Ethernet
#   ibp*s0 / ib0       - InfiniBand (if available, use for NCCL)
```

If the job hangs at initialization, the network interface is likely wrong.

### Job hangs at NCCL initialization

```bash
# Enable debug to see NCCL connection attempts
ENABLE_DEBUG=true sbatch scripts/launch_slurm.sh

# Check that nodes can reach each other on the specified interface
srun --nodes=2 --ntasks=2 bash -c 'echo "$(hostname): $(ip -4 addr show dev eth0 | grep -oP "inet \K[0-9.]+")"'
```

### "Module not found" errors

The code lives on `/lambdafs/`. If `uv` or Python can't find packages:

```bash
# Check uv is available on compute nodes
srun --nodes=1 --ntasks=1 which uv

# If uv is not in PATH, either:
# 1. Load it via modules: MODULE_LOADS="python uv" sbatch ...
# 2. Use full path: edit TRAIN_SCRIPT to use absolute path
# 3. Install uv on shared FS and add to PATH
```

### SLURM output shows nothing / job fails immediately

```bash
# Check job status and exit code
sacct -j <JOBID> --format=JobID,State,ExitCode,NodeList

# Check SLURM stderr
cat /lambdafs/users/user/logs/slurm/job_<JOBID>.err
```

### Cache/disk space issues

```bash
# Check /raid space on allocated nodes (while job is running)
srun --jobid=<JOBID> df -h /raid

# Clear old caches (requires SSH access or a separate SLURM job)
srun --nodes=1 --ntasks=1 --nodelist=node01 bash -c 'rm -rf /raid/tmp/cache'
```

______________________________________________________________________

## Useful SLURM Commands

```bash
# Submit job
sbatch scripts/launch_slurm.sh

# Check queue
squeue -u $USER

# Detailed job info (while running)
scontrol show job <JOBID>

# Cancel job
scancel <JOBID>

# Job history and exit codes
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,NodeList

# View available partitions and nodes
sinfo

# Check node status
sinfo -N -l

# Interactive session (for debugging)
salloc --nodes=1 --gpus-per-node=8 --time=1:00:00
# Then inside the session:
srun nvidia-smi
```

______________________________________________________________________

## Complete Examples

### Basic 2-node training

```bash
cd /lambdafs/users/user/new_tests/code-interp-benchmark
sbatch scripts/launch_slurm.sh
```

### Custom experiment with specific partition

```bash
TRAIN_ARGS="+experiment=keywords/v0_rl.yaml config.grpo_config.learning_rate=1e-5" \
NETWORK_INTERFACE=bond0 \
  sbatch --partition=a100 --time=24:00:00 scripts/launch_slurm.sh
```

### Resume from shared checkpoint

```bash
RESUME_FROM_RUN_DIR=/lambdafs/users/user/checkpoints/run_20250401 \
RESUME_CHECKPOINT=checkpoint-500 \
TRAIN_ARGS="+experiment=keywords/v0_rl.yaml" \
  sbatch scripts/launch_slurm.sh
```

### With debug logging

```bash
ENABLE_DEBUG=true sbatch scripts/launch_slurm.sh
```

### 4 nodes (32 GPUs)

```bash
ACCELERATE_CONFIG="pyine/configs/accelerate/deepspeed_zero3_multinode_4x8gpu.yaml" \
  sbatch --nodes=4 scripts/launch_slurm.sh
```

______________________________________________________________________

## Quick Reference

```bash
# Submit
sbatch scripts/launch_slurm.sh

# Monitor
squeue -u $USER
tail -f /lambdafs/users/user/logs/slurm/job_<JOBID>.out

# Cancel
scancel <JOBID>

# History
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed

# Override anything
TRAIN_ARGS="..." NETWORK_INTERFACE=bond0 sbatch --nodes=4 --time=72:00:00 scripts/launch_slurm.sh
```
