# RL Training Guide: GRPO with Code Execution Rewards

This guide explains how to run RL (Reinforcement Learning) training with GRPO (Group Relative Policy Optimization) for code execution prediction tasks.

## Overview

The RL training system is integrated into the main trainer framework and supports:

- **GRPO algorithm** via TRL library
- **Code execution rewards** (hard/soft matching)
- **vLLM acceleration** for fast rollouts
- **DeepSpeed/FSDP** for distributed training
- **Hydra configuration** management
- **WandB logging** and experiment tracking

## Prerequisites

- Python environment with all dependencies installed (`trl`, `transformers`, `torch`, etc.)
- TACO dataset downloaded and processed
- **vLLM server required**: 3+ GPUs for vLLM server, plus separate GPU(s) for training
- Minimum 4 GPUs total recommended (3 for vLLM, 1+ for training)

## Quick Start

### Step 1: Create an RL Experiment Config

Create a new experiment config file in `pyine/configs/experiment/`:

```yaml
# pyine/configs/experiment/my_rl_experiment.yaml
# @package _global_

defaults:
  - override /config: base  # Use RL config base (from rl_trainer_configs.py)
  - override /config/datamodule_config: shortcuts_TACO_10s10t_v1_part1to4
  - override /config/grpo_config: train_default
  - _self_

runtime:
  exp_name: my_rl_experiment
  seed: 42
  dry_run: False

config:
  use_wandb_logging: true
  base_model: Qwen/Qwen3-4B-Instruct-2507

  # LoRA settings (recommended for efficient training)
  lora_config:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.1
    bias: none
    task_type: CAUSAL_LM
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

  # RL-specific settings
  prompt_version: grpo_minimal  # Optimized prompt template
  include_prompt_examples: false  # Zero-shot by default

  # Reward configuration
  reward_config:
    reward_type: code_execution
    hard_match_reward: 1.0
    soft_match_reward: 0.5
    fail_reward: 0.0
    enable_soft_match: false  # Use hard matching only
    strip_whitespace: true

  # GRPO training arguments
  grpo_config:
    do_train: true
    do_eval: true
    do_predict: true
    per_device_train_batch_size: 1
    per_device_eval_batch_size: 4
    gradient_accumulation_steps: 8
    learning_rate: 1e-5
    warmup_steps: 100
    logging_steps: 5
    save_steps: 500
    save_total_limit: 3
    num_train_epochs: 3
    max_steps: -1  # Use epochs instead
    eval_strategy: steps
    eval_steps: 500
    # GRPO-specific
    num_generations: 4  # Number of samples per prompt
    num_generations_eval: 2  # Fewer during eval
    max_completion_length: 512
    temperature: 0.7
    top_p: 0.9
    beta: 0.05  # KL penalty coefficient
    use_vllm: true  # vLLM acceleration required
    vllm_server_port: 8000
    gradient_checkpointing: false
    gradient_checkpointing_kwargs:
      use_reentrant: false

  # Dataset caching
  cache_config:
    use_cache: true
    force_regenerate: false

  # Evaluation settings
  evals_config:
    eval_batch_size: 24
```

### Step 2: Start vLLM Server

**IMPORTANT**: vLLM and training must use different GPUs with no overlap!

```bash
# Terminal 1: Start vLLM server (use base model, TRL handles weight syncing)
CUDA_VISIBLE_DEVICES=0,1,2 trl vllm-serve \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --data_parallel_size 3 \
    --port 8000
```

Wait for server to start and show:

```
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Run RL Training

**Terminal 2: Basic training (single GPU for training):**

```bash
# Use separate GPU(s) from vLLM server
CUDA_VISIBLE_DEVICES=3 uv run python -m pyine.apps.trainers.hf_trainer \
    +experiment=my_rl_experiment
```

Note: The `hf_trainer.py` entry point handles both SFT and RL training. It automatically dispatches to the correct trainer based on the config type (determined by the `_target_` field in your experiment config).

**With custom overrides:**

```bash
CUDA_VISIBLE_DEVICES=3 uv run python -m pyine.apps.trainers.hf_trainer \
    +experiment=my_rl_experiment \
    config.grpo_config.learning_rate=5e-6 \
    config.grpo_config.num_generations=8
```

**Distributed training with Accelerate (multiple training GPUs):**

```bash
CUDA_VISIBLE_DEVICES=3,4 uv run accelerate launch \
    pyine/apps/trainers/hf_trainer.py \
    +experiment=my_rl_experiment
```

### Step 4: Monitor Training

**With WandB (if enabled):**

- Visit your WandB project dashboard
- Monitor key metrics:
  - `train/reward_mean` - Should increase (indicates learning)
  - `train/loss` - Policy loss (should decrease)
  - `train/reward_std` - Sample diversity
  - `eval/reward_mean` - Validation reward

**Key metrics to watch:**

- **Reward Mean**: Binary reward (1.0 or 0.0), so mean = accuracy
  - Start: ~0.0-0.1 (mostly incorrect)
  - Target: >0.3-0.5 after training (30-50% accuracy)
- **Loss**: Should decrease steadily
- **Reward Std**: High initially, may stabilize

## vLLM Configuration Details

### GPU Separation (Critical!)

vLLM and training **must use different GPUs** - no overlap!

```bash
# ✓ CORRECT - No overlap
CUDA_VISIBLE_DEVICES=0,1,2    # vLLM server
CUDA_VISIBLE_DEVICES=3,4      # Training

# ✗ WRONG - GPU 2 overlap!
CUDA_VISIBLE_DEVICES=0,1,2    # vLLM server
CUDA_VISIBLE_DEVICES=2,3      # Training
```

### Benefits of vLLM

- **3-10x faster generation** during rollouts
- **2-5x faster overall training**
- Better GPU utilization
- Required for efficient GRPO training

### Known Issues: TRL + vLLM

**Issue:** TRL passes unsupported `tools` parameter to vLLM chat.

**Fix:** Edit your local TRL installation:

```bash
# Find TRL location
python -c "import trl; print(trl.__file__)"

# Edit trl/trainer/grpo_trainer.py
# Add at line ~415 in __init__ method:
self.tools = None

# OR at line ~1295 in generate_single_turn:
output = self.vllm_client.chat(
    messages=ordered_set_of_prompts,
    **sampling_params,
    tools=None,  # Add this
    chat_template=self.chat_template,
)
```

This is a temporary workaround until TRL fixes vLLM compatibility.

## Distributed Training with DeepSpeed

For multi-GPU training with model sharding, use DeepSpeed via Accelerate.

### Step 1: Create Accelerate Config

```bash
accelerate config
```

Or use a pre-configured file:

```yaml
# pyine/configs/accelerate/deepspeed_zero3.yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4  # Number of GPUs
rdzv_backend: static
same_network: true
use_cpu: false
```

### Step 2: Launch with DeepSpeed

```bash
uv run accelerate launch \
    --config_file pyine/configs/accelerate/deepspeed_zero3.yaml \
    pyine/apps/trainers/hf_trainer.py \
    +experiment=my_rl_experiment
```

### When to Use DeepSpeed

- **Large models** (7B+) that don't fit in single GPU
- **Multi-GPU clusters** (4+ GPUs)
- **Memory constraints** requiring model sharding
- **Production training** with full model optimization

## Configuration Reference

### Reward Configuration

```yaml
reward_config:
  reward_type: code_execution  # Currently only type supported
  hard_match_reward: 1.0       # Exact match reward
  soft_match_reward: 0.5       # Heuristic match reward
  fail_reward: 0.0             # No match reward
  enable_soft_match: false     # Use soft matching as fallback
  strip_whitespace: true       # Strip whitespace for hard matching
  expected_outputs_key: expected_output  # Dataset key for targets
```

### GRPO Training Arguments

Key parameters from `trl.GRPOConfig`:

```yaml
grpo_config:
  # Standard training args
  learning_rate: 1e-5
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  num_train_epochs: 3
  max_steps: -1  # -1 = use epochs

  # GRPO-specific
  num_generations: 4        # Samples per prompt (higher = better exploration)
  max_completion_length: 512  # Max tokens to generate
  temperature: 0.7          # Sampling temperature
  top_p: 0.9               # Nucleus sampling
  beta: 0.05               # KL penalty coefficient

  # vLLM settings (required)
  use_vllm: true
  vllm_server_port: 8000
  vllm_importance_sampling_correction: true
```

### Prompt Templates

Available in `pyine/prompts/configs/code_execution.yaml`:

- **`grpo_minimal`** (recommended): Optimized, zero-shot, ~50% shorter
- **`unstructured_with_3_output_types`**: More verbose with examples

Configure via:

```yaml
config:
  prompt_version: grpo_minimal
  include_prompt_examples: false  # Zero-shot
```

## Troubleshooting

### ImportError for TRL

**Error:** `ModuleNotFoundError: No module named 'trl'`

**Solution:**

```bash
uv pip install trl
```

### vLLM Server Not Accessible

**Error:** `RuntimeError: vLLM server not accessible at http://localhost:8000`

**Solution:**

1. Check server is running: `curl http://localhost:8000/health`
2. Verify no firewall blocking port 8000
3. Check server logs for errors
4. Ensure model name matches between server and config

### CUDA Out of Memory

**Solutions:**

- Reduce `per_device_train_batch_size` to 1
- Reduce `num_generations` (fewer samples per prompt)
- Reduce `max_completion_length`
- Enable gradient checkpointing
- Use DeepSpeed with ZeRO-3
- Use smaller model or LoRA

### Low Reward / Not Learning

**Solutions:**

- Check dataset quality (inspect prompts)
- Increase `num_generations` for more exploration
- Adjust `temperature` (higher = more diverse)
- Reduce `beta` (lower KL penalty)
- Check reward function is working (enable debug logging)
- Verify expected outputs are correct

## Comparison: SFT vs RL Training

| Aspect             | SFT (Supervised)         | RL (GRPO)                        |
| ------------------ | ------------------------ | -------------------------------- |
| **Config**         | `HFTrainerAppMainConfig` | `RLTrainerAppMainConfig`         |
| **Training args**  | `training_args_config`   | `grpo_config`                    |
| **Main parameter** | `do_train`               | `do_train`                       |
| **Entry point**    | Same: `hf_trainer.py`    | Same: `hf_trainer.py`            |
| **Data format**    | Chat messages            | Chat messages + expected outputs |
| **Reward**         | Loss-based               | Custom reward function           |
| **vLLM**           | For eval only            | Required for training rollouts   |

## Next Steps

1. **Create your experiment config** based on the template above
2. **Set up vLLM server** on dedicated GPUs (see Step 2 in Quick Start)
3. **Test on small dataset** to verify setup (consider adding a max_samples limit for testing)
4. **Scale up** to full dataset once validated
5. **Monitor metrics** and iterate on hyperparameters

## Additional Resources

- TRL Documentation: https://huggingface.co/docs/trl
- GRPO Paper: https://arxiv.org/abs/2402.03300
- Integration Proposal: `pyine/apps/rl_trainers_proto/claude_integration_proposal.md`
- Experimentation Guide: `EXPERIMENTATION_GUIDE.md` (main repo)

## Getting Help

For issues or questions:

- Check this guide's troubleshooting section
- Review experiment logs in `<PYINE_LOGS_ROOT>/runs/`
- Check WandB dashboard for metrics
- Inspect dataset with provided tools
- Review the integration proposal documentation
