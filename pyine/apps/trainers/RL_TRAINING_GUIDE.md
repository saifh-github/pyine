# RL Training Guide: GRPO with Code Execution Rewards

This guide explains how to run RL (Reinforcement Learning) training with GRPO (Group Relative Policy Optimization) for code execution prediction tasks.

## Overview

The RL training system is integrated into the main trainer framework and supports:

- **GRPO algorithm** via TRL library
- **Code execution rewards** (hard/soft matching)
- **vLLM acceleration** for fast rollouts
- **DeepSpeed** for distributed training
- **Hydra configuration** management
- **WandB logging** and experiment tracking

## Prerequisites

- Python environment with all dependencies installed (`trl`, `transformers`, `torch`, etc.)
- TACO dataset downloaded and processed
- **vLLM server required**: Multi GPUs: at least one for the vLLM server, plus separate GPU(s) for training

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

## Distributed Training with DeepSpeed

For multi-GPU training with model sharding, use DeepSpeed ZeRO Stage 3 via Accelerate. DeepSpeed provides excellent memory efficiency and is well-tested with TRL for RL training.

### Step 1: Use Pre-configured DeepSpeed Config

The repository includes a pre-configured DeepSpeed ZeRO-3 config file:

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
num_processes: 3  # Number of training GPUs (adjust based on your setup)
rdzv_backend: static
same_network: true
use_cpu: false
```

**Important:** Adjust `num_processes` to match your number of **training GPUs** (excluding vLLM GPUs).

### Step 2: Launch with DeepSpeed

```bash
# Example: 3 GPUs for training (GPUs 3,4,5), separate from vLLM (GPUs 0,1,2)
CUDA_VISIBLE_DEVICES=3,4,5 uv run accelerate launch \
    --config_file pyine/configs/accelerate/deepspeed_zero3.yaml \
    pyine/apps/trainers/hf_trainer.py \
    +experiment=my_rl_experiment
```

### When to Use DeepSpeed

- **Large models** (7B+) that don't fit on single GPU
- **Multi-GPU clusters** (3+ training GPUs available)
- **Memory constraints** requiring model sharding
- **Production training** with full model optimization

### Important Notes

- **GPU Separation**: Remember that vLLM and training must use separate GPUs!
  - Example: vLLM on GPUs 0,1,2 → Training on GPUs 3,4,5
- **Config adjustment**: Update `num_processes` in config to match your training GPU count
- **Batch size**: With distributed training, effective batch size = `per_device_train_batch_size` × `num_processes` × `gradient_accumulation_steps`

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

## Additional Resources

- TRL Documentation: https://huggingface.co/docs/trl
- GRPO Paper: https://arxiv.org/abs/2402.03300
- Experimentation Guide: `EXPERIMENTATION_GUIDE.md` (main repo)
