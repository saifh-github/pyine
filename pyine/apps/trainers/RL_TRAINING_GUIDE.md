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
  run_name: "${runtime.exp_name}"
  tags:
    - "model:${config.base_model}"
    - "rl-training"
  seed: 0
  dry_run: False

config:
  # IMPORTANT: Explicitly specify RL config class (overrides SFT default from /config: base)
  _target_: pyine.apps.trainers.rl_trainer_configs.RLTrainerAppMainConfig

  use_wandb_logging: true
  base_model: Qwen/Qwen3-4B-Instruct-2507

  # Model settings
  auto_model_config:
    use_cache: False  # disable cache during training (auto-re-enabled during evals)

  # Datamodule configuration (detailed structure required for proper dataset loading)
  datamodule_config:
    datamodule_class_path: pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModule
    datamodule_name: null
    # Prompt configuration for GRPO (critical for RL training!)
    prompt_config:
      prompt_name: code_execution
      use_chat_template: true
      include_examples: false    # Zero-shot by default for efficiency
      target_examples: null
      version: rl_tagged_answer      # Use the GRPO-optimized template
    # Default parser config with SampleBuilder
    default_dataparser_config:
      class_path: pyine.organisms.datamodules.samples.builder.SampleBuilder
      params:
        filtering_config: {}
        selection_config:
          seed: 0
          allow_db_lookups: true
          code_type_prob_map:  # Must explicitly set all types to sum to 1.0
            original: 1.0
            hinted: 0.0
            stubbed: 0.0
            obfuscated_hinted: 0.0
            obfuscated: 0.0
          samples_per_family: 1
          draw_attempts: 5
          fallback_to_orig: false
        transform_config:
          seed: 0
          transform_strategy: never
    # Parser overrides for train/valid subsets
    dataparser_config_overrides:
      train:
        filtering_config: {}
        selection_config:
          code_type_prob_map:
            original: 1.0
            hinted: 0.0
            stubbed: 0.0
            obfuscated_hinted: 0.0
            obfuscated: 0.0
          samples_per_family: 1
          draw_attempts: 5
          fallback_to_orig: true
        transform_config:
          transform_strategy: never
      valid:
        filtering_config: {}
        selection_config:
          code_type_prob_map:
            original: 1.0
            hinted: 0.0
            stubbed: 0.0
            obfuscated_hinted: 0.0
            obfuscated: 0.0
          samples_per_family: 1
          draw_attempts: 5
          fallback_to_orig: true
        transform_config:
          transform_strategy: never
    # Dataloader config
    dataloader_config_overrides:
      train:
        shuffle: true
    # Dataset settings
    keep_generated_datasets_in_memory: false
    subset_names: [train, valid, test]
    train_subset_names: [train]
    valid_subset_names: [valid]
    eval_subset_names: [train, valid]
    instantiate_parsers_at_setup: false
    use_local_dataset_cache: true
    use_tokenized_dataset_cache: true
    cache_lock_timeout_seconds: 1800.0
    message_generator_num_workers: 4
    min_samples_with_hints: 0
    min_samples_without_hints: 0

  # GRPO training arguments
  grpo_config:
    do_train: True
    do_eval: True
    do_predict: False
    # Training parameters
    per_device_train_batch_size: 1
    per_device_eval_batch_size: 4
    gradient_accumulation_steps: 8  # Effective batch size = 1 × 8 = 8
    learning_rate: 1e-5  # Lower than SFT for RL stability
    weight_decay: 0.05
    lr_scheduler_type: "cosine"
    warmup_steps: 100
    gradient_checkpointing: False
    gradient_checkpointing_kwargs:
      use_reentrant: False
    optim: "adamw_torch_fused"
    max_grad_norm: 1.0
    # Logging and evaluation
    num_train_epochs: 3
    max_steps: -1  # Use epochs instead
    logging_steps: 5
    eval_on_start: True  # Evaluate before training starts
    eval_strategy: "steps"
    eval_steps: 500
    save_strategy: "steps"
    save_steps: 500
    save_total_limit: 3
    dataloader_num_workers: 4
    dataloader_pin_memory: False
    # GRPO-specific parameters
    num_generations: 8  # Generate 8 samples per prompt for exploration
    num_generations_eval: 2  # Use fewer generations during eval
    max_completion_length: 2048  # Allow longer completions for complex code
    temperature: 0.7
    top_p: 0.9
    beta: 0.00  # KL penalty coefficient (0.00 = no KL penalty)
    use_vllm: true  # vLLM acceleration required
    vllm_server_port: 8000  # Must match your vLLM server port
    vllm_importance_sampling_correction: true

  # Reward configuration
  reward_config:
    reward_type: code_execution
    hard_match_reward: 1.0
    soft_match_reward: 0.5
    fail_reward: 0.0
    enable_soft_match: false  # Use hard matching only
    strip_whitespace: true
    expected_outputs_key: expected_output

  # Dataset caching
  cache_config:
    use_cache: true
    force_regenerate: false

  # LoRA configuration (recommended for efficient training)
  lora_config:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.1
    bias: none
    task_type: CAUSAL_LM
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

  # Evaluation settings
  evals_config:
    eval_batch_size: 24
```

### Step 2: Start vLLM Server

**IMPORTANT**: vLLM and training must use different GPUs with no overlap!

```bash
# Terminal 1: Start vLLM server (use base model, TRL handles weight syncing)
# Port must match grpo_config.vllm_server_port in your config (default: 8000)
CUDA_VISIBLE_DEVICES=0,1,2 trl vllm-serve \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --data_parallel_size 3 \
    --port 8000
```

Wait for server to start and show:

```
INFO: Uvicorn running on http://0.0.0.0:8000
```

**Note:** Make sure the `--port` argument matches `grpo_config.vllm_server_port` in your experiment config.

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
  learning_rate: 1e-5       # Lower than SFT (typically 2e-4) for RL stability
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  weight_decay: 0.05
  lr_scheduler_type: "cosine"
  warmup_steps: 100
  optim: "adamw_torch_fused"
  max_grad_norm: 1.0
  num_train_epochs: 3
  max_steps: -1  # -1 = use epochs

  # Evaluation settings
  eval_on_start: True       # Evaluate before training starts
  eval_strategy: "steps"
  eval_steps: 500

  # GRPO-specific
  num_generations: 8        # Samples per prompt (higher = better exploration)
  num_generations_eval: 2   # Fewer during eval to save compute
  max_completion_length: 2048  # Max tokens (higher for complex code)
  temperature: 0.7          # Sampling temperature
  top_p: 0.9               # Nucleus sampling
  beta: 0.00               # KL penalty coefficient (0.00 = no penalty)

  # vLLM settings (required)
  use_vllm: true
  vllm_server_port: 8000
  vllm_importance_sampling_correction: true
```

**Important Notes:**

- **`_target_` field**: Your config must include `_target_: pyine.apps.trainers.rl_trainer_configs.RLTrainerAppMainConfig` to dispatch to the RL trainer (see example config above)
- **`beta` parameter**: Controls KL divergence penalty. Set to 0.00 for no penalty (pure reward optimization), or use small values (0.01-0.05) to stay closer to the base model
- **`num_generations`**: Higher values (8+) provide better exploration but increase compute cost
- **`max_completion_length`**: Set higher (2048+) for code generation tasks to allow complete solutions

### Prompt Templates

Available in `pyine/prompts/configs/code_execution.yaml`:

- **`rl_tagged_answer`** (recommended): Optimized, zero-shot, ~50% shorter
- **`unstructured_with_3_output_types`**: More verbose with examples

Configure via the datamodule's prompt_config:

```yaml
config:
  datamodule_config:
    prompt_config:
      prompt_name: code_execution
      use_chat_template: false  # CRITICAL: GRPO needs plain text, not chat format
      include_examples: false    # Zero-shot by default
      version: rl_tagged_answer      # Use optimized template
```

**Important:** Always set `use_chat_template: false` for GRPO training, as the algorithm expects plain text prompts rather than chat-formatted messages.

## Additional Resources

- TRL Documentation: https://huggingface.co/docs/trl
- GRPO Paper: https://arxiv.org/abs/2402.03300
- Experimentation Guide: `EXPERIMENTATION_GUIDE.md` (main repo)
