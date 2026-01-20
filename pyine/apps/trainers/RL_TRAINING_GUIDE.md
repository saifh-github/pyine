# RL Training Guide: GRPO with Code Execution Rewards

This guide explains how to run RL (Reinforcement Learning) training with GRPO (Group Relative Policy
Optimization) for code execution prediction tasks.

## Overview

The RL training system is integrated into the main trainer framework and supports:

- **GRPO algorithm** via TRL library
- **Code execution rewards** using the internal reward manager package
- **vLLM acceleration** for fast rollouts
- **DeepSpeed** for distributed training
- **Hydra configuration** management
- **WandB logging** and experiment tracking

## Prerequisites

- Python environment with all dependencies installed (`trl`, `transformers`, `torch`, etc.)
- TACO dataset downloaded and processed
- **GPU requirements**: Multi-GPU setup recommended. Mode selection depends on your use case (see vLLM Configuration section below)
  - **Colocate mode** (recommended): vLLM shares GPUs with training. Improves GPU utilization by avoiding idle phases typical of on-policy RL algorithms
  - **Server mode**: Separate GPUs dedicated to vLLM server and training
- Optional (performance): Flash Attention 2 support via `flash-attn` (see the project root [`README.md`](../../../README.md))

## Quick Start

### Step 1: Create an RL Experiment Config

Create a new experiment config file in `pyine/configs/experiment/`:

```yaml
# pyine/configs/experiment/my_rl_experiment.yaml
# @package _global_

defaults:
  - override /config: base  # Use RL config base (from hf_rl_trainer_configs.py)
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
  _target_: pyine.apps.trainers.hf_rl_trainer_configs.RLTrainerAppMainConfig

  use_wandb_logging: true
  base_model: Qwen/Qwen3-4B-Instruct-2507

  # Model settings
  auto_model_config:
    use_cache: False  # disable cache during training (auto-re-enabled during evals)
    attn_implementation: "flash_attention_2"  # optional; requires flash-attn (see repo root README)

  # Datamodule configuration (detailed structure required for proper dataset loading)
  datamodule_config:
    datamodule_class_path: pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModule
    datamodule_name: null
    # Prompt configuration for GRPO (critical for RL training!)
    prompt_config:
      prompt_name: code_execution
      use_chat_template: true    # Use chat template for proper formatting
      include_examples: false    # Zero-shot by default for efficiency
      target_examples: null
      version: rl_tagged_answer      # Use the template that requests final answers in xml tags
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
    weight_decay: 0.0
    lr_scheduler_type: "cosine"
    warmup_ratio: 0.1
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
    scale_rewards: "group"  # Scale rewards by "group" (default), "batch", or "none"
    use_vllm: true  # vLLM acceleration required
    vllm_importance_sampling_correction: true
    vllm_mode: "colocate"  # "colocate" (recommended) or "server"
    # vLLM configuration for colocated mode (shares GPUs with training)
    vllm_gpu_memory_utilization: 0.2  # Control GPU memory for vLLM (default 0.3)
    vllm_max_model_length: 8072  # Context window for vLLM (optional, inferred if omitted)
    vllm_tensor_parallel_size: 1  # Tensor parallelism size (use 1 for data parallelism)
    vllm_enable_sleep_mode: False  # Offload weights during optimizer step (adds latency)
    # vLLM configuration for server mode (separate vLLM server required)
    vllm_server_port: 8000  # Must match your vLLM server port (server mode only)

  # Reward manager configuration
  reward_manager_config:
    parsing:
      mode: tags
      final_tag: final  # Should match the tag hardcoded in the code execution prompt template
      fallback_policy: none
    terms:
      - name: hard_match
        type: hard_match
        weight: 1.0
        enabled: true
        require_parsed: true
        params:
          reward_if_match: 1.0
          reward_if_no_match: 0.0
          strip_whitespace: true
    aggregation:
      strategy: weighted_sum
      clip_total_min: 0.0
    logging:
      enabled: false

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

### Step 2: Configure vLLM Mode

Choose between two vLLM modes based on your GPU setup:

#### Option A: Colocate Mode (Recommended)

**Best for**: Most multi-GPU training setups. vLLM shares GPUs with the training process.

**Key Advantage - Improved GPU Utilization**:
On-policy RL algorithms like GRPO alternate between two phases:

1. **Generation phase**: Model generates rollout samples (GPUs busy with inference)
2. **Training phase**: Model trains on collected samples (GPUs busy with gradient updates)

In **server mode**, these phases create idle time—when training GPUs are generating, the vLLM server is idle, and vice versa. In **colocate mode**, the same GPUs handle both phases, eliminating idle time and significantly improving overall GPU utilization.

**Additional Advantages**:

- Simpler setup - no manual vLLM server management
- Automatic lifecycle management by TRL
- Supports tensor parallelism for large models via `vllm_tensor_parallel_size`
- Can shard models across multiple GPUs when needed

**Configuration**: Set in your experiment config:

```yaml
grpo_config:
  use_vllm: true
  vllm_mode: "colocate"
  vllm_gpu_memory_utilization: 0.2  # Adjust based on model size (default 0.3)
  vllm_max_model_length: 8072  # Set to max(prompt_len + completion_len)
  vllm_tensor_parallel_size: 1  # Use 1 for data parallelism, or >1 for tensor parallelism
  vllm_enable_sleep_mode: False  # Enable to save memory (adds latency)
```

**No additional setup required** - proceed directly to Step 3!

#### Option B: Server Mode

**Best for**: Specialized scenarios where dedicated vLLM hardware is beneficial.

**When to use**:

- Off-policy RL algorithms where generation and training can happen simultaneously
- Scenarios where you need persistent vLLM serving across multiple training runs
- Maximum control over vLLM server configuration and resource allocation

**Note on GPU utilization**: For on-policy algorithms like GRPO, server mode results in idle GPU time
as generation and training phases alternate. Colocate mode is generally more efficient for these algorithms.

**Configuration**: Set in your experiment config:

```yaml
grpo_config:
  use_vllm: true
  vllm_mode: "server"
  vllm_server_port: 8000  # Must match your vLLM server port
```

**Setup required**: Start vLLM server before training:

**IMPORTANT**: vLLM and training must use different GPUs with no overlap!

```bash
# Terminal 1: Start vLLM server (use base model, TRL handles weight syncing)
# Port must match grpo_config.vllm_server_port in your config
CUDA_VISIBLE_DEVICES=0,1,2 trl vllm-serve \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --data_parallel_size 3 \
    --port 8000 \
    --max_model_len 4096  # adjust based on expected prompt/output lengths!
```

Wait for server to start and show:

```
INFO: Uvicorn running on http://0.0.0.0:8000
```

**Note 1:** Make sure the `--port` argument matches `grpo_config.vllm_server_port` in your
experiment config.

**Note 2:** The `--max_model_len` argument is important because it effectively sets the
"worst-case" sequence size the engine must be ready to serve, and that choice drives memory
planning and attention-kernel behavior. If you set it much larger than you actually need (or leave
it to its default, which uses the model's full context size), vLLM will reserve more KV-cache space,
have fewer usable cache blocks for a given GPU memory budget, fit fewer concurrent sequences,
and hit cache pressure or earlier swapping/evictions, which lowers throughput and adds overhead;
if you set it close to your true longest prompt+generation length, you free KV-cache capacity,
increase batching/concurrency, reduce memory waste, and typically get better, more stable generation
speed.

### Step 3: Run RL Training

#### Colocate Mode

**Multi-GPU training:**

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run accelerate launch \
    pyine/apps/trainers/hf_trainer.py \
    +experiment=my_rl_experiment
```

**With DeepSpeed (recommended for large models):**

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run accelerate launch \
    --config_file pyine/configs/accelerate/deepspeed_zero3.yaml \
    pyine/apps/trainers/hf_trainer.py \
    +experiment=my_rl_experiment
```

#### Server Mode

**IMPORTANT**: Use separate GPUs from vLLM server!

**Multi-GPU training (server mode):**

```bash
# vLLM on GPUs 0,1,2, training on GPUs 3,4,5
CUDA_VISIBLE_DEVICES=3,4,5 uv run accelerate launch \
    pyine/apps/trainers/hf_trainer.py \
    +experiment=my_rl_experiment
```

**With DeepSpeed:**

```bash
# vLLM on GPUs 0,1,2, training on GPUs 3,4,5
CUDA_VISIBLE_DEVICES=3,4,5 uv run accelerate launch \
    --config_file pyine/configs/accelerate/deepspeed_zero3.yaml \
    pyine/apps/trainers/hf_trainer.py \
    +experiment=my_rl_experiment
```

#### General Notes

The `hf_trainer.py` entry point handles both SFT and RL training. It automatically dispatches
to the correct trainer based on the config type (determined by the `_target_` field in your
experiment config).

**With custom overrides:**

```bash
uv run python -m pyine.apps.trainers.hf_trainer \
    +experiment=my_rl_experiment \
    config.grpo_config.learning_rate=5e-6 \
    config.grpo_config.num_generations=8
```

## vLLM Configuration Details

### Mode Selection

TRL supports two modes for vLLM integration:

#### Colocate Mode (`vllm_mode: "colocate"`)

**How it works**: TRL manages a vLLM engine in the same process as training. The engine shares GPU(s)
with the training model and is automatically lifecycle-managed (started/stopped with training).

**Key Benefit - Maximum GPU Utilization**: On-policy RL algorithms like GRPO alternate between generation
and training phases. In server mode, this creates idle GPU time—when training GPUs generate rollouts,
the vLLM server idles, and vice versa. Colocate mode eliminates this inefficiency by using the same
GPUs for both phases, dramatically improving overall GPU utilization and reducing training time.

**Configuration parameters**:

- `vllm_gpu_memory_utilization`: (default `0.3`) Controls GPU memory reserved for vLLM. Lower values
  (e.g., `0.2`) leave more memory for training. Adjust based on model size and available VRAM.
- `vllm_max_model_length`: (optional) Context window size. Should be at least
  `max(prompt_length) + max_completion_length`. If omitted, inferred from model config. Setting this
  explicitly helps optimize KV-cache allocation (see Note below).
- `vllm_tensor_parallel_size`: (default `1`) Tensor parallelism size for vLLM. Use `1` for data
  parallelism across GPUs (recommended for most cases). Use higher values (2, 4, etc.) to shard large
  models across multiple GPUs when needed. Note: This is independent of training parallelism—you can
  use DeepSpeed ZeRO for training while vLLM uses tensor parallelism for inference.
- `vllm_enable_sleep_mode`: (default `False`) When enabled, vLLM offloads weights and KV-cache to CPU
  during optimizer steps, reducing GPU memory usage. However, waking the engine adds host-device
  transfer latency on each generation phase.

**When to use**: Recommended default for most RL training setups. Especially good for:

- On-policy RL algorithms (GRPO, PPO, etc.) where maximizing GPU utilization is critical
- Multi-GPU training where you want to avoid the idle-phase inefficiency of server mode
- Any setup where simplicity and efficiency are priorities

**Example**:

```yaml
grpo_config:
  use_vllm: true
  vllm_mode: "colocate"
  vllm_gpu_memory_utilization: 0.2
  vllm_max_model_length: 8072
  vllm_tensor_parallel_size: 1
  vllm_enable_sleep_mode: False
```

#### Server Mode (`vllm_mode: "server"`)

**How it works**: You manually start a separate vLLM server process on dedicated GPU(s). Training
communicates with this server via HTTP. TRL automatically syncs model weights to the server during
training.

**Configuration parameters**:

- `vllm_server_port`: (default `8000`) HTTP port of your vLLM server. Must match the `--port` argument
  used when starting the server.

**When to use**:

- Off-policy RL algorithms where generation and training can happen simultaneously
- Persistent vLLM serving across multiple training runs or experiments
- Maximum control over vLLM server configuration and resource allocation

**Note on efficiency**: For on-policy algorithms like GRPO, server mode creates idle GPU time as
generation and training phases alternate. This reduces overall GPU utilization compared to colocate mode.

**GPU Separation (Critical!)**: vLLM server and training **must use different GPUs** - no overlap!

```bash
# CORRECT - No overlap
CUDA_VISIBLE_DEVICES=0,1,2    # vLLM server
CUDA_VISIBLE_DEVICES=3,4      # Training

# WRONG - GPU 2 overlap!
CUDA_VISIBLE_DEVICES=0,1,2    # vLLM server
CUDA_VISIBLE_DEVICES=2,3      # Training
```

**Example**:

```yaml
grpo_config:
  use_vllm: true
  vllm_mode: "server"
  vllm_server_port: 8000
```

### Important Note on `vllm_max_model_length` (Colocate Mode)

The `vllm_max_model_length` parameter is important because it effectively sets the "worst-case"
sequence size the vLLM engine must be ready to serve, and that choice drives memory planning and
attention-kernel behavior. If you set it much larger than you actually need (or leave it to its
default, which uses the model's full context size), vLLM will reserve more KV-cache space, have fewer
usable cache blocks for a given GPU memory budget, fit fewer concurrent sequences, and hit cache
pressure or earlier swapping/evictions, which lowers throughput and adds overhead. If you set it close
to your true longest `prompt_length + completion_length`, you free KV-cache capacity, increase
batching/concurrency, reduce memory waste, and typically get better, more stable generation speed.

## Distributed Training with DeepSpeed

For multi-GPU training with model sharding, use DeepSpeed ZeRO Stage 3 via Accelerate. DeepSpeed
provides excellent memory efficiency and is well-tested with TRL for RL training.

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

**Important:** Adjust `num_processes` to match your number of **training GPUs**:

- **Colocate mode**: Set to total number of GPUs (vLLM shares these GPUs)
- **Server mode**: Set to number of training GPUs only (excluding vLLM server GPUs)

### Step 2: Launch with DeepSpeed

#### Colocate Mode

```bash
# Example: 3 GPUs for training with colocated vLLM
CUDA_VISIBLE_DEVICES=0,1,2 uv run accelerate launch \
    --config_file pyine/configs/accelerate/deepspeed_zero3.yaml \
    pyine/apps/trainers/hf_trainer.py \
    +experiment=my_rl_experiment
```

#### Server Mode

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

- **GPU Separation (Server Mode only)**: Remember that vLLM server and training must use separate GPUs!
  - Example: vLLM on GPUs 0,1,2 -> Training on GPUs 3,4,5
- **Colocate Mode**: vLLM shares GPUs with training, so no separation needed
- **Config adjustment**: Update `num_processes` in config to match your training GPU count
- **Batch size**: With distributed training, effective batch size = `per_device_train_batch_size` × `num_processes` × `gradient_accumulation_steps`

## Configuration Reference

### Reward Configuration

The reward configuration uses the composable rewards module:

```yaml
reward_manager_config:
  # Parsing configuration for extracting final answers
  parsing:
    mode: tags
    final_tag: final  # Extract from <final>...</final> tags
    fallback_policy: none

  # Reward terms to evaluate
  terms:
    - name: hard_match
      type: hard_match  # Exact string matching
      weight: 1.0
      enabled: true
      require_parsed: true
      params:
        reward_if_match: 1.0
        reward_if_no_match: 0.0
        strip_whitespace: true

    # Add more terms for complex reward shaping:
    # - type: soft_match  # Heuristic matching with tolerance
    # - type: llm_grader  # LLM-as-a-judge evaluation

  # Aggregation strategy
  aggregation:
    strategy: weighted_sum
    clip_total_min: 0.0  # Optional clipping

  # Logging (optional, enable for reward tracking in WandB)
  logging:
    enabled: false  # set to true to enable per-sample and run-level reward logging
    scalar_log_every_n_generations: 30  # per-sample logging frequency (default: 30)
    table_row_every_n_generations: 60  # add a table row every N generations (default: 60)
    table_flush_every_n_generations: 1000  # flush table every N generations (default: 1000)
    table_max_rows: 1000  # flush table when buffer reaches N rows (default: 1000)
    category_extraction_config:  # optional, for category-wise tracking
      enabled_fields: [code_type]  # Track rewards by code_type category
```

**Train/Eval Prefix Switching:**

The RL trainer automatically adds a `RewardLoggingCallback` that switches the logging prefix between
training and evaluation phases. Aggregate statistics are logged at phase transitions:

- **At eval entry**: Train-phase stats are flushed with `train/` prefix
- **At eval end**: Eval-phase stats are flushed with `eval/` prefix
- **At train end**: Any remaining stats are flushed (handles `do_eval=False` case)

Run-level metrics logged include:

- `{prefix}/reward/run/total/mean`, `std`, `min`, `max`, `sample_count` - Total reward stats
- `{prefix}/reward/run/terms/{term}/mean`, etc. - Per-term reward stats
- `{prefix}/reward/run/categories/{category}/mean`, etc. - Per-category reward stats (if `category_extraction_config` is set)

Per-sample metrics (when `scalar_log_every_n_generations` triggers):

- `{prefix}/reward/total` - Sample reward total
- `{prefix}/reward/terms/{term}` - Per-term weighted values
- `{prefix}/reward/metrics/{term}/{metric}` - Per-term emitted metrics
- `{prefix}/parsing/{metric}` - Parsing metrics (when parsing is enabled)
- `{prefix}/categories/{category}` - Category indicators (when `category_extraction_config` is set)

Where `{prefix}` is `train` or `eval`.

Note: Per-sample scalar logging frequency is controlled by `scalar_log_every_n_generations` (default: 30). Set to 1 to log every sample, or higher values for less frequent logging.

**Category-Wise Reward Tracking:**

When `category_extraction_config` is set, the RewardManager accumulates rewards by category
(e.g., `code_type`, `predict_type`) and logs category-wise metrics after each evaluation. This
helps identify which sample categories the model struggles with:

```yaml
logging:
  category_extraction_config:
    enabled_fields: [code_type, predict_type]  # Choose which fields to use
```

This produces metrics like:

- `eval/reward/run/categories/code_type/original/mean` - Mean reward for `code_type=original` samples
- `eval/reward/run/categories/code_type/original/std` - Std deviation
- `eval/reward/run/categories/code_type/original/min` - Min reward
- `eval/reward/run/categories/code_type/original/max` - Max reward
- `eval/reward/run/categories/code_type/original/sample_count` - Number of samples in category

For details on available reward terms and configuration options, see `pyine/organisms/models/rewards/README.md`.

### GRPO Training Arguments

Key parameters from `trl.GRPOConfig`:

```yaml
grpo_config:
  # Standard training args
  learning_rate: 1e-5       # Lower than SFT (typically 2e-4) for RL stability
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  weight_decay: 0.0
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
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
  max_prompt_length: 5000   # Truncate prompts beyond this length (last-resort)
  temperature: 0.7          # Sampling temperature
  top_p: 0.9               # Nucleus sampling
  beta: 0.00               # KL penalty coefficient (0.00 = no penalty)
  scale_rewards: "group"    # Scale rewards by "group" (default), "batch", or "none"

  # vLLM settings (required)
  use_vllm: true
  vllm_importance_sampling_correction: true
  vllm_mode: "colocate"     # "colocate" (recommended) or "server"

  # Colocate mode settings (when vllm_mode: "colocate")
  vllm_gpu_memory_utilization: 0.2  # GPU memory for vLLM (default 0.3)
  vllm_max_model_length: 8072  # Context window (optional, inferred if omitted)
  vllm_tensor_parallel_size: 1  # Tensor parallelism (use 1 for data parallelism)
  vllm_enable_sleep_mode: False  # Offload weights during optimizer step

  # Server mode settings (when vllm_mode: "server")
  vllm_server_port: 8000    # Must match your vLLM server port
```

**Important Notes:**

- **`_target_` field**: Your config must include `_target_: pyine.apps.trainers.hf_rl_trainer_configs.RLTrainerAppMainConfig` to dispatch to the RL trainer (see example config above)
- **`vllm_mode`**: Choose `"colocate"` (recommended) for maximum GPU utilization in on-policy RL, or `"server"` for specialized use cases
- **`beta` parameter**: Controls KL divergence penalty. Set to 0.00 for no penalty (pure reward optimization), or use small values (0.01-0.05) to stay closer to the base model
- **`num_generations`**: Higher values (8+) provide better exploration but increase compute cost
- **`max_completion_length`**: Set higher (2048+) for code generation tasks to allow complete solutions
- **Colocate mode parameters**: Only relevant when `vllm_mode: "colocate"`
  - `vllm_gpu_memory_utilization`: Adjust based on model size and VRAM (lower = more memory for training)
  - `vllm_max_model_length`: Set to optimize KV-cache allocation (see detailed note in vLLM Configuration section)
  - `vllm_tensor_parallel_size`: Use `1` for data parallelism, or higher values (2, 4, etc.) to shard large models across GPUs
  - `vllm_enable_sleep_mode`: Enable to reduce memory usage at the cost of added latency
- **Server mode parameters**: Only relevant when `vllm_mode: "server"`. Ensure `vllm_server_port` matches your manually started vLLM server

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
      use_chat_template: true    # Use chat template for proper formatting
      include_examples: false    # Zero-shot by default
      version: rl_tagged_answer  # Use optimized template
```

## Additional Resources

- TRL Documentation: https://huggingface.co/docs/trl
- GRPO Paper: https://arxiv.org/abs/2402.03300
- Experimentation Guide: `EXPERIMENTATION_GUIDE.md` (main repo)
