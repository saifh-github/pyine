# Probe Training Guide: Lightweight Classifiers on Frozen LLM Activations

This guide explains how to train **probe classifiers** on hidden-state activations extracted from a
frozen LLM checkpoint.

## Overview

The probe training system trains lightweight binary classifiers that read internal activations from a
frozen language model. Given a dataset of text inputs with binary labels, the trainer:

- Loads a frozen LLM and extracts hidden-state activations at specified transformer layers
- Trains **N probe models simultaneously** per forward pass (different architectures, layers, hyperparameters)
- Supports **multi-GPU DDP** via `accelerate` (each GPU holds the full frozen LLM + all probes; data is sharded)
- Logs per-probe loss (train + valid) and AUROC (valid) to W&B
- Saves per-probe checkpoints (weights + config) for downstream use
- Uses **Hydra configuration** management for full composability

## Prerequisites

- Python environment with all dependencies installed (`torch`, `transformers`, `accelerate`, `scikit-learn`, etc.)
- A pre-processed HuggingFace dataset with text inputs and binary labels (see [Dataset Format](#dataset-format))
- **GPU requirements**: Single or multi-GPU. The frozen LLM is fully replicated per GPU (no sharding needed since
  it is not trained)
- Optional (performance): Flash Attention 2 support via `flash-attn` (see the project root [`README.md`](../../../README.md))

## Quick Start

### Step 1: Prepare a Dataset

The trainer expects a **HuggingFace dataset** saved to disk (via `datasets.save_to_disk()`) with `train` and
`valid` splits. Each split must have:

- A **text field**: Either `messages` (chat-template format: list of `{"role": ..., "content": ...}` dicts) or
  a plain string column
- A **label field**: Binary integers (`0` or `1`)

#### Using the Debug Dataset (for Testing)

A built-in synthetic dataset factory generates datasets in the exact format the trainer expects. This is useful
for smoke tests and verifying your setup before training on real data:

```bash
# Generate a debug dataset to disk
uv run python -m pyine.probes.debug_dataset --output /tmp/probe-debug-dataset
```

The debug dataset contains chat-format messages with a learnable keyword-correlated signal, so probes can
actually learn (AUROC > 0.5) rather than just verifying the pipeline runs.

Options:

```bash
uv run python -m pyine.probes.debug_dataset \
    --output /tmp/probe-debug-dataset \
    --n-train 200 \    # Number of training samples (default: 200)
    --n-valid 50 \     # Number of validation samples (default: 50)
    --seed 42          # Random seed (default: 42)
```

Or from Python:

```python
from pyine.probes.debug_dataset import create_debug_probe_dataset

# In-memory (e.g., for tests)
ds = create_debug_probe_dataset(n_train=200, n_valid=50, seed=42)

# Save to disk
create_debug_probe_dataset(output_path="/tmp/probe-debug-dataset")
```

### Step 2: Create an Experiment Config

Create a new experiment config file in `pyine/configs/experiment/probes/`:

```yaml
# pyine/configs/experiment/probes/my_probe_experiment.yaml
# @package _global_
#
# Usage:
#   Single GPU:  python -m pyine.apps.trainers.probe_trainer +experiment=probes/my_probe_experiment
#   Multi-GPU:   bash scripts/run_ddp.sh -- +experiment=probes/my_probe_experiment

defaults:
  - override /config: base
  - _self_

runtime:
  exp_name: probes/my_probe_experiment
  run_name: "${runtime.exp_name}"
  tags:
    - "model:${config.base_model}"
    - "probe-training"
  seed: 42
  dry_run: false

config:
  _target_: pyine.apps.trainers.probe_trainer_configs.ProbeTrainerAppMainConfig

  use_wandb_logging: true
  base_model: Qwen/Qwen3-4B-Instruct-2507
  llm_checkpoint_path: null  # Set to /path/to/checkpoint to load from a fine-tuned checkpoint

  # Dataset
  dataset_path: /path/to/probe-dataset  # HF dataset with train/valid splits
  text_field: messages      # "messages" for chat format, or a plain text column name
  label_field: label        # Binary label column (0/1)
  max_seq_length: 3000

  # Training loop
  num_epochs: 10
  train_batch_size: 4
  eval_batch_size: 8
  gradient_accumulation_steps: 1
  logging_steps: 10
  eval_steps: 50            # Mid-epoch validation every N steps (-1 = epoch end only)
  dataloader_num_workers: 4
  save_probes: true

  auto_model_config:
    use_cache: false
    attn_implementation: "flash_attention_2"  # optional; requires flash-attn

  # Probe configurations
  probe_configs:
    - name: mean_L16
      architecture: mean
      layer: 16
      learning_rate: 1e-3
    - name: max_L16
      architecture: max
      layer: 16
      learning_rate: 1e-3
```

**Important:** The `_target_` field must point to `pyine.apps.trainers.probe_trainer_configs.ProbeTrainerAppMainConfig`
to dispatch to the probe trainer.

### Step 3: Run Training

**Single GPU:**

```bash
uv run python -m pyine.apps.trainers.probe_trainer \
    +experiment=probes/my_probe_experiment
```

**Multi-GPU (via torchrun):**

```bash
bash scripts/run_ddp.sh -- +experiment=probes/my_probe_experiment
```

**Multi-GPU (via accelerate):**

```bash
uv run accelerate launch \
    pyine/apps/trainers/probe_trainer.py \
    +experiment=probes/my_probe_experiment
```

**With command-line overrides:**

```bash
uv run python -m pyine.apps.trainers.probe_trainer \
    +experiment=probes/my_probe_experiment \
    config.dataset_path=/tmp/probe-debug-dataset \
    config.num_epochs=5 \
    config.train_batch_size=8
```

## Dataset Format

### Required Structure

The dataset must be a HuggingFace `DatasetDict` saved to disk with at least `train` and `valid` splits:

```
my-dataset/
  train/
    ...
  valid/
    ...
```

### Text Field

Two formats are supported, controlled by the `text_field` config:

**Chat format** (`text_field: messages`): Each sample has a `messages` column containing a list of chat-turn
dicts. The tokenizer's chat template is applied automatically.

````python
{
    "messages": [
        {"role": "user", "content": "Analyze the following code:\n```python\ndef foo(): ...```"},
        {"role": "assistant", "content": "This function ..."},
    ],
    "label": 1,
}
````

**Plain text** (`text_field: text`): Each sample has a string column that is tokenized directly.

```python
{
    "text": "Some input text to classify.",
    "label": 0,
}
```

**Note:** When using `text_field: messages`, the tokenizer must have a chat template. Models like Qwen, Llama,
and Mistral Instruct variants include one by default.

### Label Field

The label column must contain binary integers: `0` or `1`. The trainer validates this at load time and will
raise an error if non-binary values are found. A warning is logged if only one class is present (degenerate
training).

### Validation

The trainer validates the dataset at load time:

- Checks that `text_field` and `label_field` columns exist
- Checks that labels are in `{0, 1}`
- Warns if a split contains only one class

## Probe Architectures

All probes receive a tensor of per-token hidden states `(batch, seq_len, hidden_dim)` and an attention mask
`(batch, seq_len)`, and produce a scalar logit per sample `(batch, 1)`. The architectures differ in **how
they pool** across the sequence dimension.

| Architecture     | Config name    | Description                                                              | Extra hyperparams            |
| ---------------- | -------------- | ------------------------------------------------------------------------ | ---------------------------- |
| **Mean**         | `mean`         | Masked mean pooling over all non-padding tokens, then linear head        | -                            |
| **Max**          | `max`          | Per-token scores via linear head, then masked max over the sequence      | -                            |
| **Last-token**   | `last_token`   | Hidden state at the last non-padding position, then linear head          | -                            |
| **Rolling mean** | `rolling_mean` | 1D rolling mean of per-token scores (window of size `T`), then max       | `window_size` (default: 16)  |
| **Softmax**      | `softmax`      | Temperature-scaled softmax attention over per-token scores, weighted sum | `temperature` (default: 1.0) |
| **Attention**    | `attention`    | Learned query/value attention-weighted pooling                           | `attn_dim` (default: 64)     |

### Probe Config Fields

Each entry in `probe_configs` accepts:

```yaml
- name: mean_L16          # Unique identifier (required)
  architecture: mean       # One of: mean, max, last_token, rolling_mean, softmax, attention (required)
  layer: 16                # Transformer layer index to extract activations from (required)
  learning_rate: 1e-3      # Per-probe learning rate (default: 1e-3)
  weight_decay: 0.0        # Per-probe weight decay (default: 0.0)
  # Architecture-specific:
  window_size: 16          # For rolling_mean only (default: 16)
  temperature: 1.0         # For softmax only (default: 1.0)
  attn_dim: 64             # For attention only (default: 64)
```

**Note:** `hidden_dim` is populated automatically from the model at runtime. You do not need to specify it.

### Choosing Layers

Layer indices are 0-based and refer to transformer block outputs (after both self-attention and MLP sublayers).
For a model with `N` hidden layers, valid indices are `0` to `N-1`.

A common strategy is to probe multiple layers to find where the signal is strongest:

```yaml
probe_configs:
  # Early, middle, and late layers
  - name: mean_L8
    architecture: mean
    layer: 8
    learning_rate: 1e-3
  - name: mean_L16
    architecture: mean
    layer: 16
    learning_rate: 1e-3
  - name: mean_L24
    architecture: mean
    layer: 24
    learning_rate: 1e-3
```

All probes sharing the same layer reuse activations from a single hook (no duplicate extraction).

## Multi-GPU Training

### Strategy: Full Replication with Data Parallelism

Each GPU holds:

- A **full copy** of the frozen LLM (no sharding — not training it, so DeepSpeed/FSDP add no benefit)
- A **full copy** of the ProbeCollection (tiny — typically a few thousand parameters total)

Data is sharded across GPUs via `DistributedSampler`. Each GPU processes its data shard independently.
Probe gradients are synchronized via DDP. This is efficient because:

- No cross-GPU communication for the LLM forward pass
- Only probe gradients are synchronized (tiny payload)

### Launch Commands

**Via torchrun (recommended, aligns with existing `run_ddp.sh`):**

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/run_ddp.sh -- \
    +experiment=probes/my_probe_experiment
```

**Via accelerate:**

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run accelerate launch \
    pyine/apps/trainers/probe_trainer.py \
    +experiment=probes/my_probe_experiment
```

### When Multi-GPU Helps

- **Large datasets** where single-GPU training is slow (data is sharded across GPUs)
- **Long sequences** that need large batch sizes for stable training

Multi-GPU does **not** help if the frozen LLM doesn't fit on a single GPU. In that case, tensor parallelism
would be needed (not currently supported by the probe trainer).

## Configuration Reference

### Training Parameters

```yaml
config:
  # Model
  base_model: Qwen/Qwen3-4B-Instruct-2507  # HuggingFace model name or local path
  llm_checkpoint_path: null  # Path to a fine-tuned checkpoint (safetensors). If null, uses base_model

  # Dataset
  dataset_path: /path/to/dataset   # HF dataset with train/valid splits (required)
  text_field: messages              # "messages" for chat format, or a plain text column (default: "messages")
  label_field: label                # Binary label column name (default: "label")
  max_seq_length: 3000              # Maximum sequence length for tokenization (default: 3000)

  # Training loop
  num_epochs: 10                    # Number of training epochs (default: 10)
  train_batch_size: 4               # Per-GPU training batch size (default: 4)
  eval_batch_size: 8                # Per-GPU evaluation batch size (default: 8)
  gradient_accumulation_steps: 1    # Gradient accumulation steps (default: 1)
  logging_steps: 10                 # Log training metrics every N optimizer steps (default: 10)
  eval_steps: -1                    # Validate every N steps; -1 = epoch end only (default: -1)
  dataloader_num_workers: 4         # DataLoader workers (default: 4)
  save_probes: true                 # Save probe checkpoints after training (default: true)

  # Model loading options
  auto_model_config:
    use_cache: false                # Disable KV cache during training (recommended)
    attn_implementation: "flash_attention_2"  # Optional; requires flash-attn

  # Probe definitions (required, at least one)
  probe_configs:
    - name: ...
      architecture: ...
      layer: ...
      learning_rate: ...

  # W&B logging
  use_wandb_logging: true           # Enable W&B logging (default from base config)
```

### Effective Batch Size

With multi-GPU training:

```
effective_batch_size = train_batch_size x num_gpus x gradient_accumulation_steps
```

### Evaluation Cadence

- **`eval_steps: -1`** (default): Validation runs only at the end of each epoch
- **`eval_steps: N`** (N > 0): Validation runs every N optimizer steps (mid-epoch) **and** at epoch end

Mid-epoch validation is useful for long epochs or when you want to monitor convergence more frequently.

## Output Structure

When `save_probes: true`, probe checkpoints are saved at the end of training:

```
<output_dir>/
  probes/
    mean_L8/
      probe_state_dict.pt       # PyTorch state_dict
      probe_config.json         # ProbeConfig as JSON (architecture, layer, hyperparams)
    mean_L16/
      probe_state_dict.pt
      probe_config.json
    ...
```

### Loading Saved Probes

Saved probes can be reconstructed from their config and state dict:

```python
import json
import torch
from pyine.probes import ProbeConfig, build_probe

# Load config
config = ProbeConfig(**json.loads(open("probes/mean_L16/probe_config.json").read()))

# Build probe and load weights
probe = build_probe(config)
state_dict = torch.load("probes/mean_L16/probe_state_dict.pt", weights_only=True)
probe.load_state_dict(state_dict)
probe.eval()
```

## W&B Logging

When `use_wandb_logging: true`, the following metrics are logged:

| Metric Key                 | Phase | Description                           |
| -------------------------- | ----- | ------------------------------------- |
| `train/{probe_name}/loss`  | Train | BCE loss per logging step             |
| `train/global_step`        | Train | Global optimizer step counter         |
| `train/epoch`              | Train | Current epoch                         |
| `valid/{probe_name}/loss`  | Valid | BCE loss over the full validation set |
| `valid/{probe_name}/auroc` | Valid | AUROC over the full validation set    |

Each probe's metrics are namespaced under its name (e.g., `train/mean_L16/loss`), making it easy to compare
architectures and layers in W&B dashboards.

**Note:** If the validation set contains only one class, AUROC is reported as `NaN` with a warning.

## Example: Full Sweep Config

The included `v0_probe.yaml` experiment config demonstrates a full architecture x layer sweep:

```yaml
# pyine/configs/experiment/probes/v0_probe.yaml
probe_configs:
  # Mean probes across layers
  - {name: mean_L8,  architecture: mean,  layer: 8,  learning_rate: 1e-3}
  - {name: mean_L16, architecture: mean,  layer: 16, learning_rate: 1e-3}
  - {name: mean_L24, architecture: mean,  layer: 24, learning_rate: 1e-3}

  # Max probes
  - {name: max_L8,  architecture: max,  layer: 8,  learning_rate: 1e-3}
  - {name: max_L16, architecture: max,  layer: 16, learning_rate: 1e-3}
  - {name: max_L24, architecture: max,  layer: 24, learning_rate: 1e-3}

  # Last-token probes
  - {name: last_L8,  architecture: last_token, layer: 8,  learning_rate: 1e-3}
  - {name: last_L16, architecture: last_token, layer: 16, learning_rate: 1e-3}
  - {name: last_L24, architecture: last_token, layer: 24, learning_rate: 1e-3}

  # Specialized probes at layer 16
  - {name: rolling_L16, architecture: rolling_mean, layer: 16, learning_rate: 1e-3, window_size: 32}
  - {name: softmax_L16, architecture: softmax,      layer: 16, learning_rate: 1e-3, temperature: 0.5}
  - {name: attn_L16,    architecture: attention,     layer: 16, learning_rate: 5e-4, attn_dim: 64}
```

All 12 probes train simultaneously in a single run, sharing the LLM forward pass.

## Troubleshooting

### Common Issues

**`ValueError: text_field 'messages' requires a tokenizer with a chat template`**

Your model's tokenizer does not have a built-in chat template. Either:

- Use a model with a chat template (e.g., Qwen Instruct, Llama Instruct, Mistral Instruct)
- Set `text_field` to a preformatted string column instead of `messages`

**`ValueError: Expected binary labels {0, 1}, got ...`**

Your dataset's label column contains values other than 0 and 1. Ensure labels are binary integers.

**`ValueError: Cannot resolve transformer layer N for model type ...`**

The model architecture is not recognized by the activation extractor. The extractor supports
Llama/Qwen/Mistral-family models (`model.model.layers[i]`) and falls back to GPT-2/GPT-NeoX paths.
For other architectures, the fallback chain in `ActivationExtractor._resolve_layer()` needs to be extended.

**Out of memory on single GPU**

- Reduce `train_batch_size` and `eval_batch_size`
- Reduce `max_seq_length`
- Use a smaller base model
- If the model itself doesn't fit, multi-GPU with full replication won't help (each GPU needs to hold the
  full model). Consider using a smaller model.

## Additional Resources

- Implementation plan: `PROBES_CLAUDE.md` (repository root)
- Existing experiment config: [`pyine/configs/experiment/probes/v0_probe.yaml`](../../configs/experiment/probes/v0_probe.yaml)
- RL Training Guide: [`RL_TRAINING_GUIDE.md`](./RL_TRAINING_GUIDE.md)
