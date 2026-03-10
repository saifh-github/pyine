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
- An LMDB database exported by `DiskRewardLogger` during RL training (see [Data Source](#data-source))
- **GPU requirements**: Single or multi-GPU. The frozen LLM is fully replicated per GPU (no sharding needed since
  it is not trained)
- Optional (performance): Flash Attention 2 support via `flash-attn` (see the project root [`README.md`](../../../README.md))

## Quick Start

### Step 1: Prepare an LMDB Data Source

The trainer reads completion records from an **LMDB database** exported by `DiskRewardLogger` during RL
training. Each record contains the prompt, model output, and reward metrics. The data module produces
structured **message lists** (role-attributed conversations) and derives binary labels from reward metrics
(e.g., `reward_metrics["reward/metrics/soft_match/is_match"]`). The trainer then formats messages into
text: if the tokenizer defines a `chat_template`, it is applied; otherwise, messages are concatenated
into role-tagged plain text (e.g., `user: ...`, `assistant: ...`).

Train/valid splits are determined by LMDB key prefixes (default: `train/` and `eval/`), or via
**eval-only mode** which reads from a single prefix and splits internally (see [Eval-Only Split Mode](#eval-only-split-mode)).

#### Using the Debug LMDB (for Testing)

A built-in synthetic LMDB factory generates mock records in the exact `DiskRewardLogger` format. This is
useful for smoke tests and verifying your setup before training on real data:

```bash
# Generate a debug LMDB
uv run python -m pyine.guardrails.data.debug_dataset --output /tmp/probe-debug-lmdb
```

The debug LMDB contains records with a learnable keyword-correlated signal, so probes can actually learn
(AUROC > 0.5) rather than just verifying the pipeline runs.

Train records use flat IDs and `code_type="original"`. Eval records are organized into **families**: each
family has a base sample ID and three code-type variants (`original`, `hinted`, `misleading`) with `/a:`
augmentation suffixes matching `TraceIdentifier` conventions.

Options:

```bash
uv run python -m pyine.guardrails.data.debug_dataset \
    --output /tmp/probe-debug-lmdb \
    --n-train 200 \           # Number of training records (default: 200)
    --n-eval-families 30 \    # Number of eval families, each produces 3 records (default: 30)
    --seed 42                 # Random seed (default: 42)
```

Or from Python:

```python
from pyine.guardrails.data.debug_dataset import create_debug_probe_lmdb, create_debug_probe_dataset

# Create a debug LMDB on disk
create_debug_probe_lmdb("/tmp/probe-debug-lmdb", n_train=200, n_eval_families=30, seed=42)

# Or get a ready-to-use DatasetDict (creates temp LMDB internally)
ds = create_debug_probe_dataset(n_train=200, n_eval_families=30, seed=42)
# ds["train"] has columns: messages, label, sample_id, code_type

# Eval-only mode: use only eval records, split internally
ds = create_debug_probe_dataset(n_eval_families=30, use_eval_only_split=True)

# With code type filtering
ds = create_debug_probe_dataset(
    n_eval_families=30,
    use_eval_only_split=True,
    code_type_filter=["original", "hinted"],
)
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

  # LMDB data source
  lmdb_path: /path/to/lmdb          # LMDB exported by DiskRewardLogger
  label_metric_key: "reward/metrics/soft_match/is_match"  # Key in reward_metrics for binary label
  train_key_prefix: "train/"         # LMDB key prefix for training records
  valid_key_prefix: "eval/"          # LMDB key prefix for validation records
  selection_strategy: latest          # Deduplication: "latest" or "best_reward"
  recompute_labels: false             # Re-derive labels from expected/predicted outputs
  skip_malformed_records: false       # Skip records missing required fields
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

  # Code type metrics (logged per-code-type during validation)
  log_per_code_type_metrics: true

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
    config.lmdb_path=/tmp/probe-debug-lmdb \
    config.num_epochs=5 \
    config.train_batch_size=8
```

## Data Source

### LMDB Format

The trainer reads from an LMDB database exported by `DiskRewardLogger` during RL training. Records are
serialized with `JSON_ZSTD` (orjson + zstandard compression).

LMDB keys follow the pattern `{key_prefix}{sample_id}/{generation_count}`, for example:

```
train/TACO/train/p000001/s0000/3
eval/debug_problem_010/s0000/t0000/1
eval/debug_problem_010/s0000/t0000/a:hints_docs:000/1    (hinted variant)
eval/debug_problem_010/s0000/t0000/a:issues_docs:000/1   (misleading variant)
```

Records sharing a base sample ID (before the `/a:` suffix) belong to the same **family** — different
code-type augmentations of the same problem.

### Record Fields

Each LMDB record is a JSON dict. The fields used by the probe trainer:

| Field             | Required  | Description                                                                                                                                               |
| ----------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompt`          | Always    | The prompt text sent to the model                                                                                                                         |
| `model_output`    | Always    | The model's completion text                                                                                                                               |
| `reward_metrics`  | Default   | Dict with metric keys (e.g., `reward/metrics/soft_match/is_match`)                                                                                        |
| `expected_output` | Recompute | Expected output (required when `recompute_labels=True`)                                                                                                   |
| `final_answer`    | Recompute | Model's final answer (falls back to `model_output`)                                                                                                       |
| `reward_total`    | Optional  | Used when `selection_strategy: best_reward`                                                                                                               |
| `code_type`       | Optional  | Code augmentation type (e.g., `"original"`, `"hinted"`, `"misleading"`). Defaults to `"unknown"` if absent. Used for filtering and per-code-type metrics. |

The data module constructs structured message lists from `prompt_messages` (preferred) or `prompt`
and `model_output` (fallback with warning). The trainer then formats messages into text using the
tokenizer's chat template if available, or role-tagged plain concatenation for encoder models.
Tokenization uses `add_special_tokens=False` since the text is already formatted.

### Label Derivation

Labels are derived from LMDB records in one of two ways:

**From stored metrics** (default, `recompute_labels: false`): The label is read from
`reward_metrics[label_metric_key]` and cast to int. Common keys: `reward/metrics/soft_match/is_match`,
`reward/metrics/hard_match/is_match`.

**Re-computed** (`recompute_labels: true`): The label is re-derived by running `compute_soft_match()` or
`compute_hard_match()` on `expected_output` vs. `final_answer`. Only supported for
`label_metric_key` in `{reward/metrics/soft_match/is_match, reward/metrics/hard_match/is_match}`.

### Deduplication

When multiple generations exist per sample (same `sample_id`, different `generation_count`), a single
record is selected per sample:

- **`latest`** (default): Highest `generation_count` wins
- **`best_reward`**: Highest `reward_total` wins (raises `ValueError` if `reward_total` is `None`)

### Train/Valid Splits

**Two-prefix mode** (default): Splits are determined by LMDB key prefixes:

- `train_key_prefix: "train/"` — keys starting with `train/` become the training set
- `valid_key_prefix: "eval/"` — keys starting with `eval/` become the validation set

**Eval-only mode** (`use_eval_only_split: true`): All data is read from a single prefix
(`eval_only_source_prefix`) and split internally. See [Eval-Only Split Mode](#eval-only-split-mode).

### Validation

The trainer validates the loaded dataset at load time:

- Checks that labels are in `{0, 1}`
- Checks that text fields are non-empty
- Warns if a split contains only one class (degenerate training)
- With `skip_malformed_records: false` (default), raises `ValueError` on any record missing required fields
- With `skip_malformed_records: true`, skips malformed records and logs a count at WARNING level

### Alternative Data Source: Correctness LMDB (`CorrectnessDataModuleConfig`)

Instead of reward LMDB datasets exported during training via the `DiskRewardLogger` class, the
probe trainer can also consume eval LMDBs exported in the code exec eval pipeline by the
`DiskEvalLogger` class.

For the common TACO workflow, the trainer registers an explicit `correctness_taco_base` preset that
pre-fills `split_config.split_source: TACO` and exposes the correctness resampling presets:

```bash
uv run python -m pyine.apps.trainers.probe_trainer \
  +config/datamodule_config=correctness_taco_base \
  +config/datamodule_config/resampling=balanced

uv run python -m pyine.apps.trainers.probe_trainer \
  +config/evals_config=correctness_taco_base \
  +config/evals_config/calibration_resampling=balanced
```

If you want to spell out the full nested config manually, use a `CorrectnessDataModuleConfig` like
this:

```yaml
config:
  datamodule_config:
    _target_: pyine.evals.correctness.datamodule_configs.CorrectnessDataModuleConfig
    lmdb_paths: /path/to/eval_logs.lmdb
    label_type: soft_match
    split_config:
      split_source: /path/to/dataset/split/file.bin
      guardrail_valid_fraction: 0.5
      # include_original_train_problems: true  # opt-in: use original train data too
    resampling:  # optional
      target_positive_ratio: 0.5
```

Key differences from the default `ProbeDataModuleConfig`:

- **`lmdb_paths`** (plural) accepts one or more LMDB paths/globs instead of a single `lmdb_path`;
- **`split_config`** controls how eval records are split into guardrail train/valid/test sets;
- **`resampling`** is applied symmetrically to both train and valid splits when set;
- **`label_type`** selects the correctness label (`soft_match` or `hard_match`) directly.

The `CorrectnessDataModule` exposes the same `get_probe_dataset()` interface as `ProbeDataModule`,
returning an HF `DatasetDict` with `messages`, `label`, `sample_id`, and `code_type` columns.

## Probe Architectures

All probes receive a tensor of per-token hidden states `(batch, seq_len, hidden_dim)` and an attention mask
`(batch, seq_len)`, and produce a scalar logit per sample `(batch, 1)`. The architectures differ in **how
they pool** across the sequence dimension.

The reference paper used to pick the architecture is **Detecting High-Stakes Interactions with Activation Probes** (https://arxiv.org/pdf/2506.10805v2)

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

## Replica Training

Replica training creates multiple copies of each probe configuration, each initialized with a different
random seed, and trains them all in parallel within a single run. This enables statistically-grounded
comparisons by showing mean and standard deviation of metrics across replicas.

Since all replicas of a given architecture-layer-hyperparameter combination share the same frozen LLM
activations (already computed in the forward pass), training 1 vs. 10 replicas adds negligible overhead.

### Enabling Replicas

Add the following to your experiment config:

```yaml
config:
  num_replicas: 5             # Number of copies per probe config (default: 1 = no replication)
  replica_base_seed: 0        # Base seed for deterministic initialization (default: 0)
  log_individual_replicas: false  # Also log per-replica scalar metrics to W&B (default: false)
```

With `num_replicas: 5` and 12 probe configs, the system creates **60 probes** internally. The W&B
dashboard shows 12 curves (one per base configuration) with aggregated mean and standard deviation.

### W&B Metrics with Replicas

When `num_replicas > 1`, metrics are aggregated across replicas:

| Metric Key                     | Description                            |
| ------------------------------ | -------------------------------------- |
| `train/{base_name}/loss/mean`  | Mean train loss across replicas        |
| `train/{base_name}/loss/std`   | Std of train loss across replicas      |
| `valid/{base_name}/loss/mean`  | Mean validation loss across replicas   |
| `valid/{base_name}/loss/std`   | Std of validation loss across replicas |
| `valid/{base_name}/loss/min`   | Min validation loss across replicas    |
| `valid/{base_name}/loss/max`   | Max validation loss across replicas    |
| `valid/{base_name}/auroc/mean` | Mean AUROC across replicas             |
| `valid/{base_name}/auroc/std`  | Std of AUROC across replicas           |
| `valid/{base_name}/auroc/min`  | Min AUROC across replicas              |
| `valid/{base_name}/auroc/max`  | Max AUROC across replicas              |

Additionally, **W&B Tables** (`train/replica_details` and `valid/replica_details`) are logged at each
step with the raw per-replica values. These tables include probe metadata columns (architecture, layer,
replica index) and support filtering, sorting, and CSV export in the W&B UI.

### Output Structure with Replicas

```
<output_dir>/probes/
  mean_L16_r0/
    final/
      probe_state_dict.pt
      probe_config.json
  mean_L16_r1/
    final/
      ...
  replica_summary.json       # Aggregated final metrics per base config
```

The `replica_summary.json` file contains aggregated loss and AUROC statistics for each base probe
configuration, along with the number of replicas.

### Notes

- `num_replicas: 1` (default) produces identical behaviour to the non-replica system.
- The standard deviation uses sample std (n-1 denominator), appropriate since replicas sample from
  the population of possible initializations.
- If some replicas produce NaN AUROC (e.g., single-class validation batch in DDP), those values are
  excluded from AUROC aggregation. If all replicas produce NaN, NaN is logged.
- W&B table size is proportional to `logging_steps` frequency and `num_replicas`. For very long runs
  with frequent logging, consider increasing `logging_steps`.

## Eval-Only Split Mode

In many scenarios, code-type variants (e.g., hinted, misleading) only exist in the evaluation dataset.
To train probes that are aware of these variants, **eval-only mode** reads all data from a single LMDB
prefix and splits it internally into train/valid sets.

### When to Use

- Your LMDB has code-type variants only under the `eval/` prefix
- You want probes to see `original`, `hinted`, and `misleading` samples during training
- You need to prevent data leakage between code-type variants of the same problem

### Configuration

```yaml
config:
  use_eval_only_split: true         # Enable eval-only mode
  eval_only_source_prefix: "eval/"  # Prefix to read from (default: "eval/")
  train_split_ratio: 0.8            # 80% train, 20% valid (default: 0.8)
  split_by_family: true             # Split by problem family ID (default: true)
```

### Family-Based Splitting

When `split_by_family: true` (default), the splitter groups records by their **family ID** — the
base sample ID with the `/a:{category}:{idx}` augmentation suffix stripped. All code-type variants
of the same problem go to the same split, preventing data leakage from shared problem structure.

For example, these three records share family ID `TACO/train/p000001/s0000/t0000`:

```
eval/TACO/train/p000001/s0000/t0000/1                    (original)
eval/TACO/train/p000001/s0000/t0000/a:hints_docs:001/1   (hinted)
eval/TACO/train/p000001/s0000/t0000/a:issues_docs:001/1  (misleading)
```

All three will land in the same split (train or valid), never across splits.

The splitter guarantees both splits have at least 1 family. Requires at least 2 families in the
source data.

When `split_by_family: false`, records are shuffled and split randomly at the individual record level
(no family grouping).

## Code Type Filtering

You can filter records by code type to train probes on specific subsets:

```yaml
config:
  code_type_filter:
    - original
    - hinted
    # - misleading  # excluded
```

Setting `code_type_filter: null` (default) includes all records. The filter applies in both
two-prefix and eval-only modes.

Records with no `code_type` field are treated as `"unknown"` for filtering purposes.

## Label Balancing

Label balancing controls the label and code-type composition of training (and optionally validation)
splits via resampling. This is useful for:

- **Studying probe sensitivity** to specific failure modes (e.g., how does performance change when
  trained on only 5% misleading-incorrect samples?)
- **Correcting class imbalance** when one label dominates the raw data

### Simple Mode: Target Positive Ratio

Controls the fraction of `label=1` samples across all code types:

```yaml
config:
  label_balance:
    target_positive_ratio: 0.5    # 50/50 balanced
    strategy: subsample           # drop excess samples (default)
    apply_to: [train]             # only balance training split (default)
```

### Group Mode: (code_type, label) Composition

Controls the fraction of each `(code_type, label)` group independently. Groups not listed are
**excluded** from the resulting split:

```yaml
config:
  label_balance:
    group_proportions:
      "original:1": 0.80          # 80% correct-original
      "misleading:0": 0.05        # 5% incorrect-misleading
      "hinted:0": 0.15            # 15% incorrect-hinted
    strategy: subsample
    apply_to: [train]
```

### Subsample vs Oversample

- **`subsample`** (default): Drops excess samples from over-represented groups. The dataset size
  decreases or stays the same. Recommended when you have enough data.
- **`oversample`**: Duplicates under-represented samples (with replacement). The dataset size
  increases or stays the same. Use when minority groups are too small to subsample from.

```yaml
config:
  label_balance:
    target_positive_ratio: 0.5
    strategy: oversample          # inflate minority class
    apply_to: [train, valid]      # balance both splits
```

### Notes

- **`apply_to`** defaults to `("train",)`. Validation is typically left unbalanced for faithful
  evaluation. Set `apply_to: [train, valid]` to balance both splits.
- **Interaction with `code_type_filter`**: Filtering happens first, then balancing is applied to
  the filtered samples.
- **Interaction with `max_samples_per_split`**: Balancing happens first, then capping is applied.
- **Determinism**: Resampling is seeded by `split_seed` for reproducibility.
- **`label_balance: null`** (default) leaves splits as-is — no resampling.

## Per-Code-Type Validation Metrics

When `log_per_code_type_metrics: true` (default), the trainer computes and logs **per-code-type
loss and AUROC** during validation. This enables comparing probe performance across different
code augmentation types (e.g., does the probe perform better on `original` vs. `hinted` samples?).

Each record carries a `code_type` column (e.g., `"original"`, `"hinted"`, `"misleading"`) that is
mapped to an integer `code_type_id` for DDP-safe gathering. After gathering predictions across GPUs,
metrics are broken down by code type.

Metrics are logged to W&B under `valid/{probe_name}/loss/code_type/{ct}` and
`valid/{probe_name}/auroc/code_type/{ct}`. See [W&B Logging](#wb-logging) for the full metrics table.

Code types with fewer than 2 samples in a validation batch are skipped (no meaningful AUROC).

## Configuration Reference

### Training Parameters

```yaml
config:
  # Model
  base_model: Qwen/Qwen3-4B-Instruct-2507  # HuggingFace model name or local path
  llm_checkpoint_path: null  # Path to a fine-tuned checkpoint (safetensors). If null, uses base_model

  # LMDB data source
  lmdb_path: /path/to/lmdb          # LMDB from DiskRewardLogger (required)
  label_metric_key: "reward/metrics/soft_match/is_match"  # Key in reward_metrics for binary label (default)
  train_key_prefix: "train/"         # LMDB key prefix for training records (default: "train/")
  valid_key_prefix: "eval/"          # LMDB key prefix for validation records (default: "eval/")
  selection_strategy: latest          # "latest" or "best_reward" for dedup (default: "latest")
  recompute_labels: false             # Re-derive labels from expected/predicted (default: false)
  max_samples_per_split: null         # Cap samples per split, null = no cap (default: null)
  skip_malformed_records: false       # Skip bad records instead of raising (default: false)
  max_seq_length: 3000                # Maximum sequence length for tokenization (default: 3000)

  # Eval-only split mode
  use_eval_only_split: false          # Use single prefix + internal split (default: false)
  eval_only_source_prefix: "eval/"    # Prefix to read in eval-only mode (default: "eval/")
  train_split_ratio: 0.8             # Fraction of data for training in eval-only mode (default: 0.8)
  split_by_family: true              # Split by family ID for leakage prevention (default: true)

  # Code type filtering and metrics
  code_type_filter: null             # null = all; or list: ["original", "hinted"] (default: null)
  log_per_code_type_metrics: true    # Log per-code-type loss/AUROC during validation (default: true)

  # Label balancing (see Label Balancing section)
  label_balance: null                # null = no resampling (default: null)
  # label_balance:
  #   target_positive_ratio: 0.5    # Simple mode: target fraction of label=1
  #   # OR group_proportions: {...}  # Group mode: per-(code_type, label) proportions
  #   strategy: subsample            # "subsample" or "oversample" (default: "subsample")
  #   apply_to: [train]              # Which splits to balance (default: [train])

  # Training loop
  num_epochs: 10                    # Number of training epochs (default: 10)
  train_batch_size: 4               # Per-GPU training batch size (default: 4)
  eval_batch_size: 8                # Per-GPU evaluation batch size (default: 8)
  gradient_accumulation_steps: 1    # Gradient accumulation steps (default: 1)
  logging_steps: 10                 # Log training metrics every N optimizer steps (default: 10)
  eval_steps: -1                    # Validate every N steps; -1 = epoch end only (default: -1)
  dataloader_num_workers: 4         # DataLoader workers (default: 4)
  save_probes: true                 # Save probe checkpoints after training (default: true)
  save_steps: -1                    # Save every N optimizer steps; -1 = end only (default: -1)
  save_total_limit: null            # Max mid-training checkpoints to keep; null = all (default: null)

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

  # Replica settings
  num_replicas: 1                   # Copies per probe config, each with different init seed (default: 1)
  replica_base_seed: 0              # Base seed for deterministic replica init (default: 0)
  log_individual_replicas: false    # Also log per-replica scalar metrics to W&B (default: false)

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

## Checkpoint Saving

### Mid-Training Checkpoints

By default, probe checkpoints are only saved at the end of training. For long runs, enable
periodic mid-training saves:

```yaml
config:
  save_probes: true
  save_steps: 100           # Save every 100 optimizer steps (-1 = end only)
  save_total_limit: 5       # Keep at most 5 mid-training checkpoints (null = keep all)
```

Checkpoints are saved under each probe's architecture folder:

```
<output_dir>/probes/
  mean_L16/
    step-0100/
      probe_state_dict.pt
      probe_config.json
    step-0200/
      ...
    final/                    # Always saved at end-of-training
      probe_state_dict.pt
      probe_config.json
  max_L16/
    step-0100/
      ...
    final/
      ...
```

### Checkpoint Retention

When `save_total_limit` is set, the trainer automatically deletes the oldest mid-training
checkpoint inside each probe folder after each save to stay within the limit. The `final/`
checkpoint is never deleted by the retention policy.

### Loading Checkpoints

To load probes from the final checkpoint:

```python
from pyine.guardrails.probes.collection import ProbeCollection

collection = ProbeCollection.load_from_checkpoint(
    checkpoint_dir=Path("<output_dir>/probes"),
    hidden_dim=model.config.hidden_size,
)
```

To load from a specific mid-training checkpoint:

```python
collection = ProbeCollection.load_from_checkpoint(
    checkpoint_dir=Path("<output_dir>/probes"),
    hidden_dim=model.config.hidden_size,
    checkpoint_name="step-0100",
)
```

## Output Structure

When `save_probes: true`, probe checkpoints are saved at the end of training:

```
<output_dir>/
  probes/
    mean_L16/
      final/
        probe_state_dict.pt       # PyTorch state_dict
        probe_config.json         # ProbeConfig as JSON
    max_L16/
      final/
        probe_state_dict.pt
        probe_config.json
    ...
```

With replicas:

```
<output_dir>/probes/
  mean_L16_r0/
    final/
      probe_state_dict.pt
      probe_config.json
  mean_L16_r1/
    final/
      ...
  replica_summary.json           # Aggregated final metrics per base config
```

### Loading Saved Probes

Saved probes can be reconstructed from their config and state dict:

```python
import json
import torch
from pyine.guardrails.probes import ProbeConfig, build_probe

# Load config
config = ProbeConfig(**json.loads(open("probes/mean_L16/final/probe_config.json").read()))

# Build probe and load weights
probe = build_probe(config)
state_dict = torch.load("probes/mean_L16/final/probe_state_dict.pt", weights_only=True)
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

### Per-Code-Type Metrics

When `log_per_code_type_metrics: true` (default), additional per-code-type breakdowns are logged
during validation:

| Metric Key                                       | Description                            |
| ------------------------------------------------ | -------------------------------------- |
| `valid/{probe_name}/loss/code_type/{code_type}`  | BCE loss for samples of this code type |
| `valid/{probe_name}/auroc/code_type/{code_type}` | AUROC for samples of this code type    |

For example, with a probe named `mean_L16` and code types `original`, `hinted`, `misleading`:

- `valid/mean_L16/loss/code_type/original`
- `valid/mean_L16/auroc/code_type/hinted`
- `valid/mean_L16/auroc/code_type/misleading`

Code types with fewer than 2 samples in a validation step are skipped. Single-class code-type
subsets produce `NaN` AUROC.

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

**`ValueError: no records match key prefix 'train/' in LMDB at ...`**

The LMDB contains no keys starting with the configured `train_key_prefix` (or `valid_key_prefix`). Verify
that the prefix matches what `DiskRewardLogger` used during export. Common prefixes are `train/` and `eval/`.

**`ValueError: record for sample_id '...' is missing model_output`** or
**`ValueError: record for sample_id '...' has neither prompt_messages nor prompt`**

An LMDB record is missing required fields. Records must have `model_output` and either
`prompt_messages` (preferred) or `prompt` (fallback). Either fix the export pipeline, or set
`skip_malformed_records: true` to skip bad records.

**`ValueError: recompute_labels=True is only supported for label_metric_key in ...`**

`recompute_labels` only works with `reward/metrics/soft_match/is_match` or `reward/metrics/hard_match/is_match`. For other metrics, use
stored labels (`recompute_labels: false`).

**`ValueError: best_reward selection requires reward_total for all records`**

When using `selection_strategy: best_reward`, every record must have a non-null `reward_total`. Either ensure
`logging.log_total=True` during RL export, or switch to `selection_strategy: latest`.

**`ValueError: Expected binary labels {0, 1}, got ...`**

The derived labels contain values other than 0 and 1. Check the `label_metric_key` or underlying data.

**`ValueError: split_by_family requires at least 2 families, got ...`**

In eval-only mode with `split_by_family: true`, the source data must contain at least 2 distinct
problem families (distinct base sample IDs). Either add more data, or set `split_by_family: false`
to split randomly.

**`ValueError: no records remain after code_type_filter=...`**

The `code_type_filter` excluded all records. Check that the specified code types actually exist in
your LMDB data. Records without a `code_type` field are treated as `"unknown"`.

**`ValueError: code_type_filter must be None ... or a non-empty list; got an empty list`**

An empty list `code_type_filter: []` is invalid. Use `null` (or omit) to include all records, or
provide a non-empty list of code types.

**`ValueError: eval_only_source_prefix must be non-empty when use_eval_only_split=True`**

When enabling eval-only mode, the source prefix cannot be an empty string.

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

- Existing experiment config: [`pyine/configs/experiment/probes/v0_probe.yaml`](../../configs/experiment/probes/v0_probe.yaml)
- RL Training Guide: [`RL_TRAINING_GUIDE.md`](./RL_TRAINING_GUIDE.md)
