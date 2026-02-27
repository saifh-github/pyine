# LLM Classifier Training Guide

## What It Is

The LLM classifier trainer fine-tunes an encoder model (e.g., ModernBERT, DeBERTa-v3) as a
**binary classifier** on the same LMDB data used by the probe trainer. It predicts whether a
model's completion is correct (`label=1`) or incorrect (`label=0`).

Data is loaded as structured **message lists** (role-attributed conversations). The trainer
automatically formats messages into text: if the tokenizer defines a `chat_template` (e.g.,
chat-tuned models), it is applied; otherwise (typical for encoder models like ModernBERT,
DeBERTa), messages are concatenated into role-tagged plain text (e.g., `user: ...`, `assistant: ...`).

Unlike the probe trainer (which trains lightweight heads on frozen LLM activations), this trainer
fine-tunes the entire encoder model (or LoRA adapters) end-to-end using HuggingFace `Trainer`.

______________________________________________________________________

## Quick Start

### Single GPU

```bash
python -m pyine.apps.trainers.llm_classifier_trainer \
    +experiment=llm_classifier/v0_modernbert \
    config.datamodule_config.lmdb_path=/path/to/reward_logs.lmdb
```

### Multi-GPU

```bash
torchrun --nproc_per_node=NUM_GPUS \
    -m pyine.apps.trainers.llm_classifier_trainer \
    +experiment=llm_classifier/v0_modernbert \
    config.datamodule_config.lmdb_path=/path/to/reward_logs.lmdb
```

### Smoke Test (small model, debug data)

```bash
# 1. Generate debug LMDB
python -m pyine.guardrails.data.debug_dataset --output /tmp/probe-debug-lmdb

# 2. Train with bert-tiny
python -m pyine.apps.trainers.llm_classifier_trainer \
    +experiment=llm_classifier/v0_modernbert \
    config.datamodule_config.lmdb_path=/tmp/probe-debug-lmdb \
    config.base_model=prajjwal1/bert-tiny \
    config.max_seq_length=128
```

______________________________________________________________________

## Configuration

The main config class is `LLMClassifierTrainerAppMainConfig` in
`pyine/apps/trainers/llm_classifier_trainer_configs.py`. It inherits from `AppMainConfig` and
`ModelTokenizerConfigBase`.

### Key Config Fields

| Field                       | Default    | Description                                                |
| --------------------------- | ---------- | ---------------------------------------------------------- |
| `base_model`                | (required) | HuggingFace model ID (e.g., `answerdotai/ModernBERT-base`) |
| `datamodule_config`         | (required) | `ProbeDataModuleConfig` or `CorrectnessDataModuleConfig`   |
| `training_args_config`      | (required) | HuggingFace `TrainingArguments` wrapper                    |
| `max_seq_length`            | `3000`     | Max token length for tokenization                          |
| `num_labels`                | `2`        | Number of classification labels                            |
| `class_weight_mode`         | `"none"`   | `"none"` or `"balanced"` for inverse-frequency weighting   |
| `log_per_code_type_metrics` | `True`     | Compute per-code-type accuracy/AUROC                       |
| `truncation_side`           | `None`     | `"right"`, `"left"`, or `None` (tokenizer default)         |
| `save_model`                | `True`     | Save final model after training                            |
| `lora_config`               | `None`     | LoRA adapter config (optional)                             |

### Data Source Config (`datamodule_config`)

The trainer accepts either `ProbeDataModuleConfig` (reward LMDBs from `DiskRewardLogger`) or
`CorrectnessDataModuleConfig` (eval LMDBs from `DiskEvalLogger`).

#### Option A: Reward LMDB (`ProbeDataModuleConfig`, default)

| Field                | Default                              | Description                                        |
| -------------------- | ------------------------------------ | -------------------------------------------------- |
| `lmdb_path`          | (required)                           | Path to LMDB exported by `DiskRewardLogger`        |
| `label_metric_key`   | `reward/metrics/soft_match/is_match` | Key for binary label                               |
| `train_key_prefix`   | `train/`                             | LMDB prefix for training records                   |
| `valid_key_prefix`   | `eval/`                              | LMDB prefix for validation records                 |
| `selection_strategy` | `latest`                             | Deduplication strategy (`latest` or `best_reward`) |
| `label_balance`      | `None`                               | Optional `LabelBalanceConfig` for resampling       |
| `code_type_filter`   | `None`                               | Filter to specific code types                      |

#### Option B: Correctness LMDB (`CorrectnessDataModuleConfig`)

| Field          | Default      | Description                                                       |
| -------------- | ------------ | ----------------------------------------------------------------- |
| `lmdb_paths`   | (required)   | Path(s) or glob patterns to LMDBs exported by `DiskEvalLogger`    |
| `label_type`   | `soft_match` | Which correctness label to use (`soft_match` or `hard_match`)     |
| `split_config` | (required)   | `GuardrailSplitConfig` for building train/valid/test splits       |
| `resampling`   | `None`       | Optional `RecordResamplingConfig` applied to train and valid data |

### Training Arguments (`training_args_config`)

Standard HuggingFace `TrainingArguments`. Key defaults in `train_default`:

| Field                         | Default | Description                                         |
| ----------------------------- | ------- | --------------------------------------------------- |
| `per_device_train_batch_size` | `16`    | Training batch size per GPU                         |
| `per_device_eval_batch_size`  | `32`    | Evaluation batch size per GPU                       |
| `num_train_epochs`            | `3`     | Number of training epochs                           |
| `learning_rate`               | `2e-5`  | Learning rate                                       |
| `weight_decay`                | `0.01`  | Weight decay                                        |
| `warmup_ratio`                | `0.1`   | Warmup fraction                                     |
| `eval_strategy`               | `epoch` | Evaluate every epoch                                |
| `metric_for_best_model`       | `auroc` | Model selection metric                              |
| `save_total_limit`            | `null`  | Max checkpoints to keep on disk (`null` = keep all) |

### Checkpoint Retention

To limit disk usage from periodic checkpoints, set `save_total_limit` in
`training_args_config` (same pattern as the SFT and RL trainers):

```yaml
config:
  training_args_config:
    save_strategy: "epoch"       # Must be saving checkpoints for limit to apply
    save_total_limit: 3          # Keep at most 3 checkpoints (default: keep all)
```

HuggingFace's `Trainer` handles deletion of old `checkpoint-NNNN` directories automatically
when this limit is exceeded.

______________________________________________________________________

## YAML Examples

### Basic ModernBERT

See `pyine/configs/experiment/llm_classifier/v0_modernbert.yaml`.

### With Correctness Datamodule (records exported from the correctness eval pipeline)

```yaml
config:
  datamodule_config:
    _target_: pyine.evals.correctness.datamodule_configs.CorrectnessDataModuleConfig
    lmdb_paths: /path/to/eval_logs.lmdb
    label_type: soft_match
    split_config:
      split_source: my_dataset
      guardrail_valid_fraction: 0.5
    resampling:
      target_positive_ratio: 0.5
```

### With Probe Datamodule (records exported from the reward manager training pipeline)

```yaml
config:
  datamodule_config:
    lmdb_path: /path/to/lmdb
    label_balance:
      target_positive_ratio: 0.5
      strategy: subsample
```

### With LoRA

```yaml
config:
  base_model: answerdotai/ModernBERT-large
  lora_config:
    r: 16
    lora_alpha: 32
    target_modules: ["Wqkv"]
```

### With Class Weights

```yaml
config:
  class_weight_mode: balanced
```

______________________________________________________________________

## Metrics

Metrics computed during evaluation:

| Metric                           | Description             |
| -------------------------------- | ----------------------- |
| `eval_accuracy`                  | Classification accuracy |
| `eval_auroc`                     | Area under ROC curve    |
| `eval_f1`                        | F1 score                |
| `eval_precision`                 | Precision               |
| `eval_recall`                    | Recall                  |
| `eval_accuracy/code_type/{name}` | Per-code-type accuracy  |
| `eval_auroc/code_type/{name}`    | Per-code-type AUROC     |

AUROC returns `NaN` when only one class is present in the evaluation set.

______________________________________________________________________

## Loading Saved Models

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("path/to/checkpoint")
tokenizer = AutoTokenizer.from_pretrained("path/to/checkpoint")

inputs = tokenizer("user: prompt text\n\nassistant: completion text", return_tensors="pt", truncation=True)
outputs = model(**inputs)
probs = torch.softmax(outputs.logits, dim=-1)
predicted_class = probs.argmax(dim=-1)
```

______________________________________________________________________

## Tips

1. **Start with ModernBERT-base** — 149M params, 8K context, fast and effective.
2. **Use `max_seq_length=3000`** — fits most prompt+completion sequences.
3. **Check class distribution** — always logged before training. If imbalanced, use
   `class_weight_mode="balanced"` or `datamodule_config.label_balance` (not both).
4. **Use LoRA for larger models** — reduces memory and training time.
5. **Set `truncation_side="left"`** if completions are more informative than prompts and
   sequences are regularly truncated.
6. **Use `eval_strategy="epoch"`** — evaluates at each epoch boundary.
7. **W&B logging** — set `use_wandb_logging: true` in the config. Metrics are automatically
   logged under `eval/` prefix.
