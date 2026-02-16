# Probe Dataset Reworking — Eval-Only Mode with Code Type Awareness

## 1. Goal

Rework the probe training dataset pipeline to support training on **evaluation data only**, with visibility into the **code type** (original, hinted, misleading) of each sample. This enables training probes that distinguish model behavior across code augmentation types — a capability that requires samples from all three categories, which are only present in the eval split of RL training runs.

### What changes

| Aspect | Current | New |
|---|---|---|
| **Data source** | Separate `train/` and `eval/` LMDB prefixes for train/valid splits | Single `eval/` prefix, internally split into train/valid |
| **Code type visibility** | `code_type` field exists in LMDB records but is discarded during loading | `code_type` preserved as a dataset column and used for splitting, filtering, and per-category metrics |
| **Train/valid split** | Determined entirely by LMDB key prefix | Created from a single prefix via configurable splitting strategy (random or family-based) |
| **Validation metrics** | Aggregate loss + AUROC only | Per-code-type loss + AUROC breakdown alongside aggregates |
| **Backward compatibility** | N/A | Default behavior (`use_eval_only_split: false`) is identical to current behavior |

______________________________________________________________________

## 2. Background: Code Types in LMDB Records

### 2.1 Code Type Values

The `code_type` field in each LMDB record is a string representation of `SampleCodeTypeSet`, produced by `"_".join(sorted(self.types))`. The relevant values for this feature are:

| `code_type` string | Description |
|---|---|
| `"original"` | Unmodified code from the source dataset |
| `"hinted"` | Code with helpful execution output hints |
| `"misleading"` | Code with **misleading** execution hints |
| `"obfuscated"` | Obfuscated variable/function names |
| `"hinted_obfuscated"` | Obfuscated code with helpful hints |
| `"misleading_obfuscated"` | Obfuscated code with misleading hints |
| `"stubbed"` | Part of code hidden/stubbed |

For the primary use case (studying the effect of hints on model behavior), the three categories of interest are `"original"`, `"hinted"`, and `"misleading"` (and optionally their obfuscated variants).

### 2.2 Storage in LMDB

The `code_type` is stored in two places in each record:

1. **`code_type` field** (string): Directly written by `DiskRewardLogger.log_sample()`. This is the primary source.
2. **`tags` field** (list of strings): Contains `augment:` prefixed entries (e.g., `"augment:hinted"`, `"augment:obfuscated"`). Can be parsed via `SampleCodeTypeSet.create_from_tags()`.

Both are reliably populated for all eval records in production RL runs.

### 2.3 Sample ID Structure and Family Grouping

LMDB keys follow: `{key_prefix}{sample_id}/{generation_count}`

The `sample_id` is a string representation of a `TraceIdentifier`:

```
{dataset}/{subset}/p{problem_idx:06d}/s{solution_idx:04d}/t{test_idx:04d}
```

For augmented traces, an augmentation suffix is appended:

```
{dataset}/{subset}/p{problem_idx:06d}/s{solution_idx:04d}/t{test_idx:04d}/a:{augment_category}:{augment_idx:03d}
```

**Examples for the same underlying problem:**

| Variant | sample_id |
|---|---|
| Original | `TACO/train/p000001/s0000/t0000` |
| Hinted | `TACO/train/p000001/s0000/t0000/a:hints_docs:001` |
| Misleading | `TACO/train/p000001/s0000/t0000/a:issues_docs:001` |
| Obfuscated | `TACO/train/p000001/s0000/t0000/a:obfuscated:001` |

The **family identifier** is the augmentless prefix: `TACO/train/p000001/s0000/t0000`. All variants of the same problem share this prefix. It can be extracted by stripping the `/a:...` suffix (if present) from the sample_id.

This is important for preventing data leakage when splitting into train/valid (see Section 5.2).

______________________________________________________________________

## 3. User-Facing Interface

### 3.1 New Config Fields on `ProbeTrainerAppMainConfig`

```yaml
config:
  # Existing fields (unchanged)
  lmdb_path: /path/to/lmdb
  label_metric_key: "soft_match/is_match"
  selection_strategy: latest
  # ...

  # NEW — Eval-only split mode
  use_eval_only_split: false          # Default: false (current behavior, two-prefix mode)
  eval_only_source_prefix: "eval/"    # Which LMDB prefix to read when use_eval_only_split=true
  train_split_ratio: 0.8             # Fraction of data used for training (default: 0.8)
  split_by_family: true              # Split by family (problem) ID to prevent data leakage (default: true)

  # NEW — Code type filtering and visibility
  code_type_filter: null             # null = include all; or a list like ["original", "hinted", "misleading"]
  log_per_code_type_metrics: true    # Log per-code-type validation metrics to W&B (default: true)
```

### 3.2 Mode Summary

| Config | Behavior |
|---|---|
| `use_eval_only_split: false` (default) | **Current behavior.** Train from `train_key_prefix`, validate from `valid_key_prefix`. Two separate LMDB prefixes. `code_type` is still extracted and preserved for metrics, but splitting is prefix-based. |
| `use_eval_only_split: true` | **New mode.** Read only from `eval_only_source_prefix`. Split internally into train/valid using `train_split_ratio`. Split strategy controlled by `split_by_family`. |

### 3.3 Code Type Filtering

When `code_type_filter` is set (e.g., `["original", "hinted", "misleading"]`), only records matching one of the listed code types are included. This allows focusing on specific augmentation categories.

**Semantics:**

- Records with `code_type = null` (mapped to `"unknown"`) are excluded unless `"unknown"` is explicitly listed in the filter.
- An empty list `code_type_filter: []` is rejected at config validation time with a clear error (use `null` to include all records instead).
- The filter applies in **both modes** (prefix-based and eval-only).

### 3.4 Backward Compatibility

- `use_eval_only_split: false` (default): No behavior change whatsoever. The `eval_only_source_prefix`, `train_split_ratio`, and `split_by_family` fields are ignored.
- `code_type_filter: null` (default): All records included (same as current).
- `log_per_code_type_metrics: true` (default): This is new behavior but purely additive (extra W&B metrics). It does not affect training.
- Existing experiment YAML files continue to work without changes.

### 3.5 Example Config: Eval-Only with Hints

```yaml
config:
  lmdb_path: /path/to/rl-run/reward_logs.lmdb
  label_metric_key: "soft_match/is_match"

  # Use eval-only mode
  use_eval_only_split: true
  eval_only_source_prefix: "eval/"
  train_split_ratio: 0.8
  split_by_family: true

  # Only include original, hinted, and misleading samples
  code_type_filter:
    - original
    - hinted
    - misleading

  # Per-code-type validation metrics
  log_per_code_type_metrics: true

  # ... rest of config unchanged
```

______________________________________________________________________

## 4. Design Decisions

### 4.1 Eval-Only Mode vs. Merging Train + Eval

**Decision: Eval-only mode (single prefix, internal split).**

An alternative would be to merge records from both `train/` and `eval/` prefixes. However:

- The train prefix contains only original code samples (no hinted/misleading variants), so merging adds nothing for the hint-analysis use case.
- The eval prefix has a controlled, known distribution of code types (configured via `code_type_prob_map` in the RL datamodule).
- Keeping data from a single source simplifies provenance and avoids confounding differences in data distribution between train and eval phases.

If future needs require merging, this can be added as a separate mode later.

### 4.2 Family-Based Splitting

**Decision: Split by family ID (default), not by individual record.**

Hinted, misleading, and original samples for the same underlying problem share the same code, inputs, and expected outputs — they differ only in the hint annotations embedded in the code. If the same problem appears in train (as original) and in valid (as hinted), the probe could learn to recognize the specific problem rather than the effect of hints.

**Family ID** is derived from the sample_id by stripping the `/a:{augment_category}:{augment_idx}` suffix. All variants of problem `TACO/train/p000001/s0000/t0000` share the same family ID.

When `split_by_family: true`:
1. Group all records by family ID
2. Shuffle family IDs (deterministically, using the runtime seed)
3. Assign first `train_split_ratio` fraction of families to train, rest to valid
4. All records belonging to a family go to the same split

When `split_by_family: false`:
- Simple random split at the record level (useful if family-based splitting produces imbalanced code-type distributions in the smaller valid split)

### 4.3 Code Type as a Dataset Column

**Decision: Always extract and preserve `code_type` in the HF dataset.**

The `code_type` is extracted from every record and included as a column in the HF dataset, regardless of mode. This enables:

- Per-code-type validation metrics (Section 7)
- Downstream analysis and filtering
- No additional cost (the field is already in the record)

The column value is the raw `code_type` string from the LMDB record (e.g., `"original"`, `"hinted"`, `"misleading"`). Records with `code_type = None` get `"unknown"` as the value.

### 4.4 Where the Split Happens

**Decision: Inside `load_probe_dataset_from_lmdb()`**, matching the existing pattern where this function returns a `DatasetDict` with `"train"` and `"valid"` keys.

The function already handles loading, deduplication, sample conversion, and validation. Adding the split logic here keeps the `probe_train()` function unchanged — it still receives a `DatasetDict` with two splits. The only new information flowing downstream is the `code_type` column.

### 4.5 Per-Code-Type Metrics in Validation

**Decision: Compute per-code-type AUROC and loss during validation, log to W&B.**

When `log_per_code_type_metrics: true`, the validation loop groups predictions by `code_type` and computes separate metrics:

| Metric Key | Description |
|---|---|
| `valid/{probe_name}/loss` | Overall loss (unchanged) |
| `valid/{probe_name}/auroc` | Overall AUROC (unchanged) |
| `valid/{probe_name}/loss/code_type/{ct}` | Loss for samples with `code_type == ct` |
| `valid/{probe_name}/auroc/code_type/{ct}` | AUROC for samples with `code_type == ct` |

This provides direct visibility into whether the probe's classification accuracy varies across code types (e.g., does accuracy drop for misleading-hint samples?).

When replicas are active, per-code-type metrics follow the same aggregation pattern (mean/std across replicas).

______________________________________________________________________

## 5. Module Design

### 5.1 Changes to `pyine/probes/lmdb_dataset.py`

#### 5.1.1 Updated `_record_to_probe_sample()`

Add `code_type` extraction to each sample:

```python
def _record_to_probe_sample(
    record: dict[str, Any],
    sample_id: str,
    label_metric_key: str,
    recompute_labels: bool,
    compare_options: CompareOptions | None,
    skip_malformed: bool,
) -> dict[str, str | int] | None:
    # ... existing prompt/model_output/label extraction (unchanged) ...

    # NEW: Extract code_type
    code_type = record.get("code_type")
    if code_type is None:
        code_type = "unknown"

    return {"text": text, "label": label, "sample_id": sample_id, "code_type": code_type}
```

#### 5.1.2 New helper: `_extract_family_id()`

```python
def _extract_family_id(sample_id: str) -> str:
    """Extract the family (augmentless) identifier from a sample_id.

    Strips the '/a:{category}:{idx}' augmentation suffix if present.
    All code-type variants of the same problem share the same family ID.

    Examples:
        "TACO/train/p000001/s0000/t0000"                    → "TACO/train/p000001/s0000/t0000"
        "TACO/train/p000001/s0000/t0000/a:hints_docs:001"   → "TACO/train/p000001/s0000/t0000"
        "TACO/train/p000001/s0000/t0000/a:issues_docs:001"  → "TACO/train/p000001/s0000/t0000"
    """
    augment_marker = "/a:"
    idx = sample_id.rfind(augment_marker)
    if idx == -1:
        return sample_id
    return sample_id[:idx]
```

This is a simple string operation — no need to import or instantiate `TraceIdentifier` (which would add a heavy dependency chain). The `/a:` marker is stable across the codebase (validated by `TraceIdentifier.__repr__()` and `from_string()`). We use `rfind` (not `find`) to strip only the **last** `/a:` segment, guarding against hypothetical false positives if `/a:` ever appeared in an earlier path segment.

#### 5.1.3 New helper: `_split_records_by_family()`

```python
def _split_records_by_family(
    samples: list[dict[str, str | int]],
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
    """Split samples into train/valid by family ID.

    All samples sharing a family ID go to the same split.
    Families are shuffled deterministically before splitting.

    Rounding rule: ``n_train = max(1, int(n_families * train_ratio))``,
    ``n_valid = n_families - n_train``. Both splits are guaranteed at
    least 1 family. Raises ``ValueError`` (including the actual family
    count) if fewer than 2 families exist (cannot produce two non-empty
    splits).

    Args:
        samples: List of sample dicts (must have "sample_id" key).
        train_ratio: Fraction of families assigned to train.
        seed: Random seed for deterministic shuffling.

    Returns:
        (train_samples, valid_samples) tuple.

    Raises:
        ValueError: If fewer than 2 families exist.
    """
    ...
```

#### 5.1.4 New helper: `_filter_by_code_type()`

```python
def _filter_by_code_type(
    records: list[tuple[str, dict[str, Any]]],
    code_type_filter: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    """Filter LMDB records to include only specified code types.

    Args:
        records: List of (sample_id, record) tuples.
        code_type_filter: List of allowed code_type strings.

    Returns:
        Filtered list of (sample_id, record) tuples.
    """
    allowed = set(code_type_filter)
    return [(sid, rec) for sid, rec in records if rec.get("code_type") in allowed]
```

#### 5.1.5 Updated `load_probe_dataset_from_lmdb()` Signature

```python
def load_probe_dataset_from_lmdb(
    lmdb_path: str | Path,
    label_metric_key: str = "soft_match/is_match",
    train_key_prefix: str = "train/",
    valid_key_prefix: str = "eval/",
    selection_strategy: str = "latest",
    recompute_labels: bool = False,
    compare_options: CompareOptions | None = None,
    max_samples_per_split: int | None = None,
    skip_malformed_records: bool = False,
    seed: int = 42,
    # NEW parameters:
    use_eval_only_split: bool = False,
    eval_only_source_prefix: str = "eval/",
    train_split_ratio: float = 0.8,
    split_by_family: bool = True,
    code_type_filter: list[str] | None = None,
) -> datasets.DatasetDict:
```

#### 5.1.6 Updated Loading Logic

When `use_eval_only_split=False` (default):

1. Load records from `train_key_prefix` and `valid_key_prefix` separately (current behavior)
2. Apply `code_type_filter` to each if set
3. Convert records to samples (now includes `code_type` column)
4. Validate and return `DatasetDict`

When `use_eval_only_split=True`:

1. Load all records from `eval_only_source_prefix`
2. Apply `code_type_filter` if set
3. Convert records to samples (includes `code_type` column)
4. Split into train/valid:
   - If `split_by_family=True`: group by family ID, shuffle families, split
   - If `split_by_family=False`: random shuffle and split
5. Apply `max_samples_per_split` if set (after splitting, to each split independently)
6. Validate each split and return `DatasetDict`

```python
    if use_eval_only_split:
        with LMDBReader(lmdb_path) as reader:
            records = _load_lmdb_records(reader, eval_only_source_prefix, selection_strategy)

        if not records:
            raise ValueError(
                f"no records match key prefix '{eval_only_source_prefix}' in LMDB at {lmdb_path}"
            )

        if code_type_filter is not None:
            records = _filter_by_code_type(records, code_type_filter)
            if not records:
                raise ValueError(
                    f"no records remain after code_type_filter={code_type_filter}"
                )

        # Convert to samples
        samples, skipped = _convert_records_to_samples(
            records, label_metric_key, recompute_labels, compare_options, skip_malformed_records,
        )

        # Split
        if split_by_family:
            train_samples, valid_samples = _split_records_by_family(samples, train_split_ratio, seed)
        else:
            train_samples, valid_samples = _split_records_random(samples, train_split_ratio, seed)

        # ... max_samples_per_split, validation, return DatasetDict ...
```

### 5.2 Changes to `pyine/apps/trainers/probe_trainer_configs.py`

Add new fields to `ProbeTrainerAppMainConfig`:

```python
class ProbeTrainerAppMainConfig(common.AppMainConfig, common.ModelTokenizerConfigBase):
    # ... existing fields ...

    # --- NEW: Eval-only split mode ---
    use_eval_only_split: bool = pydantic.Field(
        default=False,
        description=(
            "When True, read data from a single LMDB prefix (eval_only_source_prefix) "
            "and split internally into train/valid. When False (default), use separate "
            "train_key_prefix and valid_key_prefix as before."
        ),
    )
    eval_only_source_prefix: str = pydantic.Field(
        default="eval/",
        description="LMDB key prefix to read from when use_eval_only_split=True.",
    )
    train_split_ratio: float = pydantic.Field(
        default=0.8,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of data used for training when use_eval_only_split=True. "
            "Remainder is used for validation."
        ),
    )
    split_by_family: bool = pydantic.Field(
        default=True,
        description=(
            "When True (default), split by family (problem) ID so that all code-type "
            "variants of the same problem go to the same split. Prevents data leakage "
            "from shared problem structure. When False, split randomly at the record level."
        ),
    )

    # --- NEW: Code type filtering and metrics ---
    code_type_filter: list[str] | None = pydantic.Field(
        default=None,
        description=(
            "If set, only include records with code_type matching one of the listed values. "
            "Example: ['original', 'hinted', 'misleading']. "
            "None (default) includes all records."
        ),
    )
    log_per_code_type_metrics: bool = pydantic.Field(
        default=True,
        description=(
            "Log per-code-type validation metrics (loss, AUROC) to W&B. "
            "Requires code_type column in the dataset (always present after this change)."
        ),
    )
```

Add a model validator for consistency:

```python
    @pydantic.model_validator(mode="after")
    def _validate_eval_only_split_config(self) -> ProbeTrainerAppMainConfig:
        if self.use_eval_only_split:
            if not self.eval_only_source_prefix:
                raise ValueError("eval_only_source_prefix must be non-empty when use_eval_only_split=True")
        if self.code_type_filter is not None and len(self.code_type_filter) == 0:
            raise ValueError(
                "code_type_filter must be None (include all) or a non-empty list; "
                "got an empty list"
            )
        return self
```

### 5.3 Changes to `pyine/apps/trainers/probe_trainer.py`

#### 5.3.1 Updated `probe_train()` — Dataset Loading

Update the dataset loading section to pass the new parameters:

```python
    raw_ds = load_probe_dataset_from_lmdb(
        lmdb_path=config.lmdb_path,
        label_metric_key=config.label_metric_key,
        train_key_prefix=config.train_key_prefix,
        valid_key_prefix=config.valid_key_prefix,
        selection_strategy=config.selection_strategy,
        recompute_labels=config.recompute_labels,
        max_samples_per_split=config.max_samples_per_split,
        skip_malformed_records=config.skip_malformed_records,
        # NEW parameters:
        use_eval_only_split=config.use_eval_only_split,
        eval_only_source_prefix=config.eval_only_source_prefix,
        train_split_ratio=config.train_split_ratio,
        split_by_family=config.split_by_family,
        code_type_filter=config.code_type_filter,
    )
```

Log code type distribution after loading:

```python
    if accelerator.is_main_process:
        for split_name in ["train", "valid"]:
            code_types = raw_ds[split_name]["code_type"]
            ct_counts = {}
            for ct in code_types:
                ct_counts[ct] = ct_counts.get(ct, 0) + 1
            logger.info(f"  {split_name} code_type distribution: {ct_counts}")
```

#### 5.3.2 Updated `_tokenize_split()` — Preserve `code_type` as Integer ID

The `code_type` must survive tokenization and be **gatherable via `accelerator.gather_for_metrics()`** in DDP mode. Strings cannot be gathered across GPU ranks, so we map code types to integer IDs during tokenization.

A `code_type_to_id` mapping (e.g., `{"original": 0, "hinted": 1, "misleading": 2, "unknown": 3}`) is built once from the full dataset's unique code types and passed to `_tokenize_split()`:

```python
def _tokenize_split(
    dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seq_length: int,
    code_type_to_id: dict[str, int],
) -> datasets.Dataset:
    def _tokenize(examples):
        tokenized = tokenizer(
            examples["text"],
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )
        tokenized["labels"] = examples["label"]
        # NEW: map code_type string → integer ID for DDP gathering
        tokenized["code_type_id"] = [code_type_to_id[ct] for ct in examples["code_type"]]
        return tokenized

    ds = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=[
            c for c in dataset.column_names
            if c not in ("input_ids", "attention_mask", "labels", "code_type_id")
        ],
    )
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels", "code_type_id"])
    return ds
```

**Why integer IDs instead of string column:** In DDP mode, `accelerator.gather_for_metrics()` gathers tensors across ranks. Since `DistributedSampler` shards and potentially reorders data, a pre-extracted string list would become misaligned with gathered predictions. By making `code_type_id` a tensor column in the batch, it is gathered alongside labels and logits, guaranteeing alignment regardless of DDP sharding.

#### 5.3.3 Updated `validate_probes()` — Per-Code-Type Metrics

The validation function needs access to `code_type` per sample to compute per-category metrics.

**Approach:** Collect `code_type_id` from each batch alongside labels, then gather it via `accelerator.gather_for_metrics()` — exactly like labels and logits. This guarantees alignment in DDP mode regardless of how `DistributedSampler` shards or reorders data.

```python
def validate_probes(
    probe_collection: ProbeCollection,
    model: torch.nn.Module,
    extractor: ActivationExtractor,
    valid_loader: DataLoader,
    loss_fn: torch.nn.Module,
    global_step: int,
    accelerator: Accelerator,
    runtime: RuntimeConfig | None,
    *,
    expanded_configs_by_name: dict[str, ProbeConfig] | None = None,
    log_individual_replicas: bool = False,
    # NEW parameters:
    id_to_code_type: dict[int, str] | None = None,
    log_per_code_type_metrics: bool = False,
) -> dict[str, dict[str, float]]:
```

The `id_to_code_type` parameter maps integer IDs back to code type strings for metric logging.

Inside the validation loop, collect `code_type_id` from each batch:

```python
    all_code_type_ids: list[torch.Tensor] = []

    for batch in valid_loader:
        # ... existing forward pass and logit collection ...
        if log_per_code_type_metrics:
            all_code_type_ids.append(batch["code_type_id"])
```

After gathering all predictions on the main process:

```python
        # Gather code_type_ids alongside labels and logits
        if log_per_code_type_metrics and id_to_code_type is not None:
            gathered_ct_ids = accelerator.gather_for_metrics(
                torch.cat(all_code_type_ids)
            ).cpu()

            unique_ct_ids = gathered_ct_ids.unique().tolist()
            for ct_id in unique_ct_ids:
                ct = id_to_code_type[int(ct_id)]
                ct_mask = (gathered_ct_ids == ct_id).nonzero(as_tuple=True)[0]
                if len(ct_mask) < 2:
                    continue  # Skip code types with too few samples for AUROC

                for name in probes_dict:
                    ct_logits = logits_cpu[ct_mask]
                    ct_labels = labels_cpu[ct_mask]

                    ct_loss = loss_fn(ct_logits, ct_labels.float()).item()
                    ct_probs = torch.sigmoid(ct_logits).numpy()
                    ct_labels_np = ct_labels.numpy()

                    ct_unique = set(int(v) for v in ct_labels_np.tolist())
                    if len(ct_unique) < 2:
                        ct_auroc = float("nan")
                    else:
                        ct_auroc = float(sklearn.metrics.roc_auc_score(ct_labels_np, ct_probs))

                    if runtime and runtime.wandb_run:
                        runtime.wandb_run.log({
                            f"valid/{name}/loss/code_type/{ct}": ct_loss,
                            f"valid/{name}/auroc/code_type/{ct}": ct_auroc,
                        }, step=global_step)
```

**DDP correctness:** Since `code_type_id` is a tensor column in the batch, it flows through `DistributedSampler` sharding and `gather_for_metrics()` gathering **identically** to labels and logits. The gathered `code_type_id` tensor is positionally aligned with gathered labels and logits by construction — no ordering assumptions needed.

#### 5.3.4 Building the `code_type` Mapping in `probe_train()`

Before tokenization, build the `code_type_to_id` mapping from all unique code types across both splits. This ensures a consistent mapping for both train and valid datasets:

```python
    # Build code_type → integer ID mapping (consistent across splits)
    all_code_types = sorted(
        set(raw_ds["train"]["code_type"]) | set(raw_ds["valid"]["code_type"])
    )
    code_type_to_id: dict[str, int] = {ct: i for i, ct in enumerate(all_code_types)}
    id_to_code_type: dict[int, str] = {i: ct for ct, i in code_type_to_id.items()}

    # Pass mapping to _tokenize_split()
    train_ds = _tokenize_split(raw_ds["train"], tokenizer, config.max_seq_length, code_type_to_id)
    valid_ds = _tokenize_split(raw_ds["valid"], tokenizer, config.max_seq_length, code_type_to_id)
```

Pass the reverse mapping to `validate_probes()`:

```python
    validate_probes(
        ...,
        id_to_code_type=id_to_code_type if config.log_per_code_type_metrics else None,
        log_per_code_type_metrics=config.log_per_code_type_metrics,
    )
```

### 5.4 Changes to `pyine/probes/debug_dataset.py`

The current debug dataset is **insufficient** for testing the new eval-only mode, family-based splitting, and code type filtering. All records have `"code_type": "original"`, flat sample IDs (`debug_sample_0042/1`), and no family structure. This section describes the required rework.

#### 5.4.1 Problem Summary

| Feature to test | Current debug dataset support | What's missing |
|---|---|---|
| **Code type filtering** | All records are `"original"` | No `"hinted"` or `"misleading"` records to filter on |
| **Family-based splitting** | IDs are flat (`debug_sample_0042/1`) | No `/a:` augmentation suffixes — `_extract_family_id()` has nothing to strip, so every record is its own family |
| **Per-code-type metrics** | Only `"original"` code type | Can't verify per-category breakdown |
| **Eval-only mode integration** | 50 eval records, all identical code type | Would produce a degenerate single-code-type dataset |

#### 5.4.2 Updated `_make_record()` Signature

Add `code_type` and `sample_id` parameters so the caller controls these per-record:

```python
def _make_record(
    label: int,
    rng: random.Random,
    key_prefix: str,
    sample_id: str,         # NEW: full sample_id (caller-constructed)
    code_type: str = "original",  # NEW: code type string
) -> tuple[str, dict[str, typing.Any]]:
    """Create a single mock LMDB record in DiskRewardLogger format.

    Args:
        label: Binary label (0 or 1).
        rng: Random number generator for content selection.
        key_prefix: LMDB key prefix (e.g., "train/", "eval/").
        sample_id: Full sample_id string (e.g., "debug_problem_010/s0000/t0000").
        code_type: Code type string (e.g., "original", "hinted", "misleading").

    Returns:
        Tuple of (lmdb_key, record_dict).
    """
    # ... content generation unchanged ...

    # Tags include augmentation info when code_type is not "original"
    tags = ["debug"]
    if code_type == "hinted":
        tags.append("augment:hinted")
    elif code_type == "misleading":
        tags.append("augment:misleading")

    record: dict[str, typing.Any] = {
        # ... prompt, model_output, expected_output, etc. unchanged ...
        "code_type": code_type,  # WAS: hardcoded "original"
        "tags": tags,            # WAS: always ["debug"]
        "key_prefix": key_prefix,
    }

    lmdb_key = f"{key_prefix}{sample_id}/1"  # WAS: f"{key_prefix}debug_sample_{sample_idx:04d}/1"
    return lmdb_key, record
```

#### 5.4.3 Updated `create_debug_probe_lmdb()` — Family-Structured Eval Records

The key change is in how eval records are generated. Each eval "problem" becomes a **family** with up to three code-type variants (original, hinted, misleading), sharing a base sample ID.

**Train records** remain unchanged in structure (all `"original"`, flat IDs) since the train prefix doesn't need code type diversity.

```python
# Code types to distribute across eval families
_EVAL_CODE_TYPES = ["original", "hinted", "misleading"]

# Augmentation category mapping for constructing /a: suffixes
_CODE_TYPE_TO_AUGMENT = {
    "hinted": "hints_docs",
    "misleading": "issues_docs",
}


def create_debug_probe_lmdb(
    output_path: str | Path,
    n_train: int = 200,
    n_eval_families: int = 30,  # NEW: replaces n_valid; number of problem families
    seed: int = 42,
    noise_rate: float = _NOISE_RATE,
) -> Path:
    """Generate a mock LMDB database for probe training testing.

    Eval records are organized into families. Each family has a base
    sample ID (e.g., ``debug_problem_010/s0000/t0000``) and up to three
    code-type variants:

    - Original: ``eval/debug_problem_010/s0000/t0000/1``
    - Hinted:   ``eval/debug_problem_010/s0000/t0000/a:hints_docs:000/1``
    - Misleading: ``eval/debug_problem_010/s0000/t0000/a:issues_docs:000/1``

    This structure exercises:
    - ``_extract_family_id()`` (strips ``/a:`` suffix)
    - ``_split_records_by_family()`` (all variants in same split)
    - ``_filter_by_code_type()`` (multiple code types present)
    - Per-code-type validation metrics
    """
    output_path = Path(output_path)
    rng = random.Random(seed)
    serialization_config = SerializationConfig(method=SerializationMethod.JSON_ZSTD)

    with LMDBWriter(output_path, serialization_config=serialization_config) as writer:
        # --- Train records: flat structure, all "original" ---
        for i in range(n_train):
            base_label = rng.randint(0, 1)
            label = 1 - base_label if rng.random() < noise_rate else base_label
            sample_id = f"debug_sample_{i:04d}"
            key, record = _make_record(label, rng, "train/", sample_id, code_type="original")
            writer.put(key, record)

        # --- Eval records: family-structured with code type variants ---
        for family_idx in range(n_eval_families):
            base_id = f"debug_problem_{family_idx:03d}/s0000/t0000"
            # Each family gets one label for the original (can vary per variant)
            family_base_label = rng.randint(0, 1)

            for code_type in _EVAL_CODE_TYPES:
                # Construct sample_id with augmentation suffix for non-original
                if code_type == "original":
                    sample_id = base_id
                else:
                    augment_cat = _CODE_TYPE_TO_AUGMENT[code_type]
                    sample_id = f"{base_id}/a:{augment_cat}:000"

                # Label may differ per variant (hinted → more correct, misleading → less)
                if code_type == "hinted":
                    label = 1 if rng.random() > 0.2 else 0  # ~80% correct
                elif code_type == "misleading":
                    label = 0 if rng.random() > 0.3 else 1  # ~70% incorrect
                else:
                    base_label = family_base_label
                    label = 1 - base_label if rng.random() < noise_rate else base_label

                key, record = _make_record(label, rng, "eval/", sample_id, code_type=code_type)
                writer.put(key, record)

    return output_path
```

**Key design choices for the debug data:**

1. **`n_eval_families=30` (default):** With 3 code types per family → 90 eval records. Enough for a meaningful 80/20 family-based split (24 families train / 6 families valid → 72/18 records).

2. **Label correlation with code type:** Hinted samples are biased toward label=1 (correct), misleading toward label=0 (incorrect). This mimics real data and makes per-code-type AUROC testing meaningful — the probe should show varying accuracy across code types.

3. **`/a:hints_docs:000` and `/a:issues_docs:000` suffixes:** These match the real augmentation categories used in `TraceIdentifier`, so `_extract_family_id()` is exercised with realistic suffix patterns.

4. **Train records unchanged:** Flat IDs, all `"original"`. This matches production behavior where training data has no code type diversity.

#### 5.4.4 Updated `create_debug_probe_dataset()` — Support Eval-Only Mode

The convenience wrapper should accept eval-only mode parameters:

```python
def create_debug_probe_dataset(
    output_path: str | Path | None = None,
    n_train: int = 200,
    n_eval_families: int = 30,
    seed: int = 42,
    # NEW: pass-through to load_probe_dataset_from_lmdb
    use_eval_only_split: bool = False,
    code_type_filter: list[str] | None = None,
) -> datasets.DatasetDict:
    """Generate a debug probe dataset (convenience wrapper).

    When use_eval_only_split=True, loads only eval records and splits
    them internally (exercises the full new pipeline).
    """
    ...
    create_debug_probe_lmdb(lmdb_path, n_train=n_train, n_eval_families=n_eval_families, seed=seed)
    return load_probe_dataset_from_lmdb(
        lmdb_path,
        use_eval_only_split=use_eval_only_split,
        code_type_filter=code_type_filter,
    )
```

#### 5.4.5 Backward Compatibility

The parameter rename `n_valid` → `n_eval_families` is a breaking change for callers. Since the only known callers are:

1. **CLI `__main__` block**: Updated in the same change.
2. **Test fixtures**: Updated in the same change.
3. **`create_debug_probe_dataset()`**: Updated in the same change.

No external callers are expected. If needed for safety, we can keep `n_valid` as a deprecated alias that maps to `n_eval_families // 3` (since each family produces ~3 records).

______________________________________________________________________

## 6. Data Flow: Current vs. New

### 6.1 Current Flow (Unchanged when `use_eval_only_split=False`)

```
LMDB
 ├── "train/" records → train split
 └── "eval/" records  → valid split
      ↓
 _tokenize_split()
      ↓
 DataLoader → LLM forward → probes → train/validate
```

### 6.2 New Flow (when `use_eval_only_split=True`)

```
LMDB
 └── "eval/" records (all code types: original, hinted, misleading, ...)
      ↓
 code_type_filter (optional: keep only ["original", "hinted", "misleading"])
      ↓
 _split_records_by_family() or _split_records_random()
 ├── train samples (80%) — contains all code types
 └── valid samples (20%) — contains all code types
      ↓
 _tokenize_split() (preserves code_type column)
      ↓
 DataLoader → LLM forward → probes → train/validate
                                          ↓
                              Per-code-type AUROC/loss breakdown
```

______________________________________________________________________

## 7. W&B Metrics

### 7.1 New Per-Code-Type Metrics

When `log_per_code_type_metrics: true` (default):

| Metric Key | Description |
|---|---|
| `valid/{probe_name}/loss/code_type/original` | Loss on original-code samples |
| `valid/{probe_name}/loss/code_type/hinted` | Loss on hinted-code samples |
| `valid/{probe_name}/loss/code_type/misleading` | Loss on misleading-code samples |
| `valid/{probe_name}/auroc/code_type/original` | AUROC on original-code samples |
| `valid/{probe_name}/auroc/code_type/hinted` | AUROC on hinted-code samples |
| `valid/{probe_name}/auroc/code_type/misleading` | AUROC on misleading-code samples |

If additional code types are present (e.g., `obfuscated`, `hinted_obfuscated`), they get their own metrics automatically.

### 7.2 Interaction with Replica Aggregation

When both replicas and per-code-type metrics are active:

| Metric Key | Description |
|---|---|
| `valid/{base_name}/auroc/code_type/{ct}/mean` | Mean AUROC across replicas for code type `ct` |
| `valid/{base_name}/auroc/code_type/{ct}/std` | Std of AUROC across replicas for code type `ct` |

This follows the existing replica aggregation pattern.

### 7.3 Code Type Distribution Summary

At dataset load time, log the code type distribution as a W&B summary:

```python
if runtime and runtime.wandb_run:
    for split_name in ["train", "valid"]:
        ct_counts = ...  # computed above
        for ct, count in ct_counts.items():
            runtime.wandb_run.summary[f"data/{split_name}/code_type/{ct}"] = count
        runtime.wandb_run.summary[f"data/{split_name}/total"] = sum(ct_counts.values())
```

______________________________________________________________________

## 8. Edge Cases and Error Handling

| Scenario | Handling |
|---|---|
| `code_type` is `None` in LMDB record | Mapped to `"unknown"`. If `code_type_filter` is active and doesn't include `"unknown"`, the record is excluded. |
| `code_type_filter` excludes all records | Raise `ValueError` with a clear message listing the filter and available code types. |
| Code type has too few samples for AUROC (< 2 unique labels) | Log NaN for that code type's AUROC; log a warning. |
| `split_by_family=True` but all families have only one sample | Works correctly — each family is a single record. Degenerates to random split by record. |
| `use_eval_only_split=True` but `eval_only_source_prefix` has no records | Raise `ValueError` (same as current behavior for missing prefix). |
| `train_split_ratio` produces fewer than 2 families | `_split_records_by_family()` raises `ValueError`. Rounding rule: `n_train = max(1, int(n_families * ratio))`, `n_valid = n_families - n_train`. Both guaranteed ≥ 1. |
| DDP mode with `code_type_id` alignment | `code_type_id` is a tensor column gathered via `gather_for_metrics()` alongside labels and logits. Alignment is guaranteed by construction — no ordering assumptions. |
| Family-based split produces unbalanced code type distribution | Log a warning with per-split code type counts and the exact missing code types. Suggest `split_by_family: false` as fallback. |
| `code_type_filter: []` (empty list) | Rejected at config validation time with a clear error. Use `null` to include all records. |

______________________________________________________________________

## 9. File Changes Summary

| File | Change type | Description |
|---|---|---|
| `pyine/probes/lmdb_dataset.py` | **Modify** | Add `code_type` extraction to samples, add `_extract_family_id()`, `_split_records_by_family()`, `_split_records_random()`, `_filter_by_code_type()` helpers, update `load_probe_dataset_from_lmdb()` with new parameters and eval-only split logic |
| `pyine/apps/trainers/probe_trainer_configs.py` | **Modify** | Add `use_eval_only_split`, `eval_only_source_prefix`, `train_split_ratio`, `split_by_family`, `code_type_filter`, `log_per_code_type_metrics` fields |
| `pyine/apps/trainers/probe_trainer.py` | **Modify** | Pass new config fields to `load_probe_dataset_from_lmdb()`, log code type distribution, extract `valid_code_types` list, update `_tokenize_split()` to preserve `code_type`, update `validate_probes()` with per-code-type metrics, update `build_dataloader()` to handle `code_type` column |
| `pyine/probes/debug_dataset.py` | **Modify** | Rework `_make_record()` to accept `code_type`/`sample_id`, restructure `create_debug_probe_lmdb()` to generate family-structured eval records with 3 code-type variants per family (original/hinted/misleading), add augmentation suffixes to sample IDs, update `create_debug_probe_dataset()` for eval-only passthrough (see Section 5.4) |
| `pyine/configs/experiment/probes/v0_probe.yaml` | **Modify** | Add commented-out examples of new config fields |
| `pyine/apps/trainers/PROBE_TRAINING_GUIDE.md` | **Modify** | Add section on eval-only mode and code type metrics |

______________________________________________________________________

## 10. Test Plan

### 10.1 New Tests: `tests/probes/test_lmdb_dataset.py`

```python
class TestExtractFamilyId:
    """Tests for _extract_family_id helper."""

    def test_original_sample_id(self):
        """sample_id without augmentation returns itself."""

    def test_augmented_sample_id(self):
        """sample_id with /a:... suffix is stripped to family ID."""

    def test_multi_segment_sample_id(self):
        """Multi-segment sample_id (TACO/train/p000001/s0000/t0000) is handled correctly."""

    def test_no_false_positive_on_a_in_path(self):
        """The string '/a:' must be the augmentation marker, not part of a dataset name."""


class TestFilterByCodeType:
    """Tests for _filter_by_code_type helper."""

    def test_filters_to_specified_types(self):
        """Only records with matching code_type are kept."""

    def test_empty_filter_list_returns_empty(self):
        """An empty filter list returns no records."""

    def test_null_code_type_excluded(self):
        """Records with code_type=None are excluded when filter is active."""


class TestSplitRecordsByFamily:
    """Tests for _split_records_by_family helper."""

    def test_all_family_members_in_same_split(self):
        """All variants of the same family go to train or valid, never split."""

    def test_train_ratio_approximate(self):
        """Train/valid sizes roughly match train_split_ratio."""

    def test_deterministic_with_seed(self):
        """Same seed produces same split."""

    def test_different_seed_different_split(self):
        """Different seed produces different split."""

    def test_single_member_families(self):
        """Works when each family has exactly one record."""


class TestSplitRecordsRandom:
    """Tests for _split_records_random helper."""

    def test_split_sizes_match_ratio(self):
        """Train/valid sizes match train_split_ratio."""

    def test_deterministic_with_seed(self):
        """Same seed produces same split."""


class TestLoadProbeDatasetEvalOnly:
    """Tests for eval-only mode in load_probe_dataset_from_lmdb."""

    def test_returns_train_and_valid_splits(self, debug_lmdb):
        """Returns DatasetDict with both splits from a single prefix."""

    def test_code_type_column_present(self, debug_lmdb):
        """Each split has a 'code_type' column."""

    def test_code_type_filter_applied(self, debug_lmdb):
        """Only specified code types appear in the dataset."""

    def test_family_split_no_leakage(self, debug_lmdb):
        """No family ID appears in both train and valid."""

    def test_code_type_filter_all_excluded_raises(self, debug_lmdb):
        """ValueError when filter excludes all records."""

    def test_both_labels_in_each_split(self, debug_lmdb):
        """Both label=0 and label=1 present in each split."""
```

### 10.2 Modified Tests: `tests/probes/test_lmdb_dataset.py`

```python
class TestRecordToSample:
    # EXISTING tests unchanged

    def test_code_type_extracted(self):
        """Output sample includes code_type from record."""

    def test_code_type_none_becomes_unknown(self):
        """Records with code_type=None get 'unknown' in sample."""
```

### 10.3 Modified Tests: `tests/apps/trainers/test_probe_trainer_configs.py`

```python
class TestProbeTrainerConfigEvalOnly:
    """Tests for eval-only split config fields."""

    def test_use_eval_only_split_default_false(self):
        """Default is False."""

    def test_train_split_ratio_bounds(self):
        """Rejects ratio <= 0 or >= 1."""

    def test_eval_only_fields_accepted(self):
        """All new fields are accepted without error."""

    def test_code_type_filter_accepts_list(self):
        """code_type_filter accepts a list of strings."""

    def test_code_type_filter_default_none(self):
        """Default code_type_filter is None."""

    def test_code_type_filter_empty_list_rejected(self):
        """Empty list code_type_filter raises ValidationError."""
```

### 10.4 Modified Tests: `tests/apps/trainers/test_probe_trainer.py`

```python
class TestValidateProbesWithCodeTypes:
    """Tests for per-code-type validation metrics."""

    def test_per_code_type_metrics_computed(self):
        """validate_probes computes per-code-type loss and AUROC."""

    def test_per_code_type_single_class_nan(self):
        """NaN AUROC for code types with single-class labels."""

    def test_per_code_type_disabled(self):
        """No per-code-type metrics when log_per_code_type_metrics=False."""
```

### 10.5 Reworked Tests: `tests/probes/test_debug_dataset.py`

The debug dataset tests need significant expansion to cover the new family-structured eval records.

```python
class TestDebugLmdbStructure:
    """Tests for the structural properties of the generated LMDB."""

    def test_train_records_all_original(self, tmp_path):
        """Train-prefix records all have code_type='original'."""

    def test_train_records_flat_ids(self, tmp_path):
        """Train-prefix sample IDs have no /a: augmentation suffix."""

    def test_eval_records_have_three_code_types(self, tmp_path):
        """Eval-prefix records include 'original', 'hinted', and 'misleading' code_type values."""

    def test_eval_records_per_family_count(self, tmp_path):
        """Each eval family produces exactly 3 records (one per code type)."""

    def test_eval_total_record_count(self, tmp_path):
        """Total eval records = n_eval_families * 3."""

    def test_eval_family_ids_shared(self, tmp_path):
        """All code-type variants of the same problem share the same family ID
        when processed through _extract_family_id()."""

    def test_eval_original_no_augment_suffix(self, tmp_path):
        """Original-code eval sample_ids have no /a: suffix."""

    def test_eval_hinted_has_hints_docs_suffix(self, tmp_path):
        """Hinted eval sample_ids have /a:hints_docs:000 suffix."""

    def test_eval_misleading_has_issues_docs_suffix(self, tmp_path):
        """Misleading eval sample_ids have /a:issues_docs:000 suffix."""


class TestDebugLmdbCodeTypeTags:
    """Tests for code_type and tags field consistency."""

    def test_original_records_no_augment_tag(self, tmp_path):
        """Records with code_type='original' have no 'augment:' tags."""

    def test_hinted_records_have_augment_hinted_tag(self, tmp_path):
        """Records with code_type='hinted' have 'augment:hinted' in tags."""

    def test_misleading_records_have_augment_misleading_tag(self, tmp_path):
        """Records with code_type='misleading' have 'augment:misleading' in tags."""


class TestDebugLmdbLabelDistribution:
    """Tests for label correlation with code type (hinted → more correct, misleading → less)."""

    def test_hinted_biased_toward_label_1(self, tmp_path):
        """Hinted records have label=1 in >60% of cases (biased toward correct)."""

    def test_misleading_biased_toward_label_0(self, tmp_path):
        """Misleading records have label=0 in >50% of cases (biased toward incorrect)."""


class TestDebugLmdbIntegrationWithEvalOnly:
    """Integration tests: debug LMDB → load_probe_dataset_from_lmdb(use_eval_only_split=True)."""

    def test_eval_only_loads_both_splits(self, tmp_path):
        """load_probe_dataset_from_lmdb with eval-only mode returns train+valid from eval records."""

    def test_eval_only_code_type_column_present(self, tmp_path):
        """Both splits have a 'code_type' column with non-empty values."""

    def test_eval_only_family_split_no_leakage(self, tmp_path):
        """With split_by_family=True, no family ID appears in both train and valid."""

    def test_eval_only_code_type_filter(self, tmp_path):
        """code_type_filter=['original', 'hinted'] excludes misleading records."""

    def test_eval_only_all_code_types_in_train_split(self, tmp_path):
        """With enough families, the train split contains all three code types."""


class TestDebugProbeDatasetConvenience:
    """Tests for create_debug_probe_dataset() with new parameters."""

    def test_eval_only_mode(self, tmp_path):
        """create_debug_probe_dataset(use_eval_only_split=True) returns valid DatasetDict."""

    def test_code_type_filter_passthrough(self, tmp_path):
        """code_type_filter parameter is passed through to load_probe_dataset_from_lmdb."""

    def test_backward_compat_default_mode(self, tmp_path):
        """Default parameters produce a DatasetDict with train/valid (same as before)."""
```

### 10.6 Test Summary

| Target file | New/Modified tests | # Tests |
|---|---|---|
| `tests/probes/test_lmdb_dataset.py` | `TestExtractFamilyId` | 4 |
| `tests/probes/test_lmdb_dataset.py` | `TestFilterByCodeType` | 3 |
| `tests/probes/test_lmdb_dataset.py` | `TestSplitRecordsByFamily` | 5 |
| `tests/probes/test_lmdb_dataset.py` | `TestSplitRecordsRandom` | 2 |
| `tests/probes/test_lmdb_dataset.py` | `TestLoadProbeDatasetEvalOnly` | 6 |
| `tests/probes/test_lmdb_dataset.py` | `TestRecordToSample` (modified) | 2 |
| `tests/apps/trainers/test_probe_trainer_configs.py` | `TestProbeTrainerConfigEvalOnly` | 6 |
| `tests/apps/trainers/test_probe_trainer.py` | `TestValidateProbesWithCodeTypes` | 3 |
| `tests/probes/test_debug_dataset.py` | `TestDebugLmdbStructure` | 9 |
| `tests/probes/test_debug_dataset.py` | `TestDebugLmdbCodeTypeTags` | 3 |
| `tests/probes/test_debug_dataset.py` | `TestDebugLmdbLabelDistribution` | 2 |
| `tests/probes/test_debug_dataset.py` | `TestDebugLmdbIntegrationWithEvalOnly` | 5 |
| `tests/probes/test_debug_dataset.py` | `TestDebugProbeDatasetConvenience` | 3 |
| **Total** | | **53** |

______________________________________________________________________

## 11. Implementation Steps

### Step 1: Add `code_type` extraction to `_record_to_probe_sample()`

Modify `pyine/probes/lmdb_dataset.py` to include `code_type` in every sample dict. This is a minimal, backward-compatible change — the extra column doesn't affect downstream code that doesn't use it.

**Tests:** Verify `code_type` appears in output samples.

### Step 2: Add helper functions to `lmdb_dataset.py`

Implement `_extract_family_id()`, `_filter_by_code_type()`, `_split_records_by_family()`, `_split_records_random()`.

**Tests:** Full test suites for all helpers (10.1).

### Step 3: Update `load_probe_dataset_from_lmdb()` with eval-only mode

Add the new parameters and the conditional loading/splitting logic.

**Tests:** `TestLoadProbeDatasetEvalOnly` (10.1).

### Step 4: Add config fields to `ProbeTrainerAppMainConfig`

Add `use_eval_only_split`, `eval_only_source_prefix`, `train_split_ratio`, `split_by_family`, `code_type_filter`, `log_per_code_type_metrics`.

**Tests:** Config validation tests (10.3).

### Step 5: Update `probe_trainer.py` — dataset loading and tokenization

Pass new config fields to `load_probe_dataset_from_lmdb()`. Update `_tokenize_split()` to preserve `code_type`. Log code type distribution. Extract `valid_code_types` for metrics.

### Step 6: Update `validate_probes()` — per-code-type metrics

Add `code_types` and `log_per_code_type_metrics` parameters. Compute and log per-code-type loss and AUROC.

**Tests:** `TestValidateProbesWithCodeTypes` (10.4).

### Step 7: Rework debug dataset

Rework `pyine/probes/debug_dataset.py` per Section 5.4:

1. Update `_make_record()` to accept `code_type` and `sample_id` parameters, set tags accordingly.
2. Restructure `create_debug_probe_lmdb()` to generate family-structured eval records with three code-type variants per family (`original`, `hinted`, `misleading`), using `/a:hints_docs:000` and `/a:issues_docs:000` suffixes.
3. Update `create_debug_probe_dataset()` to accept and pass through `use_eval_only_split` and `code_type_filter`.
4. Update CLI `__main__` block for the new `n_eval_families` parameter.

**Tests:** Full debug dataset test suites (10.5) — structure, tags, label distribution, integration with eval-only mode, and convenience wrapper.

### Step 8: Update experiment config and documentation

- Add commented-out examples to `v0_probe.yaml`
- Add eval-only mode section to `PROBE_TRAINING_GUIDE.md`

### Step 9: Run full test suite

```bash
uv run pytest tests/probes/ tests/apps/trainers/test_probe_trainer.py tests/apps/trainers/test_probe_trainer_configs.py -v
```

______________________________________________________________________

## 12. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Data leakage** from same-problem variants in train/valid | High (without family split) | High | `split_by_family: true` is default. All variants of a problem go to the same split. |
| **Small valid set** if few families exist | Medium | Medium | Log split sizes and code_type distributions; warn if valid has < 50 samples or missing code types. |
| **Unbalanced code types** after family-based split | Medium | Low | Log per-split code type distribution. Offer `split_by_family: false` as escape hatch. Document the tradeoff. |
| **DDP code_type alignment** | Very low | High | `code_type_id` is a tensor column gathered via `gather_for_metrics()` alongside labels/logits — alignment by construction. No ordering assumptions. |
| **`code_type` column breaks DataLoader/collator** | Low | Medium | `set_format("torch")` applied only to numeric columns. `DataCollatorWithPadding` ignores non-tensor columns. |
| **Backward compatibility** | Low | High | All new fields have defaults matching current behavior. Existing configs work unchanged. |
| **`/a:` substring in dataset name** | Very low | Low | `_extract_family_id()` uses `rfind` to strip only the last `/a:` segment. The marker is standardized by `TraceIdentifier.__repr__()`. No known dataset uses `/a:` in its name. |

______________________________________________________________________

## 13. Future Extensions (Out of Scope)

- **Merging train + eval prefixes**: Support reading from multiple LMDB prefixes and combining records. Useful if train data also has some code type diversity.
- **Stratified family split**: Ensure each split has proportional code type representation, not just proportional family count.
- **Per-code-type training loss**: Also compute per-code-type training loss (currently only validation). Lower priority since training loss is noisier.
- **Code type as a label**: Train probes to predict code type directly (is this sample hinted or not?) rather than correctness. Would require a different label column.
- **Counterfactual pairing**: Leverage the family grouping to create paired train samples (same problem, original vs. hinted) for contrastive probe training.
- **W&B Table with code type**: Add `code_type` as a column in the existing replica details tables for combined analysis.
