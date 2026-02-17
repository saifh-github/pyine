# Probe Trainer — Datamodule/LMDB Integration Plan

## 1. Goal

Replace the probe trainer's current HuggingFace dataset interface (`dataset_path` with pre-processed `messages` + `label` columns) with a direct LMDB-based pipeline that reads **completion records** exported by `DiskRewardLogger` during RL training runs. Each LMDB record already contains the prompt, the model's completion, the expected output, and pre-computed reward metrics (including `is_match` from soft/hard match terms). The probe trainer will use this data to assemble input-label pairs for training probes on frozen LLM activations.

### What changes

| Aspect                | Current                                                      | New                                                                                      |
| --------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| **Data source**       | Pre-processed HF dataset on disk (`datasets.load_from_disk`) | LMDB database(s) exported by `DiskRewardLogger`                                          |
| **Input text**        | `messages` column (chat format) or plain `text` column       | `prompt` + `model_output` concatenated from LMDB record (plain text, post-chat-template) |
| **Label**             | Pre-computed `label` column (0/1)                            | Derived from `reward_metrics["reward/metrics/soft_match/is_match"]` in LMDB record       |
| **Train/valid split** | Separate HF dataset splits (`train`/`valid`)                 | Key prefix filtering on LMDB keys (e.g., `"train/"`, `"eval/"`)                          |
| **Config**            | `dataset_path`, `text_field`, `label_field`                  | `lmdb_path`, `label_metric_key`, `train_key_prefix`, `valid_key_prefix`                  |
| **Debug dataset**     | Chat-format messages with keyword signal                     | Mock LMDB records matching `DiskRewardLogger` format                                     |

______________________________________________________________________

## 2. Data Flow: Current vs. New

### 2.1 Current Flow

```
HF Dataset (disk)
  ├── train split: {messages: [...], label: 0|1}
  └── valid split: {messages: [...], label: 0|1}
       ↓
  tokenize_for_probes() — apply chat template or plain tokenize
       ↓
  DataLoader with DataCollatorWithPadding
       ↓
  LLM forward pass → activations → probe training
```

### 2.2 New Flow

```
LMDB (from DiskRewardLogger)
  └── Records: {prompt, model_output, expected_output, reward_metrics, ...}
       ↓
  load_probe_dataset_from_lmdb()
    - Filter by key_prefix → train/valid splits
    - Extract text = prompt + model_output
    - Extract label = reward_metrics[label_metric_key] (e.g., "reward/metrics/soft_match/is_match")
    - Validate labels are binary
    - Return HF DatasetDict with "train"/"valid" splits
       ↓
  _tokenize_split() — plain text tokenization (add_special_tokens=False)
       ↓
  DataLoader with DataCollatorWithPadding
       ↓
  LLM forward pass → activations → probe training
```

______________________________________________________________________

## 3. LMDB Record Structure (Reference)

Each LMDB record written by `DiskRewardLogger` contains (relevant fields):

```python
record = {
    "prompt": str,                      # Full formatted prompt (post-chat-template)
    "model_output": str,                # Model's raw completion
    "expected_output": str,             # Ground-truth output
    "final_answer": str | None,         # Extracted answer from model output
    "reasoning": str | None,            # Extracted reasoning
    "reward_total": float | None,       # Total reward value
    "reward_terms": dict | None,        # Per-term weighted rewards
    "reward_metrics": dict | None,      # Term-emitted metrics (contains is_match)
    "reward_terms_raw": dict | None,    # Pre-clipping values
    "predict_type": str | None,         # "program_output", etc.
    "code_type": str | None,            # "original", "obfuscated", etc.
    "tags": list[str] | None,           # Sample tags
    "key_prefix": str,                  # Phase prefix (e.g., "train/", "eval/")
}
```

**Key format**: `{key_prefix}{sample_id}/{generation_count}`
**Example**: `"train/TACO/train/p000001/s0000/3"`

**Key parsing rule** (reuses the convention from `BiasDataModuleBase._load_pregenerated_outputs()`):

```python
# Split at the LAST "/" — everything before is sample_id, after is generation_count
suffix = key[len(key_prefix):]
sep = suffix.rfind("/")
sample_id = suffix[:sep]
gen_str = suffix[sep + 1:]
generation_count = 0 if gen_str == "none" else int(gen_str)
```

This handles multi-segment `sample_id` values (e.g., `"TACO/train/p000001/s0000"`) correctly. The `"none"` sentinel for `generation_count` is emitted by `DiskRewardLogger` when `generation_count is None` and is treated as `0`.

**Serialization**: Records are stored using `SerializationMethod.JSON_ZSTD` (orjson + zstandard compression), matching `DiskRewardLogger`'s write format.

**Metric key for labels**: `reward_metrics["reward/metrics/soft_match/is_match"]` — value is `1` (correct) or `0` (incorrect). The `reward/metrics/` prefix is added by `RewardManager` during aggregation, producing full keys like `"reward/metrics/soft_match/is_match"`. Hard match is also available as `"reward/metrics/hard_match/is_match"`.

______________________________________________________________________

## 4. Label Derivation Strategy

### 4.1 Primary: Pre-computed `is_match` from LMDB

The `SoftMatchTerm` (and `HardMatchTerm`) emit an `is_match` metric that is stored in the LMDB record under `reward_metrics`. The full key path in the record is:

```
reward_metrics["reward/metrics/soft_match/is_match"] → 1 or 0
reward_metrics["reward/metrics/hard_match/is_match"] → 1 or 0
```

This is the recommended approach because:

- It's fast (no re-computation needed)
- It reflects the exact evaluation used during RL training
- It's already available in every record

### 4.2 Fallback: Re-compute Match

If `reward_metrics` is `None` or the specified metric key is missing, fall back to re-computing the match. **Re-computation is only supported when `label_metric_key` is `"reward/metrics/soft_match/is_match"` or `"reward/metrics/hard_match/is_match"`**. If `recompute_labels=True` is set with any other `label_metric_key`, a `ValueError` is raised at startup.

```python
from pyine.organisms.models.rewards.terms.code_exec.utils import compute_soft_match, compute_hard_match

if label_metric_key == "reward/metrics/soft_match/is_match":
    result = compute_soft_match(
        expected=record["expected_output"],
        predicted=record["final_answer"],  # or model_output if final_answer is None
    )
elif label_metric_key == "reward/metrics/hard_match/is_match":
    result = compute_hard_match(
        expected=record["expected_output"],
        predicted=record["final_answer"],
    )
else:
    raise ValueError(
        f"recompute_labels=True is only supported for 'reward/metrics/soft_match/is_match' and "
        f"'reward/metrics/hard_match/is_match', got '{label_metric_key}'"
    )
label = int(result.equal)
```

This ensures that the recompute path always produces labels consistent with `label_metric_key`.

### 4.3 Configurable Label Source

The config will expose a `label_metric_key` field (default: `"reward/metrics/soft_match/is_match"`) that specifies which metric to use as the binary label. This allows:

- `"reward/metrics/soft_match/is_match"` — semantic match with numeric tolerance (default)
- `"reward/metrics/hard_match/is_match"` — exact string match
- Any other binary metric stored in `reward_metrics` (but `recompute_labels` is not supported for arbitrary metrics)

A boolean `recompute_labels` flag (default: `False`) forces re-computation instead of using stored metrics. **Only valid when `label_metric_key` is `"reward/metrics/soft_match/is_match"` or `"reward/metrics/hard_match/is_match"`** — enforced in two places:

1. **Config validation** (pydantic `model_validator`): fails early before any data loading.
2. **Runtime check** at the top of `load_probe_dataset_from_lmdb()`: defensive guard for direct callers that bypass the config layer.

`label_metric_key` itself is **permissive** — any string is accepted as long as the key exists in `reward_metrics`. The restriction only applies when `recompute_labels=True`.

Note: `compare_options` is only used for soft match recomputation. Hard match (`compute_hard_match`) is a strict string comparison with no tolerance parameters and ignores `compare_options`.

______________________________________________________________________

## 5. Input Assembly

### 5.1 Plain Text Concatenation

The LMDB `prompt` field contains the **post-chat-template** formatted prompt string — exactly what the tokenizer produced during RL inference. The `model_output` field contains the raw generated text.

For probe training, the input is the full sequence the model processed:

```python
text = record["prompt"] + record["model_output"]
```

This is tokenized as plain text with **`add_special_tokens=False`** (no chat template re-application needed), producing token IDs that match what the model saw during generation. Setting `add_special_tokens=False` prevents the tokenizer from inserting BOS/EOS tokens or chat markers around the already-formatted text — this is consistent with the convention used throughout the codebase for post-template text (see `_load_pregenerated_outputs()`, `apply_chat_template()`). This ensures activation extraction produces representative hidden states.

### 5.2 Why Not Reconstruct Chat Messages?

Reconstructing structured messages would require:

1. Access to the original trace LMDB (to get `SampleData`)
2. The prompt template configuration
3. Re-applying the chat template

This adds complexity with no benefit — the stored `prompt` already IS the post-template string. Plain text tokenization of `prompt + model_output` produces identical tokens.

______________________________________________________________________

## 6. Train/Valid Split Strategy

### 6.1 Key Prefix Filtering

LMDB keys follow the pattern `{key_prefix}{sample_id}/{generation_count}`. The `key_prefix` encodes the phase (e.g., `"train/"`, `"eval/"`). We filter records by prefix to determine splits:

```python
train_key_prefix: str = "train/"   # Records from RL training phase
valid_key_prefix: str = "eval/"    # Records from RL evaluation phase
```

This matches the existing `DiskRewardLogger` convention where the `key_prefix` is set per phase during RL training.

**Source of truth**: The LMDB key prefix is the authoritative source for split assignment. Each record also stores a `key_prefix` field — we optionally validate that `record["key_prefix"]` matches the key-derived prefix and log a warning on mismatch (but always trust the LMDB key). This guards against export bugs where the stored field could diverge from the actual key.

### 6.2 Deduplication

When multiple generations exist for the same `sample_id` (different `generation_count` values), we need a selection strategy:

```python
selection_strategy: str = "latest"  # "latest" or "best_reward"
```

- `"latest"`: Use the record with the highest `generation_count`
- `"best_reward"`: Use the record with the highest `reward_total`. **If `reward_total` is `None` for any record under this strategy, raise `ValueError`** — this indicates a data export issue that should be fixed at the source rather than silently handled. This matches the codebase's fail-fast convention (the existing `_load_pregenerated_outputs()` also does not handle `None` reward gracefully).

This reuses the same deduplication logic already implemented in `BiasDataModuleBase._load_pregenerated_outputs()`.

______________________________________________________________________

## 7. Module Design

### 7.1 New File: `pyine/probes/lmdb_dataset.py`

This module handles loading LMDB completion records and converting them into an HF dataset suitable for probe training.

```python
def load_probe_dataset_from_lmdb(
    lmdb_path: str | Path,
    label_metric_key: str = "reward/metrics/soft_match/is_match",
    train_key_prefix: str = "train/",
    valid_key_prefix: str = "eval/",
    selection_strategy: str = "latest",
    recompute_labels: bool = False,
    compare_options: CompareOptions | None = None,
    max_samples_per_split: int | None = None,
    skip_malformed_records: bool = False,
    seed: int = 42,
) -> datasets.DatasetDict:
    """Load LMDB completion records and create a probe training dataset.

    Reads records exported by DiskRewardLogger (JSON_ZSTD serialization),
    extracts prompt+completion as input text and derives binary labels
    from reward metrics.

    Args:
        lmdb_path: Path to LMDB database from DiskRewardLogger.
        label_metric_key: Key in reward_metrics for binary label. Permissive — any
            string is accepted as long as the key exists in record reward_metrics.
        train_key_prefix: Key prefix for training records.
        valid_key_prefix: Key prefix for validation records.
        selection_strategy: "latest" or "best_reward" for deduplication.
        recompute_labels: If True, re-compute match instead of using stored metrics.
            Only supported for label_metric_key in {"reward/metrics/soft_match/is_match",
            "reward/metrics/hard_match/is_match"} — raises ValueError otherwise.
        compare_options: CompareOptions for soft match re-computation. Ignored when
            recompute_labels=False or when label_metric_key is "reward/metrics/hard_match/is_match"
            (hard match is a strict string comparison with no tolerance parameters).
        max_samples_per_split: Cap samples per split (for debugging/fast iteration).
        skip_malformed_records: If True, skip records missing required fields and log
            count at WARNING level. If False (default), raise ValueError. Required
            fields per mode:
            - recompute_labels=False: prompt, model_output, reward_metrics[label_metric_key]
            - recompute_labels=True: prompt, model_output, expected_output
            (final_answer may be None — falls back to model_output for recompute)
        seed: Random seed for any shuffling/sampling.

    Returns:
        DatasetDict with "train" and "valid" splits, each containing:
        - "text": str (prompt + model_output)
        - "label": int (0 or 1)
        - "sample_id": str (for provenance tracking)

    Raises:
        ValueError: If recompute_labels=True with unsupported label_metric_key,
            if selection_strategy='best_reward' and reward_total is None,
            if skip_malformed_records=False and a malformed record is encountered,
            or if no records match a key prefix.
    """
    ...
```

**Internal helpers:**

```python
def _load_lmdb_records(
    lmdb_path: Path,
    key_prefix: str,
    selection_strategy: str,
) -> list[dict[str, Any]]:
    """Load and deduplicate LMDB records for a given key prefix."""
    ...

def _record_to_probe_sample(
    record: dict[str, Any],
    sample_id: str,
    label_metric_key: str,
    recompute_labels: bool,
    compare_options: CompareOptions | None,
) -> dict[str, str | int]:
    """Convert a single LMDB record to a probe training sample.

    Returns:
        {"text": str, "label": int, "sample_id": str}
    """
    ...

def _validate_probe_dataset(dataset: datasets.Dataset) -> None:
    """Validate the probe dataset has correct structure and binary labels."""
    ...
```

### 7.2 Modified: `pyine/apps/trainers/probe_trainer_configs.py`

Replace dataset fields with LMDB fields:

```python
class ProbeTrainerAppMainConfig(common.AppMainConfig, common.ModelTokenizerConfigBase):
    # --- REMOVED ---
    # dataset_path: str
    # text_field: str = "messages"
    # label_field: str = "label"

    # --- NEW: LMDB data source ---
    lmdb_path: str = pydantic.Field(
        ...,
        description="Path to LMDB database exported by DiskRewardLogger.",
    )
    label_metric_key: str = pydantic.Field(
        default="reward/metrics/soft_match/is_match",
        description=(
            "Key in reward_metrics dict for binary label derivation. "
            "Common values: 'reward/metrics/soft_match/is_match', 'reward/metrics/hard_match/is_match'."
        ),
    )
    train_key_prefix: str = pydantic.Field(
        default="train/",
        description="LMDB key prefix for training records.",
    )
    valid_key_prefix: str = pydantic.Field(
        default="eval/",
        description="LMDB key prefix for validation records.",
    )
    selection_strategy: typing.Literal["latest", "best_reward"] = pydantic.Field(
        default="latest",
        description=(
            "Strategy for deduplicating multiple generations per sample. "
            "'latest' uses highest generation_count, 'best_reward' uses highest reward_total."
        ),
    )
    recompute_labels: bool = pydantic.Field(
        default=False,
        description=(
            "If True, re-compute labels instead of using stored reward_metrics. "
            "Only valid when label_metric_key is 'reward/metrics/soft_match/is_match' or "
            "'reward/metrics/hard_match/is_match' — validated at config construction time."
        ),
    )
    max_samples_per_split: int | None = pydantic.Field(
        default=None,
        description="Cap samples per split. Useful for debugging or fast iteration.",
    )
    skip_malformed_records: bool = pydantic.Field(
        default=False,
        description=(
            "If True, skip records missing required fields instead of raising. "
            "Required fields depend on mode: "
            "recompute_labels=False requires prompt, model_output, reward_metrics[label_metric_key]; "
            "recompute_labels=True requires prompt, model_output, expected_output. "
            "Skipped records are counted and logged at WARNING level. "
            "If False (default), raise ValueError on any malformed record."
        ),
    )
    # ... (existing fields: num_epochs, train_batch_size, etc. unchanged)

    @pydantic.model_validator(mode="after")
    def _validate_recompute_label_metric(self) -> "ProbeTrainerAppMainConfig":
        _RECOMPUTABLE = {"reward/metrics/soft_match/is_match", "reward/metrics/hard_match/is_match"}
        if self.recompute_labels and self.label_metric_key not in _RECOMPUTABLE:
            raise ValueError(
                f"recompute_labels=True is only supported for label_metric_key in "
                f"{_RECOMPUTABLE}, got '{self.label_metric_key}'"
            )
        return self
```

### 7.3 Modified: `pyine/apps/trainers/probe_trainer.py`

**Changes to `probe_train()`:**

1. Replace `load_and_tokenize()` calls with LMDB loading:

```python
# --- OLD ---
# train_ds = load_and_tokenize(config.dataset_path, "train", tokenizer, config)
# valid_ds = load_and_tokenize(config.dataset_path, "valid", tokenizer, config)

# --- NEW ---
from pyine.probes.lmdb_dataset import load_probe_dataset_from_lmdb

raw_ds = load_probe_dataset_from_lmdb(
    lmdb_path=config.lmdb_path,
    label_metric_key=config.label_metric_key,
    train_key_prefix=config.train_key_prefix,
    valid_key_prefix=config.valid_key_prefix,
    selection_strategy=config.selection_strategy,
    recompute_labels=config.recompute_labels,
    max_samples_per_split=config.max_samples_per_split,
    skip_malformed_records=config.skip_malformed_records,
)
train_ds = _tokenize_split(raw_ds["train"], tokenizer, config.max_seq_length)
valid_ds = _tokenize_split(raw_ds["valid"], tokenizer, config.max_seq_length)
```

2. Simplify tokenization (always plain text now):

```python
def _tokenize_split(
    dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seq_length: int,
) -> datasets.Dataset:
    """Tokenize a probe dataset split (plain text, no special tokens)."""
    def _tokenize(examples):
        tokenized = tokenizer(
            examples["text"],
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,  # Post-template text — don't add BOS/EOS/chat markers
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    ds = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=[c for c in dataset.column_names if c not in ("input_ids", "attention_mask", "labels")],
    )
    ds.set_format("torch")
    return ds
```

3. Remove `tokenize_for_probes()`, `load_and_tokenize()`, and `validate_probe_dataset()` (replaced by LMDB loading + simplified tokenization).

### 7.4 Modified: `pyine/probes/debug_dataset.py`

Replace the chat-message-based debug dataset with one that produces a **mock LMDB** matching the `DiskRewardLogger` format. This enables end-to-end testing of the LMDB pipeline without a real RL run.

```python
def create_debug_probe_lmdb(
    output_path: str | Path,
    n_train: int = 200,
    n_valid: int = 50,
    seed: int = 42,
    noise_rate: float = 0.1,
) -> Path:
    """Generate a mock LMDB database for probe training testing.

    Uses LMDBWriter with SerializationMethod.JSON_ZSTD to match the exact
    serialization format used by DiskRewardLogger in production. This ensures
    records are readable by the same LMDBReader used in the real pipeline.

    Creates LMDB records in DiskRewardLogger format with:
    - Synthetic prompts (code analysis questions)
    - Synthetic model completions (with learnable signal)
    - Pre-computed reward_metrics with reward/metrics/soft_match/is_match labels
    - Train/eval key prefixes for split filtering

    The signal: label=1 samples contain the keyword "helper" in the
    completion, making the classification task learnable by probes.

    Args:
        output_path: Directory for the LMDB database.
        n_train: Number of training records.
        n_valid: Number of validation records.
        seed: Random seed.
        noise_rate: Fraction of labels to flip (adds noise).

    Returns:
        Path to the created LMDB directory.
    """
    ...
```

**Record format** (mirrors `DiskRewardLogger`):

````python
record = {
    "prompt": "You are an AI assistant...\n\nAnalyze the following code:\n```python\ndef helper(x): ...\n```",
    "model_output": "The output of this program is: 42\n\nThis uses the helper pattern...",
    "expected_output": "42",
    "final_answer": "42",
    "reward_total": 1.0,
    "reward_metrics": {
        "reward/metrics/soft_match/is_match": 1,      # Binary label
        "reward/metrics/hard_match/is_match": 1,
    },
    "reward_terms": {"reward/terms/soft_match": 1.0, "reward/terms/hard_match": 1.0},
    "predict_type": "program_output",
    "code_type": "original",
    "tags": ["debug"],
    "key_prefix": "train/",            # or "eval/" for valid split
}
````

**LMDB key**: `"train/debug_sample_042/1"` or `"eval/debug_sample_042/1"`

The module also keeps a `create_debug_probe_dataset()` convenience function that creates the LMDB and immediately loads it via `load_probe_dataset_from_lmdb()`, returning an HF `DatasetDict` for direct use in tests:

```python
def create_debug_probe_dataset(
    output_path: str | Path | None = None,
    n_train: int = 200,
    n_valid: int = 50,
    seed: int = 42,
) -> datasets.DatasetDict:
    """Generate a debug probe dataset (convenience wrapper).

    Creates a temporary LMDB, loads it via load_probe_dataset_from_lmdb(),
    and returns the resulting DatasetDict. If output_path is provided,
    the LMDB is persisted there; otherwise it uses a temporary directory.
    """
    ...
```

### 7.5 Modified: `pyine/configs/experiment/probes/v0_probe.yaml`

Update to use LMDB fields:

```yaml
config:
  _target_: pyine.apps.trainers.probe_trainer_configs.ProbeTrainerAppMainConfig

  # Data source (LMDB from DiskRewardLogger)
  lmdb_path: /path/to/rl-run/reward_logs.lmdb
  label_metric_key: reward/metrics/soft_match/is_match
  train_key_prefix: train/
  valid_key_prefix: eval/
  selection_strategy: latest
  # recompute_labels: false
  # max_samples_per_split: null

  # ... (rest unchanged)
```

______________________________________________________________________

## 8. Removed / Replaced Code

### 8.1 Functions Removed from `probe_trainer.py`

| Function                   | Reason                                                                                                     |
| -------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `validate_probe_dataset()` | Replaced by validation in `load_probe_dataset_from_lmdb()`                                                 |
| `tokenize_for_probes()`    | Replaced by simpler `_tokenize_split()` (always plain text)                                                |
| `load_and_tokenize()`      | Replaced by LMDB loading + `_tokenize_split()`                                                             |
| `build_dataloader()`       | Kept but simplified (no longer needs `tokenizer` arg for collation — still uses `DataCollatorWithPadding`) |

### 8.2 Config Fields Removed from `ProbeTrainerAppMainConfig`

| Field          | Reason                                        |
| -------------- | --------------------------------------------- |
| `dataset_path` | Replaced by `lmdb_path`                       |
| `text_field`   | No longer needed (always `"text"` from LMDB)  |
| `label_field`  | No longer needed (always `"label"` from LMDB) |

______________________________________________________________________

## 9. Test Plan

### 9.1 New Tests: `tests/probes/test_lmdb_dataset.py`

Tests for the new LMDB dataset loading module.

```python
class TestLoadLmdbRecords:
    """Tests for _load_lmdb_records helper."""

    def test_loads_records_by_prefix(self, debug_lmdb):
        """Records are filtered by key prefix (train/ vs eval/)."""

    def test_deduplication_latest(self, debug_lmdb):
        """With selection_strategy='latest', highest generation_count wins."""

    def test_deduplication_best_reward(self, debug_lmdb):
        """With selection_strategy='best_reward', highest reward_total wins."""

    def test_best_reward_none_raises(self, debug_lmdb):
        """With selection_strategy='best_reward' and reward_total=None, raises ValueError."""

    def test_empty_prefix_returns_empty(self, debug_lmdb):
        """Non-matching prefix returns empty list."""

    def test_key_parsing_multi_segment_sample_id(self, debug_lmdb):
        """sample_id with multiple '/' segments is parsed correctly via rfind."""

    def test_key_parsing_none_generation_count(self, debug_lmdb):
        """Generation count 'none' is treated as 0."""

    def test_key_parsing_non_numeric_raises(self, debug_lmdb):
        """Non-numeric, non-'none' generation count segment raises ValueError."""

    def test_key_prefix_mismatch_warns(self, debug_lmdb):
        """Mismatched record['key_prefix'] vs LMDB key logs a warning."""


class TestRecordToSample:
    """Tests for _record_to_probe_sample conversion."""

    def test_text_is_prompt_plus_completion(self):
        """Output text is prompt + model_output concatenation."""

    def test_label_from_reward_metrics(self):
        """Label extracted from reward_metrics[label_metric_key]."""

    def test_label_from_hard_match(self):
        """Can use reward/metrics/hard_match/is_match as label source."""

    def test_missing_metrics_with_recompute(self):
        """When reward_metrics is None and recompute_labels=True, label is re-computed."""

    def test_missing_metrics_without_recompute_raises(self):
        """When reward_metrics is None and recompute_labels=False, raises ValueError."""

    def test_missing_metric_key_with_recompute(self):
        """When label_metric_key not in reward_metrics and recompute=True, falls back."""

    def test_missing_metric_key_without_recompute_raises(self):
        """When label_metric_key not in reward_metrics and recompute=False, raises ValueError."""

    def test_sample_id_preserved(self):
        """sample_id field matches the LMDB key-derived identifier."""

    def test_recompute_uses_soft_match(self):
        """recompute_labels=True calls compute_soft_match with expected and final_answer."""

    def test_recompute_uses_hard_match(self):
        """recompute_labels=True with label_metric_key='reward/metrics/hard_match/is_match' uses compute_hard_match."""

    def test_recompute_unsupported_metric_raises(self):
        """recompute_labels=True with non-soft/hard label_metric_key raises ValueError."""

    def test_recompute_falls_back_to_model_output(self):
        """When final_answer is None, recompute uses model_output as predicted."""

    def test_malformed_record_raises_by_default(self):
        """Record with None prompt/model_output raises ValueError by default."""

    def test_malformed_record_skipped_with_flag(self):
        """Record with None prompt/model_output is skipped when skip_malformed_records=True."""

    def test_missing_label_key_raises_by_default(self):
        """Record with reward_metrics missing label_metric_key raises ValueError."""

    def test_missing_label_key_skipped_with_flag(self):
        """Record with missing label_metric_key is skipped when skip_malformed_records=True."""

    def test_missing_expected_output_with_recompute_raises(self):
        """Record with None expected_output raises ValueError when recompute_labels=True."""

    def test_missing_expected_output_with_recompute_skipped(self):
        """Record with None expected_output is skipped when recompute + skip_malformed."""


class TestLoadProbeDatasetFromLmdb:
    """Integration tests for the main loading function."""

    def test_returns_dataset_dict_with_splits(self, debug_lmdb):
        """Returns DatasetDict with 'train' and 'valid' keys."""

    def test_train_valid_sizes_match_prefixes(self, debug_lmdb):
        """Split sizes match the number of records with each prefix."""

    def test_dataset_has_required_columns(self, debug_lmdb):
        """Each split has 'text', 'label', 'sample_id' columns."""

    def test_labels_are_binary(self, debug_lmdb):
        """All labels are 0 or 1."""

    def test_both_labels_present(self, debug_lmdb):
        """Both label=0 and label=1 are present in train split."""

    def test_max_samples_per_split(self, debug_lmdb):
        """max_samples_per_split caps the number of samples."""

    def test_text_not_empty(self, debug_lmdb):
        """All text fields are non-empty strings."""

    def test_warns_on_single_class_split(self, debug_lmdb_single_class):
        """Warns when a split has only one label class."""


class TestValidation:
    """Tests for dataset validation."""

    def test_rejects_non_binary_labels(self):
        """Raises on labels outside {0, 1}."""

    def test_rejects_empty_text(self):
        """Raises on empty text fields."""
```

### 9.2 Modified Tests: `tests/probes/test_debug_dataset.py`

Update to test the new LMDB-based debug dataset:

```python
class TestDebugProbeLmdb:
    """Tests for the mock LMDB creation."""

    def test_creates_valid_lmdb(self, tmp_path):
        """create_debug_probe_lmdb produces a readable LMDB."""

    def test_round_trip_serialization(self, tmp_path):
        """Records written by create_debug_probe_lmdb are readable by LMDBReader (JSON_ZSTD)."""

    def test_record_has_required_fields(self, tmp_path):
        """Each record has prompt, model_output, expected_output, reward_metrics."""

    def test_train_and_eval_prefixes(self, tmp_path):
        """Records use 'train/' and 'eval/' key prefixes."""

    def test_sample_counts(self, tmp_path):
        """n_train and n_valid control record counts per prefix."""

    def test_deterministic_with_seed(self, tmp_path):
        """Same seed produces identical LMDB contents."""

    def test_labels_have_signal(self, tmp_path):
        """Keyword signal produces correlated labels (not random)."""

    def test_noise_flips_some_labels(self, tmp_path):
        """noise_rate > 0 flips some labels for non-trivial task."""


class TestDebugProbeDataset:
    """Tests for the convenience wrapper."""

    def test_returns_dataset_dict(self, tmp_path):
        """create_debug_probe_dataset returns DatasetDict with train/valid."""

    def test_dataset_has_text_and_label(self, tmp_path):
        """Each split has 'text' and 'label' columns."""

    def test_labels_are_binary(self, tmp_path):
        """All labels are 0 or 1."""

    def test_both_labels_present(self, tmp_path):
        """Both classes present in train split."""
```

### 9.3 Modified Tests: `tests/apps/trainers/test_probe_trainer.py`

**Removed test classes:**

- `TestDatasetValidation` — replaced by `tests/probes/test_lmdb_dataset.py`

**Modified fixtures:**

```python
@pytest.fixture
def debug_lmdb(tmp_path) -> Path:
    """Create a debug LMDB for probe training tests."""
    from pyine.probes.debug_dataset import create_debug_probe_lmdb
    return create_debug_probe_lmdb(tmp_path / "debug.lmdb", n_train=40, n_valid=10, seed=42)

@pytest.fixture
def probe_hf_dataset(debug_lmdb) -> datasets.DatasetDict:
    """Load the debug LMDB as an HF DatasetDict."""
    from pyine.probes.lmdb_dataset import load_probe_dataset_from_lmdb
    return load_probe_dataset_from_lmdb(debug_lmdb)
```

**Modified `TestProbeTrainUnit`:**

- `test_train_step_reduces_loss`: Use tokenized `probe_hf_dataset` instead of random `input_ids`
- `test_validation_produces_metrics`: Build dataloader from `probe_hf_dataset`
- `test_save_probe_checkpoints`, `test_probe_checkpoints_loadable`: Unchanged (don't depend on dataset format)
- `test_auroc_guard_single_class`: Use a single-class subset of `probe_hf_dataset`

**New tests on `TestProbeTrainUnit`:**

```python
def test_tokenize_split(self):
    """_tokenize_split produces correct columns and format."""

def test_tokenize_split_no_special_tokens(self):
    """_tokenize_split does not add BOS/EOS or chat markers (add_special_tokens=False)."""
```

**Unchanged test classes** (no dataset dependency):

- `TestStableReplicaSeed`
- `TestExpandProbeConfigsWithReplicas`
- `TestAggregateReplicaMetrics`
- `TestReplicaTables`
- `TestTrainStepWithReplicas`

### 9.4 Modified Tests: `tests/apps/trainers/test_probe_trainer_configs.py`

Update `TestProbeTrainerAppMainConfig`:

```python
def test_valid_config_construction(self):
    """Config with lmdb_path validates successfully."""
    # Replace dataset_path with lmdb_path

def test_lmdb_path_required(self):
    """Config without lmdb_path raises."""

def test_label_metric_key_default(self):
    """Default label_metric_key is 'reward/metrics/soft_match/is_match'."""

def test_selection_strategy_values(self):
    """selection_strategy accepts 'latest' and 'best_reward'."""

def test_selection_strategy_invalid_raises(self):
    """Invalid selection_strategy raises validation error."""

def test_skip_malformed_records_default_false(self):
    """Default skip_malformed_records is False."""

def test_recompute_with_unsupported_metric_rejected(self):
    """recompute_labels=True with arbitrary label_metric_key is rejected by model_validator."""

def test_recompute_with_soft_match_accepted(self):
    """recompute_labels=True with 'reward/metrics/soft_match/is_match' passes config validation."""

def test_recompute_with_hard_match_accepted(self):
    """recompute_labels=True with 'reward/metrics/hard_match/is_match' passes config validation."""
```

### 9.5 New Fixtures: `tests/probes/conftest.py`

```python
@pytest.fixture
def debug_lmdb(tmp_path: Path) -> Path:
    """Minimal debug LMDB for fast probe tests."""
    from pyine.probes.debug_dataset import create_debug_probe_lmdb
    return create_debug_probe_lmdb(tmp_path / "debug.lmdb", n_train=40, n_valid=10, seed=42)
```

### 9.6 Test Summary

| Target file                                         | Change type  | # New/Modified tests |
| --------------------------------------------------- | ------------ | -------------------- |
| `tests/probes/test_lmdb_dataset.py`                 | **New file** | ~32 tests            |
| `tests/probes/test_debug_dataset.py`                | **Rewrite**  | ~12 tests            |
| `tests/probes/conftest.py`                          | **Modify**   | 1 new fixture        |
| `tests/apps/trainers/test_probe_trainer.py`         | **Modify**   | ~6 modified, 2 new   |
| `tests/apps/trainers/test_probe_trainer_configs.py` | **Modify**   | ~9 modified          |
| **Total**                                           |              | ~61 tests            |

______________________________________________________________________

## 10. File Changes Summary

| File                                                | Change type | Description                                                               |
| --------------------------------------------------- | ----------- | ------------------------------------------------------------------------- |
| `pyine/probes/lmdb_dataset.py`                      | **New**     | LMDB → HF dataset loading, record conversion, validation                  |
| `pyine/probes/debug_dataset.py`                     | **Rewrite** | Generate mock LMDB instead of chat-format HF dataset                      |
| `pyine/probes/__init__.py`                          | **Modify**  | Export `load_probe_dataset_from_lmdb`                                     |
| `pyine/apps/trainers/probe_trainer.py`              | **Modify**  | Replace dataset loading with LMDB pipeline, simplify tokenization         |
| `pyine/apps/trainers/probe_trainer_configs.py`      | **Modify**  | Replace `dataset_path`/`text_field`/`label_field` with LMDB config fields |
| `pyine/configs/experiment/probes/v0_probe.yaml`     | **Modify**  | Update config fields                                                      |
| `tests/probes/test_lmdb_dataset.py`                 | **New**     | LMDB dataset loading tests                                                |
| `tests/probes/test_debug_dataset.py`                | **Rewrite** | Mock LMDB + convenience wrapper tests                                     |
| `tests/probes/conftest.py`                          | **Modify**  | Add `debug_lmdb` fixture                                                  |
| `tests/apps/trainers/test_probe_trainer.py`         | **Modify**  | Use LMDB-based fixtures, remove old dataset validation tests              |
| `tests/apps/trainers/test_probe_trainer_configs.py` | **Modify**  | Update config validation tests                                            |

______________________________________________________________________

## 11. Implementation Steps

### Step 1: Create `pyine/probes/lmdb_dataset.py`

Implement the LMDB loading module with:

- `_load_lmdb_records()` — key prefix filtering + deduplication
- `_record_to_probe_sample()` — text assembly + label extraction
- `_validate_probe_dataset()` — binary label check, both-classes warning
- `load_probe_dataset_from_lmdb()` — main entry point

**Dependencies**: `pyine.data.utils.lmdb_io.LMDBReader`, `pyine.utils.code.output_compare` (for fallback re-computation).

**Tests**: `tests/probes/test_lmdb_dataset.py` — run and pass all unit tests.

### Step 2: Rewrite `pyine/probes/debug_dataset.py`

Replace chat-message generation with mock LMDB creation:

- `create_debug_probe_lmdb()` — writes mock records using `LMDBWriter` with `SerializationMethod.JSON_ZSTD` (matching `DiskRewardLogger`'s production format)
- `create_debug_probe_dataset()` — convenience wrapper that creates LMDB + loads it

**Tests**: `tests/probes/test_debug_dataset.py` — run and pass all tests (including round-trip read via `LMDBReader`).

### Step 3: Update `pyine/apps/trainers/probe_trainer_configs.py`

- Remove `dataset_path`, `text_field`, `label_field`
- Add `lmdb_path`, `label_metric_key`, `train_key_prefix`, `valid_key_prefix`, `selection_strategy`, `recompute_labels`, `max_samples_per_split`

**Tests**: `tests/apps/trainers/test_probe_trainer_configs.py` — update and pass.

### Step 4: Update `pyine/apps/trainers/probe_trainer.py`

- Remove `validate_probe_dataset()`, `tokenize_for_probes()`, `load_and_tokenize()`
- Add `_tokenize_split()` (simple plain-text tokenization)
- Update `probe_train()` to use `load_probe_dataset_from_lmdb()` + `_tokenize_split()`
- Keep `build_dataloader()` (still needed for collation)

**Tests**: `tests/apps/trainers/test_probe_trainer.py` — update fixtures and run.

### Step 5: Update experiment config and exports

- Update `pyine/configs/experiment/probes/v0_probe.yaml`
- Update `pyine/probes/__init__.py` exports

### Step 6: Run full test suite

```bash
uv run pytest tests/probes/ tests/apps/trainers/test_probe_trainer.py tests/apps/trainers/test_probe_trainer_configs.py -v
```

### Step 7: Manual smoke test

```bash
# Generate debug LMDB
python -m pyine.probes.debug_dataset --output /tmp/probe-debug-lmdb

# Single GPU
python -m pyine.apps.trainers.probe_trainer +experiment=probes/v0_probe \
    config.lmdb_path=/tmp/probe-debug-lmdb

# With real RL output LMDB
python -m pyine.apps.trainers.probe_trainer +experiment=probes/v0_probe \
    config.lmdb_path=/path/to/rl-run/reward_logs.lmdb
```

______________________________________________________________________

## 12. Design Decisions

### 12.1 Direct LMDB Reading vs. Full Datamodule Integration

**Decision: Direct LMDB reading.**

The full datamodule pipeline (`BiasDataModuleBase` → `SampleBuilder` → transforms) is designed for RL training with complex trace selection, code augmentation, and prompt construction. For probe training, we only need the already-computed completion records. Using the full datamodule would add unnecessary complexity:

- Requires the original trace LMDB(s) in addition to the completion LMDB
- Requires prompt template configuration matching the RL run
- Requires the split file for train/valid partitioning
- The `SampleBuilder` pipeline is not designed for this use case

The completion LMDB already contains everything needed: the formatted prompt, the model's output, and the reward metrics. Direct reading is simpler, faster, and self-contained.

### 12.2 Plain Text vs. Chat Messages

**Decision: Plain text (prompt + model_output concatenation).**

The stored `prompt` field is already post-chat-template. Re-applying the chat template to reconstructed messages would:

- Require access to the model's chat template at dataset creation time
- Risk tokenization mismatches if the template differs from the RL run
- Add complexity with no benefit

Plain text tokenization of `prompt + model_output` produces tokens identical to what the model saw.

### 12.3 Pre-computed Labels vs. Re-computation

**Decision: Pre-computed by default, re-computation as fallback.**

Pre-computed `reward_metrics["reward/metrics/soft_match/is_match"]` is:

- Always available when `DiskRewardLogger` logged with `log_metrics=True`
- Exactly what was used during RL (no evaluation drift)
- Zero additional computation

Re-computation is available via `recompute_labels=True` for cases where:

- The LMDB was exported without metrics
- Different comparison settings are desired
- A different match criterion is needed

### 12.4 Why Replace Rather Than Extend the Interface

**Decision: Replace the old `dataset_path` interface entirely.**

The old interface was a temporary bootstrapping mechanism. With the LMDB pipeline:

- All production probe training will use RL completion data
- The debug dataset now produces mock LMDBs (same interface)
- Maintaining two data loading paths doubles testing/maintenance burden
- The LMDB interface is strictly more capable

______________________________________________________________________

## 13. Edge Cases and Error Handling

| Scenario                                                        | Handling                                                                                                                                                                                                                                                                                                                                                                                        |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LMDB record has `reward_metrics = None`                         | If `recompute_labels=True` (and `label_metric_key` is `reward/metrics/soft_match/is_match` or `reward/metrics/hard_match/is_match`), re-compute via the matching function. Otherwise raise `ValueError` with clear message.                                                                                                                                                                     |
| `label_metric_key` not found in `reward_metrics`                | Same as above — fallback or raise.                                                                                                                                                                                                                                                                                                                                                              |
| `recompute_labels=True` with unsupported `label_metric_key`     | Raise `ValueError` — enforced in config validation (pydantic `model_validator`) and mirrored by runtime check in `load_probe_dataset_from_lmdb()`.                                                                                                                                                                                                                                              |
| Record missing required fields (see below)                      | **Default**: raise `ValueError` (fail fast). If `skip_malformed_records=True`, skip and log count at `WARNING` level. **Required fields per mode**: `recompute_labels=False` requires `prompt`, `model_output`, `reward_metrics[label_metric_key]`; `recompute_labels=True` requires `prompt`, `model_output`, `expected_output` (`final_answer` may be `None` — falls back to `model_output`). |
| `selection_strategy="best_reward"` and `reward_total` is `None` | Raise `ValueError` — cannot compare `None` rewards. Indicates data export issue.                                                                                                                                                                                                                                                                                                                |
| LMDB key `key_prefix` disagrees with `record["key_prefix"]`     | Log a warning (using `logging.getLogger(__name__).warning()`). Trust the LMDB key for split assignment.                                                                                                                                                                                                                                                                                         |
| Generation count segment is `"none"`                            | Treat as `generation_count = 0` (matches `DiskRewardLogger` convention for `None`).                                                                                                                                                                                                                                                                                                             |
| Generation count segment is non-numeric and not `"none"`        | Raise `ValueError` with the offending key for diagnosis.                                                                                                                                                                                                                                                                                                                                        |
| All records for a split have the same label                     | Log warning via `logging.getLogger(__name__).warning()` with split name and class counts. Training may be degenerate.                                                                                                                                                                                                                                                                           |
| No records match `train_key_prefix`                             | Raise `ValueError` — cannot train without training data.                                                                                                                                                                                                                                                                                                                                        |
| No records match `valid_key_prefix`                             | Raise `ValueError` — cannot validate without validation data.                                                                                                                                                                                                                                                                                                                                   |
| `final_answer` is `None` during re-computation                  | Fall back to `model_output` as predicted value.                                                                                                                                                                                                                                                                                                                                                 |
| Multiple LMDB paths needed                                      | Out of scope for v1. User can merge LMDBs externally or we add `lmdb_paths: list[str]` later.                                                                                                                                                                                                                                                                                                   |

______________________________________________________________________

## 14. Future Extensions (Out of Scope)

- **Multiple LMDB sources**: Support `lmdb_paths: list[str]` for merging data from multiple RL runs.
- **Full datamodule integration**: Use `BiasDataModuleBase` for trace-level filtering and augmentation-aware probe training.
- **On-the-fly label derivation**: Compute labels during training (e.g., with an LLM grader) instead of at dataset load time.
- **Streaming LMDB loading**: For very large LMDBs, stream records instead of loading all into memory.
- **Multi-class labels**: Support non-binary labels (e.g., reward buckets or graded scores).
- **Prompt-only probing**: Train probes on prompt activations only (without completion) to detect properties of the input.

______________________________________________________________________

## Appendix A: Codex Review Assessment

The following summarizes the review feedback from `INTEGRATION_PLAN_CODEX.md` and the changes applied to this plan.

### A.1 Accepted Changes

| #   | Finding                                                                                                               | Assessment                                                                                                   | Action Taken                                                                                                                                                                                                                 |
| --- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Ambiguous key parsing** — plan didn't specify how multi-segment `sample_id` is split from `generation_count`        | Valid. The existing codebase uses `rfind("/")` in `_load_pregenerated_outputs()` and handles `"none"` → `0`. | Added explicit key parsing rules (Section 3), new tests for multi-segment IDs, `"none"` generation count, and non-numeric errors (Section 9.1).                                                                              |
| 2   | **Recompute/label_metric_key mismatch** — recompute always used `compute_soft_match` regardless of `label_metric_key` | Valid. Restricting recompute to supported metrics is cleaner than computing all and selecting.               | Restricted `recompute_labels=True` to `reward/metrics/soft_match/is_match` and `reward/metrics/hard_match/is_match` only; raises `ValueError` otherwise (Section 4.2, 4.3). Added test for unsupported metric (Section 9.1). |
| 3   | **Prefix filtering source of truth** — LMDB key vs. record field `key_prefix` could diverge                           | Valid but low-risk. LMDB key is authoritative since it determines iteration order.                           | LMDB key is source of truth; record field is validated with warning on mismatch (Section 6.1). Added test for mismatch warning (Section 9.1).                                                                                |
| 4   | **Missing `add_special_tokens=False`** — tokenization didn't specify special token handling                           | Valid and important. Codebase consistently uses `add_special_tokens=False` for post-template text.           | Added `add_special_tokens=False` to `_tokenize_split()` with documented rationale (Sections 5.1, 7.3). Added no-special-tokens test (Section 9.3).                                                                           |
| 5   | **`reward_total=None` with `best_reward` strategy** — undefined sort behavior                                         | Valid. `None` comparison would raise `TypeError` in Python.                                                  | Raise `ValueError` when `best_reward` encounters `None` reward (Section 6.2, 13). Added test (Section 9.1).                                                                                                                  |
| 6   | **"Skip with warning" conflicts with fail-fast ethos** — silent data loss on `None` prompt/model_output               | Partially valid. Fail fast is correct default, but a permissive mode is useful for noisy exports.            | Default: raise `ValueError`. Added `skip_malformed_records` config flag for permissive mode with logged counts (Sections 7.1, 7.2, 13). Added tests for both paths (Section 9.1).                                            |
| 7   | **Debug LMDB serialization unspecified** — could differ from production `DiskRewardLogger` format                     | Valid. Serialization mismatch would make debug data unreadable.                                              | Specified `SerializationMethod.JSON_ZSTD` for debug LMDB writer (Section 7.4, Step 2). Added round-trip read test (Section 9.2).                                                                                             |
| 8   | **Warning location unspecified** — unclear logging convention                                                         | Minor but valid.                                                                                             | Specified `logging.getLogger(__name__).warning()` in all warning locations (Section 13).                                                                                                                                     |

### A.2 Open Questions Resolved

| #   | Question                                                         | Answer                                                                                                                                                                                                |
| --- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Should split selection use LMDB key prefixes exclusively?        | Yes. LMDB key is source of truth. Record `key_prefix` field is validated with warning on mismatch but not used for split assignment.                                                                  |
| 2   | Can generation count be non-numeric (e.g., `"final"`)?           | Only `"none"` is a known non-numeric value (emitted by `DiskRewardLogger` when `generation_count is None`). Any other non-numeric value raises `ValueError`.                                          |
| 3   | Is `reward_metrics` a flat dict?                                 | Yes. `RewardManager` stores metrics as `{"reward/metrics/{term_name}/{metric_name}": value}` — always a flat dict with string keys. `label_metric_key` is a simple key lookup, not a path expression. |
| 4   | Should we preserve additional fields (tags, predict_type, etc.)? | Not for v1. The dataset schema is `{text, label, sample_id}`. Additional fields can be added later if needed for downstream analysis (see Section 14, Future Extensions).                             |

### A.3 Second Review — Accepted Changes

| #   | Finding                                                                                                | Assessment                           | Action Taken                                                                                                                                                                                                                                                                 |
| --- | ------------------------------------------------------------------------------------------------------ | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Key parsing snippet has offset bug** — `sep` is relative to sliced string but used as absolute index | Valid. The code was wrong.           | Fixed snippet to use `suffix` variable: `suffix = key[len(key_prefix):]`, then `suffix[:sep]` / `suffix[sep+1:]` (Section 3).                                                                                                                                                |
| 2   | **`recompute_labels` validation location ambiguous** — "at startup" could mean config or runtime       | Valid. Should be enforced early.     | Added pydantic `model_validator` on `ProbeTrainerAppMainConfig` for config-time enforcement, plus defensive runtime check in `load_probe_dataset_from_lmdb()` (Sections 4.3, 7.2). Added config validation tests (Section 9.4).                                              |
| 3   | **`best_reward` + `None` test already exists** — reviewer missed `test_best_reward_none_raises`        | Already addressed in prior revision. | No change needed — test was already in Section 9.1.                                                                                                                                                                                                                          |
| 4   | **"Malformed" definition unclear** — which fields are required under which mode?                       | Valid and useful.                    | Explicitly listed required fields per mode in `skip_malformed_records` description (Section 7.2), function docstring (Section 7.1), and edge cases table (Section 13). Added tests for missing `label_metric_key`, missing `expected_output` during recompute (Section 9.1). |
| 5   | **Flow diagram references removed function** — `tokenize_for_probes()` in Section 2.2                  | Valid, trivial fix.                  | Updated to `_tokenize_split()` (Section 2.2).                                                                                                                                                                                                                                |

### A.4 Second Review — Open Questions Resolved

| #   | Question                                                                           | Answer                                                                                                                                                                                |
| --- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Should `skip_malformed_records=True` also skip records missing `label_metric_key`? | Yes. Missing required fields are treated uniformly — `skip_malformed_records` covers all of them.                                                                                     |
| 2   | Are `compare_options` required for hard match recompute?                           | No. `compute_hard_match` is a strict string comparison with no tolerance parameters. `compare_options` is only used for soft match. Noted in docstring (Section 7.1) and Section 4.3. |
| 3   | Is `label_metric_key` validation strict or permissive?                             | Permissive. Any key that exists in `reward_metrics` is accepted. The restriction only applies when `recompute_labels=True`. Clarified in Section 4.3.                                 |
