# Guardrail Correctness Evaluation (`pyine.evals.correctness`)

Evaluates **guardrails**, i.e. classifiers that predict whether a target model's output is correct.
The pipeline loads pregenerated outputs and labels from existing LMDB eval exports, applies
problem-level splits, runs a provided guardrail scorer, calibrates a decision threshold on a
dev set, computes a ton of metrics, and aggregates across multiple independent runs.

## Pipeline Overview

```
CorrectnessEvalsConfig
    │
    ├────► apply optional calibration data resampling
    │      (depends on whether calibration_resampling is set)
    │
    ├────────────────► CorrectnessDataModuleConfig
    │                              ├────► apply optional training data resampling
    │                              │      (depends on whether resampling is set)
    │                              │
    │                    prepare_data() + setup()
    │                              │
    │                              ▼
    │                    CorrectnessDataModule
    │                      ├── load_records_from_lmdb()  ──►  list[EvalRecord]
    │                      ├── build_guardrail_splits()  ──►  GuardrailSplits
    │                      ├── get_records_for_subset("guardrail_train"|"..._valid"|"..._test")
    │                      ├── get_records_for_calibration(resampling_config=...)
    │                      ├── get_records_for_training()   (applies resampling if provided)
    │                      ├── get_records_for_validation() (applies resampling if provided)
    │                      └── get_probe_dataset()          (train/valid splits with messages schema)
    │
    ├── text_field ──► selects which EvalRecord field to score (default: "model_output")
    │
    ▼
evaluate_guardrail_replicas(guardrails, datamodule, eval_subset_name)
    │
    ├──► guardrail.score_records(calibration_records)  ──►  calibrate_threshold()
    │                                                            │
    ├──► guardrail.score_records(eval_records)          ◄────────┘ (threshold)
    │         │
    │         ├──► compute_threshold_free_metrics()
    │         ├──► compute_thresholded_metrics()       (per target_fpr)
    │         ├──► compute_sample_level_metrics()      (per target_fpr)
    │         ├──► compute_category_results()
    │         ├──► compute_clustered_bootstrap_cis()
    │         ├──► compute_difficulty_stats()
    │         └──► compute_verification_cost_stats()   (per target_fpr)
    │
    ▼
SingleRunResult  ──►  (repeat for R guardrail runs)
    │
    ▼
_aggregate_runs()  ──►  AggregatedResult
    │
    ▼
CorrectnessEvalResult (metrics=to_flat_dict(), aggregated=AggregatedResult)
```

## Split Strategy

Splits are at the **problem level**: all samples (and their solutions, traces, augmentations) for a
coding problem stay together. This prevents data leakage between guardrail training and evaluation.

- Original **train** problems -> discarded entirely (these might be overfit, so unreliable for
  evaluations or calibration).
- Original **valid** problems -> re-split into `guardrail_train` + `guardrail_valid`:
  - when `stratify_by_label=True`: stratified by per-problem correctness rate
    (all-correct / all-incorrect / mixed strata);
  - the split module (`splits.py`) is standalone and importable by training pipelines.
- Original **test** problems -> `guardrail_test` (final evaluation).

## Threshold Calibration

The pipeline uses an FPR-constrained approach: given `target_fpr` (e.g. 0.01), select the
highest threshold such that FPR \<= `target_fpr` on the validation set.

- Convention: FPR = P(score >= threshold | label=False).
- Tie-safety: if ties at the boundary cause FPR to exceed the target, the threshold is nudged
  upward until the constraint holds.
- Degenerate case: when `floor(target_fpr * n_incorrect) = 0`, returns a threshold just above
  the highest negative score; positives scored above all negatives are still accepted, but
  positives tied with the highest negative are blocked.

## Metrics Glossary

**Positive class = "correct"** (label=True means the model output is correct and should be accepted
by the guardrail).

### Threshold-Free Metrics

| Metric            | Description                                                                                                                   |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| AUROC             | Area under the ROC curve. 1.0 = perfect ranking, 0.5 = random                                                                 |
| Average Precision | Average precision score (weighted mean of precisions at each recall threshold, via `sklearn.metrics.average_precision_score`) |
| TPR@FPR           | TPR interpolated at specific FPR levels                                                                                       |

None when the subset contains only one label class (e.g. all-incorrect biasing category).

### Thresholded Metrics (per target FPR)

| Metric         | Description                                               |
| -------------- | --------------------------------------------------------- |
| TP, FP, TN, FN | Confusion matrix counts                                   |
| TPR            | TP / (TP + FN); None when no correct attempts             |
| FPR            | FP / (FP + TN); None when no incorrect attempts           |
| FNR            | FN / (TP + FN); None when no correct attempts (= 1 - TPR) |
| Precision      | TP / (TP + FP); None when all blocked                     |
| NPV            | TN / (TN + FN); None when none blocked                    |

### Sample-Level Metrics (per target FPR)

| Metric                 | Description                                                                              |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| base_pass_rate         | Fraction of samples with >=1 correct attempt (baseline)                                  |
| guarded_pass_rate      | Fraction with >=1 correct AND accepted attempt                                           |
| unsafe_slip_rate       | Fraction with >=1 incorrect AND accepted attempt                                         |
| total_block_rate       | Fraction where all attempts blocked                                                      |
| best_of_k_success_rate | Among non-blocked, fraction where any top-scored accepted attempt is correct (tie-aware) |

### Conservative Deployment Strategy Metrics (per target FPR)

The sample-level metrics above model an **optimistic** deployment strategy: accept a sample if
ANY of its K attempts passes the guardrail. The conservative strategy below takes the opposite
posture: reject the entire sample if ANY attempt is flagged as incorrect.

| Metric                     | Description                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| cons_pass_rate             | Fraction of samples where ALL K attempts are accepted by the guardrail                                             |
| cons_unsafe_slip_rate      | Among cons-accepted samples, fraction with >=1 incorrect attempt (None when cons_pass_rate == 0)                   |
| cons_justified_reject_rate | Among cons-rejected samples, fraction where >=1 blocked attempt is truly incorrect (None when cons_pass_rate == 1) |

Note: `cons_pass_rate` is NOT bounded by `base_pass_rate` a sample can have all attempts accepted
(guardrail-decision-based) yet have no correct attempts (label-based).

### Category-Wise Metrics

Category extraction uses two complementary systems:

- **`RecordCategoryConfig`** handles code_type grouping into `regular` (non-biasing code types:
  original, obfuscated, stubbed) vs `biasing/*` (bias-induced failures: hinted, misleading,
  bugged). Compound code types (e.g. `bugged_hinted`) appear in each matching biasing category
  and in a combined category using the canonical code type set string (e.g.
  `biasing/bugged_hinted`). When
  `report_per_code_type=True`, raw per-code_type groupings are also emitted (e.g.
  `code_type/misleading`).
- **`category_extraction_config`** (inherited from `BaseEvalsConfig`) extracts categories from
  other record fields: `predict_type`, `has_keyword`, `identifier_suffix`, and `tags`. The
  `code_type` field is automatically filtered out of this extractor to avoid duplicating the
  grouped code_type categories above.

Per-record categories from both systems are merged and deduplicated before indexing.

The threshold is shared (calibrated on the full dev set) so performance is comparable across
categories.

## Uncertainty Quantification

### Problem-Clustered Bootstrap CIs

Per replicate: resample problem_ids with replacement -> include all samples and their K attempts
-> recompute all metrics. Clustering at the problem level respects the correlation between
samples from the same coding problem.

**Calibration note**: CIs are **conditional on fixed calibrated thresholds**. Thresholds are
not re-calibrated per bootstrap replicate. This captures the dominant uncertainty source
(test set sampling variance). For threshold selection variance, the hierarchical bootstrap
across R independent runs provides indirect coverage.

### Hierarchical Bootstrap CIs (Cross-Run)

Per replicate: resample run indices with replacement -> within each, resample problems with
replacement -> compute mean metric across resampled draws. Accounts for both run variance
(different guardrail training seeds) and test set sampling variance.

## GuardrailScorer Protocol

This is the interface for guardrail scorers that must be implemented in upstream apps:

```python
# given such records:
@dataclasses.dataclass(frozen=True)
class EvalRecord:
    sample_id: str                    # trace-level identifier from LMDB
    problem_id: str                   # coding-problem-level ID (e.g. "TACO/TRAIN/p000001")
    attempt_index: int                # zero-based attempt index
    model_output: str                 # full raw model output (includes reasoning)
    final_answer: str | None          # parsed final answer, or None
    expected_output: str              # ground-truth expected output
    label: bool                       # correctness label (from hard_match or soft_match)
    code_type: str                    # e.g. "original", "misleading", "bugged_hinted"
    tags: list[str]                   # from LMDB record
    difficulty_score: float | None    # from LMDB when present, else None
    record: dict[str, typing.Any]     # full LMDB record for ad-hoc access

# guardrails should implement this:
class GuardrailScorer(typing.Protocol):
    def score_records(self, records: list[EvalRecord]) -> ScoringResult: ...
    def get_metadata(self) -> dict[str, typing.Any]: ...
    def get_verification_cost_unit(self) -> str | None: ...

# returning this:
class ScoringResult(pydantic.BaseModel):
    scores: list[float]
    """One continuous score per record (higher = more likely correct)."""
    verification_costs: list[float] | None = None
    """Per-record compute cost (e.g. token counts). None when the scorer does not report costs."""
```

Higher scores = higher confidence the output is correct (should be accepted);
`get_verification_cost_unit()` returns the unit label for costs (e.g. `'tokens'`, `'FLOPs'`,
`'turns'`), or `None` when the scorer does not report costs.

Three implementations exist:

- **`ProbeScorer`** and **`LLMClassifierScorer`** (in `scorers.py`) — trained model adapters (see below).
- **`PromptedLLMGuardrailScorer`** (in `pyine.guardrails.prompted_llm`) — inference-only scorer
  using a prompted LLM judge (see below).

## DataModule

`CorrectnessDataModule` wraps LMDB loading and split construction behind the standard Lightning
DataModule lifecycle (`prepare_data()` -> `setup()` -> data accessors -> `teardown()`).

`CorrectnessDataModuleConfig` holds:

- `lmdb_paths`: paths (or glob patterns) to pregenerated eval records from the code exec pipeline;
- `label_type`: which correctness label to use (`SOFT_MATCH` by default);
- `split_config`: split source, valid fraction, seed, stratification;
- `resampling`: optional `RecordResamplingConfig` for training and validation data composition
  control (e.g. label balance, code type diversity). When set, `get_records_for_training()`,
  `get_records_for_validation()`, and `get_probe_dataset()` all apply it symmetrically.

`CorrectnessEvalsConfig` additionally holds:

- `calibration_resampling`: optional `RecordResamplingConfig` for calibration data resampling.
  When set, `get_records_for_calibration()` resamples the validation records before threshold
  calibration (e.g. to study guardrail robustness to skewed calibration sets).

After `setup()`, the datamodule provides:

- `get_records_for_subset(name)`: returns raw records for `"guardrail_train"`,
  `"guardrail_valid"`, or `"guardrail_test"`;
- `get_records_for_calibration(resampling_config=None)`: always draws from `guardrail_valid`
  records, optionally resampled (threshold calibration must never use test data);
- `get_records_for_training()`: returns `guardrail_train` records, resampled if
  `config.resampling` is set;
- `get_records_for_validation()`: returns `guardrail_valid` records, resampled if
  `config.resampling` is set (symmetric with training);
- `get_guardrail_splits()`: returns the full `GuardrailSplits` object;
- `get_all_records()`: returns all records before splitting;
- `get_probe_dataset(text_field)`: returns an HF `DatasetDict` with `messages`, `label`,
  `sample_id`, `code_type` columns and `train`/`valid` split names (probe-compatible);
- `get_stats()`: returns per-subset record/label/code_type counts. When `resampling` is
  set, also includes `guardrail_train_resampled/...` and `guardrail_valid_resampled/...` stats.

`CorrectnessEvalsConfig.prepare_eval_datamodule()` instantiates and sets up the datamodule
automatically; callers should not provide their own datamodule for this evaluation pipeline.

## Scorer Adapters

`scorers.py` provides `GuardrailScorer` implementations for trained models:

**`ProbeScorer`**: wraps a single probe (from a `ProbeCollection`) + base model + tokenizer +
`ActivationExtractor`. Tokenizes input text, forwards through the base model, extracts activations
at the configured layer, runs the probe, and applies sigmoid to produce scores.

**`LLMClassifierScorer`**: wraps a fine-tuned encoder classifier + tokenizer. Tokenizes input text,
forwards through the classifier, and extracts the positive class probability via softmax.

Both scorers receive `max_seq_length` and `text_field` from the training/eval configs.

**`PromptedLLMGuardrailScorer`** (`pyine.guardrails.prompted_llm.scorer`): inference-only scorer
that uses a LangChain chain (`prompt | model | parser`) to ask an LLM to judge whether a model's
prediction is correct. Returns continuous confidence scores (0–1) and tracks token costs via
`CaptureLLMHandler`. Requires no training step — the scorer is built directly from a
`PromptedLLMGuardrailConfig` specifying an `LLMProviderConfig` (supports OpenAI, vLLM, DeepSeek).
Has a standalone evaluation app (`pyine.apps.guardrail_eval.prompted_llm_eval`) with Hydra
experiment configs for each provider.

## Multi-Type Evaluation

`evaluate_guardrail_types()` evaluates multiple guardrail types independently, each with optional
replicas for cross-run aggregation. This is used by the probe trainer where multiple architectures
and layers are trained and evaluated simultaneously:

```python
# each key is a type name; the list contains replica instances
scorers_by_type = {
    "mean_pool_L8": [scorer_replica_0, scorer_replica_1, ...],
    "cls_L12": [scorer_replica_0, scorer_replica_1, ...],
}
results = await evaluate_guardrail_types(
    config=evals_config,
    guardrails_by_type=scorers_by_type,
    datamodule=eval_dm,
    eval_subset_name="guardrail_valid",
)
```

Results and W&B metrics are prefixed with the type name (e.g. `mean_pool_L8/auroc/mean`).

## Standalone Split Usage

Training pipelines can import the split logic independently to get access to the same train/valid set:

```python
import pyine.evals.correctness.splits as correctness_splits
import pyine.evals.correctness.data_loading as correctness_data

records = correctness_data.load_records_from_lmdb(lmdb_paths, label_type)
splits = correctness_splits.build_guardrail_splits(records, split_config)
train_records = splits.guardrail_train
valid_records = splits.guardrail_valid
```

## Difficulty and Cost Analysis

- `EvalRecord.difficulty_score`: read from the LMDB record when the `difficulty_score` field
  is present; otherwise `None`. When all records have scores, `DifficultyStats` reports
  per-tercile AUROC/TPR and `difficulty_accuracy_rank_correlation` (Spearman rank correlation
  between per-sample difficulty and guardrail classification accuracy, using the most conservative
  calibrated threshold).
- `ScoringResult.verification_costs`: reported by the scorer. `VerificationCostStats` aggregates
  total/mean/median/std, cost-per-TP/cost-per-TN breakdowns, and two Spearman rank correlations:
  `cost_accuracy_rank_correlation` (per-record cost vs. classification correctness) and
  `cost_difficulty_rank_correlation` (per-sample mean cost vs. mean difficulty, when difficulty
  scores are available). `cost_unit` carries the unit label from
  `GuardrailScorer.get_verification_cost_unit()` for axis labels in plots.
