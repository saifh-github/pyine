"""Core data types for the guardrail correctness evaluation pipeline."""

from __future__ import annotations

import dataclasses
import enum
import pathlib  # noqa: TC003
import typing

import numpy as np  # noqa: TC002
import numpy.typing as npt  # noqa: TC002
import pydantic

import pyine.utils.metrics.confidence  # noqa: TC001


class RecordResamplingConfig(pydantic.BaseModel):
    """Controls record resampling for calibration and training data.

    Supports three independent axes of control, applied in order:
    1. **Code type filtering/rebalancing** (``code_type_proportions``): removes records whose
       code_type is not listed, then resamples each listed code_type to match the target
       proportions.
    2. **Label ratio adjustment** (``target_positive_ratio``): downsamples (or oversamples) to
       achieve the target fraction of correct (label=True) records.
    3. **Size cap** (``max_records``): if the result exceeds this limit, a stratified random
       sample is drawn to the cap.

    After all steps, a validation check ensures both label classes are present with at least
    ``min_records_per_label`` each.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    seed: int = 0
    """Random seed for deterministic resampling."""
    target_positive_ratio: float | None = pydantic.Field(default=None, gt=0.0, lt=1.0)
    """Target fraction of label=True records. None preserves natural distribution.

    Bounds (0, 1) exclusive, but with small datasets, rounding may still drive a class count to
    zero, in which case ``resample_records`` raises a ``ValueError``.
    """
    code_type_proportions: dict[str, float] | None = None
    """Target relative weights for code_type groups (normalized internally).

    Keys are code_type strings. Unlisted code types are excluded. None preserves all code types at
    natural proportions.
    """
    strategy: typing.Literal["subsample", "oversample"] = "subsample"
    """'subsample' drops excess records (no duplicates); 'oversample' duplicates minority."""
    max_records: int | None = pydantic.Field(default=None, gt=0)
    """Cap on output size (applied after all other steps). None means no cap."""
    min_records_per_label: int = pydantic.Field(default=2, ge=1)
    """Minimum records per label class in the output (raises ValueError if violated)."""

    @property
    def is_noop(self) -> bool:
        """True when this config will not alter the record list."""
        return self.target_positive_ratio is None and self.code_type_proportions is None and self.max_records is None

    @classmethod
    def for_skewed_positive(cls, seed: int = 0) -> RecordResamplingConfig:
        """Preset for 80% positive label ratio instead of the original class balance."""
        return cls(seed=seed, target_positive_ratio=0.8)

    @classmethod
    def for_weak_bias(cls, seed: int = 0) -> RecordResamplingConfig:
        """Preset for a weak hint bias (10% hinted, with 9% helpful) and original class balance."""
        return cls(seed=seed, code_type_proportions={"original": 0.9, "hinted": 0.09, "misleading": 0.01})

    @classmethod
    def for_skewed_weak_bias(cls, seed: int = 0) -> RecordResamplingConfig:
        """Preset for a weak hint bias (10% hinted, with 9% helpful) with 80% correct labels."""
        return cls(
            seed=seed,
            target_positive_ratio=0.8,
            code_type_proportions={"original": 0.9, "hinted": 0.09, "misleading": 0.01},
        )

    @classmethod
    def for_moderate_bias(cls, seed: int = 0) -> RecordResamplingConfig:
        """Preset for a moderate hint bias (20% hinted, with 18% helpful) and original class balance."""
        return cls(seed=seed, code_type_proportions={"original": 0.8, "hinted": 0.18, "misleading": 0.02})

    @classmethod
    def for_skewed_moderate_bias(cls, seed: int = 0) -> RecordResamplingConfig:
        """Preset for a moderate hint bias (20% hinted, with 18% helpful) with 80% correct labels."""
        return cls(
            seed=seed,
            target_positive_ratio=0.8,
            code_type_proportions={"original": 0.8, "hinted": 0.18, "misleading": 0.02},
        )

    @classmethod
    def for_strong_bias(cls, seed: int = 0) -> RecordResamplingConfig:
        """Preset for a strong hint bias (50% hinted, with 45% helpful) and original class balance."""
        return cls(seed=seed, code_type_proportions={"original": 0.5, "hinted": 0.45, "misleading": 0.05})

    @classmethod
    def for_skewed_strong_bias(cls, seed: int = 0) -> RecordResamplingConfig:
        """Preset for a strong hint bias (50% hinted, with 45% helpful) with 80% correct labels."""
        return cls(
            seed=seed,
            target_positive_ratio=0.8,
            code_type_proportions={"original": 0.5, "hinted": 0.45, "misleading": 0.05},
        )

    @pydantic.field_validator("code_type_proportions")
    @classmethod
    def _validate_code_type_proportions(
        cls,
        value: dict[str, float] | None,
    ) -> dict[str, float] | None:
        """Validates that code_type_proportions is non-empty with finite positive values when set."""
        if value is None:
            return value
        if not value:
            raise ValueError("code_type_proportions must be non-empty when set")
        for key, weight in value.items():
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError(
                    f"code_type_proportions values must be finite and positive, got {weight} for key {key!r}"
                )
        return value


class LabelType(enum.StrEnum):
    """Selects which correctness label to use from LMDB records."""

    HARD_MATCH = enum.auto()
    """Use the hard_match field (exact string equality)."""
    SOFT_MATCH = enum.auto()
    """Use the soft_match field (relaxed matching with tolerance)."""


class GuardrailSplitConfig(pydantic.BaseModel):
    """Standalone config for building guardrail splits from an existing code problem split.

    Designed to be importable and usable independently by training pipelines.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    split_source: str | pathlib.Path
    """Dataset name or path to a split file, resolved via get_dataset_split_result()."""
    guardrail_valid_fraction: float = 0.8
    """Fraction of original validation problems assigned to guardrail_valid (rest to guardrail_train).

    Must be in (0, 1) exclusive.
    """
    seed: int = 42
    """Random seed for the valid-to-train/valid re-split."""
    stratify_by_label: bool = True
    """Stratify the valid to train/valid re-split by per-problem correctness rate."""
    include_original_train_problems: bool = False
    """When True, include original train problems in guardrail_train instead of discarding them.

    By default, original train problems are discarded from all guardrail splits. Setting this to True
    adds them to the guardrail_train split alongside the re-split validation problems, increasing the
    training set size, with the risk of using samples that models might have been exposed to directly
    during their original training.
    """

    @pydantic.field_validator("guardrail_valid_fraction")
    @classmethod
    def _validate_fraction(
        cls,
        value: float,
    ) -> float:
        """Validates guardrail validation dataset fraction."""
        if value <= 0.0 or value >= 1.0:
            raise ValueError(f"guardrail_valid_fraction must be in (0, 1), got {value}")
        return value


@dataclasses.dataclass(frozen=True)
class EvalRecord:
    """One eval attempt at one data sample, loaded from an eval LMDB export."""

    sample_id: str
    """Sample identifier (which should contain trace-level info) found in the LMDB."""
    problem_id: str
    """Coding-problem-level ID (e.g. TACO/TRAIN/p000001), extracted from sample id."""
    attempt_index: int
    """Zero-based attempt index for evaluations using the associated data sample."""
    model_output: str
    """Full raw model output (includes reasoning); guardrails typically analyze this."""
    final_answer: str | None
    """Parsed final answer when output parsing was used; None otherwise."""
    expected_output: str
    """Ground-truth expected output."""
    label: bool
    """Correctness label (from hard_match or soft_match on final_answer or model_output)."""
    code_type: str
    """From LMDB record (e.g. 'original', 'misleading', 'bugged_hinted')."""
    tags: list[str]
    """From LMDB record."""
    difficulty_score: float | None
    """How hard this problem is for the target model (None until wired in)."""
    record: dict[str, typing.Any]
    """Full LMDB record (so that consumers can access whatever they need)."""


type AttemptKey = tuple[str, int]
"""Base attempt identifier key: ``(sample_id, attempt_index)`` from the source LMDB."""


type ScoredAttemptKey = tuple[str, int, int]
"""Per-scored-row key: ``(sample_id, attempt_index, draw_index)``.

The ``draw_index`` disambiguates duplicated attempts created by resampling with replacement.
"""


class AttemptInspectionRecord(pydantic.BaseModel):
    """Per-attempt record for one-by-one qualitative correctness analysis."""

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""

    sample_id: str
    """Sample identifier."""
    problem_id: str
    """Problem identifier."""
    attempt_index: int
    """Zero-based attempt index."""
    draw_index: int
    """Zero-based index in the scored record list (post-resampling)."""
    label: bool
    """Ground-truth correctness label for this attempt."""
    score: float
    """Guardrail score for this attempt (higher = more likely correct)."""
    verification_cost: float | None = None
    """Optional per-attempt verification cost reported by the scorer."""
    final_answer: str | None
    """Parsed final answer when available."""
    code_type: str
    """Code type from LMDB metadata."""
    difficulty_score: float | None
    """Per-attempt difficulty score when available."""
    attempt_metadata: dict[str, typing.Any] | None = None
    """Optional scorer-provided metadata for this attempt."""


class ScoringResult(pydantic.BaseModel):
    """Output of GuardrailScorer.score_records().

    Bundles continuous scores with optional per-record verification costs and metadata.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""

    scores: list[float]
    """One continuous score per record (higher = more likely correct)."""
    verification_costs: list[float] | None = None
    """Per-record compute cost of the guardrail's assessment (e.g. token counts, FLOPs).

    None when the scorer does not report costs. When present, must be same length as scores and all
    values must be non-negative. The actual `cost` unit solely depends on the guardrail implementation.
    """
    attempt_metadata: dict[ScoredAttemptKey, dict[str, typing.Any]] | None = None
    """Optional per-attempt metadata keyed by ``(sample_id, attempt_index, draw_index)``.

    This is intended for lightweight forensic/debug context (for example input token counts).
    When present, it must contain exactly one entry per scored attempt.
    """

    @pydantic.model_validator(mode="after")
    def _validate_costs(self) -> ScoringResult:
        """Validates that optional arrays/maps align with ``scores`` length and constraints."""
        if self.verification_costs is not None:
            if len(self.verification_costs) != len(self.scores):
                raise ValueError("verification_costs must have same length as scores")
            if any(cost < 0 for cost in self.verification_costs):
                raise ValueError("verification_costs must be non-negative")
        if self.attempt_metadata is not None and len(self.attempt_metadata) != len(self.scores):
            raise ValueError("attempt_metadata must have same length as scores")
        return self


class GuardrailScorer(typing.Protocol):
    """Protocol for guardrails that produce continuous correctness scores.

    Higher scores = higher confidence the output is correct (i.e. that it should be accepted).
    """

    def score_records(
        self,
        records: list[EvalRecord],
    ) -> ScoringResult:
        """Score records and optionally report verification costs and per-attempt metadata."""
        ...

    def get_metadata(self) -> dict[str, typing.Any]:
        """Return guardrail metadata (architecture details, training seed, hyperparameters, etc.)."""
        ...

    def get_verification_cost_unit(self) -> str | None:
        """Return the unit label for verification costs (e.g. 'tokens', 'FLOPs', 'turns').

        Returns None when the scorer does not report verification costs. Used for
        axis labels in plots and metric descriptions in reports.
        """
        ...


class ThresholdFreeMetrics(pydantic.BaseModel):
    """Threshold-independent metrics computed over all score/label pairs.

    Measures the guardrail's ranking quality irrespective of any operating threshold. Curve grids
    are stored for downstream plotting (ROC/PR with confidence bands).
    """

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)
    """Pydantic model configuration (freezes the dataclass)."""

    auroc: float | None
    """Area under the ROC curve. 1.0 = perfect ranking, 0.5 = random.

    None when the subset contains only one label class.
    """
    average_precision: float | None
    """Average precision score (positive class = correct, label=True).

    Computed via `sklearn.metrics.average_precision_score` (weighted mean of precisions at each
    recall threshold). None when the subset contains only one label class.
    """
    tpr_at_fpr: dict[float, float] | None
    """Descriptive TPR at specific FPR levels, e.g. {0.001: 0.82, 0.01: 0.91}.

    None when auroc is None (single-class subset).
    """
    fpr_grid: npt.NDArray[np.float64]
    """FPR values for ROC curve plotting (ascending, from 0.0 to 1.0)."""
    tpr_grid: npt.NDArray[np.float64]
    """TPR values corresponding to fpr_grid."""
    precision_grid: npt.NDArray[np.float64]
    """Precision values for PR curve plotting."""
    recall_grid: npt.NDArray[np.float64]
    """Recall values corresponding to precision_grid."""

    @pydantic.model_validator(mode="after")
    def _validate_grids_and_nullability(self) -> ThresholdFreeMetrics:
        """Validates that curve grids are consistent and compatible with nullability of metrics."""
        none_flags = [self.auroc is None, self.average_precision is None, self.tpr_at_fpr is None]
        if any(none_flags) and not all(none_flags):
            raise ValueError("auroc, average_precision, tpr_at_fpr must all be None or all non-None")
        if len(self.fpr_grid) != len(self.tpr_grid):
            raise ValueError("fpr_grid and tpr_grid must have the same length")
        if len(self.precision_grid) != len(self.recall_grid):
            raise ValueError("precision_grid and recall_grid must have the same length")
        if self.auroc is None and (len(self.fpr_grid) > 0 or len(self.precision_grid) > 0):
            raise ValueError("curve grids must be empty when auroc/average_precision are None")
        return self


class ThresholdedMetrics(pydantic.BaseModel):
    """Attempt-level confusion matrix and derived rates at a calibrated threshold.

    The threshold is chosen on the dev set to satisfy FPR <= target_fpr. All counts and rates are
    computed on the test set at that threshold.

    Rates are None when their denominator is zero:
    - tpr, fnr: None when tp+fn==0 (no correct attempts in subset)
    - fpr: None when fp+tn==0 (no incorrect attempts in subset)
    - precision: None when tp+fp==0 (all attempts blocked)
    - npv: None when tn+fn==0 (no attempts blocked)
    """

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""

    target_fpr: float
    """The FPR constraint (alpha) used to calibrate the threshold."""
    threshold: float
    """The calibrated decision threshold (accept if score >= threshold)."""
    tp: int
    """True positives: correct attempts accepted by the guardrail."""
    fp: int
    """False positives: incorrect attempts accepted by the guardrail."""
    tn: int
    """True negatives: incorrect attempts blocked by the guardrail."""
    fn: int
    """False negatives: correct attempts blocked by the guardrail."""
    tpr: float | None
    """True positive rate = TP / (TP + FN). None when tp+fn==0."""
    fpr: float | None
    """False positive rate = FP / (FP + TN). None when fp+tn==0."""
    fnr: float | None
    """False negative rate = FN / (TP + FN). None when tp+fn==0."""
    precision: float | None
    """Precision = TP / (TP + FP). None when tp+fp==0 (all blocked)."""
    npv: float | None
    """Negative predictive value = TN / (TN + FN). None when tn+fn==0 (none blocked)."""

    @pydantic.model_validator(mode="after")
    def _validate_confusion_matrix(self) -> ThresholdedMetrics:
        """Validates that confusion matrix counts are non-negative and rates are consistent with counts."""
        if self.tp < 0 or self.fp < 0 or self.tn < 0 or self.fn < 0:
            raise ValueError("confusion matrix counts must be non-negative")
        if self.tp + self.fn > 0:
            if self.tpr is None or self.fnr is None:
                raise ValueError("tpr/fnr must not be None when tp+fn > 0")
            expected_tpr = self.tp / (self.tp + self.fn)
            if abs(self.tpr - expected_tpr) > 1e-9:
                raise ValueError(f"tpr={self.tpr} inconsistent with tp={self.tp}, fn={self.fn}")
            expected_fnr = self.fn / (self.tp + self.fn)
            if abs(self.fnr - expected_fnr) > 1e-9:
                raise ValueError(f"fnr={self.fnr} inconsistent with fn={self.fn}, tp={self.tp}")
        else:
            if self.tpr is not None or self.fnr is not None:
                raise ValueError("tpr/fnr must be None when tp+fn == 0")
        if self.fp + self.tn > 0:
            if self.fpr is None:
                raise ValueError("fpr must not be None when fp+tn > 0")
            expected_fpr = self.fp / (self.fp + self.tn)
            if abs(self.fpr - expected_fpr) > 1e-9:
                raise ValueError(f"fpr={self.fpr} inconsistent with fp={self.fp}, tn={self.tn}")
        else:
            if self.fpr is not None:
                raise ValueError("fpr must be None when fp+tn == 0")
        if self.tp + self.fp > 0:
            if self.precision is None:
                raise ValueError("precision must not be None when tp+fp > 0")
            expected_precision = self.tp / (self.tp + self.fp)
            if abs(self.precision - expected_precision) > 1e-9:
                raise ValueError(f"precision={self.precision} inconsistent with tp={self.tp}, fp={self.fp}")
        else:
            if self.precision is not None:
                raise ValueError("precision must be None when tp+fp == 0")
        if self.tn + self.fn > 0:
            if self.npv is None:
                raise ValueError("npv must not be None when tn+fn > 0")
            expected_npv = self.tn / (self.tn + self.fn)
            if abs(self.npv - expected_npv) > 1e-9:
                raise ValueError(f"npv={self.npv} inconsistent with tn={self.tn}, fn={self.fn}")
        else:
            if self.npv is not None:
                raise ValueError("npv must be None when tn+fn == 0")
        return self


class SampleLevelMetrics(pydantic.BaseModel):
    """Metrics aggregated at the sample level (across K attempts per sample).

    Each indicator is computed per-sample then averaged across all samples. These reflect the
    user-facing experience: 'given K outputs, does the guardrail help me find a correct one?'
    """

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""

    target_fpr: float
    """The FPR constraint (alpha) used to calibrate the threshold."""
    base_pass_rate: float
    """Fraction of samples with >=1 correct attempt. Baseline without any guardrail."""
    guarded_pass_rate: float
    """Fraction of samples with >=1 correct AND accepted attempt. Always <= base_pass_rate."""
    unsafe_slip_rate: float
    """Fraction of samples with >=1 incorrect AND accepted attempt (unsafe slip-through)."""
    total_block_rate: float
    """Fraction of samples where all K attempts are blocked by the guardrail."""
    best_of_k_success_rate: float | None
    """Among samples with >=1 accepted attempt, fraction where the highest-scored accepted is correct.

    This models the deployment scenario where the guardrail's scores are used to rank K outputs and
    the top-scoring accepted output is decisive (i.e. we enact decisions based on the top score only).

    None when all attempts across all samples are blocked (total_block_rate == 1.0).
    """
    cons_pass_rate: float
    """Fraction of samples where ALL K attempts are accepted by the guardrail.

    This is the conservative pass rate: the sample is only trusted when every attempt passes the
    guardrail's threshold. Note: this is NOT bounded by base_pass_rate; a sample can be all-accepted
    yet have no actual correct attempts.
    """
    cons_unsafe_slip_rate: float | None
    """Among conservatively-accepted samples (all K accepted), fraction where >= 1 attempt is incorrect.

    Measures how often the conservative strategy lets through a sample that contains an incorrect
    output. This is a key metric for understanding guardrail performance in safety-critical contexts.

    None when cons_pass_rate == 0.0 (no conservatively-accepted samples).
    """
    cons_justified_reject_rate: float | None
    """Among conservatively-rejected samples (at least one attempt blocked), fraction where at
    least one blocked attempt is truly incorrect (label=False).

    Measures how often the conservative rejection was 'justified', i.e. the guardrail correctly
    identified at least one bad output in the sample.

    None when cons_pass_rate == 1.0 (no conservatively-rejected samples).
    """

    @pydantic.model_validator(mode="after")
    def _validate_rates(self) -> SampleLevelMetrics:
        """Validates that rates are consistent with each other and the base_pass_rate."""
        if self.guarded_pass_rate > self.base_pass_rate + 1e-9:
            raise ValueError(
                f"guarded_pass_rate ({self.guarded_pass_rate}) cannot exceed base_pass_rate ({self.base_pass_rate})"
            )
        if self.cons_pass_rate == 0.0 and self.cons_unsafe_slip_rate is not None:
            raise ValueError(
                "cons_unsafe_slip_rate must be None when cons_pass_rate == 0.0 (no conservatively-accepted samples)"
            )
        if self.cons_pass_rate == 1.0 and self.cons_justified_reject_rate is not None:
            raise ValueError(
                "cons_justified_reject_rate must be None when cons_pass_rate == 1.0 "
                "(no conservatively-rejected samples)"
            )
        return self


class ClassBalanceStats(pydantic.BaseModel):
    """Descriptive statistics about the correctness label distribution.

    Important covariate for interpreting all other metrics (e.g. high base rate makes AUROC less
    informative; many all-correct samples inflate base_pass_rate).
    """

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""

    overall_positive_rate: float
    """Fraction of all attempts that are correct (label=True)."""
    per_sample_positive_rates: list[float]
    """One correctness rate per sample (fraction of K attempts that are correct)."""
    num_all_correct_samples: int
    """Number of samples where all K attempts are correct."""
    num_all_incorrect_samples: int
    """Number of samples where all K attempts are incorrect."""
    code_type_proportions: dict[str, float]
    """Fraction of records belonging to each code_type.

    Includes both raw compound types (e.g. 'bugged_hinted') and their individual components
    (e.g. 'bugged', 'hinted'), so component proportions may sum to more than 1.0 when compounds
    are present.
    """
    predict_type_proportions: dict[str, float]
    """Fraction of records belonging to each predict_type.

    Empty when predict_type is not available in the LMDB records.
    """

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def per_sample_positive_rate_mean(self) -> float:
        """Mean of per_sample_positive_rates. Returns 0.0 when the list is empty."""
        if not self.per_sample_positive_rates:
            return 0.0
        return sum(self.per_sample_positive_rates) / len(self.per_sample_positive_rates)

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def per_sample_positive_rate_std(self) -> float:
        """Population standard deviation of per_sample_positive_rates. Returns 0.0 when the list is empty."""
        if not self.per_sample_positive_rates:
            return 0.0
        mean = self.per_sample_positive_rate_mean
        variance = sum((rate - mean) ** 2 for rate in self.per_sample_positive_rates) / len(
            self.per_sample_positive_rates
        )
        return variance**0.5


class DifficultyStats(pydantic.BaseModel):
    """Guardrail performance conditioned on sample difficulty.

    Difficulty scores quantify how hard each coding problem is for the target model.
    Records are bucketed into terciles (easy/medium/hard) by difficulty score, and key
    guardrail metrics are computed per bucket. All fields are None when difficulty
    scores are unavailable.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""

    bucket_boundaries: tuple[float, float] | None
    """33rd and 67th percentile thresholds used to define easy/medium/hard buckets."""
    per_bucket_auroc: dict[str, float | None] | None
    """AUROC per difficulty bucket ('easy', 'medium', 'hard')."""
    per_bucket_tpr: dict[str, dict[float, float]] | None
    """TPR per bucket at each target_fpr."""
    per_bucket_sample_count: dict[str, int] | None
    """Number of unique samples in each difficulty bucket."""
    difficulty_accuracy_rank_correlation: float | None
    """Spearman rank correlation between per-sample difficulty and guardrail classification accuracy."""


class VerificationCostStats(pydantic.BaseModel):
    """Summary statistics of the compute cost incurred by the guardrail at a specific threshold.

    Captures per-record verification costs and aggregates them. All fields are None when the scorer
    does not report costs.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""

    target_fpr: float
    """The FPR constraint used to calibrate the threshold for accept/reject decisions."""
    cost_unit: str | None
    """Unit label for verification costs (e.g. 'tokens', 'FLOPs', 'turns').

    None when the scorer does not report costs; from `GuardrailScorer.get_verification_cost_unit()`.
    """
    total_cost: float | None
    """Sum of verification costs across all scored records."""
    mean_cost_per_record: float | None
    """Mean verification cost per record."""
    median_cost_per_record: float | None
    """Median verification cost per record."""
    std_cost_per_record: float | None
    """Standard deviation of per-record verification costs."""
    cost_per_correct_acceptance: float | None
    """Mean cost among records that were correct and accepted (TP). None if no TPs."""
    cost_per_incorrect_block: float | None
    """Mean cost among records that were incorrect and blocked (TN). None if no TNs."""
    cost_accuracy_rank_correlation: float | None
    """Spearman rank correlation between per-record verif cost and guardrail correctness.

    None when fewer than 3 records or costs are unavailable.
    """
    cost_difficulty_rank_correlation: float | None
    """Spearman rank correlation between per-sample mean verif cost and per-sample difficulty.

    None when difficulty scores are unavailable or fewer than 3 samples.
    """


class CategoryResult(pydantic.BaseModel):
    """All metrics for a single record category (e.g. 'regular', 'biasing/misleading').

    Categories group code_types from EvalRecords into semantically meaningful buckets (see
    ``RecordCategoryConfig.code_type_to_category``). The default grouping maps non-biasing code
    types ('original', 'obfuscated', 'stubbed') to the 'regular' category, while bias-inducing
    code types ('hinted', 'misleading', 'bugged') each get their own 'biasing/...' category.
    Compound code types (e.g. 'bugged_hinted') appear in each matching biasing category and in a
    combined category.

    Contains the same metric types as the global results, computed on the subset of records
    belonging to this category. The threshold is shared with the global result (calibrated on the
    full dev set, not per-category).
    """

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)
    """Pydantic model configuration (freezes the dataclass)."""

    category: str
    """Category name (e.g. 'regular', 'biasing/misleading', 'code_type/bugged')."""
    record_count: int
    """Number of EvalRecords in this category."""
    sample_count: int
    """Number of unique sample_ids in this category."""
    class_balance: ClassBalanceStats
    """Label distribution within this category."""
    threshold_free: ThresholdFreeMetrics
    """Threshold-free metrics (AUROC, average precision) on this category's records."""
    attempt_metrics: dict[float, ThresholdedMetrics]
    """Thresholded attempt-level metrics, keyed by target_fpr."""
    sample_metrics: dict[float, SampleLevelMetrics]
    """Sample-level metrics, keyed by target_fpr."""


class SingleRunResult(pydantic.BaseModel):
    """Complete evaluation result for one guardrail instance (one training run).

    Contains global metrics (across all categories), per-category breakdowns, and problem-clustered
    bootstrap CIs.
    """

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)
    """Pydantic model configuration (freezes the dataclass)."""

    guardrail_metadata: dict[str, typing.Any]
    """Metadata from GuardrailScorer.get_metadata()."""
    attempt_metadata: dict[ScoredAttemptKey, dict[str, typing.Any]] | None = None
    """Per-attempt metadata from ``ScoringResult.attempt_metadata`` for the eval set."""
    attempt_records: list[AttemptInspectionRecord] = pydantic.Field(default_factory=lambda: [])
    """Per-attempt rows with scores + sample metadata for notebook browsing."""
    threshold_free: ThresholdFreeMetrics
    """Global threshold-free metrics (all categories combined)."""
    attempt_metrics: dict[float, ThresholdedMetrics]
    """Global thresholded attempt-level metrics, keyed by target_fpr."""
    sample_metrics: dict[float, SampleLevelMetrics]
    """Global sample-level metrics, keyed by target_fpr."""
    category_results: dict[str, CategoryResult]
    """Per-category metric breakdowns."""
    bootstrap_cis: dict[str, pyine.utils.metrics.confidence.ConfidenceInterval]
    """Problem-clustered bootstrap CIs for global metrics only (not per-category).

    Keys follow the naming scheme: 'auroc', 'average_precision' for threshold-free metrics;
    'fpr_0_01/tpr', 'fpr_0_01/precision', etc. for attempt-level metrics at each target FPR;
    'fpr_0_01/guarded_pass_rate', 'fpr_0_01/unsafe_slip_rate', etc. for sample-level metrics.
    FPR values are formatted with underscores (0.01 -> fpr_0_01). Metrics that are None in a
    replicate (e.g. single-class subsets) are excluded from that replicate's contribution.
    """
    difficulty_stats: DifficultyStats | None
    """Difficulty-conditioned metrics. None when unavailable."""
    verification_cost_stats: dict[float, VerificationCostStats] | None
    """Guardrail cost summary per target_fpr. None when unavailable."""


class AggregatedResult(pydantic.BaseModel):
    """Aggregated results across R independent guardrail runs.

    Contains per-run results, cross-run summary statistics, and hierarchical bootstrap CIs that
    account for both pipeline randomness and test set sampling.
    """

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)
    """Pydantic model configuration (freezes the dataclass)."""

    split_summary: dict[str, typing.Any]
    """Lightweight split metadata from ``splits.GuardrailSplits.to_summary()``."""
    class_balance: ClassBalanceStats
    """Label distribution on the test set."""
    per_run: list[SingleRunResult]
    """Full results for each guardrail run."""
    attempt_records_by_key: dict[ScoredAttemptKey, dict[str, typing.Any]] | None = None
    """Shared raw record payloads keyed by ``(sample_id, attempt_index, draw_index)``.

    This stores heavyweight LMDB records once at the aggregated level so per-run
    ``SingleRunResult.attempt_records`` can stay lightweight.
    """
    cross_run_mean: dict[str, float]
    """Mean of each metric across runs (None-valued metrics excluded)."""
    cross_run_std: dict[str, float]
    """Standard deviation of each metric across runs."""
    cross_run_p5: dict[str, float]
    """5th percentile of each metric across runs."""
    cross_run_num_valid: dict[str, int]
    """Number of runs that produced a non-None value for each metric."""
    hierarchical_cis: dict[str, pyine.utils.metrics.confidence.ConfidenceInterval]
    """Hierarchical bootstrap CIs for the pipeline mean.

    Accounts for both run-level variance (each guardrail trains on a different seed) and test-set
    sampling variance (problem-clustered resampling within each run).

    Uses the same key naming scheme as SingleRunResult.bootstrap_cis (e.g. 'auroc', 'fpr_0_01/tpr',
    'fpr_0_01/guarded_pass_rate'). Metrics that are None in all runs are omitted entirely; metrics
    with fewer than 10 valid replicates are also omitted."""
    difficulty_stats: DifficultyStats | None
    """Difficulty-conditioned metrics aggregated across runs. None when unavailable."""
    verification_cost_stats: dict[float, VerificationCostStats] | None
    """Cross-run verification cost summary per target_fpr. None when unavailable."""

    @pydantic.field_validator("per_run")
    @classmethod
    def _validate_per_run_nonempty(
        cls,
        value: list[SingleRunResult],
    ) -> list[SingleRunResult]:
        """Validates that per_run is non-empty."""
        if not value:
            raise ValueError("per_run must contain at least one run result")
        return value

    def to_flat_dict(self) -> dict[str, float | int | str]:
        """Flatten to a namespaced dict for W&B logging and MetricsDictType compatibility.

        All keys use a metric-first naming convention: the metric path comes first, followed by the
        statistic type as a suffix.

        Naming scheme:
        - Threshold-free: 'auroc/mean', 'auroc/std', 'auroc/p5', 'average_precision/mean', etc.
        - TPR@FPR: 'tpr_at_fpr_0_01/mean'
        - Thresholded: 'fpr_0_01/tpr/mean', 'fpr_0_01/fpr/mean', 'fpr_0_01/precision/mean', etc.
        - Sample-level: 'fpr_0_01/base_pass_rate/mean', 'fpr_0_01/guarded_pass_rate/mean', etc.
        - Costs: 'fpr_0_01/cost_total/mean', 'fpr_0_01/cost_mean/mean', etc.
        - Bootstrap CIs: 'auroc/bootstrap_ci_lower', 'auroc/bootstrap_ci_upper',
          'auroc/bootstrap_ci_point', 'fpr_0_01/tpr/bootstrap_ci_lower', etc.
        - Run validity: 'auroc/num_valid_runs', 'fpr_0_01/tpr/num_valid_runs', etc.
        - Class balance: 'class_balance/overall_positive_rate', etc.
        - Counts: 'sample_count', 'record_count'
        - Category: 'category/{safe_cat}/auroc/mean', 'category/{safe_cat}/fpr_0_01/tpr/mean', etc.
          (slashes in category names are replaced with underscores for flat key
          compatibility, e.g. 'biasing/misleading' becomes 'biasing_misleading')
        """
        flat: dict[str, float | int | str] = {}
        # cross-run aggregates (metric-first: {metric}/mean, {metric}/std, {metric}/p5)
        for suffix_key, source in [
            ("mean", self.cross_run_mean),
            ("std", self.cross_run_std),
            ("p5", self.cross_run_p5),
        ]:
            for metric_name, metric_val in source.items():
                flat[f"{metric_name}/{suffix_key}"] = metric_val
        # run validity counts ({metric}/num_valid_runs)
        for metric_name, num_valid in self.cross_run_num_valid.items():
            flat[f"{metric_name}/num_valid_runs"] = num_valid
        # bootstrap CIs ({metric}/bootstrap_ci_lower, etc.)
        for ci_name, ci_val in self.hierarchical_cis.items():
            flat[f"{ci_name}/bootstrap_ci_point"] = ci_val.point_estimate
            flat[f"{ci_name}/bootstrap_ci_lower"] = ci_val.lower_bound
            flat[f"{ci_name}/bootstrap_ci_upper"] = ci_val.upper_bound
        # class balance
        flat["class_balance/overall_positive_rate"] = self.class_balance.overall_positive_rate
        flat["class_balance/per_sample_positive_rate_mean"] = self.class_balance.per_sample_positive_rate_mean
        flat["class_balance/per_sample_positive_rate_std"] = self.class_balance.per_sample_positive_rate_std
        flat["class_balance/num_all_correct_samples"] = self.class_balance.num_all_correct_samples
        flat["class_balance/num_all_incorrect_samples"] = self.class_balance.num_all_incorrect_samples
        for code_type, proportion in self.class_balance.code_type_proportions.items():
            flat[f"class_balance/code_type/{code_type}"] = proportion
        for predict_type, proportion in self.class_balance.predict_type_proportions.items():
            flat[f"class_balance/predict_type/{predict_type}"] = proportion
        # counts from split summary and test set
        flat["record_count"] = self.split_summary.get("test_record_count", 0)
        # per_sample_positive_rates has one entry per unique sample in the test set
        flat["sample_count"] = len(self.class_balance.per_sample_positive_rates)
        # note: cost and category metrics are already included via cross_run_mean/std/p5
        # above (e.g. fpr_0_01/cost_total/mean, category/regular/auroc/mean, etc.)
        # the aggregated VerificationCostStats is available on self for programmatic access
        return flat


# ---- Reserved type-name tokens (shared between _impl validation and analysis parsing) ----

RESERVED_TYPE_NAME_PREFIXES: frozenset[str] = frozenset({"fpr_", "tpr_at_fpr_"})
"""String prefixes that guardrail type names must NOT start with (case-insensitive).

These collide with the FPR-keyed metric namespace (e.g. ``fpr_0_01/tpr/mean``).
"""

RESERVED_TYPE_NAME_EXACT: frozenset[str] = frozenset(
    {
        "category",
        "class_balance",
        "auroc",
        "average_precision",
        "sample_count",
        "record_count",
        "_guardrail_type_names",
    }
)
"""Exact tokens that guardrail type names must NOT match (case-insensitive).

These collide with top-level metric keys or metadata keys logged under ``benchmark/{subset}/`` in
W&B summary.
"""
