from __future__ import annotations

import asyncio
import dataclasses
import typing

import numpy as np
import pydantic

import pyine.evals.common
import pyine.evals.constants
import pyine.evals.utils
import pyine.organisms.datamodules.samples
import pyine.utils.code.complexity_metrics
import pyine.utils.code.output_compare
import pyine.utils.parsing

if typing.TYPE_CHECKING:
    import pyine.evals.code_exec.evaluator  # noqa

type LLMScoreFuture = asyncio.Task[float]
"""Type alias for pending LLM score computations."""
type LLMGraderResponse = float | pyine.utils.code.output_compare.GradingResult
"""Raw response type emitted by the LLM grading chain."""
type AttemptKey = tuple[str, int]
"""Key type for attempt-level token usage: (identifier, attempt_index)."""
type MatchType = typing.Literal["hard", "soft", "grader"]
"""Match type for evaluation metrics (hard exact match, soft heuristic match, LLM grader)."""
MATCH_TYPES: tuple[MatchType, ...] = ("hard", "soft", "grader")
"""All match types (including grader), used for extraction and plotting."""
DETERMINISTIC_MATCH_TYPES: tuple[typing.Literal["hard", "soft"], ...] = ("hard", "soft")
"""Match types that are always available (no LLM grader required)."""
type AccuracyType = typing.Literal["accuracy_hard", "accuracy_soft", "accuracy_grader"]
"""Type of accuracy to compute (hard, soft, or grader-based); prefixed form of MatchType."""

AggregatedGraderMetricNames: typing.Final[tuple[str, ...]] = tuple(
    f"grader_{aggr_name}" for aggr_name in pyine.evals.constants.AGGREGATION_STAT_NAMES
)
"""List of aggregated grader score metrics over all samples."""


@dataclasses.dataclass
class SampleEval:
    """Container holding cached evaluation artifacts for a single code exec sample."""

    identifier: str
    """Unique sample id associated with the executed code snippet (used for lookups)."""
    expected: str
    """Ground-truth string that corresponds to the expected execution result."""
    predicted: str
    """Model prediction string that we hope is the same as the expected result."""
    hard_match: bool
    """Exact match result (following potential string normalization)."""
    soft_match: pyine.utils.code.output_compare.CompareResult
    """Soft match result (using the framework's output comparison function)."""
    _llm_score: float | LLMScoreFuture | None
    """Score in [0, 1] returned by a LLM grader, if used."""
    tags: list[str]
    """Arbitrary tags used for grouping/filtering (e.g., difficulty, source)."""
    attempt_index: int = 0
    """Index of this attempt within multi-sample generation (0 for single-sample)."""
    predict_type: str = "unknown"
    """Prediction type for this sample (authoritative source for invariant validation)."""

    @property
    def llm_score(self) -> float | None:
        """Returns the LLM score, if available; raises if the score is a future."""
        if self._llm_score is None or isinstance(self._llm_score, float):
            return self._llm_score
        if isinstance(self._llm_score, asyncio.Task):
            raise RuntimeError("attempting to access llm score before it is ready")
        raise ValueError(f"unexpected LLM score type: {type(self._llm_score)}")

    @property
    def has_gathered_llm_score(self) -> bool:
        """Returns whether the LLM score (if used) has been gathered; if unused, always returns True."""
        return self._llm_score is None or isinstance(self._llm_score, float)

    @staticmethod
    async def gather_llm_scores(eval_objs: typing.Iterable[SampleEval]) -> None:
        """Awaits all pending LLM grader score futures and updates the objects in-place.

        Args:
            eval_objs: Iterable of SampleEval objects whose LLM scores may be pending.
        """
        objs_with_future = [obj for obj in eval_objs if not obj.has_gathered_llm_score]
        if not objs_with_future:
            return
        tasks: list[LLMScoreFuture] = [typing.cast("LLMScoreFuture", obj._llm_score) for obj in objs_with_future]
        scores = await asyncio.gather(*tasks)
        for obj, score in zip(objs_with_future, scores, strict=False):
            if not isinstance(score, float):
                raise TypeError(f"expected float LLM score, got {type(score)}")
            obj._llm_score = score


@dataclasses.dataclass
class SampleEvalGroup:
    """A group of SampleEval attempts for a single sample identifier."""

    sample_identifier: str
    """The shared identifier across all attempts in this group."""
    attempts: list[SampleEval]
    """List of attempts sorted by attempt_index."""
    tags: list[str]
    """Tags from the first attempt (should be consistent across attempts)."""

    @property
    def num_attempts(self) -> int:
        """Returns the number of attempts in this group."""
        return len(self.attempts)

    @property
    def num_hard_correct(self) -> int:
        """Returns the number of hard-match correct attempts."""
        return sum(1 for a in self.attempts if a.hard_match)

    @property
    def num_soft_correct(self) -> int:
        """Returns the number of soft-match correct attempts."""
        return sum(1 for a in self.attempts if a.soft_match.equal)

    @property
    def unique_predictions(self) -> set[str]:
        """Returns the set of unique stripped predictions.

        Uses .strip() normalization for diversity/uniqueness measures,
        regardless of the evaluator's strip_hard_checks setting.
        """
        return {a.predicted.strip() for a in self.attempts}

    @property
    def output_diversity(self) -> float:
        """Returns |unique_stripped_outputs| / num_attempts."""
        if self.num_attempts == 0:
            return 0.0
        return len(self.unique_predictions) / self.num_attempts


def get_sample_token_usage(
    attempt_token_usage: dict[AttemptKey, pyine.evals.utils.TokenUsageInfo],
    identifier: str,
) -> list[pyine.evals.utils.TokenUsageInfo]:
    """Returns all token usage entries for a given sample identifier, sorted by attempt index."""
    entries = [(key[1], val) for key, val in attempt_token_usage.items() if key[0] == identifier]
    entries.sort(key=lambda x: x[0])
    return [val for _, val in entries]


def get_single_attempt_token_usage(
    attempt_token_usage: dict[AttemptKey, pyine.evals.utils.TokenUsageInfo],
    identifier: str,
    attempt_index: int,
) -> pyine.evals.utils.TokenUsageInfo:
    """Returns token usage for a specific (identifier, attempt_index) pair."""
    return attempt_token_usage[(identifier, attempt_index)]


class AgreementTable(typing.TypedDict):
    """Agreement table for a given set of samples."""

    hard_vs_soft: float
    """Agreement rate between hard (exact) and soft (heuristic-based) evaluators."""
    hard_vs_grader: float
    """Agreement rate between hard (exact) and grader (LLM-based) evaluators."""
    soft_vs_grader: float
    """Agreement rate between soft (heuristic-based) and grader (LLM-based) evaluators."""


class CodeExecEvalArtifact(pydantic.BaseModel):
    """Evaluation artifact resulting from a single data sample."""

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)
    """Pydantic model configuration (immutable)."""

    sample: pyine.organisms.datamodules.samples.SampleData
    """Sample data associated with the prediction."""
    token_usage: pyine.evals.utils.TokenUsageInfo
    """Token usage information associated with the prediction."""
    eval_result: SampleEval
    """Evaluation result associated with the prediction."""
    parsed_output: pyine.utils.parsing.ParsedOutput | None = None
    """Structured parsed output (reasoning, final_answer) when output parsing is enabled."""
    difficulty_score: float | None = None
    """Per-sample difficulty score (when difficulty estimation is enabled)."""

    @property
    def sample_identifier(self) -> str:
        """Sample-level identifier (shared across attempts for the same sample)."""
        return self.sample.identifier

    @property
    def attempt_key(self) -> AttemptKey:
        """Attempt-level key ``(sample_identifier, attempt_index)`` for unambiguous identification."""
        return self.sample.identifier, self.eval_result.attempt_index

    @pydantic.model_validator(mode="after")
    def _post_validation(self) -> CodeExecEvalArtifact:
        """Validates inter-field attributes."""
        assert self.sample.identifier == self.eval_result.identifier, "sample id mismatch"
        assert self.eval_result.llm_score is None or isinstance(self.eval_result.llm_score, float), "invalid llm score"
        return self


class CodeExecEvalResult(pyine.evals.common.EvalResult):
    """Container for evaluation metrics and captured artifacts."""

    metrics: pyine.evals.utils.MetricsDictType
    """Dictionary of aggregated evaluation metrics; keys are metric names, values are eval outcomes.

    Note that the metrics may be global ones as well as category-wise ones, depending on their prefix.
    """
    artifacts: list[CodeExecEvalArtifact]
    """List of captured evaluation artifacts (include sample data and eval result)."""
    category_to_identifiers: dict[str, list[str]]
    """Dictionary mapping category names to lists of sample identifiers associated with them."""

    @property
    def sample_identifiers(self) -> list[str]:
        """Returns a list of sample identifiers (may have duplicates when K>1)."""
        return [s.sample_identifier for s in self.artifacts]

    @property
    def unique_sample_identifiers(self) -> list[str]:
        """Returns a deduplicated list of sample identifiers."""
        return list(dict.fromkeys(s.sample_identifier for s in self.artifacts))

    @property
    def categories(self) -> list[str]:
        """Returns a list of categories associated with the evaluation results."""
        return list(self.category_to_identifiers.keys())

    @property
    def num_samples(self) -> int:
        """Returns the number of unique samples (identifiers)."""
        return len({a.sample_identifier for a in self.artifacts})

    @property
    def num_attempts(self) -> int:
        """Returns the total number of evaluation attempts (= K * num_samples when uniform)."""
        return len(self.artifacts)

    @property
    def attempt_keys(self) -> list[AttemptKey]:
        """Returns attempt-level keys for unambiguous identification when K>1."""
        return [a.attempt_key for a in self.artifacts]


@typing.no_type_check  # because wandb sucks at typing
def define_metrics_for_wandb(
    wandb_run: typing.Any,
    metric_prefix: str,
    pass_at_k_values: list[int] | None = None,
    num_attempts_per_sample: int = 1,
) -> None:
    """Defines the evaluation metrics for the given wandb run.

    Args:
        wandb_run: The wandb run to define metrics on.
        metric_prefix: The metric name prefix to use (which should specify the eval subset).
        pass_at_k_values: Resolved list of k values, or None for base metrics only.
        num_attempts_per_sample: Number of attempts per sample.
    """
    from pyine.evals.code_exec.evaluator import OutcomeEvaluator

    if pass_at_k_values is not None and max(pass_at_k_values) > num_attempts_per_sample:
        raise ValueError(
            f"max(pass_at_k_values)={max(pass_at_k_values)} > num_attempts_per_sample={num_attempts_per_sample}"
        )
    step_metric = "train/global_step"  # the global step for the run, logged by the hf trainer
    # catch-all for category-wise metrics (e.g. code_type/original/accuracy_hard) whose
    # category values are data-dependent and unknown at definition time. more-specific
    # definitions below take precedence over this glob in wandb.
    wandb_run.define_metric(name=f"{metric_prefix}/*", step_metric=step_metric)
    for metric_name in OutcomeEvaluator.get_supported_metric_names(pass_at_k_values, num_attempts_per_sample):
        wandb_run.define_metric(name=f"{metric_prefix}/{metric_name}", step_metric=step_metric)
    # define token usage metrics (matching get_metrics output format)
    for token_metric_name in pyine.evals.utils.TokenUsageInfo.get_metric_names():
        # total token usage metrics (total_token_usage/{metric})
        token_usage_name = f"{metric_prefix}/total_token_usage/{token_metric_name}"
        wandb_run.define_metric(name=token_usage_name, step_metric=step_metric)
        # per-attempt aggregated token usage metrics (attempt_token_usage/{metric}_{aggr})
        for aggr_name in pyine.evals.constants.AGGREGATION_STAT_NAMES:
            attempt_token_name = f"{metric_prefix}/attempt_token_usage/{token_metric_name}_{aggr_name}"
            wandb_run.define_metric(name=attempt_token_name, step_metric=step_metric)
    # define complexity metrics (matching get_metrics output format)
    for complexity_metric_name in pyine.utils.code.complexity_metrics.COMPLEXITY_METRICS:
        for aggr_name in pyine.evals.constants.AGGREGATION_STAT_NAMES:
            complexity_name = f"{metric_prefix}/complexity/{complexity_metric_name}_{aggr_name}"
            # these are interesting for analyses, but not for plotting
            wandb_run.define_metric(
                name=complexity_name,
                step_metric=step_metric,
                hidden=True,
                summary="none",
            )
    for aggr_name in pyine.evals.constants.AGGREGATION_STAT_NAMES:
        difficulty_name = f"{metric_prefix}/difficulty/score_{aggr_name}"
        wandb_run.define_metric(
            name=difficulty_name,
            step_metric=step_metric,
            hidden=True,
            summary="none",
        )


def compute_aggregated_complexity_stats(
    sample_data: typing.Iterable[pyine.organisms.datamodules.samples.SampleData],
) -> dict[str, float]:
    """Computes aggregated complexity statistics across all samples.

    All code complexity metrics are aggregated using mean, median, std, min, and max operators
    across all samples. If the sample list is empty, an empty dict is returned.

    Args:
        sample_data: List of samples containing code complexity metrics to aggregate.

    Returns:
        Dict mapping metric names (with an aggregation suffix) to value of aggregated metric.
    """
    metric_values: dict[str, list[float]] = {
        metric: [] for metric in pyine.utils.code.complexity_metrics.COMPLEXITY_METRICS
    }
    for sample in sample_data:
        for metric_name in pyine.utils.code.complexity_metrics.COMPLEXITY_METRICS:
            metric_values[metric_name].append(float(sample.complexity_metrics[metric_name]))
    output: dict[str, float] = {}
    for metric_name, values in metric_values.items():
        if not values:
            continue
        arr = np.array(values)
        for aggr_name, aggr_func in pyine.evals.constants.AGGREGATION_STAT_FUNCS.items():
            output[f"{metric_name}_{aggr_name}"] = float(aggr_func(arr))
    return output


def compute_aggregated_difficulty_stats(
    difficulty_scores: typing.Iterable[float],
) -> dict[str, float]:
    """Compute aggregated difficulty statistics across all samples.

    Args:
        difficulty_scores: Iterable of difficulty score values.

    Returns:
        Dict mapping metric names (with aggregation suffix) to values.
    """
    values = list(difficulty_scores)
    if not values:
        return {}
    arr = np.array(values)
    output: dict[str, float] = {}
    for aggr_name, aggr_func in pyine.evals.constants.AGGREGATION_STAT_FUNCS.items():
        output[f"score_{aggr_name}"] = float(aggr_func(arr))
    return output


def _check_metrics_inputs(
    evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    attempt_token_usage: dict[AttemptKey, pyine.evals.utils.TokenUsageInfo],
    sample_data_store: dict[str, pyine.organisms.datamodules.samples.SampleData],
    partial: bool = False,
) -> None:
    """Validates arguments used to compute metrics.

    Args:
        evaluator: The outcome evaluator with results.
        attempt_token_usage: Token usage keyed by (identifier, attempt_index).
        sample_data_store: Sample data keyed by bare identifier.
        partial: When True (progress callback path), skip strict equality checks.
    """
    evaluated_identifiers = {r.identifier for r in evaluator.results}
    if not partial:
        # strict: attempt_token_usage keys must exactly match evaluator results (attempt-level)
        expected_keys = {(r.identifier, r.attempt_index) for r in evaluator.results}
        if set(attempt_token_usage) != expected_keys:
            raise ValueError(
                f"token usage keys mismatch: "
                f"extra={set(attempt_token_usage) - expected_keys}, "
                f"missing={expected_keys - set(attempt_token_usage)}"
            )
        # strict: sample_data_store must exactly match evaluated identifiers (sample-level)
        if evaluated_identifiers != set(sample_data_store):
            extra = set(sample_data_store) - evaluated_identifiers
            missing = evaluated_identifiers - set(sample_data_store)
            raise ValueError(f"sample data store mismatch: extra={extra}, missing={missing}")
    else:
        # partial: only check that evaluated identifiers exist in sample_data_store (subset ok)
        if not evaluated_identifiers.issubset(set(sample_data_store)):
            missing = evaluated_identifiers - set(sample_data_store)
            raise ValueError(f"missing sample data for identifiers: {missing}")


def _float_or_nan(
    value: typing.Any,
) -> float:
    """Tries to convert a given value to a float, returning NaN if it fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


async def get_metrics(
    evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    token_usage: pyine.evals.utils.TokenUsageInfo,
    attempt_token_usage: dict[AttemptKey, pyine.evals.utils.TokenUsageInfo],
    sample_data_store: dict[str, pyine.organisms.datamodules.samples.SampleData],
    difficulty_scores: dict[str, float | None] | None = None,
    pass_at_k_values: list[int] | None = None,
    num_attempts_per_sample: int = 1,
    partial: bool = False,
) -> pyine.evals.utils.MetricsDictType:
    """Compute and returns metrics associated with the current evaluation result.

    Args:
        evaluator: The outcome evaluator with results.
        token_usage: Total token usage across all attempts.
        attempt_token_usage: Per-attempt token usage keyed by (identifier, attempt_index).
        sample_data_store: Sample data keyed by bare identifier.
        difficulty_scores: Optional mapping from sample identifier to difficulty score.
        pass_at_k_values: Resolved list of k values for Pass@K, or None to skip Pass@K.
        num_attempts_per_sample: Number of attempts per sample for validation.
        partial: When True, relaxes validation (subset checks only). Use for progress
            callbacks where not all samples have been evaluated yet.
    """
    _check_metrics_inputs(evaluator, attempt_token_usage, sample_data_store, partial=partial)
    output_metrics: pyine.evals.utils.MetricsDictType = await evaluator.compute_metrics(
        pass_at_k_values=pass_at_k_values,
        num_attempts_per_sample=num_attempts_per_sample,
    )
    for key, value in token_usage.asdict().items():
        output_metrics[f"total_token_usage/{key}"] = _float_or_nan(value)  # convert to bypass 'unknown'
    for key, value in pyine.evals.utils.compute_aggregated_token_usage_metrics(attempt_token_usage.values()).items():
        output_metrics[f"attempt_token_usage/{key}"] = value  # these should always be floats
    for key, value in compute_aggregated_complexity_stats(sample_data_store.values()).items():
        output_metrics[f"complexity/{key}"] = value  # these should always be floats
    if difficulty_scores is not None:
        valid_scores = [score for score in difficulty_scores.values() if score is not None]
        for key, value in compute_aggregated_difficulty_stats(valid_scores).items():
            output_metrics[f"difficulty/{key}"] = value
    return output_metrics


async def get_category_wise_metrics(
    evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    attempt_token_usage: dict[AttemptKey, pyine.evals.utils.TokenUsageInfo],
    sample_data_store: dict[str, pyine.organisms.datamodules.samples.SampleData],
    category_to_identifiers: dict[str, list[str]],
    difficulty_scores: dict[str, float | None] | None = None,
    pass_at_k_values: list[int] | None = None,
    num_attempts_per_sample: int = 1,
) -> pyine.evals.utils.MetricsDictType:
    """Compute and returns metrics associated with the current evaluation result grouped by category.

    Args:
        evaluator: The outcome evaluator with results.
        attempt_token_usage: Per-attempt token usage keyed by (identifier, attempt_index).
        sample_data_store: Sample data keyed by bare identifier.
        category_to_identifiers: Per-sample identifiers grouped by category.
        difficulty_scores: Optional mapping from sample identifier to difficulty score.
        pass_at_k_values: Resolved list of k values for Pass@K, or None.
        num_attempts_per_sample: Number of attempts per sample for validation.
    """
    _check_metrics_inputs(evaluator, attempt_token_usage, sample_data_store)
    output_metrics: pyine.evals.utils.MetricsDictType = {}
    category_wise_metrics = await evaluator.compute_category_wise_metrics(
        category_to_identifiers,
        pass_at_k_values=pass_at_k_values,
        num_attempts_per_sample=num_attempts_per_sample,
    )
    for category, metrics in category_wise_metrics.items():
        for metric_name, metric_value in metrics.items():
            output_metrics[f"{category}/{metric_name}"] = metric_value
        category_token_usage = pyine.evals.utils.TokenUsageInfo.get_default()
        category_token_usage_data: list[pyine.evals.utils.TokenUsageInfo] = []
        category_sample_data: list[pyine.organisms.datamodules.samples.SampleData] = []
        identifiers = list(dict.fromkeys(category_to_identifiers[category]))
        for sample_identifier in identifiers:
            sample_token_entries = get_sample_token_usage(attempt_token_usage, sample_identifier)
            for entry in sample_token_entries:
                category_token_usage += entry
                category_token_usage_data.append(entry)
            if sample_identifier in sample_data_store:
                category_sample_data.append(sample_data_store[sample_identifier])
        for key, value in category_token_usage.asdict().items():
            output_metrics[f"{category}/total_token_usage/{key}"] = _float_or_nan(value)
        for key, value in pyine.evals.utils.compute_aggregated_token_usage_metrics(category_token_usage_data).items():
            output_metrics[f"{category}/attempt_token_usage/{key}"] = value
        for key, value in compute_aggregated_complexity_stats(category_sample_data).items():
            output_metrics[f"{category}/complexity/{key}"] = value
        if difficulty_scores is not None:
            category_scores: list[float] = []
            for sample_identifier in identifiers:
                score = difficulty_scores.get(sample_identifier)
                if score is not None:
                    category_scores.append(score)
            for key, value in compute_aggregated_difficulty_stats(category_scores).items():
                output_metrics[f"{category}/difficulty/{key}"] = value
    return output_metrics


# -------------------------------- sample metrics table schema --------------------------------
# these functions define the shared schema for per-sample metrics tables, used by both
# `log_sample_metrics` (wandb logging) and `eval_result_to_dataframe` (offline analysis).


def get_sample_metrics_columns() -> list[str]:
    """Returns the column names for per-sample metrics tables.

    This is the canonical schema for tables logged by `log_sample_metrics` and DataFrames
    created by `eval_result_to_dataframe`. Both functions use this to ensure consistency.

    Returns:
        List of column names in the order they should appear.
    """
    token_usage_columns = pyine.evals.utils.TokenUsageInfo.get_metric_names()
    return [
        # sample identification
        "identifier",
        "attempt_index",
        "code_type",
        "predict_type",
        "tags",
        # evaluation results
        "hard_match",
        "soft_match",
        "grader_score",
        # sample structure
        "trace_step_count",
        "first_line",
        "last_line",
        "has_code_override",
        "code_line_count",
        "code_length",
        "inputs_length",
        "expected_output_length",
        # keyword content
        "bias_keyword",
        "has_bias_keyword",
        "keyword_injected",
        "keyword_refactored",
        # token usage
        *token_usage_columns,
        # complexity metrics
        *pyine.utils.code.complexity_metrics.COMPLEXITY_METRICS,
        # difficulty metrics
        "difficulty_score",
    ]


def artifact_to_sample_metrics_row(artifact: CodeExecEvalArtifact) -> dict[str, typing.Any]:
    """Extracts a row dict from a CodeExecEvalArtifact for sample metrics tables.

    This is the canonical extraction logic used by both `log_sample_metrics` (wandb logging)
    and `eval_result_to_dataframe` (offline analysis).

    Args:
        artifact: The evaluation artifact containing sample data and eval results.

    Returns:
        Dict mapping column names to values, ready for DataFrame or wandb.Table row.
    """
    sample = artifact.sample
    eval_res = artifact.eval_result
    token_usage = artifact.token_usage.asdict()
    token_usage_columns = pyine.evals.utils.TokenUsageInfo.get_metric_names()
    bias_keyword = pyine.evals.utils.parse_bias_keyword_from_sample(sample)
    return {
        # sample identification
        "identifier": artifact.sample_identifier,
        "attempt_index": eval_res.attempt_index,
        "code_type": str(sample.code_type),
        "predict_type": str(sample.predict_type),
        "tags": sample.comma_separated_tags,
        # evaluation results
        "hard_match": int(eval_res.hard_match),
        "soft_match": int(eval_res.soft_match.equal),
        "grader_score": eval_res.llm_score,
        # sample structure
        "trace_step_count": sample.trace_step_count,
        "first_line": sample.first_line,
        "last_line": sample.last_line,
        "has_code_override": int(sample.has_code_override),
        "code_line_count": len(sample.code.splitlines()),
        "code_length": len(sample.code),
        "inputs_length": len(sample.inputs),
        "expected_output_length": len(sample.expected_output),
        # keyword content
        "bias_keyword": "" if bias_keyword is None else bias_keyword,
        "has_bias_keyword": int("has_bias_keyword:1" in sample.comma_separated_tags),
        "keyword_injected": int("keyword_injected:1" in sample.comma_separated_tags),
        "keyword_refactored": int("keyword_refactored:1" in sample.comma_separated_tags),
        # token usage (convert 'unknown' to None)
        **{t: token_usage[t] if token_usage[t] != "unknown" else None for t in token_usage_columns},
        # complexity metrics
        **sample.complexity_metrics,
        # difficulty metrics
        "difficulty_score": artifact.difficulty_score,
    }
