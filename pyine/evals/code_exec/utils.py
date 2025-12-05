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

if typing.TYPE_CHECKING:
    import pyine.evals.code_exec.evaluator  # noqa

type LLMScoreFuture = asyncio.Task[float]
"""Type alias for pending LLM score computations."""
type LLMGraderResponse = float | pyine.utils.code.output_compare.GradingResult
"""Raw response type emitted by the LLM grading chain."""


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


type AccuracyType = typing.Literal["accuracy_hard", "accuracy_soft", "accuracy_grader"]
"""Type of accuracy to compute (hard, soft, or grader-based)."""

AggregatedGraderMetricNames: typing.Final[tuple[str, ...]] = tuple(
    f"grader_{aggr_name}" for aggr_name in pyine.evals.constants.AGGREGATION_STAT_NAMES
)
"""List of aggregated grader score metrics over all samples."""


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

    @property
    def identifier(self) -> str:
        """Unique sample id associated with the data sample (used for lookups)."""
        return self.sample.identifier

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
    def identifiers(self) -> list[str]:
        """Returns a list of sample identifiers associated with the evaluation results."""
        return [s.identifier for s in self.artifacts]

    @property
    def categories(self) -> list[str]:
        """Returns a list of categories associated with the evaluation results."""
        return list(self.category_to_identifiers.keys())

    @property
    def num_samples(self) -> int:
        """Returns the number of samples associated with the evaluation results."""
        return len(self.artifacts)


@typing.no_type_check  # because wandb sucks at typing
def define_metrics_for_wandb(
    wandb_run: typing.Any,
    prefix: str | None = None,
) -> None:
    """Defines the evaluation metrics for the given wandb run."""
    from pyine.evals.code_exec.evaluator import OutcomeEvaluator

    step_metric = "train/global_step"  # the global step for the run, logged by the hf trainer
    for metric_name in OutcomeEvaluator.get_supported_metric_names():
        metric_name = f"{prefix}/{metric_name}" if prefix else metric_name
        wandb_run.define_metric(name=metric_name, step_metric=step_metric)
    # define token usage metrics (matching get_metrics output format)
    for token_metric_name in pyine.evals.utils.TokenUsageInfo.get_metric_names():
        # total token usage metrics (token_usage/{metric})
        token_usage_name = f"token_usage/{token_metric_name}"
        if prefix:
            token_usage_name = f"{prefix}/{token_usage_name}"
        wandb_run.define_metric(name=token_usage_name, step_metric=step_metric)
        # per-sample aggregated token usage metrics (sample_token_usage/{metric}_{aggr})
        for aggr_name in pyine.evals.constants.AGGREGATION_STAT_NAMES:
            sample_token_name = f"sample_token_usage/{token_metric_name}_{aggr_name}"
            if prefix:
                sample_token_name = f"{prefix}/{sample_token_name}"
            wandb_run.define_metric(name=sample_token_name, step_metric=step_metric)
    # define complexity metrics (matching get_metrics output format)
    for complexity_metric_name in pyine.utils.code.complexity_metrics.COMPLEXITY_METRICS:
        for aggr_name in pyine.evals.constants.AGGREGATION_STAT_NAMES:
            complexity_name = f"complexity/{complexity_metric_name}_{aggr_name}"
            if prefix:
                complexity_name = f"{prefix}/{complexity_name}"
            # these are interesting for analyses, but not for plotting
            wandb_run.define_metric(
                name=complexity_name,
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


def _check_metrics_inputs(
    evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    sample_token_usage: dict[str, pyine.evals.utils.TokenUsageInfo],
    sample_data_store: dict[str, pyine.organisms.datamodules.samples.SampleData],
) -> None:
    """Validates arguments used to compute metrics."""
    if evaluator.get_sample_count() != len(sample_data_store):
        raise ValueError(
            f"evaluator sample count ({evaluator.get_sample_count()}) != data store size ({len(sample_data_store)})"
        )
    if len(sample_token_usage) != len(sample_data_store):
        raise ValueError(f"token usage count ({len(sample_token_usage)}) != data store size ({len(sample_data_store)})")
    if set(sample_token_usage) != set(sample_data_store):
        raise ValueError("token usage and data store have different sample identifiers")


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
    sample_token_usage: dict[str, pyine.evals.utils.TokenUsageInfo],
    sample_data_store: dict[str, pyine.organisms.datamodules.samples.SampleData],
) -> pyine.evals.utils.MetricsDictType:
    """Compute and returns metrics associated with the current evaluation result."""
    _check_metrics_inputs(evaluator, sample_token_usage, sample_data_store)
    output_metrics: pyine.evals.utils.MetricsDictType = await evaluator.compute_metrics()
    for key, value in token_usage.asdict().items():
        output_metrics[f"token_usage/{key}"] = _float_or_nan(value)  # convert to bypass 'unknown'
    for key, value in pyine.evals.utils.compute_aggregated_token_usage_metrics(sample_token_usage.values()).items():
        output_metrics[f"sample_token_usage/{key}"] = value  # these should always be floats
    for key, value in compute_aggregated_complexity_stats(sample_data_store.values()).items():
        output_metrics[f"complexity/{key}"] = value  # these should always be floats
    return output_metrics


async def get_category_wise_metrics(
    evaluator: pyine.evals.code_exec.evaluator.OutcomeEvaluator,
    sample_token_usage: dict[str, pyine.evals.utils.TokenUsageInfo],
    sample_data_store: dict[str, pyine.organisms.datamodules.samples.SampleData],
    category_to_identifiers: dict[str, list[str]],
) -> pyine.evals.utils.MetricsDictType:
    """Compute and returns metrics associated with the current evaluation result grouped by category."""
    _check_metrics_inputs(evaluator, sample_token_usage, sample_data_store)
    output_metrics: pyine.evals.utils.MetricsDictType = {}
    category_wise_metrics = await evaluator.compute_category_wise_metrics(category_to_identifiers)
    for category, metrics in category_wise_metrics.items():
        for metric_name, metric_value in metrics.items():
            output_metrics[f"{category}/{metric_name}"] = metric_value
        category_token_usage = pyine.evals.utils.TokenUsageInfo.get_default()
        category_token_usage_data: list[pyine.evals.utils.TokenUsageInfo] = []
        category_sample_data: list[pyine.organisms.datamodules.samples.SampleData] = []
        for sample_identifier in category_to_identifiers[category]:
            category_token_usage += sample_token_usage[sample_identifier]
            category_token_usage_data.append(sample_token_usage[sample_identifier])
            category_sample_data.append(sample_data_store[sample_identifier])
        for key, value in category_token_usage.asdict().items():
            output_metrics[f"{category}/token_usage/{key}"] = _float_or_nan(value)  # convert to bypass 'unknown'
        for key, value in pyine.evals.utils.compute_aggregated_token_usage_metrics(category_token_usage_data).items():
            output_metrics[f"{category}/sample_token_usage/{key}"] = value  # these should always be floats
        for key, value in compute_aggregated_complexity_stats(category_sample_data).items():
            output_metrics[f"{category}/complexity/{key}"] = value  # these should always be floats
    return output_metrics
