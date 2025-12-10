import collections.abc
import typing

import pydantic
import radon.complexity
import radon.metrics
import radon.raw


class ComplexityMetrics(pydantic.BaseModel, collections.abc.Mapping[str, float | int]):
    """Container for radon-derived code complexity metrics."""

    model_config = pydantic.ConfigDict(frozen=True)

    cyclomatic_complexity_avg: float
    """Average cyclomatic complexity across analyzed blocks."""
    cyclomatic_complexity_max: int
    """Highest cyclomatic complexity across analyzed blocks."""
    cyclomatic_complexity_sum: int
    """Total cyclomatic complexity summed over all code blocks."""
    loc: int
    """Total number of lines of code."""
    lloc: int
    """Number of logical lines of code, excluding comments and blank lines."""
    sloc: int
    """Number of source lines of executable code (not necessarily the same as LLOC)."""
    comments: int
    """Number of comment lines."""
    multi: int
    """Number of multi-line strings (typically docstrings)."""
    blank: int
    """Number of blank lines."""
    halstead_volume: float
    """Halstead program volume."""
    halstead_difficulty: float
    """Halstead program difficulty."""
    halstead_effort: float
    """Halstead effort metric, computed as volume multiplied by difficulty."""
    maintainability_index: float
    """Radon maintainability index; higher values imply easier maintenance."""

    def __getitem__(
        self,
        metric_name: str,
    ) -> float | int:
        """Returns the value of the specified metric (by name)."""
        if metric_name not in type(self).model_fields:
            raise KeyError(metric_name) from None
        return typing.cast("float | int", getattr(self, metric_name))

    def __iter__(self) -> collections.abc.Iterator[str]:  # type: ignore[override]
        """Returns an iterator over the metric names (overrides BaseModel's tuple iterator)."""
        return iter(COMPLEXITY_METRICS)

    def __len__(self) -> int:
        """Returns the number of metrics contained in this model."""
        return len(COMPLEXITY_METRICS)

    def as_dict(self) -> dict[str, float | int]:
        """Returns a plain dictionary representation of the metrics."""
        return typing.cast("dict[str, float | int]", self.model_dump())


COMPLEXITY_METRICS = list(ComplexityMetrics.model_fields.keys())
"""Ordered list of metric names matching the model fields."""


@typing.no_type_check
def _untyped_get_complexity_metrics(
    code: str,
) -> ComplexityMetrics:
    """Compute radon-derived structural metrics for a Python code snippet.

    Args:
        code: Python code snippet to analyze

    Returns:
        ComplexityMetrics dataclass containing the measured values
    """
    cc_results = radon.complexity.cc_visit(code)
    if cc_results:
        cc_sum = sum(block.complexity for block in cc_results)
        cc_avg = cc_sum / len(cc_results)
        cc_max = max(block.complexity for block in cc_results)
    else:
        cc_sum = 0
        cc_avg = 0.0
        cc_max = 0

    raw_metrics = radon.raw.analyze(code)

    halstead = radon.metrics.h_visit(code)
    if halstead:
        halstead_volume = float(halstead.total.volume)
        halstead_difficulty = float(halstead.total.difficulty)
        halstead_effort = float(halstead.total.effort)
    else:
        halstead_volume = 0.0
        halstead_difficulty = 0.0
        halstead_effort = 0.0

    maintainability_index = float(radon.metrics.mi_visit(code, multi=True))

    return ComplexityMetrics(
        cyclomatic_complexity_avg=float(cc_avg),
        cyclomatic_complexity_max=int(cc_max),
        cyclomatic_complexity_sum=int(cc_sum),
        loc=int(raw_metrics.loc),
        lloc=int(raw_metrics.lloc),
        sloc=int(raw_metrics.sloc),
        comments=int(raw_metrics.comments),
        multi=int(raw_metrics.multi),
        blank=int(raw_metrics.blank),
        halstead_volume=halstead_volume,
        halstead_difficulty=halstead_difficulty,
        halstead_effort=halstead_effort,
        maintainability_index=maintainability_index,
    )


def get_complexity_metrics(
    code: str,
) -> ComplexityMetrics:
    """Compute radon-derived structural metrics for a Python code snippet.

    Args:
        code: Python code snippet to analyze

    Returns:
        ComplexityMetrics dataclass containing the measured values
    """
    return _untyped_get_complexity_metrics(code)
