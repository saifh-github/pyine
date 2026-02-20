"""Confidence interval utilities for proportion-based metrics.

Provides Wilson score intervals and z-score computation via the standard library's
``statistics.NormalDist``. All functions are stateless and operate on simple numeric inputs.
"""

from __future__ import annotations

import dataclasses
import math
import statistics


@dataclasses.dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate with associated confidence bounds."""

    point_estimate: float
    """The central estimate of the metric."""
    lower_bound: float
    """Lower bound of the confidence interval."""
    upper_bound: float
    """Upper bound of the confidence interval."""


FULL_UNCERTAINTY_INTERVAL = ConfidenceInterval(point_estimate=0.0, lower_bound=0.0, upper_bound=1.0)
"""Default interval used when the estimate is undefined or uninformative."""


def _validate_confidence_level(
    confidence_level: float,
) -> None:
    """Validates that the confidence level is a probability in (0, 1)."""
    if not (0 < confidence_level < 1):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")


def compute_accuracy_with_ci(
    num_correct: int,
    num_total: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Computes accuracy with Wilson score confidence interval from counts.

    Args:
        num_correct: Number of correct items.
        num_total: Total number of items.
        confidence_level: Confidence level for the CI (default 0.95).

    Returns:
        ConfidenceInterval with accuracy and Wilson score bounds.

    Raises:
        ValueError: If confidence_level is not in (0, 1), num_correct or num_total is negative,
            or num_correct > num_total.
    """
    _validate_confidence_level(confidence_level)
    if num_correct < 0:
        raise ValueError(f"num_correct must be non-negative, got {num_correct}")
    if num_total < 0:
        raise ValueError(f"num_total must be non-negative, got {num_total}")
    if num_correct > num_total:
        raise ValueError(f"num_correct ({num_correct}) > num_total ({num_total})")
    if num_total == 0:
        return FULL_UNCERTAINTY_INTERVAL
    proportion = num_correct / num_total
    lower, upper = wilson_score_interval(proportion, num_total, confidence_level)
    return ConfidenceInterval(point_estimate=proportion, lower_bound=lower, upper_bound=upper)


def compute_proportion_ci(
    proportion: float | None,
    sample_count: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Computes Wilson score confidence interval from a proportion and sample count.

    Edge case behavior: when proportion is None, sample_count == 0, or proportion is not in [0, 1],
    returns full uncertainty interval (0.0, 0.0, 1.0).

    Args:
        proportion: The observed proportion in [0, 1], or None.
        sample_count: Number of samples (must be non-negative).
        confidence_level: Confidence level for the CI (default 0.95).

    Returns:
        ConfidenceInterval with the proportion and Wilson score bounds.

    Raises:
        ValueError: If confidence_level is not in (0, 1) or sample_count is negative.
    """
    _validate_confidence_level(confidence_level)
    if sample_count < 0:
        raise ValueError(f"sample_count must be non-negative, got {sample_count}")
    if sample_count == 0 or proportion is None or not (0 <= proportion <= 1):
        return FULL_UNCERTAINTY_INTERVAL
    lower, upper = wilson_score_interval(proportion, sample_count, confidence_level)
    return ConfidenceInterval(point_estimate=proportion, lower_bound=lower, upper_bound=upper)


def wilson_score_interval(
    proportion: float,
    sample_count: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Computes Wilson score interval for a binomial proportion.

    Args:
        proportion: The observed proportion in [0, 1].
        sample_count: Number of samples (n).
        confidence_level: Confidence level (default 0.95).

    Returns:
        Tuple of (lower_bound, upper_bound), clamped to [0, 1].

    Raises:
        ValueError: If sample_count is not positive, proportion is outside [0, 1], or
            confidence_level is not in (0, 1).
    """
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    if not (0 <= proportion <= 1):
        raise ValueError(f"proportion must be in [0, 1], got {proportion}")
    z_score = z_score_for_confidence(confidence_level)
    n = sample_count
    p = proportion
    z2 = z_score * z_score
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    margin = z_score * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


_NORMAL_DIST = statistics.NormalDist()
"""Standard normal distribution for z-score computation."""


def z_score_for_confidence(
    confidence_level: float,
) -> float:
    """Returns the z-score for a given two-tailed confidence level.

    Uses the standard library's NormalDist inverse CDF, so any confidence level in (0, 1) is
    supported without a hardcoded lookup table.

    Args:
        confidence_level: The desired confidence level (e.g. 0.90, 0.95, 0.99).

    Raises:
        ValueError: If the confidence level is not in (0, 1).
    """
    _validate_confidence_level(confidence_level)
    return _NORMAL_DIST.inv_cdf((1 + confidence_level) / 2)
