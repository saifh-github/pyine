"""Multi-attempt metrics for generation-based evaluations.

Implements the standard Pass@K estimator from Chen et al. (2021), majority correct, output
diversity, and mean unique outputs. These metrics operate on lists of ``SampleAttemptSummary``
objects that capture per-sample attempt counts.

All functions are stateless and I/O-free.
"""

from __future__ import annotations

import math
import typing

import numpy as np

import pyine.utils.metrics.confidence


class SampleAttemptSummary(typing.NamedTuple):
    """Per-sample summary of attempt outcomes for multi-sample metrics.

    This is the input type for all aggregate metric functions in this module. It decouples
    the metric computations from any specific evaluation framework (e.g. hard/soft matching
    is resolved by the caller before constructing this summary).
    """

    num_total: int
    """Total number of attempts for this sample."""
    num_correct: int
    """Number of correct attempts for this sample."""
    num_unique_outputs: int = 0
    """Number of unique (stripped) outputs across attempts. Used for diversity metrics."""


def _validate_counts(
    summary: SampleAttemptSummary,
) -> None:
    """Validates num_total and num_correct fields only (not num_unique_outputs)."""
    if summary.num_total < 0:
        raise ValueError(f"num_total must be non-negative, got {summary.num_total}")
    if summary.num_correct < 0:
        raise ValueError(f"num_correct must be non-negative, got {summary.num_correct}")
    if summary.num_correct > summary.num_total:
        raise ValueError(f"num_correct ({summary.num_correct}) > num_total ({summary.num_total})")


def _validate_summary(
    summary: SampleAttemptSummary,
) -> None:
    """Validates all fields including num_unique_outputs (for diversity metrics)."""
    _validate_counts(summary)
    if summary.num_unique_outputs < 0:
        raise ValueError(f"num_unique_outputs must be non-negative, got {summary.num_unique_outputs}")
    if summary.num_unique_outputs > summary.num_total:
        raise ValueError(f"num_unique_outputs ({summary.num_unique_outputs}) > num_total ({summary.num_total})")


def compute_pass_at_k(
    num_total: int,
    num_correct: int,
    k: int,
) -> float:
    """Computes the unbiased Pass@K estimator for a single sample.

    Uses the standard formula: pass@k = 1 - C(n-c, k) / C(n, k), computed via log-gamma
    for numerical stability (using ``expm1`` to avoid catastrophic cancellation when the
    ratio is close to 1). For k=1, this simplifies to c/n.

    Args:
        num_total: Total number of attempts (n).
        num_correct: Number of correct attempts (c).
        k: The k value for Pass@K.

    Returns:
        The Pass@K estimate in [0, 1]. When num_total is 0, returns 0.0 regardless of k
        (no attempts means no chance of passing; the k > num_total check is skipped).

    Raises:
        ValueError: If inputs are invalid (negative counts, k <= 0, num_correct > num_total,
            or k > num_total when num_total > 0).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if num_total < 0:
        raise ValueError(f"num_total must be non-negative, got {num_total}")
    if num_correct < 0:
        raise ValueError(f"num_correct must be non-negative, got {num_correct}")
    if num_correct > num_total:
        raise ValueError(f"num_correct ({num_correct}) > num_total ({num_total})")
    if num_total == 0:
        return 0.0
    if num_correct == 0:
        return 0.0
    if k > num_total:
        raise ValueError(f"k ({k}) > num_total ({num_total}); invalid for Pass@K")
    if num_total - num_correct < k:
        return 1.0  # guaranteed to pick at least one correct
    if k == 1:
        return num_correct / num_total  # exact for k=1
    # log-gamma computation of log(C(n-c, k) / C(n, k)); O(1) instead of O(k)
    # use -expm1 instead of 1-exp to avoid catastrophic cancellation near 1
    log_ratio = (
        math.lgamma(num_total - num_correct + 1)
        - math.lgamma(num_total - num_correct - k + 1)
        - math.lgamma(num_total + 1)
        + math.lgamma(num_total - k + 1)
    )
    return float(-np.expm1(log_ratio))


def compute_pass_at_k_with_ci(
    summaries: list[SampleAttemptSummary],
    k: int,
    confidence_level: float = 0.95,
) -> pyine.utils.metrics.confidence.ConfidenceInterval:
    """Computes mean Pass@K across samples with SEM-based confidence interval.

    Per-sample pass@k estimates are real-valued for all k (including k=1 where pass@1 = c/n).
    The CI uses a normal approximation: mean +/- z * std / sqrt(n_samples), clamped to [0, 1].
    For small sample counts (n < 30), a t-distribution would be more appropriate; this function
    uses the normal approximation regardless of sample size.

    Args:
        summaries: Non-empty list of per-sample attempt summaries. Summaries with num_total == 0
            contribute 0.0 to the mean (no attempts = no chance of passing); summaries with
            num_total > 0 must have num_total >= k.
        k: The k value for Pass@K (must be positive).
        confidence_level: Confidence level for the CI (default 0.95).

    Returns:
        ConfidenceInterval with mean Pass@K and SEM-based bounds.

    Raises:
        ValueError: If summaries is empty, k <= 0, confidence_level is invalid, any summary
            has invalid counts, or any summary with num_total > 0 has num_total < k.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not summaries:
        raise ValueError("summaries must not be empty")
    z_score = pyine.utils.metrics.confidence.z_score_for_confidence(confidence_level)  # validates confidence_level
    for summary in summaries:
        _validate_counts(summary)
    per_sample_estimates = [compute_pass_at_k(s.num_total, s.num_correct, k) for s in summaries]
    arr = np.array(per_sample_estimates)
    mean_val = float(np.mean(arr))
    n_samples = len(per_sample_estimates)
    if n_samples < 2:
        return pyine.utils.metrics.confidence.ConfidenceInterval(
            point_estimate=mean_val,
            lower_bound=max(0.0, mean_val),
            upper_bound=min(1.0, mean_val),
        )
    std_val = float(np.std(arr, ddof=1))
    sem = std_val / math.sqrt(n_samples)
    lower = max(0.0, mean_val - z_score * sem)
    upper = min(1.0, mean_val + z_score * sem)
    return pyine.utils.metrics.confidence.ConfidenceInterval(
        point_estimate=mean_val,
        lower_bound=lower,
        upper_bound=upper,
    )


def compute_majority_correct(
    summaries: list[SampleAttemptSummary],
) -> float:
    """Computes fraction of samples where a strict majority (>K/2) of all K attempts are correct.

    Uses all K attempts for each sample (no per-k variants). This is NOT true majority-vote-
    on-output (which would pick the most common output then check correctness).

    Args:
        summaries: List of per-sample attempt summaries.

    Returns:
        Fraction in [0, 1] of samples with majority correct.

    Raises:
        ValueError: If any summary has invalid counts.
    """
    for summary in summaries:
        _validate_counts(summary)
    if not summaries:
        return 0.0
    majority_count = sum(1 for s in summaries if s.num_correct > s.num_total / 2)
    return majority_count / len(summaries)


def compute_mean_output_diversity(
    summaries: list[SampleAttemptSummary],
) -> float:
    """Computes mean of num_unique_outputs / num_total per sample.

    Summaries with num_total == 0 contribute 0.0 to the mean (no attempts = no diversity).

    Args:
        summaries: List of per-sample attempt summaries.

    Returns:
        Mean diversity ratio in [0, 1]. Returns 0.0 for an empty list.

    Raises:
        ValueError: If any summary has invalid counts.
    """
    for summary in summaries:
        _validate_summary(summary)
    if not summaries:
        return 0.0
    diversities = [s.num_unique_outputs / s.num_total if s.num_total > 0 else 0.0 for s in summaries]
    return float(np.mean(diversities))


def compute_mean_unique_outputs(
    summaries: list[SampleAttemptSummary],
) -> float:
    """Computes mean of num_unique_outputs per sample (absolute count, not ratio).

    Args:
        summaries: List of per-sample attempt summaries.

    Returns:
        Mean unique output count across samples.

    Raises:
        ValueError: If any summary has invalid counts.
    """
    for summary in summaries:
        _validate_summary(summary)
    if not summaries:
        return 0.0
    return float(np.mean([s.num_unique_outputs for s in summaries]))
