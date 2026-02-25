"""FPR-constrained threshold calibration for classifier evaluations.

Provides ``calibrate_threshold`` which selects a decision threshold satisfying
FPR <= target_fpr on a validation set. Convention: FPR = P(score >= threshold | label=False).
"""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt


def calibrate_threshold(
    scores: npt.NDArray[np.floating[typing.Any]],
    labels: npt.NDArray[np.bool_],
    target_fpr: float,
) -> float:
    """Select a threshold such that FPR <= target_fpr on the provided scores and labels.

    Convention: a record is accepted if score >= threshold. FPR is the fraction of incorrect
    records (label=False) that are accepted.

    Args:
        scores: Continuous classifier scores (higher = more likely correct).
        labels: Boolean correctness labels (True = correct).
        target_fpr: Maximum allowed false positive rate. Must be in (0, 1).

    Returns:
        The calibrated threshold.

    Raises:
        ValueError: If all labels are the same class (calibration is meaningless),
            or if target_fpr is out of range.
    """
    if target_fpr <= 0.0 or target_fpr >= 1.0:
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    if len(scores) != len(labels):
        raise ValueError(f"scores and labels must have the same length, got {len(scores)} and {len(labels)}")
    if len(scores) == 0:
        raise ValueError("scores and labels must not be empty")
    if labels.dtype != np.bool_:
        raise ValueError(f"labels must have dtype bool, got {labels.dtype}")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite (no NaN or inf)")
    n_correct = int(np.sum(labels))
    n_incorrect = len(labels) - n_correct
    if n_correct == 0 or n_incorrect == 0:
        raise ValueError(
            f"calibration requires both positive and negative labels; "
            f"got {n_correct} correct and {n_incorrect} incorrect"
        )
    neg_scores = scores[~labels]
    max_fp = int(np.floor(target_fpr * n_incorrect))
    if max_fp == 0:
        # can't allow any false positives: set threshold above the highest negative score
        # (not above all scores; positives above all negatives should still be accepted)
        return float(np.nextafter(np.max(neg_scores), np.inf))
    # sort negative scores descending
    neg_scores_sorted = np.sort(neg_scores)[::-1]
    candidate = float(neg_scores_sorted[max_fp - 1])
    # tie-safety: verify FPR constraint
    threshold = candidate
    while int(np.sum(neg_scores >= threshold)) > max_fp:
        threshold = float(np.nextafter(threshold, np.inf))
    # postcondition: FPR <= target_fpr
    actual_fpr = float(np.mean(neg_scores >= threshold))
    if actual_fpr > target_fpr + 1e-12:
        raise RuntimeError(f"calibration postcondition violated: actual_fpr={actual_fpr} > target_fpr={target_fpr}")
    return threshold
