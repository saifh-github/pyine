"""Tests for pyine.evals.correctness.calibration."""

from __future__ import annotations

import numpy as np
import pytest

import pyine.evals.correctness.calibration as correctness_calibration


class TestCalibrateThreshold:
    def test_perfect_separation(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        labels = np.array([False, False, False, True, True, True])
        threshold = correctness_calibration.calibrate_threshold(scores, labels, target_fpr=0.05)
        # with perfect separation, threshold should be between 0.3 and 0.8
        fpr = float(np.mean(scores[~labels] >= threshold))
        assert fpr == 0.0  # no false positives

    def test_overlapping_scores(self) -> None:
        scores = np.array([0.1, 0.4, 0.5, 0.6, 0.8, 0.9])
        labels = np.array([False, False, True, False, True, True])
        threshold = correctness_calibration.calibrate_threshold(scores, labels, target_fpr=0.5)
        fpr = float(np.mean(scores[~labels] >= threshold))
        assert fpr <= 0.5 + 1e-9

    def test_strict_fpr_constraint(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.random(100)
        labels = rng.random(100) > 0.5
        for target_fpr in [0.01, 0.05, 0.1]:
            threshold = correctness_calibration.calibrate_threshold(scores, labels, target_fpr)
            fpr = float(np.mean(scores[~labels] >= threshold))
            assert fpr <= target_fpr + 1e-9

    def test_all_correct_raises(self) -> None:
        scores = np.array([0.5, 0.6, 0.7])
        labels = np.array([True, True, True])
        with pytest.raises(ValueError, match="both positive and negative"):
            correctness_calibration.calibrate_threshold(scores, labels, target_fpr=0.05)

    def test_all_incorrect_raises(self) -> None:
        scores = np.array([0.5, 0.6, 0.7])
        labels = np.array([False, False, False])
        with pytest.raises(ValueError, match="both positive and negative"):
            correctness_calibration.calibrate_threshold(scores, labels, target_fpr=0.05)

    def test_tied_scores_at_boundary(self) -> None:
        # all negatives have the same score
        scores = np.array([0.5, 0.5, 0.5, 0.8, 0.9])
        labels = np.array([False, False, False, True, True])
        threshold = correctness_calibration.calibrate_threshold(scores, labels, target_fpr=0.01)
        fpr = float(np.mean(scores[~labels] >= threshold))
        assert fpr <= 0.01 + 1e-9

    def test_very_low_target_fpr_zero_fp_still_accepts_positives(self) -> None:
        scores = np.array([0.5, 0.5, 0.5, 0.8, 0.9])
        labels = np.array([False, False, False, True, True])
        threshold = correctness_calibration.calibrate_threshold(scores, labels, target_fpr=0.001)
        # max_fp = floor(0.001 * 3) = 0, so threshold must block all negatives...
        assert float(np.mean(scores[~labels] >= threshold)) == 0.0
        # ...but positives above all negatives should still be accepted
        assert threshold > np.max(scores[~labels])
        assert threshold < np.min(scores[labels])

    def test_zero_fp_with_overlapping_scores_blocks_some_positives(self) -> None:
        scores = np.array([0.3, 0.5, 0.7, 0.5, 0.8])
        labels = np.array([False, False, True, True, True])
        threshold = correctness_calibration.calibrate_threshold(scores, labels, target_fpr=0.001)
        # threshold must be above max negative (0.5), so the positive at 0.5 is also blocked
        assert float(np.mean(scores[~labels] >= threshold)) == 0.0
        assert threshold > 0.5
        # but positives at 0.7 and 0.8 should be accepted
        assert scores[labels][scores[labels] >= threshold].tolist() == pytest.approx([0.7, 0.8])

    def test_target_fpr_out_of_range(self) -> None:
        scores = np.array([0.5, 0.6])
        labels = np.array([True, False])
        with pytest.raises(ValueError, match="must be in"):
            correctness_calibration.calibrate_threshold(scores, labels, target_fpr=0.0)
        with pytest.raises(ValueError, match="must be in"):
            correctness_calibration.calibrate_threshold(scores, labels, target_fpr=1.0)
