"""Tests for pyine.utils.metrics.confidence."""

from __future__ import annotations

import dataclasses

import pytest

import pyine.utils.metrics.confidence as confidence


class TestConfidenceInterval:
    """Tests for ConfidenceInterval dataclass."""

    def test_frozen(self) -> None:
        ci = confidence.ConfidenceInterval(point_estimate=0.5, lower_bound=0.3, upper_bound=0.7)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ci.point_estimate = 0.6  # type: ignore[misc]


class TestComputeAccuracyWithCi:
    """Tests for compute_accuracy_with_ci (Wilson score from counts)."""

    def test_perfect_accuracy(self) -> None:
        ci = confidence.compute_accuracy_with_ci(num_correct=100, num_total=100)
        assert ci.point_estimate == 1.0
        assert ci.lower_bound > 0.95
        assert ci.upper_bound == pytest.approx(1.0)

    def test_zero_accuracy(self) -> None:
        ci = confidence.compute_accuracy_with_ci(num_correct=0, num_total=100)
        assert ci.point_estimate == 0.0
        assert ci.lower_bound == 0.0
        assert ci.upper_bound < 0.05

    def test_zero_total_returns_full_uncertainty(self) -> None:
        ci = confidence.compute_accuracy_with_ci(num_correct=0, num_total=0)
        assert ci.point_estimate == 0.0
        assert ci.lower_bound == 0.0
        assert ci.upper_bound == 1.0

    def test_ci_narrows_with_more_samples(self) -> None:
        ci_small = confidence.compute_accuracy_with_ci(num_correct=5, num_total=10)
        ci_large = confidence.compute_accuracy_with_ci(num_correct=50, num_total=100)
        small_width = ci_small.upper_bound - ci_small.lower_bound
        large_width = ci_large.upper_bound - ci_large.lower_bound
        assert large_width < small_width

    def test_bounds_clamped(self) -> None:
        ci = confidence.compute_accuracy_with_ci(num_correct=1, num_total=1)
        assert ci.lower_bound >= 0.0
        assert ci.upper_bound <= 1.0


class TestComputeProportionCi:
    """Tests for compute_proportion_ci (Wilson score from proportion)."""

    def test_none_proportion_returns_full_uncertainty(self) -> None:
        ci = confidence.compute_proportion_ci(proportion=None, sample_count=100)
        assert ci.point_estimate == 0.0
        assert ci.lower_bound == 0.0
        assert ci.upper_bound == 1.0

    def test_zero_sample_count_returns_full_uncertainty(self) -> None:
        ci = confidence.compute_proportion_ci(proportion=0.5, sample_count=0)
        assert ci.point_estimate == 0.0
        assert ci.lower_bound == 0.0
        assert ci.upper_bound == 1.0

    def test_negative_sample_count_raises(self) -> None:
        with pytest.raises(ValueError, match="sample_count must be non-negative"):
            confidence.compute_proportion_ci(proportion=0.5, sample_count=-1)

    def test_out_of_range_proportion_returns_full_uncertainty(self) -> None:
        ci = confidence.compute_proportion_ci(proportion=1.5, sample_count=100)
        assert ci.point_estimate == 0.0
        assert ci.lower_bound == 0.0
        assert ci.upper_bound == 1.0

    def test_negative_proportion_returns_full_uncertainty(self) -> None:
        ci = confidence.compute_proportion_ci(proportion=-0.1, sample_count=100)
        assert ci.point_estimate == 0.0
        assert ci.lower_bound == 0.0
        assert ci.upper_bound == 1.0

    def test_valid_proportion(self) -> None:
        ci = confidence.compute_proportion_ci(proportion=0.5, sample_count=100)
        assert ci.point_estimate == 0.5
        assert ci.lower_bound < 0.5
        assert ci.upper_bound > 0.5
        assert ci.lower_bound >= 0.0
        assert ci.upper_bound <= 1.0

    def test_matches_accuracy_with_ci(self) -> None:
        ci_prop = confidence.compute_proportion_ci(proportion=0.5, sample_count=100)
        ci_count = confidence.compute_accuracy_with_ci(num_correct=50, num_total=100)
        assert ci_prop.lower_bound == pytest.approx(ci_count.lower_bound)
        assert ci_prop.upper_bound == pytest.approx(ci_count.upper_bound)


class TestComputeAccuracyWithCiValidation:
    """Tests for input validation in compute_accuracy_with_ci."""

    def test_negative_num_correct_raises(self) -> None:
        with pytest.raises(ValueError, match="num_correct must be non-negative"):
            confidence.compute_accuracy_with_ci(num_correct=-1, num_total=10)

    def test_negative_num_total_raises(self) -> None:
        with pytest.raises(ValueError, match="num_total must be non-negative"):
            confidence.compute_accuracy_with_ci(num_correct=0, num_total=-5)

    def test_num_correct_greater_than_num_total_raises(self) -> None:
        with pytest.raises(ValueError, match="num_correct .* > num_total"):
            confidence.compute_accuracy_with_ci(num_correct=11, num_total=10)

    def test_invalid_confidence_level_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_level must be in \\(0, 1\\)"):
            confidence.compute_accuracy_with_ci(num_correct=1, num_total=10, confidence_level=1.0)


class TestComputeProportionCiValidation:
    """Tests for input validation in compute_proportion_ci."""

    def test_invalid_confidence_level_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_level must be in \\(0, 1\\)"):
            confidence.compute_proportion_ci(proportion=0.5, sample_count=10, confidence_level=0.0)


class TestZScoreForConfidence:
    """Tests for z_score_for_confidence."""

    def test_95_confidence_matches_known_value(self) -> None:
        z = confidence.z_score_for_confidence(0.95)
        assert z == pytest.approx(1.95996, abs=1e-4)

    def test_99_confidence_matches_known_value(self) -> None:
        z = confidence.z_score_for_confidence(0.99)
        assert z == pytest.approx(2.57583, abs=1e-4)

    def test_returns_pure_float(self) -> None:
        z = confidence.z_score_for_confidence(0.95)
        assert type(z) is float


class TestWilsonScoreInterval:
    """Tests for wilson_score_interval known values."""

    def test_known_value_p50_n100(self) -> None:
        # p=0.5, n=100, 95% CI: well-known reference interval
        lower, upper = confidence.wilson_score_interval(proportion=0.5, sample_count=100)
        assert lower == pytest.approx(0.4038, abs=1e-3)
        assert upper == pytest.approx(0.5962, abs=1e-3)

    def test_returns_pure_floats(self) -> None:
        lower, upper = confidence.wilson_score_interval(proportion=0.5, sample_count=100)
        assert type(lower) is float
        assert type(upper) is float


class TestWilsonScoreIntervalValidation:
    """Tests for input validation in wilson_score_interval."""

    def test_invalid_confidence_level_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_level must be in \\(0, 1\\)"):
            confidence.wilson_score_interval(proportion=0.5, sample_count=10, confidence_level=-0.1)

    def test_non_positive_sample_count_raises(self) -> None:
        with pytest.raises(ValueError, match="sample_count must be positive"):
            confidence.wilson_score_interval(proportion=0.5, sample_count=0)

    def test_out_of_range_proportion_raises(self) -> None:
        with pytest.raises(ValueError, match="proportion must be in \\[0, 1\\]"):
            confidence.wilson_score_interval(proportion=1.5, sample_count=10)
