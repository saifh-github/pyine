"""Tests for pyine.utils.metrics.multi_sample."""

from __future__ import annotations

import pytest

import pyine.utils.metrics.multi_sample as msm


class TestComputePassAtK:
    """Tests for compute_pass_at_k (single-sample estimator)."""

    def test_all_correct_returns_one(self) -> None:
        assert msm.compute_pass_at_k(num_total=10, num_correct=10, k=1) == 1.0
        assert msm.compute_pass_at_k(num_total=10, num_correct=10, k=5) == 1.0

    def test_none_correct_returns_zero(self) -> None:
        assert msm.compute_pass_at_k(num_total=10, num_correct=0, k=1) == 0.0
        assert msm.compute_pass_at_k(num_total=10, num_correct=0, k=5) == 0.0

    def test_zero_total_returns_zero(self) -> None:
        assert msm.compute_pass_at_k(num_total=0, num_correct=0, k=1) == 0.0

    def test_zero_total_with_large_k_returns_zero(self) -> None:
        # num_total=0 returns 0.0 regardless of k (k > num_total check is skipped)
        assert msm.compute_pass_at_k(num_total=0, num_correct=0, k=5) == 0.0
        assert msm.compute_pass_at_k(num_total=0, num_correct=0, k=100) == 0.0

    def test_k_greater_than_total_raises(self) -> None:
        with pytest.raises(ValueError, match="k .* > num_total"):
            msm.compute_pass_at_k(num_total=5, num_correct=3, k=10)

    def test_pass_at_1_equals_fraction(self) -> None:
        assert msm.compute_pass_at_k(num_total=10, num_correct=3, k=1) == pytest.approx(0.3)
        assert msm.compute_pass_at_k(num_total=10, num_correct=7, k=1) == pytest.approx(0.7)

    def test_guaranteed_correct_returns_one(self) -> None:
        # n-c < k means guaranteed at least one correct in k picks
        assert msm.compute_pass_at_k(num_total=5, num_correct=4, k=2) == 1.0

    def test_known_combinatorial_value(self) -> None:
        # n=10, c=1, k=5: pass@5 = 1 - C(9,5)/C(10,5) = 1 - 126/252 = 0.5
        assert msm.compute_pass_at_k(num_total=10, num_correct=1, k=5) == pytest.approx(0.5)

    def test_n_equals_k(self) -> None:
        assert msm.compute_pass_at_k(num_total=5, num_correct=1, k=5) == 1.0
        assert msm.compute_pass_at_k(num_total=5, num_correct=0, k=5) == 0.0

    def test_large_n_numerical_stability(self) -> None:
        result = msm.compute_pass_at_k(num_total=1000, num_correct=10, k=100)
        assert 0.0 <= result <= 1.0

    def test_k_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            msm.compute_pass_at_k(num_total=10, num_correct=3, k=0)

    def test_k_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            msm.compute_pass_at_k(num_total=10, num_correct=3, k=-1)

    def test_negative_num_total_raises(self) -> None:
        with pytest.raises(ValueError, match="num_total must be non-negative"):
            msm.compute_pass_at_k(num_total=-1, num_correct=0, k=1)

    def test_negative_num_correct_raises(self) -> None:
        with pytest.raises(ValueError, match="num_correct must be non-negative"):
            msm.compute_pass_at_k(num_total=10, num_correct=-1, k=1)

    def test_num_correct_greater_than_num_total_raises(self) -> None:
        with pytest.raises(ValueError, match="num_correct .* > num_total"):
            msm.compute_pass_at_k(num_total=5, num_correct=10, k=1)


class TestPassAtKWithCi:
    """Tests for compute_pass_at_k_with_ci."""

    def test_empty_summaries_raises(self) -> None:
        with pytest.raises(ValueError, match="summaries must not be empty"):
            msm.compute_pass_at_k_with_ci([], k=1)

    def test_pass_at_1_equals_mean_fraction(self) -> None:
        # sample1: 3/5 correct, sample2: 1/5 correct -> mean pass@1 = (0.6 + 0.2) / 2 = 0.4
        summaries = [
            msm.SampleAttemptSummary(num_total=5, num_correct=3),
            msm.SampleAttemptSummary(num_total=5, num_correct=1),
        ]
        ci = msm.compute_pass_at_k_with_ci(summaries, k=1)
        assert ci.point_estimate == pytest.approx(0.4)
        assert ci.lower_bound <= ci.point_estimate
        assert ci.upper_bound >= ci.point_estimate

    def test_pass_at_k_all_correct(self) -> None:
        summaries = [msm.SampleAttemptSummary(num_total=5, num_correct=5)]
        ci = msm.compute_pass_at_k_with_ci(summaries, k=3)
        assert ci.point_estimate == 1.0

    def test_ci_bounds_clamped(self) -> None:
        summaries = [
            msm.SampleAttemptSummary(num_total=10, num_correct=1),
            msm.SampleAttemptSummary(num_total=10, num_correct=9),
        ]
        ci = msm.compute_pass_at_k_with_ci(summaries, k=5)
        assert ci.lower_bound >= 0.0
        assert ci.upper_bound <= 1.0

    def test_single_sample_bounds_equal_mean(self) -> None:
        # with n_samples=1, CI degenerates: bounds = clamped point estimate
        summaries = [msm.SampleAttemptSummary(num_total=10, num_correct=3)]
        ci = msm.compute_pass_at_k_with_ci(summaries, k=1)
        assert ci.point_estimate == pytest.approx(0.3)
        assert ci.lower_bound == pytest.approx(0.3)
        assert ci.upper_bound == pytest.approx(0.3)

    def test_zero_attempt_sample_contributes_zero(self) -> None:
        # num_total=0 contributes 0.0 to the mean (no attempts = no chance of passing)
        summaries = [
            msm.SampleAttemptSummary(num_total=0, num_correct=0),
            msm.SampleAttemptSummary(num_total=10, num_correct=10),
        ]
        ci = msm.compute_pass_at_k_with_ci(summaries, k=1)
        assert ci.point_estimate == pytest.approx(0.5)  # mean of [0.0, 1.0]


class TestMajorityCorrect:
    """Tests for compute_majority_correct."""

    def test_empty_summaries_returns_zero(self) -> None:
        assert msm.compute_majority_correct([]) == 0.0

    def test_all_majority_correct(self) -> None:
        summaries = [
            msm.SampleAttemptSummary(num_total=5, num_correct=4),
            msm.SampleAttemptSummary(num_total=5, num_correct=3),
        ]
        assert msm.compute_majority_correct(summaries) == 1.0

    def test_no_majority_correct(self) -> None:
        summaries = [
            msm.SampleAttemptSummary(num_total=5, num_correct=2),
            msm.SampleAttemptSummary(num_total=5, num_correct=1),
        ]
        assert msm.compute_majority_correct(summaries) == 0.0

    def test_mixed_majority(self) -> None:
        summaries = [
            msm.SampleAttemptSummary(num_total=5, num_correct=4),
            msm.SampleAttemptSummary(num_total=5, num_correct=1),
        ]
        assert msm.compute_majority_correct(summaries) == pytest.approx(0.5)

    def test_exact_half_not_majority(self) -> None:
        # 2/4 is NOT a strict majority (need >K/2)
        summaries = [msm.SampleAttemptSummary(num_total=4, num_correct=2)]
        assert msm.compute_majority_correct(summaries) == 0.0


class TestOutputDiversity:
    """Tests for compute_mean_output_diversity."""

    def test_empty_summaries_returns_zero(self) -> None:
        assert msm.compute_mean_output_diversity([]) == 0.0

    def test_all_identical_outputs(self) -> None:
        # 1 unique output out of 5 -> diversity = 1/5
        summaries = [msm.SampleAttemptSummary(num_total=5, num_correct=5, num_unique_outputs=1)]
        assert msm.compute_mean_output_diversity(summaries) == pytest.approx(1 / 5)

    def test_all_unique_outputs(self) -> None:
        # 5 unique outputs out of 5 -> diversity = 1.0
        summaries = [msm.SampleAttemptSummary(num_total=5, num_correct=0, num_unique_outputs=5)]
        assert msm.compute_mean_output_diversity(summaries) == pytest.approx(1.0)

    def test_zero_attempt_group_counts_as_zero_diversity(self) -> None:
        zero_summary = msm.SampleAttemptSummary(num_total=0, num_correct=0, num_unique_outputs=0)
        normal_summary = msm.SampleAttemptSummary(num_total=5, num_correct=3, num_unique_outputs=3)
        # mean of [0.0, 3/5] = 0.3
        result = msm.compute_mean_output_diversity([zero_summary, normal_summary])
        assert result == pytest.approx(0.3)


class TestMeanUniqueOutputs:
    """Tests for compute_mean_unique_outputs."""

    def test_empty_summaries_returns_zero(self) -> None:
        assert msm.compute_mean_unique_outputs([]) == 0.0

    def test_counts_unique_outputs(self) -> None:
        summaries = [msm.SampleAttemptSummary(num_total=3, num_correct=1, num_unique_outputs=2)]
        assert msm.compute_mean_unique_outputs(summaries) == pytest.approx(2.0)

    def test_mean_across_samples(self) -> None:
        summaries = [
            msm.SampleAttemptSummary(num_total=5, num_correct=3, num_unique_outputs=2),
            msm.SampleAttemptSummary(num_total=5, num_correct=1, num_unique_outputs=4),
        ]
        assert msm.compute_mean_unique_outputs(summaries) == pytest.approx(3.0)


class TestSampleAttemptSummaryValidation:
    """Tests for _validate_counts / _validate_summary on invalid inputs."""

    def test_negative_num_total_raises(self) -> None:
        bad = [msm.SampleAttemptSummary(num_total=-1, num_correct=0)]
        with pytest.raises(ValueError, match="num_total must be non-negative"):
            msm.compute_majority_correct(bad)

    def test_negative_num_correct_raises(self) -> None:
        bad = [msm.SampleAttemptSummary(num_total=5, num_correct=-1)]
        with pytest.raises(ValueError, match="num_correct must be non-negative"):
            msm.compute_majority_correct(bad)

    def test_negative_num_unique_outputs_raises(self) -> None:
        bad = [msm.SampleAttemptSummary(num_total=5, num_correct=1, num_unique_outputs=-1)]
        with pytest.raises(ValueError, match="num_unique_outputs must be non-negative"):
            msm.compute_mean_output_diversity(bad)

    def test_num_correct_greater_than_num_total_raises(self) -> None:
        bad = [msm.SampleAttemptSummary(num_total=5, num_correct=10)]
        with pytest.raises(ValueError, match="num_correct .* > num_total"):
            msm.compute_majority_correct(bad)

    def test_num_unique_outputs_greater_than_num_total_raises(self) -> None:
        bad = [msm.SampleAttemptSummary(num_total=5, num_correct=1, num_unique_outputs=10)]
        with pytest.raises(ValueError, match="num_unique_outputs .* > num_total"):
            msm.compute_mean_output_diversity(bad)

    def test_bad_unique_outputs_does_not_affect_pass_at_k(self) -> None:
        # num_unique_outputs > num_total should NOT cause pass@k or majority to fail
        summaries = [msm.SampleAttemptSummary(num_total=5, num_correct=3, num_unique_outputs=99)]
        ci = msm.compute_pass_at_k_with_ci(summaries, k=1)
        assert ci.point_estimate == pytest.approx(0.6)
        assert msm.compute_majority_correct(summaries) == 1.0


class TestPassAtKWithCiValidation:
    """Tests for validation in compute_pass_at_k_with_ci."""

    def test_k_zero_raises(self) -> None:
        summaries = [msm.SampleAttemptSummary(num_total=5, num_correct=3)]
        with pytest.raises(ValueError, match="k must be positive"):
            msm.compute_pass_at_k_with_ci(summaries, k=0)

    def test_k_negative_raises(self) -> None:
        summaries = [msm.SampleAttemptSummary(num_total=5, num_correct=3)]
        with pytest.raises(ValueError, match="k must be positive"):
            msm.compute_pass_at_k_with_ci(summaries, k=-1)

    def test_k_zero_on_empty_raises_before_empty_check(self) -> None:
        # k validation happens before the empty-summaries check
        with pytest.raises(ValueError, match="k must be positive"):
            msm.compute_pass_at_k_with_ci([], k=0)

    def test_empty_summaries_raises(self) -> None:
        with pytest.raises(ValueError, match="summaries must not be empty"):
            msm.compute_pass_at_k_with_ci([], k=1)

    def test_invalid_confidence_level_raises_on_single_sample(self) -> None:
        summaries = [msm.SampleAttemptSummary(num_total=5, num_correct=3)]
        with pytest.raises(ValueError, match="confidence_level must be in"):
            msm.compute_pass_at_k_with_ci(summaries, k=1, confidence_level=-0.5)

    def test_invalid_summary_in_list_raises(self) -> None:
        bad = [msm.SampleAttemptSummary(num_total=5, num_correct=10)]
        with pytest.raises(ValueError, match="num_correct .* > num_total"):
            msm.compute_pass_at_k_with_ci(bad, k=1)

    def test_num_total_less_than_k_raises(self) -> None:
        summaries = [
            msm.SampleAttemptSummary(num_total=10, num_correct=5),
            msm.SampleAttemptSummary(num_total=3, num_correct=1),  # num_total < k
        ]
        with pytest.raises(ValueError, match="k .* > num_total"):
            msm.compute_pass_at_k_with_ci(summaries, k=5)


class TestPassAtKExpm1Precision:
    """Regression test for expm1-based computation avoiding catastrophic cancellation."""

    def test_small_pass_rate_precision(self) -> None:
        # n=1000, c=1, k=1 -> pass@1 = 1/1000 = 0.001
        result = msm.compute_pass_at_k(num_total=1000, num_correct=1, k=1)
        assert result == pytest.approx(0.001, abs=1e-10)

    def test_very_small_pass_rate(self) -> None:
        # n=10000, c=1, k=1 -> pass@1 = 0.0001
        result = msm.compute_pass_at_k(num_total=10000, num_correct=1, k=1)
        assert result == pytest.approx(0.0001, abs=1e-12)
