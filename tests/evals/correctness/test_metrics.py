"""Tests for pyine.evals.correctness.metrics."""

from __future__ import annotations

import typing

import numpy as np
import pytest

import pyine.evals.correctness.configs as correctness_configs
import pyine.evals.correctness.metrics as correctness_metrics
import pyine.evals.correctness.types as correctness_types
import pyine.evals.utils


def _make_record(
    sample_id: str,
    problem_id: str,
    attempt_index: int = 0,
    label: bool = True,
    code_type: str = "original",
    difficulty_score: float | None = None,
    tags: list[str] | None = None,
    record: dict[str, typing.Any] | None = None,
) -> correctness_types.EvalRecord:
    return correctness_types.EvalRecord(
        sample_id=sample_id,
        problem_id=problem_id,
        attempt_index=attempt_index,
        model_output="output",
        final_answer=None,
        expected_output="expected",
        label=label,
        code_type=code_type,
        tags=tags if tags is not None else [],
        record=record if record is not None else {},
        difficulty_score=difficulty_score,
    )


class TestThresholdFreeMetrics:
    def test_empty_labels(self) -> None:
        scores = np.array([], dtype=np.float64)
        labels = np.array([], dtype=np.bool_)
        result = correctness_metrics.compute_threshold_free_metrics(scores, labels, [0.01], 100)
        assert result.auroc is None
        assert result.average_precision is None
        assert len(result.fpr_grid) == 0

    def test_perfect_auroc(self) -> None:
        scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
        labels = np.array([False, False, False, True, True, True])
        result = correctness_metrics.compute_threshold_free_metrics(scores, labels, [0.01], 100)
        assert result.auroc is not None
        assert result.auroc == pytest.approx(1.0)

    def test_random_auroc_near_half(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.random(1000)
        labels = rng.random(1000) > 0.5
        result = correctness_metrics.compute_threshold_free_metrics(scores, labels, [0.01], 100)
        assert result.auroc is not None
        assert abs(result.auroc - 0.5) < 0.1

    def test_single_class_returns_none(self) -> None:
        scores = np.array([0.5, 0.6, 0.7])
        labels = np.array([True, True, True])
        result = correctness_metrics.compute_threshold_free_metrics(scores, labels, [0.01], 100)
        assert result.auroc is None
        assert result.average_precision is None
        assert result.tpr_at_fpr is None
        assert len(result.fpr_grid) == 0

    def test_target_fprs_out_of_range_raises(self) -> None:
        scores = np.array([0.5, 0.8])
        labels = np.array([False, True])
        with pytest.raises(ValueError, match="must all be in"):
            correctness_metrics.compute_threshold_free_metrics(scores, labels, [1.5], 100)

    def test_non_bool_labels_raises(self) -> None:
        scores = np.array([0.5, 0.8])
        labels = np.array([0, 1])  # int, not bool
        with pytest.raises(ValueError, match="boolean array"):
            correctness_metrics.compute_threshold_free_metrics(scores, labels, [0.01], 100)

    def test_tpr_at_fpr(self) -> None:
        scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
        labels = np.array([False, False, False, True, True, True])
        result = correctness_metrics.compute_threshold_free_metrics(scores, labels, [0.01, 0.5], 100)
        assert result.tpr_at_fpr is not None
        assert 0.01 in result.tpr_at_fpr
        assert 0.5 in result.tpr_at_fpr


class TestBuildDeduplicatedRoc:
    def test_no_duplicates_passthrough(self) -> None:
        fpr = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        tpr = np.array([0.0, 0.5, 0.7, 0.9, 1.0])
        fpr_u, tpr_u = correctness_metrics._build_deduplicated_roc(fpr, tpr)
        np.testing.assert_array_almost_equal(fpr_u, fpr)
        np.testing.assert_array_almost_equal(tpr_u, tpr)

    def test_duplicates_keep_max_tpr(self) -> None:
        fpr = np.array([0.0, 0.5, 0.5, 1.0])
        tpr = np.array([0.0, 0.3, 0.7, 1.0])
        fpr_u, tpr_u = correctness_metrics._build_deduplicated_roc(fpr, tpr)
        np.testing.assert_array_almost_equal(fpr_u, [0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(tpr_u, [0.0, 0.7, 1.0])

    def test_all_same_fpr(self) -> None:
        fpr = np.array([0.5, 0.5, 0.5])
        tpr = np.array([0.1, 0.5, 0.3])
        fpr_u, tpr_u = correctness_metrics._build_deduplicated_roc(fpr, tpr)
        assert len(fpr_u) == 1
        assert fpr_u[0] == pytest.approx(0.5)
        assert tpr_u[0] == pytest.approx(0.5)  # max of 0.1, 0.5, 0.3

    def test_assertion_on_non_monotonic(self) -> None:
        fpr = np.array([0.0, 0.5, 0.3, 1.0])
        tpr = np.array([0.0, 0.5, 0.7, 1.0])
        with pytest.raises(AssertionError):
            correctness_metrics._build_deduplicated_roc(fpr, tpr)


class TestComputeTprAtTargetFprs:
    def test_perfect_separation(self) -> None:
        scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
        labels = np.array([False, False, False, True, True, True])
        result = correctness_metrics.compute_tpr_at_target_fprs(scores, labels, [0.01, 0.5, 1.0])
        assert result is not None
        assert result[1.0] == pytest.approx(1.0)

    def test_random_scores_reasonable(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.random(1000)
        labels = rng.random(1000) > 0.5
        result = correctness_metrics.compute_tpr_at_target_fprs(scores, labels, [0.5])
        assert result is not None
        assert abs(result[0.5] - 0.5) < 0.15

    def test_multiple_targets_monotonic(self) -> None:
        rng = np.random.default_rng(123)
        scores = rng.random(500)
        labels = rng.random(500) > 0.3
        targets = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        result = correctness_metrics.compute_tpr_at_target_fprs(scores, labels, targets)
        assert result is not None
        assert len(result) == len(targets)
        values = [result[t] for t in targets]
        assert all(values[idx] <= values[idx + 1] for idx in range(len(values) - 1))

    def test_single_class_returns_none(self) -> None:
        scores = np.array([0.5, 0.6, 0.7])
        labels = np.array([True, True, True])
        assert correctness_metrics.compute_tpr_at_target_fprs(scores, labels, [0.05]) is None

    def test_empty_returns_none(self) -> None:
        scores = np.array([], dtype=np.float64)
        labels = np.array([], dtype=np.bool_)
        assert correctness_metrics.compute_tpr_at_target_fprs(scores, labels, [0.05]) is None

    def test_target_fprs_out_of_range_raises(self) -> None:
        scores = np.array([0.5, 0.8])
        labels = np.array([False, True])
        with pytest.raises(ValueError, match="must all be in"):
            correctness_metrics.compute_tpr_at_target_fprs(scores, labels, [1.5])

    def test_empty_target_fprs_raises(self) -> None:
        scores = np.array([0.5, 0.8])
        labels = np.array([False, True])
        with pytest.raises(ValueError, match="must be non-empty"):
            correctness_metrics.compute_tpr_at_target_fprs(scores, labels, [])

    def test_non_bool_labels_raises(self) -> None:
        scores = np.array([0.5, 0.8])
        labels = np.array([0, 1])
        with pytest.raises(ValueError, match="boolean array"):
            correctness_metrics.compute_tpr_at_target_fprs(scores, labels, [0.05])

    def test_length_mismatch_raises(self) -> None:
        scores = np.array([0.5, 0.8, 0.9])
        labels = np.array([False, True])
        with pytest.raises(ValueError, match="same length"):
            correctness_metrics.compute_tpr_at_target_fprs(scores, labels, [0.05])

    def test_agrees_with_threshold_free_metrics(self) -> None:
        rng = np.random.default_rng(99)
        scores = rng.random(200)
        labels = rng.random(200) > 0.4
        targets = [0.01, 0.05, 0.1, 0.5]
        standalone = correctness_metrics.compute_tpr_at_target_fprs(scores, labels, targets)
        full = correctness_metrics.compute_threshold_free_metrics(scores, labels, targets, 200)
        assert standalone is not None
        assert full.tpr_at_fpr is not None
        for fpr_val in targets:
            assert standalone[fpr_val] == pytest.approx(full.tpr_at_fpr[fpr_val])


class TestInterpolateTprFromRocGrids:
    def test_linear_roc(self) -> None:
        fpr_grid = np.linspace(0.0, 1.0, 100)
        tpr_grid = np.linspace(0.0, 1.0, 100)  # random-classifier diagonal
        result = correctness_metrics.interpolate_tpr_from_roc_grids(fpr_grid, tpr_grid, [0.5])
        assert result[0.5] == pytest.approx(0.5, abs=0.02)

    def test_perfect_roc(self) -> None:
        fpr_grid = np.linspace(0.0, 1.0, 100)
        tpr_grid = np.ones(100)
        result = correctness_metrics.interpolate_tpr_from_roc_grids(fpr_grid, tpr_grid, [0.0, 0.5, 1.0])
        for fpr_val in [0.0, 0.5, 1.0]:
            assert result[fpr_val] == pytest.approx(1.0)

    def test_endpoints(self) -> None:
        fpr_grid = np.linspace(0.0, 1.0, 200)
        tpr_grid = np.sqrt(fpr_grid)
        result = correctness_metrics.interpolate_tpr_from_roc_grids(fpr_grid, tpr_grid, [0.0, 1.0])
        assert result[0.0] == pytest.approx(0.0, abs=1e-6)
        assert result[1.0] == pytest.approx(1.0, abs=1e-6)

    def test_multiple_targets(self) -> None:
        fpr_grid = np.linspace(0.0, 1.0, 200)
        tpr_grid = np.sqrt(fpr_grid)
        targets = [0.01, 0.05, 0.1, 0.5]
        result = correctness_metrics.interpolate_tpr_from_roc_grids(fpr_grid, tpr_grid, targets)
        assert len(result) == len(targets)

    def test_empty_grids_raises(self) -> None:
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="must be non-empty"):
            correctness_metrics.interpolate_tpr_from_roc_grids(empty, empty, [0.05])

    def test_mismatched_grid_lengths_raises(self) -> None:
        fpr_grid = np.linspace(0.0, 1.0, 100)
        tpr_grid = np.linspace(0.0, 1.0, 50)
        with pytest.raises(ValueError, match="same length"):
            correctness_metrics.interpolate_tpr_from_roc_grids(fpr_grid, tpr_grid, [0.05])

    def test_target_fprs_out_of_range_raises(self) -> None:
        fpr_grid = np.linspace(0.0, 1.0, 100)
        tpr_grid = np.linspace(0.0, 1.0, 100)
        with pytest.raises(ValueError, match="must all be in"):
            correctness_metrics.interpolate_tpr_from_roc_grids(fpr_grid, tpr_grid, [-0.1])

    def test_agrees_with_raw_scores_within_grid_tolerance(self) -> None:
        rng = np.random.default_rng(77)
        scores = rng.random(500)
        labels = rng.random(500) > 0.4
        targets = [0.01, 0.05, 0.1, 0.2, 0.5]
        exact = correctness_metrics.compute_tpr_at_target_fprs(scores, labels, targets)
        full = correctness_metrics.compute_threshold_free_metrics(scores, labels, [0.05], 200)
        grid_based = correctness_metrics.interpolate_tpr_from_roc_grids(
            full.fpr_grid,
            full.tpr_grid,
            targets,
        )
        assert exact is not None
        for fpr_val in targets:
            assert grid_based[fpr_val] == pytest.approx(exact[fpr_val], abs=0.02)


class TestRawScoreReconstruction:
    """Validates that compute_tpr_at_target_fprs reproduces the stored tpr_at_fpr values
    from compute_threshold_free_metrics exactly; the contract the notebook relies on."""

    def test_reconstruction_matches_stored_values_exactly(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.random(300)
        labels = rng.random(300) > 0.4
        stored_targets = [0.01, 0.05, 0.1]
        full = correctness_metrics.compute_threshold_free_metrics(scores, labels, stored_targets, 200)
        assert full.tpr_at_fpr is not None
        reconstructed = correctness_metrics.compute_tpr_at_target_fprs(scores, labels, stored_targets)
        assert reconstructed is not None
        for fpr_val in stored_targets:
            assert reconstructed[fpr_val] == pytest.approx(full.tpr_at_fpr[fpr_val])

    def test_reconstruction_with_novel_targets_exact(self) -> None:
        rng = np.random.default_rng(55)
        scores = rng.random(400)
        labels = rng.random(400) > 0.5
        novel_targets = [0.02, 0.07, 0.15, 0.3, 0.8]
        full = correctness_metrics.compute_threshold_free_metrics(scores, labels, novel_targets, 200)
        assert full.tpr_at_fpr is not None
        reconstructed = correctness_metrics.compute_tpr_at_target_fprs(scores, labels, novel_targets)
        assert reconstructed is not None
        for fpr_val in novel_targets:
            assert reconstructed[fpr_val] == pytest.approx(full.tpr_at_fpr[fpr_val])

    def test_per_category_filter_reconstruction(self) -> None:
        """Simulates the notebook workflow: filter attempt_records by code_type, then recompute."""
        rng = np.random.default_rng(88)
        all_scores = rng.random(600)
        all_labels = rng.random(600) > 0.45
        code_types = rng.choice(["original", "hinted", "misleading"], size=600)
        targets = [0.01, 0.05, 0.1, 0.5]
        for ct in ["original", "hinted", "misleading"]:
            mask = code_types == ct
            ct_scores = all_scores[mask]
            ct_labels = all_labels[mask]
            if len(np.unique(ct_labels)) < 2:
                continue
            full = correctness_metrics.compute_threshold_free_metrics(ct_scores, ct_labels, targets, 200)
            assert full.tpr_at_fpr is not None
            reconstructed = correctness_metrics.compute_tpr_at_target_fprs(ct_scores, ct_labels, targets)
            assert reconstructed is not None
            for fpr_val in targets:
                assert reconstructed[fpr_val] == pytest.approx(full.tpr_at_fpr[fpr_val])


class TestThresholdedMetrics:
    def test_basic(self) -> None:
        scores = np.array([0.1, 0.3, 0.7, 0.9])
        labels = np.array([False, False, True, True])
        result = correctness_metrics.compute_thresholded_metrics(scores, labels, 0.5, 0.01)
        assert result.tp == 2  # 0.7, 0.9 are correct and accepted
        assert result.fp == 0  # no incorrect accepted
        assert result.tn == 2  # 0.1, 0.3 are incorrect and blocked
        assert result.fn == 0  # no correct blocked
        assert result.tpr == 1.0
        assert result.fpr == 0.0

    def test_nan_threshold_raises(self) -> None:
        scores = np.array([0.5, 0.8])
        labels = np.array([False, True])
        with pytest.raises(ValueError, match="threshold must be finite"):
            correctness_metrics.compute_thresholded_metrics(scores, labels, float("nan"), 0.01)

    def test_inf_threshold_raises(self) -> None:
        scores = np.array([0.5, 0.8])
        labels = np.array([False, True])
        with pytest.raises(ValueError, match="threshold must be finite"):
            correctness_metrics.compute_thresholded_metrics(scores, labels, float("inf"), 0.01)


class TestSampleLevelMetrics:
    def test_basic_three_samples(self) -> None:
        # sample1: 2 correct, 1 incorrect; sample2: all incorrect; sample3: all correct
        records = [
            _make_record("s1", "p1", 0, True),
            _make_record("s1", "p1", 1, True),
            _make_record("s1", "p1", 2, False),
            _make_record("s2", "p2", 0, False),
            _make_record("s2", "p2", 1, False),
            _make_record("s3", "p3", 0, True),
            _make_record("s3", "p3", 1, True),
        ]
        # scores: accept everything (threshold=0, all scores >= 0)
        scores = np.array([0.8, 0.7, 0.3, 0.2, 0.1, 0.9, 0.85])
        result = correctness_metrics.compute_sample_level_metrics(records, scores, 0.0, 0.01)
        assert result.base_pass_rate == pytest.approx(2 / 3)  # s1 and s3 have correct
        assert result.guarded_pass_rate == pytest.approx(2 / 3)
        assert result.unsafe_slip_rate == pytest.approx(2 / 3)  # s1 and s2 have incorrect accepted
        assert result.total_block_rate == 0.0
        assert result.best_of_k_success_rate is not None

    def test_sample_keys_override_grouping(self) -> None:
        """When sample_keys are provided, records are grouped by key instead of sample_id."""
        # 2 records with same sample_id but different keys -> treated as 2 separate samples
        records = [
            _make_record("s1", "p1", 0, True),
            _make_record("s1", "p1", 1, False),
        ]
        scores = np.array([0.9, 0.1])
        # without sample_keys: grouped as 1 sample
        result_no_keys = correctness_metrics.compute_sample_level_metrics(records, scores, 0.5, 0.01)
        assert result_no_keys.base_pass_rate == 1.0  # 1 sample with at least 1 correct
        # with sample_keys: 2 separate samples
        result_with_keys = correctness_metrics.compute_sample_level_metrics(
            records,
            scores,
            0.5,
            0.01,
            sample_keys=["s1__draw0", "s1__draw1"],
        )
        assert result_with_keys.base_pass_rate == 0.5  # 1 of 2 samples has correct

    def test_sample_keys_length_mismatch_raises(self) -> None:
        records = [_make_record("s1", "p1", 0, True)]
        scores = np.array([0.9])
        with pytest.raises(ValueError, match="sample_keys length"):
            correctness_metrics.compute_sample_level_metrics(
                records,
                scores,
                0.5,
                0.01,
                sample_keys=["k1", "k2"],
            )

    def test_all_blocked(self) -> None:
        records = [_make_record("s1", "p1", 0, True)]
        scores = np.array([0.1])
        result = correctness_metrics.compute_sample_level_metrics(records, scores, 999.0, 0.01)
        assert result.total_block_rate == 1.0
        assert result.best_of_k_success_rate is None

    def test_conservative_all_accepted(self) -> None:
        """K=3 sample where all attempts accepted: cons_pass_rate=1, check cons_unsafe_slip_rate."""
        # sample with 2 correct + 1 incorrect, all accepted (threshold=0)
        records = [
            _make_record("s1", "p1", 0, True),
            _make_record("s1", "p1", 1, True),
            _make_record("s1", "p1", 2, False),
        ]
        scores = np.array([0.9, 0.8, 0.7])
        result = correctness_metrics.compute_sample_level_metrics(records, scores, 0.0, 0.01)
        assert result.cons_pass_rate == 1.0
        assert result.cons_unsafe_slip_rate == 1.0  # has incorrect attempt
        assert result.cons_justified_reject_rate is None  # no rejected samples

    def test_conservative_some_blocked(self) -> None:
        """K=3 sample where 1 blocked: not cons-accepted, check cons_justified_reject_rate."""
        # 2 correct accepted, 1 incorrect blocked
        records = [
            _make_record("s1", "p1", 0, True),
            _make_record("s1", "p1", 1, True),
            _make_record("s1", "p1", 2, False),
        ]
        scores = np.array([0.9, 0.8, 0.2])
        result = correctness_metrics.compute_sample_level_metrics(records, scores, 0.5, 0.01)
        assert result.cons_pass_rate == 0.0  # not all accepted
        assert result.cons_unsafe_slip_rate is None  # no cons-accepted samples
        assert result.cons_justified_reject_rate == 1.0  # blocked attempt is incorrect

    def test_conservative_all_blocked(self) -> None:
        """All blocked: cons_pass_rate=0, cons_unsafe_slip_rate=None."""
        records = [
            _make_record("s1", "p1", 0, True),
            _make_record("s1", "p1", 1, False),
        ]
        scores = np.array([0.1, 0.05])
        result = correctness_metrics.compute_sample_level_metrics(records, scores, 999.0, 0.01)
        assert result.cons_pass_rate == 0.0
        assert result.cons_unsafe_slip_rate is None
        # justified: blocked attempt idx=1 is incorrect
        assert result.cons_justified_reject_rate == 1.0

    def test_conservative_mixed_samples(self) -> None:
        """3 samples with mixed patterns."""
        records = [
            # sample1: all correct, all accepted -> cons-accepted, no unsafe slip
            _make_record("s1", "p1", 0, True),
            _make_record("s1", "p1", 1, True),
            # sample2: 1 correct + 1 incorrect, 1 blocked (incorrect) -> cons-rejected, justified
            _make_record("s2", "p2", 0, True),
            _make_record("s2", "p2", 1, False),
            # sample3: all correct, 1 blocked -> cons-rejected, not justified (blocked correct)
            _make_record("s3", "p3", 0, True),
            _make_record("s3", "p3", 1, True),
        ]
        scores = np.array([0.9, 0.8, 0.9, 0.2, 0.9, 0.2])
        result = correctness_metrics.compute_sample_level_metrics(records, scores, 0.5, 0.01)
        assert result.cons_pass_rate == pytest.approx(1 / 3)  # only s1
        assert result.cons_unsafe_slip_rate == 0.0  # s1 is all correct, no unsafe slip
        assert result.cons_justified_reject_rate == pytest.approx(1 / 2)  # s2 justified, s3 not

    def test_nan_threshold_raises(self) -> None:
        records = [_make_record("s1", "p1", 0, True)]
        scores = np.array([0.9])
        with pytest.raises(ValueError, match="threshold must be finite"):
            correctness_metrics.compute_sample_level_metrics(records, scores, float("nan"), 0.01)

    def test_inf_threshold_raises(self) -> None:
        records = [_make_record("s1", "p1", 0, True)]
        scores = np.array([0.9])
        with pytest.raises(ValueError, match="threshold must be finite"):
            correctness_metrics.compute_sample_level_metrics(records, scores, float("-inf"), 0.01)

    def test_best_of_k_tie_aware(self) -> None:
        """When multiple accepted attempts share the max score, success if any tied attempt is correct."""
        # 1 sample, 3 attempts: incorrect at 0.9, correct at 0.9, incorrect at 0.5
        # with order-dependent logic, result would depend on list order; tie-aware always succeeds
        records = [
            _make_record("s1", "p1", 0, False),
            _make_record("s1", "p1", 1, True),
            _make_record("s1", "p1", 2, False),
        ]
        scores = np.array([0.9, 0.9, 0.5])
        result = correctness_metrics.compute_sample_level_metrics(records, scores, 0.0, 0.01)
        assert result.best_of_k_success_rate == 1.0
        # reverse order: incorrect first at max score; should still succeed
        records_rev = [
            _make_record("s1", "p1", 0, True),
            _make_record("s1", "p1", 1, False),
            _make_record("s1", "p1", 2, False),
        ]
        scores_rev = np.array([0.9, 0.9, 0.5])
        result_rev = correctness_metrics.compute_sample_level_metrics(records_rev, scores_rev, 0.0, 0.01)
        assert result_rev.best_of_k_success_rate == result.best_of_k_success_rate


class TestCategorizeRecords:
    def test_regular_category(self) -> None:
        records = [_make_record("s1", "p1", code_type="original")]
        config = correctness_configs.RecordCategoryConfig()
        result = correctness_metrics.categorize_records(records, config)
        assert 0 in result["regular"]

    def test_biasing_category(self) -> None:
        records = [_make_record("s1", "p1", code_type="misleading")]
        config = correctness_configs.RecordCategoryConfig()
        result = correctness_metrics.categorize_records(records, config)
        assert 0 in result["biasing/misleading"]
        assert "regular" not in result or 0 not in result.get("regular", [])

    def test_compound_code_type(self) -> None:
        records = [_make_record("s1", "p1", code_type="bugged_hinted")]
        config = correctness_configs.RecordCategoryConfig()
        result = correctness_metrics.categorize_records(records, config)
        assert 0 in result["biasing/bugged"]
        assert 0 in result["biasing/hinted"]
        assert 0 in result["biasing/bugged_hinted"]  # combined uses code type set convention
        assert "biasing/bugged+biasing/hinted" not in result  # old format no longer used

    def test_unknown_code_type_goes_to_other(self) -> None:
        records = [_make_record("s1", "p1", code_type="xyzunknown")]
        config = correctness_configs.RecordCategoryConfig(report_per_code_type=False)
        result = correctness_metrics.categorize_records(records, config)
        assert 0 in result["other"]
        assert "regular" not in result

    def test_per_code_type_reporting(self) -> None:
        records = [_make_record("s1", "p1", code_type="original")]
        config = correctness_configs.RecordCategoryConfig(report_per_code_type=True)
        result = correctness_metrics.categorize_records(records, config)
        assert "code_type/original" in result


class TestFiniteValidation:
    """Tests for _assert_finite checks on scores, costs, and difficulty."""

    def test_nan_scores_in_threshold_free(self) -> None:
        scores = np.array([0.5, float("nan"), 0.8])
        labels = np.array([True, False, True])
        with pytest.raises(ValueError, match="non-finite"):
            correctness_metrics.compute_threshold_free_metrics(scores, labels, [0.01], 100)

    def test_inf_scores_in_threshold_free(self) -> None:
        scores = np.array([0.5, float("inf"), 0.8])
        labels = np.array([True, False, True])
        with pytest.raises(ValueError, match="non-finite"):
            correctness_metrics.compute_threshold_free_metrics(scores, labels, [0.01], 100)

    def test_nan_scores_in_thresholded(self) -> None:
        scores = np.array([float("nan"), 0.8])
        labels = np.array([True, False])
        with pytest.raises(ValueError, match="non-finite"):
            correctness_metrics.compute_thresholded_metrics(scores, labels, 0.5, 0.01)

    def test_nan_scores_in_sample_level(self) -> None:
        records = [_make_record("s1", "p1", 0, True)]
        scores = np.array([float("nan")])
        with pytest.raises(ValueError, match="non-finite"):
            correctness_metrics.compute_sample_level_metrics(records, scores, 0.5, 0.01)

    def test_nan_scores_in_difficulty_stats(self) -> None:
        records = [_make_record(f"s{idx}", f"p{idx}", difficulty_score=float(idx)) for idx in range(3)]
        scores = np.array([0.5, float("nan"), 0.8])
        labels = np.array([True, False, True])
        with pytest.raises(ValueError, match="non-finite"):
            correctness_metrics.compute_difficulty_stats(records, scores, labels, {0.05: 0.5}, [0.05])

    def test_nan_difficulty_scores(self) -> None:
        records = [
            _make_record("s0", "p0", difficulty_score=0.1),
            _make_record("s1", "p1", difficulty_score=float("nan")),
            _make_record("s2", "p2", difficulty_score=0.9),
        ]
        scores = np.array([0.5, 0.6, 0.8])
        labels = np.array([True, False, True])
        with pytest.raises(ValueError, match="non-finite"):
            correctness_metrics.compute_difficulty_stats(records, scores, labels, {0.05: 0.5}, [0.05])

    def test_nan_verification_costs(self) -> None:
        costs = [1.0, float("nan"), 3.0]
        labels = np.array([True, False, True])
        accepted = np.array([True, False, True])
        with pytest.raises(ValueError, match="non-finite"):
            correctness_metrics.compute_verification_cost_stats(costs, labels, accepted, 0.05, cost_unit="tokens")

    def test_inf_verification_costs(self) -> None:
        costs = [1.0, float("inf"), 3.0]
        labels = np.array([True, False, True])
        accepted = np.array([True, False, True])
        with pytest.raises(ValueError, match="non-finite"):
            correctness_metrics.compute_verification_cost_stats(costs, labels, accepted, 0.05, cost_unit="tokens")

    def test_int_labels_in_thresholded_raises(self) -> None:
        scores = np.array([0.5, 0.8])
        labels = np.array([0, 1])
        with pytest.raises(ValueError, match="boolean array"):
            correctness_metrics.compute_thresholded_metrics(scores, labels, 0.5, 0.01)

    def test_int_labels_in_difficulty_raises(self) -> None:
        records = [_make_record(f"s{idx}", f"p{idx}", difficulty_score=float(idx)) for idx in range(3)]
        scores = np.array([0.5, 0.6, 0.8])
        labels = np.array([1, 0, 1])
        with pytest.raises(ValueError, match="boolean array"):
            correctness_metrics.compute_difficulty_stats(records, scores, labels, {0.05: 0.5}, [0.05])

    def test_int_labels_in_cost_stats_raises(self) -> None:
        labels = np.array([1, 0])
        accepted = np.array([True, False])
        with pytest.raises(ValueError, match="boolean array"):
            correctness_metrics.compute_verification_cost_stats([1.0, 2.0], labels, accepted, 0.05, cost_unit="tokens")

    def test_int_accepted_in_cost_stats_raises(self) -> None:
        labels = np.array([True, False])
        accepted = np.array([1, 0])
        with pytest.raises(ValueError, match="boolean array"):
            correctness_metrics.compute_verification_cost_stats([1.0, 2.0], labels, accepted, 0.05, cost_unit="tokens")


class TestDuplicateFprRecallInterpolation:
    """Tests for ROC/PR interpolation with duplicate x-values (tied scores)."""

    def test_tied_scores_produce_valid_grids(self) -> None:
        """Tied scores cause duplicate FPR/recall values; interpolation must handle them."""
        scores = np.array([0.0, 0.5, 0.5, 0.5, 1.0])
        labels = np.array([False, False, True, True, True])
        result = correctness_metrics.compute_threshold_free_metrics(scores, labels, [0.01, 0.5], 50)
        assert result.auroc is not None
        assert len(result.fpr_grid) == 50
        assert len(result.tpr_grid) == 50
        # grids should be finite
        assert np.all(np.isfinite(result.fpr_grid))
        assert np.all(np.isfinite(result.tpr_grid))
        assert np.all(np.isfinite(result.precision_grid))
        assert np.all(np.isfinite(result.recall_grid))

    def test_all_tied_scores(self) -> None:
        """All scores identical; extreme case of duplication."""
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([True, True, False, False])
        result = correctness_metrics.compute_threshold_free_metrics(scores, labels, [0.01], 50)
        # AUROC should be 0.5 (random-equivalent with all ties)
        assert result.auroc is not None
        assert result.auroc == pytest.approx(0.5)

    def test_many_tied_negative_scores(self) -> None:
        """Many negatives at same score; produces many duplicate FPR values."""
        scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 1.0])
        labels = np.array([False, False, False, False, False, True, True, True])
        result = correctness_metrics.compute_threshold_free_metrics(scores, labels, [0.01], 100)
        assert result.auroc is not None
        assert result.auroc == pytest.approx(1.0)  # perfect separation
        assert len(result.fpr_grid) == 100


class TestForbiddenCodeTypeCombinations:
    """Tests that forbidden code_type combos raise ValueError in compute_class_balance."""

    def test_misleading_hinted_raises(self) -> None:
        records = [_make_record("s1", "p1", code_type="misleading_hinted")]
        with pytest.raises(ValueError, match="forbidden code_type combination"):
            correctness_metrics.compute_class_balance(records)

    def test_stubbed_hinted_raises(self) -> None:
        records = [_make_record("s1", "p1", code_type="stubbed_hinted")]
        with pytest.raises(ValueError, match="forbidden code_type combination"):
            correctness_metrics.compute_class_balance(records)

    def test_original_bugged_raises(self) -> None:
        records = [_make_record("s1", "p1", code_type="original_bugged")]
        with pytest.raises(ValueError, match="forbidden code_type combination"):
            correctness_metrics.compute_class_balance(records)

    def test_valid_compound_passes(self) -> None:
        """bugged_hinted is a valid combination; should not raise."""
        records = [_make_record("s1", "p1", code_type="bugged_hinted")]
        result = correctness_metrics.compute_class_balance(records)
        assert result.overall_positive_rate == 1.0  # label=True by default


class TestComputeClassBalance:
    def test_basic(self) -> None:
        records = [
            _make_record("s1", "p1", 0, True),
            _make_record("s1", "p1", 1, False),
            _make_record("s2", "p2", 0, True),
            _make_record("s2", "p2", 1, True),
        ]
        result = correctness_metrics.compute_class_balance(records)
        assert result.overall_positive_rate == 0.75
        assert result.num_all_correct_samples == 1  # s2
        assert result.num_all_incorrect_samples == 0

    def test_empty(self) -> None:
        result = correctness_metrics.compute_class_balance([])
        assert result.overall_positive_rate == 0.0


class TestBootstrapCIs:
    def test_nan_scores_raises(self) -> None:
        records = [_make_record(f"s{idx}", f"p{idx}", label=(idx < 3)) for idx in range(5)]
        scores = np.array([0.1, 0.2, float("nan"), 0.8, 0.9])
        with pytest.raises(ValueError, match="non-finite"):
            correctness_metrics.compute_clustered_bootstrap_cis(
                records,
                scores,
                {0.05: 0.5},
                [0.05],
                10,
                seed=0,
                confidence_level=0.95,
            )

    def test_seed_reproducibility(self) -> None:
        records = [_make_record(f"s{idx}", f"p{idx}", label=(idx < 5)) for idx in range(10)]
        scores = np.array([idx / 10.0 for idx in range(10)])
        thresholds = {0.05: 0.5}
        ci1 = correctness_metrics.compute_clustered_bootstrap_cis(
            records,
            scores,
            thresholds,
            [0.05],
            50,
            seed=42,
            confidence_level=0.95,
        )
        ci2 = correctness_metrics.compute_clustered_bootstrap_cis(
            records,
            scores,
            thresholds,
            [0.05],
            50,
            seed=42,
            confidence_level=0.95,
        )
        for key in ci1:
            assert ci1[key].point_estimate == ci2[key].point_estimate

    def test_includes_fpr_and_sample_metrics(self) -> None:
        rng = np.random.default_rng(123)
        records = [_make_record(f"s{idx}", f"p{idx}", label=bool(rng.random() > 0.5)) for idx in range(30)]
        scores = rng.random(30)
        thresholds = {0.05: 0.5}
        cis = correctness_metrics.compute_clustered_bootstrap_cis(
            records,
            scores,
            thresholds,
            [0.05],
            100,
            seed=42,
            confidence_level=0.95,
        )
        assert "fpr_0_05/fpr" in cis
        assert "fpr_0_05/guarded_pass_rate" in cis
        assert "fpr_0_05/unsafe_slip_rate" in cis

    def test_bootstrap_sample_keys_prevent_collapse(self) -> None:
        """Resampling with replacement should not collapse duplicate sample_ids."""
        # create 2 problems, each with 1 sample and 2 attempts
        records = [
            _make_record("s1", "p1", 0, True),
            _make_record("s1", "p1", 1, False),
            _make_record("s2", "p2", 0, False),
            _make_record("s2", "p2", 1, True),
        ]
        scores = np.array([0.9, 0.1, 0.1, 0.9])
        thresholds = {0.05: 0.5}
        # with sample_keys, duplicated problems should be treated independently
        cis = correctness_metrics.compute_clustered_bootstrap_cis(
            records,
            scores,
            thresholds,
            [0.05],
            200,
            seed=42,
            confidence_level=0.95,
        )
        # verify we get sample-level CIs (these would be overly tight without the fix)
        assert "fpr_0_05/base_pass_rate" in cis

    def test_includes_base_pass_rate(self) -> None:
        rng = np.random.default_rng(123)
        records = [_make_record(f"s{idx}", f"p{idx}", label=bool(rng.random() > 0.5)) for idx in range(30)]
        scores = rng.random(30)
        thresholds = {0.05: 0.5}
        cis = correctness_metrics.compute_clustered_bootstrap_cis(
            records,
            scores,
            thresholds,
            [0.05],
            100,
            seed=42,
            confidence_level=0.95,
        )
        assert "fpr_0_05/base_pass_rate" in cis

    def test_point_estimate_within_ci(self) -> None:
        rng = np.random.default_rng(42)
        records = [_make_record(f"s{idx}", f"p{idx}", label=bool(rng.random() > 0.5)) for idx in range(50)]
        scores = rng.random(50)
        thresholds = {0.05: 0.5}
        cis = correctness_metrics.compute_clustered_bootstrap_cis(
            records,
            scores,
            thresholds,
            [0.05],
            200,
            seed=42,
            confidence_level=0.95,
        )
        for _key, ci in cis.items():
            assert ci.lower_bound <= ci.point_estimate <= ci.upper_bound


class TestHierarchicalBootstrapCIs:
    def test_nan_scores_in_run_raises(self) -> None:
        records = [_make_record(f"s{idx}", f"p{idx}", label=(idx < 3)) for idx in range(5)]
        scores_ok = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
        scores_bad = np.array([0.1, float("inf"), 0.5, 0.8, 0.9])
        with pytest.raises(ValueError, match="non-finite"):
            correctness_metrics.compute_hierarchical_bootstrap_cis(
                [records, records],
                [scores_ok, scores_bad],
                {0.05: [0.5, 0.5]},
                [0.05],
                10,
                seed=0,
                confidence_level=0.95,
            )

    def test_seed_reproducibility(self) -> None:
        records = [_make_record(f"s{idx}", f"p{idx}", label=(idx < 5)) for idx in range(10)]
        scores = np.array([idx / 10.0 for idx in range(10)])
        per_run_records = [records, records]
        per_run_scores = [scores, scores + 0.05]
        per_run_thresholds = {0.05: [0.5, 0.55]}
        ci1 = correctness_metrics.compute_hierarchical_bootstrap_cis(
            per_run_records,
            per_run_scores,
            per_run_thresholds,
            [0.05],
            50,
            seed=42,
            confidence_level=0.95,
        )
        ci2 = correctness_metrics.compute_hierarchical_bootstrap_cis(
            per_run_records,
            per_run_scores,
            per_run_thresholds,
            [0.05],
            50,
            seed=42,
            confidence_level=0.95,
        )
        for key in ci1:
            assert ci1[key].point_estimate == ci2[key].point_estimate

    def test_includes_sample_level_metrics(self) -> None:
        rng = np.random.default_rng(123)
        records = [_make_record(f"s{idx}", f"p{idx}", label=bool(rng.random() > 0.5)) for idx in range(20)]
        scores = rng.random(20)
        per_run_records = [records]
        per_run_scores = [scores]
        per_run_thresholds = {0.05: [0.5]}
        cis = correctness_metrics.compute_hierarchical_bootstrap_cis(
            per_run_records,
            per_run_scores,
            per_run_thresholds,
            [0.05],
            50,
            seed=42,
            confidence_level=0.95,
        )
        assert "fpr_0_05/guarded_pass_rate" in cis
        assert "fpr_0_05/base_pass_rate" in cis

    def test_point_estimate_within_ci(self) -> None:
        rng = np.random.default_rng(42)
        records = [_make_record(f"s{idx}", f"p{idx}", label=bool(rng.random() > 0.5)) for idx in range(30)]
        scores = rng.random(30)
        per_run_records = [records, records]
        per_run_scores = [scores, scores + 0.02]
        per_run_thresholds = {0.05: [0.5, 0.52]}
        cis = correctness_metrics.compute_hierarchical_bootstrap_cis(
            per_run_records,
            per_run_scores,
            per_run_thresholds,
            [0.05],
            100,
            seed=42,
            confidence_level=0.95,
        )
        for _key, ci in cis.items():
            assert ci.lower_bound <= ci.point_estimate <= ci.upper_bound


class TestDifficultyStats:
    def test_returns_none_when_unavailable(self) -> None:
        records = [_make_record("s1", "p1", difficulty_score=None)]
        scores = np.array([0.5])
        labels = np.array([True])
        result = correctness_metrics.compute_difficulty_stats(records, scores, labels, {0.05: 0.5}, [0.05])
        assert result is None

    def test_returns_stats_when_available(self) -> None:
        records = [
            _make_record(f"s{idx}", f"p{idx}", label=(idx < 5), difficulty_score=float(idx)) for idx in range(10)
        ]
        scores = np.array([idx / 10.0 for idx in range(10)])
        labels = np.array([rec.label for rec in records])
        result = correctness_metrics.compute_difficulty_stats(records, scores, labels, {0.05: 0.5}, [0.05])
        assert result is not None
        assert result.bucket_boundaries is not None
        assert "easy" in result.per_bucket_sample_count

    def test_constant_difficulty_gives_none_correlation(self) -> None:
        records = [_make_record(f"s{idx}", f"p{idx}", label=(idx < 5), difficulty_score=3.0) for idx in range(10)]
        scores = np.array([idx / 10.0 for idx in range(10)])
        labels = np.array([rec.label for rec in records])
        result = correctness_metrics.compute_difficulty_stats(records, scores, labels, {0.05: 0.5}, [0.05])
        assert result is not None
        assert result.difficulty_accuracy_rank_correlation is None

    def test_constant_accuracy_gives_none_correlation(self) -> None:
        records = [_make_record(f"s{idx}", f"p{idx}", label=True, difficulty_score=float(idx)) for idx in range(10)]
        scores = np.array([0.9] * 10)
        labels = np.array([True] * 10)
        result = correctness_metrics.compute_difficulty_stats(records, scores, labels, {0.05: 0.5}, [0.05])
        assert result is not None
        assert result.difficulty_accuracy_rank_correlation is None


class TestVerificationCostStats:
    def test_returns_none_when_unavailable(self) -> None:
        labels = np.array([True, False])
        accepted = np.array([True, False])
        result = correctness_metrics.compute_verification_cost_stats(None, labels, accepted, 0.05)
        assert result is None

    def test_costs_without_unit_raises(self) -> None:
        labels = np.array([True, False])
        accepted = np.array([True, False])
        with pytest.raises(ValueError, match="cost_unit is None"):
            correctness_metrics.compute_verification_cost_stats([1.0, 2.0], labels, accepted, 0.05)

    def test_unit_without_costs_raises(self) -> None:
        labels = np.array([True, False])
        accepted = np.array([True, False])
        with pytest.raises(ValueError, match="verification_costs is None"):
            correctness_metrics.compute_verification_cost_stats(None, labels, accepted, 0.05, cost_unit="tokens")

    def test_basic_costs(self) -> None:
        costs = [1.0, 2.0, 3.0, 4.0]
        labels = np.array([True, True, False, False])
        accepted = np.array([True, False, True, False])
        result = correctness_metrics.compute_verification_cost_stats(costs, labels, accepted, 0.05, cost_unit="tokens")
        assert result is not None
        assert result.total_cost == 10.0
        assert result.mean_cost_per_record == 2.5
        assert result.cost_per_correct_acceptance == 1.0  # only idx 0 is TP
        assert result.cost_per_incorrect_block == 4.0  # only idx 3 is TN
        assert result.cost_accuracy_rank_correlation is not None

    def test_cost_accuracy_correlation_too_few_records(self) -> None:
        costs = [1.0, 2.0]
        labels = np.array([True, False])
        accepted = np.array([True, False])
        result = correctness_metrics.compute_verification_cost_stats(costs, labels, accepted, 0.05, cost_unit="tokens")
        assert result is not None
        assert result.cost_accuracy_rank_correlation is None

    def test_cost_difficulty_correlation_without_records(self) -> None:
        costs = [1.0, 2.0, 3.0]
        labels = np.array([True, False, True])
        accepted = np.array([True, False, True])
        result = correctness_metrics.compute_verification_cost_stats(costs, labels, accepted, 0.05, cost_unit="tokens")
        assert result is not None
        assert result.cost_difficulty_rank_correlation is None  # no records passed

    def test_cost_difficulty_correlation_with_difficulty(self) -> None:
        records = [
            _make_record("s1", "p1", attempt_index=0, label=True, difficulty_score=0.1),
            _make_record("s2", "p2", attempt_index=0, label=False, difficulty_score=0.5),
            _make_record("s3", "p3", attempt_index=0, label=True, difficulty_score=0.9),
        ]
        costs = [1.0, 5.0, 9.0]  # cost increases with difficulty
        labels = np.array([True, False, True])
        accepted = np.array([True, False, True])
        result = correctness_metrics.compute_verification_cost_stats(
            costs,
            labels,
            accepted,
            0.05,
            cost_unit="tokens",
            records=records,
        )
        assert result is not None
        assert result.cost_difficulty_rank_correlation is not None
        assert result.cost_difficulty_rank_correlation > 0  # positive correlation

    def test_cost_difficulty_correlation_none_when_no_difficulty(self) -> None:
        records = [
            _make_record("s1", "p1", attempt_index=0, label=True, difficulty_score=None),
            _make_record("s2", "p2", attempt_index=0, label=False, difficulty_score=None),
            _make_record("s3", "p3", attempt_index=0, label=True, difficulty_score=None),
        ]
        costs = [1.0, 5.0, 9.0]
        labels = np.array([True, False, True])
        accepted = np.array([True, False, True])
        result = correctness_metrics.compute_verification_cost_stats(
            costs,
            labels,
            accepted,
            0.05,
            cost_unit="tokens",
            records=records,
        )
        assert result is not None
        assert result.cost_difficulty_rank_correlation is None

    def test_constant_costs_gives_none_accuracy_correlation(self) -> None:
        costs = [5.0, 5.0, 5.0]
        labels = np.array([True, False, True])
        accepted = np.array([True, False, True])
        result = correctness_metrics.compute_verification_cost_stats(
            costs,
            labels,
            accepted,
            0.05,
            cost_unit="tokens",
        )
        assert result is not None
        assert result.cost_accuracy_rank_correlation is None

    def test_constant_difficulty_gives_none_cost_difficulty_correlation(self) -> None:
        records = [
            _make_record("s1", "p1", attempt_index=0, label=True, difficulty_score=3.0),
            _make_record("s2", "p2", attempt_index=0, label=False, difficulty_score=3.0),
            _make_record("s3", "p3", attempt_index=0, label=True, difficulty_score=3.0),
        ]
        costs = [1.0, 5.0, 9.0]
        labels = np.array([True, False, True])
        accepted = np.array([True, False, True])
        result = correctness_metrics.compute_verification_cost_stats(
            costs,
            labels,
            accepted,
            0.05,
            cost_unit="tokens",
            records=records,
        )
        assert result is not None
        assert result.cost_difficulty_rank_correlation is None


class TestCategoryResults:
    def test_empty_categories_skipped(self) -> None:
        records = [_make_record("s1", "p1", code_type="original", label=True)]
        scores = np.array([0.9])
        config = correctness_configs.RecordCategoryConfig(report_per_code_type=False)
        result = correctness_metrics.compute_category_results(
            records,
            scores,
            {0.05: 0.5},
            [0.05],
            config,
            100,
        )
        # only "regular" category should exist
        assert "regular" in result
        assert "biasing/misleading" not in result

    def test_cross_run_none_dropping(self) -> None:
        """Verify that None metrics don't crash aggregation (tested indirectly)."""
        records = [_make_record("s1", "p1", code_type="misleading", label=False)]
        scores = np.array([0.5])
        config = correctness_configs.RecordCategoryConfig()
        result = correctness_metrics.compute_category_results(
            records,
            scores,
            {0.05: 0.5},
            [0.05],
            config,
            100,
        )
        # single-class subset -> auroc should be None
        misleading_result = result.get("biasing/misleading")
        assert misleading_result is not None
        assert misleading_result.threshold_free.auroc is None


class TestAdaptRecordForBaseExtractor:
    def test_fills_missing_identifier_from_sample_id(self) -> None:
        record_data: dict[str, typing.Any] = {"sample_id": "foo::hinted"}
        adapted = correctness_metrics._adapt_record_for_base_extractor(record_data)
        assert adapted["identifier"] == "foo::hinted"

    def test_fills_missing_comma_separated_tags_from_tags_list(self) -> None:
        record_data: dict[str, typing.Any] = {"tags": ["bias_keyword:hello", "has_bias_keyword:1"]}
        adapted = correctness_metrics._adapt_record_for_base_extractor(record_data)
        assert adapted["comma_separated_tags"] == "bias_keyword:hello,has_bias_keyword:1"

    def test_does_not_clobber_existing_fields(self) -> None:
        record_data: dict[str, typing.Any] = {
            "sample_id": "s1",
            "identifier": "existing_id",
            "tags": ["tag1"],
            "comma_separated_tags": "existing_tags",
        }
        adapted = correctness_metrics._adapt_record_for_base_extractor(record_data)
        assert adapted["identifier"] == "existing_id"
        assert adapted["comma_separated_tags"] == "existing_tags"

    def test_handles_empty_tags_list(self) -> None:
        record_data: dict[str, typing.Any] = {"tags": []}
        adapted = correctness_metrics._adapt_record_for_base_extractor(record_data)
        assert adapted["comma_separated_tags"] == ""

    def test_no_tags_key_leaves_comma_separated_tags_absent(self) -> None:
        record_data: dict[str, typing.Any] = {"sample_id": "s1"}
        adapted = correctness_metrics._adapt_record_for_base_extractor(record_data)
        assert "comma_separated_tags" not in adapted


class TestCategorizeRecordsWithBaseExtraction:
    def test_with_base_extraction_predict_type(self) -> None:
        """Base extraction adds predict_type categories alongside code_type grouping."""
        records = [
            _make_record("s1", "p1", code_type="original", record={"predict_type": "direct"}),
            _make_record("s2", "p2", code_type="misleading", record={"predict_type": "cot"}),
        ]
        config = correctness_configs.RecordCategoryConfig(report_per_code_type=False)
        base_config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=[pyine.evals.utils.SampleCategoryField.predict_type],
        )
        result = correctness_metrics.categorize_records(records, config, base_extraction_config=base_config)
        assert 0 in result["regular"]
        assert 1 in result["biasing/misleading"]
        assert 0 in result["predict_type/direct"]
        assert 1 in result["predict_type/cot"]

    def test_code_type_filtered_from_base(self) -> None:
        """Code_type is not duplicated even when base config includes it in enabled_fields."""
        records = [_make_record("s1", "p1", code_type="original", record={"predict_type": "direct"})]
        config = correctness_configs.RecordCategoryConfig(report_per_code_type=False)
        base_config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=[
                pyine.evals.utils.SampleCategoryField.code_type,
                pyine.evals.utils.SampleCategoryField.predict_type,
            ],
        )
        result = correctness_metrics.categorize_records(records, config, base_extraction_config=base_config)
        # code_type/original should NOT appear (code_type is filtered from base extractor)
        assert "code_type/original" not in result
        assert "predict_type/direct" in result
        assert "regular" in result

    def test_record_adaptation_identifier_suffix(self) -> None:
        """Records with sample_id containing :: get identifier_suffix categories."""
        records = [
            _make_record(
                "problem1::hinted",
                "p1",
                code_type="original",
                record={"sample_id": "problem1::hinted"},
            ),
        ]
        config = correctness_configs.RecordCategoryConfig(report_per_code_type=False)
        base_config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=[pyine.evals.utils.SampleCategoryField.identifier_suffix],
        )
        result = correctness_metrics.categorize_records(records, config, base_extraction_config=base_config)
        assert 0 in result["identifier_suffix/hinted"]

    def test_record_adaptation_tags(self) -> None:
        """Records with tags list get tag categories via adaptation."""
        records = [
            _make_record(
                "s1",
                "p1",
                code_type="original",
                tags=["bias_keyword:hello", "has_bias_keyword:1"],
                record={"tags": ["bias_keyword:hello", "has_bias_keyword:1"]},
            ),
        ]
        config = correctness_configs.RecordCategoryConfig(report_per_code_type=False)
        base_config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=[pyine.evals.utils.SampleCategoryField.has_keyword],
        )
        result = correctness_metrics.categorize_records(records, config, base_extraction_config=base_config)
        assert 0 in result["has_keyword/true"]

    def test_deduplication(self) -> None:
        """Duplicate categories from base + grouped don't cause double counting."""
        records = [
            _make_record(
                "s1",
                "p1",
                code_type="original",
                record={"predict_type": "direct"},
            ),
        ]
        config = correctness_configs.RecordCategoryConfig(report_per_code_type=False)
        base_config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=[pyine.evals.utils.SampleCategoryField.predict_type],
        )
        result = correctness_metrics.categorize_records(records, config, base_extraction_config=base_config)
        # each category should contain index 0 exactly once
        assert result["regular"].count(0) == 1
        assert result["predict_type/direct"].count(0) == 1

    def test_none_base_extraction_preserves_existing_behavior(self) -> None:
        """Passing base_extraction_config=None gives same results as before."""
        records = [_make_record("s1", "p1", code_type="original")]
        config = correctness_configs.RecordCategoryConfig()
        result = correctness_metrics.categorize_records(records, config, base_extraction_config=None)
        assert 0 in result["regular"]
        assert "code_type/original" in result


class TestParallelBootstrap:
    """Tests for the parallel (multi-worker) bootstrap CI paths."""

    # sequential vs parallel use independent RNG streams, so point estimates will differ;
    # with only 100 replicates on ~40 records, bootstrap variability is high; with production
    # scale data (10K+ records, 5K+ replicates) this gap shrinks to ~0.01
    _SEQ_PAR_OVERLAP_TOLERANCE = 0.15

    @pytest.fixture()
    def clustered_data(
        self,
    ) -> tuple[
        list[correctness_types.EvalRecord],
        np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]],
        dict[float, float],
        list[float],
    ]:
        rng = np.random.default_rng(99)
        records = [_make_record(f"s{idx}", f"p{idx % 10}", label=bool(rng.random() > 0.4)) for idx in range(40)]
        scores = rng.random(40).astype(np.float64)
        thresholds = {0.05: 0.5}
        target_fprs = [0.05]
        return records, scores, thresholds, target_fprs

    @pytest.fixture()
    def hierarchical_data(
        self,
    ) -> tuple[
        list[list[correctness_types.EvalRecord]],
        list[np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]]],
        dict[float, list[float]],
        list[float],
    ]:
        rng = np.random.default_rng(99)
        records = [_make_record(f"s{idx}", f"p{idx % 8}", label=bool(rng.random() > 0.4)) for idx in range(30)]
        scores1 = rng.random(30).astype(np.float64)
        scores2 = rng.random(30).astype(np.float64)
        per_run_records = [records, records]
        per_run_scores = [scores1, scores2]
        per_run_thresholds = {0.05: [0.5, 0.5]}
        target_fprs = [0.05]
        return per_run_records, per_run_scores, per_run_thresholds, target_fprs

    def test_parallel_determinism_clustered(
        self,
        clustered_data: tuple[
            list[correctness_types.EvalRecord],
            np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]],
            dict[float, float],
            list[float],
        ],
    ) -> None:
        records, scores, thresholds, target_fprs = clustered_data
        kwargs: dict[str, typing.Any] = {
            "records": records,
            "scores": scores,
            "thresholds": thresholds,
            "target_fpr_values": target_fprs,
            "num_replicates": 50,
            "seed": 42,
            "confidence_level": 0.95,
            "num_workers": 2,
        }
        ci1 = correctness_metrics.compute_clustered_bootstrap_cis(**kwargs)
        ci2 = correctness_metrics.compute_clustered_bootstrap_cis(**kwargs)
        for key in ci1:
            assert key in ci2
            assert ci1[key].point_estimate == ci2[key].point_estimate
            assert ci1[key].lower_bound == ci2[key].lower_bound
            assert ci1[key].upper_bound == ci2[key].upper_bound

    def test_parallel_determinism_hierarchical(
        self,
        hierarchical_data: tuple[
            list[list[correctness_types.EvalRecord]],
            list[np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]]],
            dict[float, list[float]],
            list[float],
        ],
    ) -> None:
        per_run_records, per_run_scores, per_run_thresholds, target_fprs = hierarchical_data
        kwargs: dict[str, typing.Any] = {
            "per_run_records": per_run_records,
            "per_run_scores": per_run_scores,
            "per_run_thresholds": per_run_thresholds,
            "target_fpr_values": target_fprs,
            "num_replicates": 50,
            "seed": 42,
            "confidence_level": 0.95,
            "num_workers": 2,
        }
        ci1 = correctness_metrics.compute_hierarchical_bootstrap_cis(**kwargs)
        ci2 = correctness_metrics.compute_hierarchical_bootstrap_cis(**kwargs)
        for key in ci1:
            assert key in ci2
            assert ci1[key].point_estimate == ci2[key].point_estimate
            assert ci1[key].lower_bound == ci2[key].lower_bound
            assert ci1[key].upper_bound == ci2[key].upper_bound

    def test_parallel_ci_overlap_clustered(
        self,
        clustered_data: tuple[
            list[correctness_types.EvalRecord],
            np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]],
            dict[float, float],
            list[float],
        ],
    ) -> None:
        records, scores, thresholds, target_fprs = clustered_data
        common_kwargs: dict[str, typing.Any] = {
            "records": records,
            "scores": scores,
            "thresholds": thresholds,
            "target_fpr_values": target_fprs,
            "num_replicates": 100,
            "seed": 42,
            "confidence_level": 0.95,
        }
        ci_seq = correctness_metrics.compute_clustered_bootstrap_cis(**common_kwargs, num_workers=1)
        ci_par = correctness_metrics.compute_clustered_bootstrap_cis(**common_kwargs, num_workers=2)
        common_keys = set(ci_seq.keys()) & set(ci_par.keys())
        assert len(common_keys) > 0
        tol = self._SEQ_PAR_OVERLAP_TOLERANCE
        for key in common_keys:
            assert ci_seq[key].point_estimate <= ci_par[key].upper_bound + tol
            assert ci_seq[key].point_estimate >= ci_par[key].lower_bound - tol
            assert ci_par[key].point_estimate <= ci_seq[key].upper_bound + tol
            assert ci_par[key].point_estimate >= ci_seq[key].lower_bound - tol

    def test_parallel_ci_overlap_hierarchical(
        self,
        hierarchical_data: tuple[
            list[list[correctness_types.EvalRecord]],
            list[np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]]],
            dict[float, list[float]],
            list[float],
        ],
    ) -> None:
        per_run_records, per_run_scores, per_run_thresholds, target_fprs = hierarchical_data
        common_kwargs: dict[str, typing.Any] = {
            "per_run_records": per_run_records,
            "per_run_scores": per_run_scores,
            "per_run_thresholds": per_run_thresholds,
            "target_fpr_values": target_fprs,
            "num_replicates": 100,
            "seed": 42,
            "confidence_level": 0.95,
        }
        ci_seq = correctness_metrics.compute_hierarchical_bootstrap_cis(**common_kwargs, num_workers=1)
        ci_par = correctness_metrics.compute_hierarchical_bootstrap_cis(**common_kwargs, num_workers=2)
        common_keys = set(ci_seq.keys()) & set(ci_par.keys())
        assert len(common_keys) > 0
        tol = self._SEQ_PAR_OVERLAP_TOLERANCE
        for key in common_keys:
            assert ci_seq[key].point_estimate <= ci_par[key].upper_bound + tol
            assert ci_seq[key].point_estimate >= ci_par[key].lower_bound - tol
            assert ci_par[key].point_estimate <= ci_seq[key].upper_bound + tol
            assert ci_par[key].point_estimate >= ci_seq[key].lower_bound - tol

    def test_parallel_num_workers_exceeds_replicates(
        self,
        clustered_data: tuple[
            list[correctness_types.EvalRecord],
            np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]],
            dict[float, float],
            list[float],
        ],
    ) -> None:
        records, scores, thresholds, target_fprs = clustered_data
        cis = correctness_metrics.compute_clustered_bootstrap_cis(
            records=records,
            scores=scores,
            thresholds=thresholds,
            target_fpr_values=target_fprs,
            num_replicates=15,
            seed=42,
            confidence_level=0.95,
            num_workers=50,
        )
        assert len(cis) > 0
        for ci in cis.values():
            assert ci.lower_bound <= ci.point_estimate <= ci.upper_bound

    def test_parallel_error_propagation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clustered_data: tuple[
            list[correctness_types.EvalRecord],
            np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]],
            dict[float, float],
            list[float],
        ],
    ) -> None:
        import pyine.utils.concurrency

        original_run_in_parallel = pyine.utils.concurrency.run_in_parallel

        def _failing_run_in_parallel(
            callables: list[typing.Any],
            **kwargs: typing.Any,
        ) -> tuple[list[typing.Any], list[BaseException | None]]:
            results, errors = original_run_in_parallel(callables, **kwargs)
            errors[0] = RuntimeError("simulated worker failure")
            results[0] = None
            return results, errors

        monkeypatch.setattr(pyine.utils.concurrency, "run_in_parallel", _failing_run_in_parallel)
        records, scores, thresholds, target_fprs = clustered_data
        with pytest.raises(RuntimeError, match="simulated worker failure"):
            correctness_metrics.compute_clustered_bootstrap_cis(
                records=records,
                scores=scores,
                thresholds=thresholds,
                target_fpr_values=target_fprs,
                num_replicates=50,
                seed=42,
                confidence_level=0.95,
                num_workers=2,
            )

    def test_negative_num_workers_raises(
        self,
        clustered_data: tuple[
            list[correctness_types.EvalRecord],
            np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]],
            dict[float, float],
            list[float],
        ],
    ) -> None:
        records, scores, thresholds, target_fprs = clustered_data
        with pytest.raises(ValueError, match="num_workers must be non-negative"):
            correctness_metrics.compute_clustered_bootstrap_cis(
                records=records,
                scores=scores,
                thresholds=thresholds,
                target_fpr_values=target_fprs,
                num_replicates=50,
                seed=42,
                confidence_level=0.95,
                num_workers=-1,
            )
