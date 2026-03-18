"""Tests for pyine.evals.correctness.types."""

from __future__ import annotations

import typing

import numpy as np
import pydantic
import pytest

import pyine.evals.correctness.types as correctness_types


class TestEvalRecord:
    def test_construction(self) -> None:
        record = correctness_types.EvalRecord(
            sample_id="TACO/TRAIN/p000001/s0000/t0000",
            problem_id="TACO/TRAIN/p000001",
            attempt_index=0,
            model_output="print(42)",
            final_answer="42",
            expected_output="42",
            label=True,
            code_type="original",
            tags=["tag1"],
            record={"sample_id": "TACO/TRAIN/p000001/s0000/t0000"},
            difficulty_score=None,
        )
        assert record.sample_id == "TACO/TRAIN/p000001/s0000/t0000"
        assert record.label is True
        assert record.difficulty_score is None

    def test_frozen(self) -> None:
        record = correctness_types.EvalRecord(
            sample_id="id",
            problem_id="pid",
            attempt_index=0,
            model_output="out",
            final_answer=None,
            expected_output="exp",
            label=False,
            code_type="original",
            tags=[],
            record={},
            difficulty_score=None,
        )
        with pytest.raises(AttributeError):
            record.label = True  # type: ignore[misc]


class TestScoringResult:
    def test_valid_without_costs(self) -> None:
        result = correctness_types.ScoringResult(scores=[0.1, 0.9])
        assert result.verification_costs is None

    def test_valid_with_costs(self) -> None:
        result = correctness_types.ScoringResult(
            scores=[0.1, 0.9],
            verification_costs=[1.0, 2.0],
        )
        assert result.verification_costs == [1.0, 2.0]

    def test_costs_length_mismatch(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="same length"):
            correctness_types.ScoringResult(
                scores=[0.1, 0.9],
                verification_costs=[1.0],
            )

    def test_negative_costs(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="non-negative"):
            correctness_types.ScoringResult(
                scores=[0.1, 0.9],
                verification_costs=[1.0, -0.5],
            )

    def test_valid_with_attempt_metadata(self) -> None:
        result = correctness_types.ScoringResult(
            scores=[0.1, 0.9],
            attempt_metadata={
                ("s1", 0, 0): {"input_token_count": 12},
                ("s2", 1, 1): {"input_token_count": 8},
            },
        )
        assert result.attempt_metadata is not None
        assert result.attempt_metadata[("s1", 0, 0)]["input_token_count"] == 12

    def test_attempt_metadata_length_mismatch(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="attempt_metadata must have same length"):
            correctness_types.ScoringResult(
                scores=[0.1, 0.9],
                attempt_metadata={
                    ("s1", 0, 0): {"input_token_count": 12},
                },
            )


class TestGuardrailScorerProtocol:
    def test_protocol_conformance(self) -> None:
        class MockScorer:
            def score_records(
                self,
                records: list[correctness_types.EvalRecord],
            ) -> correctness_types.ScoringResult:
                return correctness_types.ScoringResult(scores=[0.5] * len(records))

            def get_metadata(self) -> dict[str, typing.Any]:
                return {"name": "mock"}

            def get_verification_cost_unit(self) -> str | None:
                return None

        scorer: correctness_types.GuardrailScorer = MockScorer()
        assert scorer.get_metadata()["name"] == "mock"
        assert scorer.get_verification_cost_unit() is None


class TestThresholdFreeMetrics:
    def test_valid_with_curves(self) -> None:
        correctness_types.ThresholdFreeMetrics(
            auroc=0.85,
            average_precision=0.9,
            tpr_at_fpr={0.01: 0.7},
            fpr_grid=np.array([0.0, 1.0]),
            tpr_grid=np.array([0.0, 1.0]),
            precision_grid=np.array([1.0, 0.5]),
            recall_grid=np.array([0.0, 1.0]),
        )

    def test_all_none_for_single_class(self) -> None:
        empty = np.array([], dtype=np.float64)
        correctness_types.ThresholdFreeMetrics(
            auroc=None,
            average_precision=None,
            tpr_at_fpr=None,
            fpr_grid=empty,
            tpr_grid=empty,
            precision_grid=empty,
            recall_grid=empty,
        )

    def test_partial_none_raises(self) -> None:
        empty = np.array([], dtype=np.float64)
        with pytest.raises(pydantic.ValidationError, match="all be None or all non-None"):
            correctness_types.ThresholdFreeMetrics(
                auroc=0.5,
                average_precision=None,
                tpr_at_fpr=None,
                fpr_grid=empty,
                tpr_grid=empty,
                precision_grid=empty,
                recall_grid=empty,
            )

    def test_grid_length_mismatch(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="same length"):
            correctness_types.ThresholdFreeMetrics(
                auroc=0.5,
                average_precision=0.5,
                tpr_at_fpr={0.01: 0.5},
                fpr_grid=np.array([0.0, 1.0]),
                tpr_grid=np.array([0.0]),
                precision_grid=np.array([1.0]),
                recall_grid=np.array([1.0]),
            )


class TestThresholdedMetrics:
    def test_valid_confusion_matrix(self) -> None:
        metrics = correctness_types.ThresholdedMetrics(
            target_fpr=0.01,
            threshold=0.5,
            tp=80,
            fp=5,
            tn=95,
            fn=20,
            tpr=80 / 100,
            fpr=5 / 100,
            fnr=20 / 100,
            precision=80 / 85,
            npv=95 / 115,
        )
        assert metrics.tp == 80

    def test_zero_positive_denom(self) -> None:
        correctness_types.ThresholdedMetrics(
            target_fpr=0.01,
            threshold=0.5,
            tp=0,
            fp=5,
            tn=95,
            fn=0,
            tpr=None,
            fpr=5 / 100,
            fnr=None,
            precision=0.0,
            npv=95 / 95,
        )

    def test_precision_none_when_nonzero_denom_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="precision must not be None"):
            correctness_types.ThresholdedMetrics(
                target_fpr=0.01,
                threshold=0.5,
                tp=80,
                fp=5,
                tn=95,
                fn=20,
                tpr=80 / 100,
                fpr=5 / 100,
                fnr=20 / 100,
                precision=None,
                npv=95 / 115,
            )

    def test_npv_none_when_nonzero_denom_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="npv must not be None"):
            correctness_types.ThresholdedMetrics(
                target_fpr=0.01,
                threshold=0.5,
                tp=80,
                fp=5,
                tn=95,
                fn=20,
                tpr=80 / 100,
                fpr=5 / 100,
                fnr=20 / 100,
                precision=80 / 85,
                npv=None,
            )

    def test_inconsistent_precision_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="precision.*inconsistent"):
            correctness_types.ThresholdedMetrics(
                target_fpr=0.01,
                threshold=0.5,
                tp=80,
                fp=5,
                tn=95,
                fn=20,
                tpr=80 / 100,
                fpr=5 / 100,
                fnr=20 / 100,
                precision=0.5,
                npv=95 / 115,
            )

    def test_inconsistent_tpr_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="inconsistent"):
            correctness_types.ThresholdedMetrics(
                target_fpr=0.01,
                threshold=0.5,
                tp=80,
                fp=5,
                tn=95,
                fn=20,
                tpr=0.5,
                fpr=5 / 100,
                fnr=20 / 100,
                precision=80 / 85,
                npv=95 / 115,
            )


class TestSampleLevelMetrics:
    def test_valid(self) -> None:
        correctness_types.SampleLevelMetrics(
            target_fpr=0.01,
            base_pass_rate=0.8,
            guarded_pass_rate=0.7,
            unsafe_slip_rate=0.1,
            total_block_rate=0.05,
            best_of_k_success_rate=0.9,
            cons_pass_rate=0.6,
            cons_unsafe_slip_rate=0.1,
            cons_justified_reject_rate=0.8,
        )

    def test_guarded_exceeds_base_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="cannot exceed"):
            correctness_types.SampleLevelMetrics(
                target_fpr=0.01,
                base_pass_rate=0.5,
                guarded_pass_rate=0.6,
                unsafe_slip_rate=0.1,
                total_block_rate=0.0,
                best_of_k_success_rate=0.9,
                cons_pass_rate=0.4,
                cons_unsafe_slip_rate=None,
                cons_justified_reject_rate=0.5,
            )

    def test_cons_unsafe_slip_not_none_when_cons_zero_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="cons_unsafe_slip_rate must be None"):
            correctness_types.SampleLevelMetrics(
                target_fpr=0.01,
                base_pass_rate=0.8,
                guarded_pass_rate=0.7,
                unsafe_slip_rate=0.1,
                total_block_rate=0.05,
                best_of_k_success_rate=0.9,
                cons_pass_rate=0.0,
                cons_unsafe_slip_rate=0.5,
                cons_justified_reject_rate=0.8,
            )

    def test_cons_justified_not_none_when_all_pass_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="cons_justified_reject_rate must be None"):
            correctness_types.SampleLevelMetrics(
                target_fpr=0.01,
                base_pass_rate=0.8,
                guarded_pass_rate=0.7,
                unsafe_slip_rate=0.1,
                total_block_rate=0.0,
                best_of_k_success_rate=0.9,
                cons_pass_rate=1.0,
                cons_unsafe_slip_rate=0.1,
                cons_justified_reject_rate=0.5,
            )

    def test_cons_pass_zero_with_none_unsafe_valid(self) -> None:
        correctness_types.SampleLevelMetrics(
            target_fpr=0.01,
            base_pass_rate=0.8,
            guarded_pass_rate=0.7,
            unsafe_slip_rate=0.1,
            total_block_rate=0.05,
            best_of_k_success_rate=0.9,
            cons_pass_rate=0.0,
            cons_unsafe_slip_rate=None,
            cons_justified_reject_rate=0.8,
        )

    def test_cons_pass_one_with_none_justified_valid(self) -> None:
        correctness_types.SampleLevelMetrics(
            target_fpr=0.01,
            base_pass_rate=0.8,
            guarded_pass_rate=0.7,
            unsafe_slip_rate=0.1,
            total_block_rate=0.0,
            best_of_k_success_rate=0.9,
            cons_pass_rate=1.0,
            cons_unsafe_slip_rate=0.1,
            cons_justified_reject_rate=None,
        )


class TestAggregatedResult:
    def test_empty_per_run_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="at least one"):
            correctness_types.AggregatedResult(
                split_summary={},
                class_balance=_make_class_balance(),
                per_run=[],
                cross_run_mean={},
                cross_run_std={},
                cross_run_p5={},
                cross_run_num_valid={},
                hierarchical_cis={},
                difficulty_stats=None,
                verification_cost_stats=None,
            )

    def test_to_flat_dict_includes_sample_count(self) -> None:
        agg = _make_minimal_aggregated_result()
        flat = agg.to_flat_dict()
        assert "sample_count" in flat
        assert flat["sample_count"] == 1  # 1 entry in per_sample_positive_rates

    def test_to_flat_dict_includes_num_valid(self) -> None:
        agg = _make_minimal_aggregated_result(cross_run_num_valid={"auroc": 1})
        flat = agg.to_flat_dict()
        assert "auroc/num_valid_runs" in flat
        assert flat["auroc/num_valid_runs"] == 1

    def test_to_flat_dict_includes_cost_metrics_from_cross_run(self) -> None:
        agg = _make_minimal_aggregated_result(
            cross_run_mean={"fpr_0_01/cost_total": 100.0, "fpr_0_01/cost_mean": 5.0},
        )
        flat = agg.to_flat_dict()
        assert flat["fpr_0_01/cost_total/mean"] == 100.0
        assert flat["fpr_0_01/cost_mean/mean"] == 5.0

    def test_to_flat_dict_no_duplicate_cost_namespace(self) -> None:
        """Verify there is no separate 'cost/...' namespace (finding 3 fix)."""
        agg = _make_minimal_aggregated_result(
            verification_cost_stats={
                0.01: correctness_types.VerificationCostStats(
                    target_fpr=0.01,
                    cost_unit="tokens",
                    total_cost=100.0,
                    mean_cost_per_record=5.0,
                    median_cost_per_record=4.0,
                    std_cost_per_record=2.0,
                    cost_per_correct_acceptance=3.0,
                    cost_per_incorrect_block=7.0,
                    cost_accuracy_rank_correlation=0.3,
                    cost_difficulty_rank_correlation=None,
                ),
            },
        )
        flat = agg.to_flat_dict()
        assert not any(key.startswith("cost/") for key in flat)


def _make_minimal_aggregated_result(
    cross_run_mean: dict[str, float] | None = None,
    cross_run_num_valid: dict[str, int] | None = None,
    verification_cost_stats: dict[float, correctness_types.VerificationCostStats] | None = None,
) -> correctness_types.AggregatedResult:
    """Build a minimal AggregatedResult for to_flat_dict tests."""
    empty = np.array([], dtype=np.float64)
    run = correctness_types.SingleRunResult(
        guardrail_metadata={},
        threshold_free=correctness_types.ThresholdFreeMetrics(
            auroc=None,
            average_precision=None,
            tpr_at_fpr=None,
            fpr_grid=empty,
            tpr_grid=empty,
            precision_grid=empty,
            recall_grid=empty,
        ),
        attempt_metrics={},
        sample_metrics={},
        category_results={},
        bootstrap_cis={},
        difficulty_stats=None,
        verification_cost_stats=None,
    )
    return correctness_types.AggregatedResult(
        split_summary={"test_record_count": 10},
        class_balance=_make_class_balance(),
        per_run=[run],
        cross_run_mean=cross_run_mean or {},
        cross_run_std={},
        cross_run_p5={},
        cross_run_num_valid=cross_run_num_valid or {},
        hierarchical_cis={},
        difficulty_stats=None,
        verification_cost_stats=verification_cost_stats,
    )


def _make_class_balance() -> correctness_types.ClassBalanceStats:
    return correctness_types.ClassBalanceStats(
        overall_positive_rate=0.5,
        per_sample_positive_rates=[0.5],
        num_all_correct_samples=0,
        num_all_incorrect_samples=0,
        code_type_proportions={"original": 1.0},
        predict_type_proportions={"program_output": 1.0},
    )


def _make_aggregated_with_metrics() -> correctness_types.AggregatedResult:
    """Build an AggregatedResult with actual metrics and categories for table method tests."""
    attempt_metrics = {
        0.01: correctness_types.ThresholdedMetrics(
            target_fpr=0.01,
            threshold=0.5,
            tp=40,
            fp=2,
            tn=198,
            fn=10,
            tpr=0.8,
            fpr=0.01,
            fnr=0.2,
            precision=40 / 42,
            npv=198 / 208,
        ),
    }
    sample_metrics = {
        0.01: correctness_types.SampleLevelMetrics(
            target_fpr=0.01,
            base_pass_rate=0.9,
            guarded_pass_rate=0.84,
            unsafe_slip_rate=0.05,
            total_block_rate=0.02,
            best_of_k_success_rate=0.95,
            cons_pass_rate=0.7,
            cons_unsafe_slip_rate=0.03,
            cons_justified_reject_rate=0.8,
        ),
    }
    cat_result = correctness_types.CategoryResult(
        category="regular",
        record_count=400,
        sample_count=80,
        class_balance=correctness_types.ClassBalanceStats(
            overall_positive_rate=0.6,
            per_sample_positive_rates=[0.6] * 80,
            num_all_correct_samples=0,
            num_all_incorrect_samples=0,
            code_type_proportions={"original": 1.0},
            predict_type_proportions={},
        ),
        threshold_free=correctness_types.ThresholdFreeMetrics(
            auroc=0.94,
            average_precision=0.90,
            tpr_at_fpr={0.01: 0.75},
            fpr_grid=np.linspace(0, 1, 10),
            tpr_grid=np.linspace(0, 1, 10),
            precision_grid=np.linspace(1, 0.5, 10),
            recall_grid=np.linspace(0, 1, 10),
        ),
        attempt_metrics=attempt_metrics,
        sample_metrics=sample_metrics,
    )
    run = correctness_types.SingleRunResult(
        guardrail_metadata={},
        threshold_free=correctness_types.ThresholdFreeMetrics(
            auroc=0.92,
            average_precision=0.87,
            tpr_at_fpr={0.01: 0.72},
            fpr_grid=np.linspace(0, 1, 10),
            tpr_grid=np.linspace(0, 1, 10),
            precision_grid=np.linspace(1, 0.5, 10),
            recall_grid=np.linspace(0, 1, 10),
        ),
        attempt_metrics=attempt_metrics,
        sample_metrics=sample_metrics,
        category_results={"regular": cat_result},
        bootstrap_cis={},
        difficulty_stats=None,
        verification_cost_stats=None,
    )
    cross_run_mean = {
        "auroc": 0.92,
        "average_precision": 0.87,
        "tpr_at_fpr_0_01": 0.72,
        "fpr_0_01/tpr": 0.80,
        "fpr_0_01/fpr": 0.01,
        "fpr_0_01/guarded_pass_rate": 0.84,
        "fpr_0_01/unsafe_slip_rate": 0.05,
        "fpr_0_01/best_of_k_success_rate": 0.95,
        "fpr_0_01/cons_pass_rate": 0.7,
        "fpr_0_01/cons_unsafe_slip_rate": 0.03,
        "fpr_0_01/cons_justified_reject_rate": 0.8,
        "category/regular/auroc": 0.94,
        "category/regular/fpr_0_01/tpr": 0.75,
    }
    import pyine.utils.metrics.confidence

    hierarchical_cis = {
        "auroc": pyine.utils.metrics.confidence.ConfidenceInterval(
            point_estimate=0.92,
            lower_bound=0.88,
            upper_bound=0.95,
        ),
    }
    return correctness_types.AggregatedResult(
        split_summary={"test_record_count": 500},
        class_balance=_make_class_balance(),
        per_run=[run],
        cross_run_mean=cross_run_mean,
        cross_run_std={"auroc": 0.01},
        cross_run_p5={"auroc": 0.90},
        cross_run_num_valid={"auroc": 1},
        hierarchical_cis=hierarchical_cis,
        difficulty_stats=None,
        verification_cost_stats=None,
    )


class TestToCompactSummaryDict:
    def test_contains_auroc_with_cis(self) -> None:
        agg = _make_aggregated_with_metrics()
        compact = agg.to_compact_summary_dict()
        assert compact["auroc/mean"] == pytest.approx(0.92)
        assert compact["auroc/bootstrap_ci_lower"] == pytest.approx(0.88)
        assert compact["auroc/bootstrap_ci_upper"] == pytest.approx(0.95)

    def test_contains_per_fpr_metrics(self) -> None:
        agg = _make_aggregated_with_metrics()
        compact = agg.to_compact_summary_dict()
        assert compact["fpr_0_01/tpr/mean"] == pytest.approx(0.80)
        assert compact["fpr_0_01/guarded_pass_rate/mean"] == pytest.approx(0.84)
        assert compact["fpr_0_01/unsafe_slip_rate/mean"] == pytest.approx(0.05)

    def test_excludes_category_keys(self) -> None:
        agg = _make_aggregated_with_metrics()
        compact = agg.to_compact_summary_dict()
        assert not any(key.startswith("category/") for key in compact)

    def test_excludes_record_and_sample_count(self) -> None:
        agg = _make_aggregated_with_metrics()
        compact = agg.to_compact_summary_dict()
        assert "record_count" not in compact
        assert "sample_count" not in compact

    def test_contains_class_balance(self) -> None:
        agg = _make_aggregated_with_metrics()
        compact = agg.to_compact_summary_dict()
        assert "class_balance/overall_positive_rate" in compact

    def test_contains_tpr_at_fpr(self) -> None:
        agg = _make_aggregated_with_metrics()
        compact = agg.to_compact_summary_dict()
        assert compact["tpr_at_fpr_0_01/mean"] == pytest.approx(0.72)

    def test_is_subset_of_flat_dict(self) -> None:
        agg = _make_aggregated_with_metrics()
        compact = agg.to_compact_summary_dict()
        flat = agg.to_flat_dict()
        for key, _val in compact.items():
            assert key in flat, f"compact key {key!r} not in flat dict"


class TestToDetailedMetricsTable:
    def test_row_count_excludes_categories(self) -> None:
        agg = _make_aggregated_with_metrics()
        rows = agg.to_detailed_metrics_table()
        metric_names = [row[correctness_types.COL_METRIC_NAME] for row in rows]
        assert not any(name.startswith("category/") for name in metric_names)
        non_cat_keys = [key for key in agg.cross_run_mean if not key.startswith("category/")]
        assert len(rows) == len(non_cat_keys)

    def test_columns_match_schema(self) -> None:
        agg = _make_aggregated_with_metrics()
        rows = agg.to_detailed_metrics_table()
        for row in rows:
            assert set(row.keys()) == set(correctness_types.DETAILED_METRICS_COLUMNS)

    def test_values_match_cross_run_mean(self) -> None:
        agg = _make_aggregated_with_metrics()
        rows = agg.to_detailed_metrics_table()
        auroc_row = next(row for row in rows if row[correctness_types.COL_METRIC_NAME] == "auroc")
        assert auroc_row[correctness_types.COL_MEAN] == pytest.approx(0.92)

    def test_bootstrap_ci_present_for_auroc(self) -> None:
        agg = _make_aggregated_with_metrics()
        rows = agg.to_detailed_metrics_table()
        auroc_row = next(row for row in rows if row[correctness_types.COL_METRIC_NAME] == "auroc")
        assert auroc_row[correctness_types.COL_BOOTSTRAP_CI_LOWER] == pytest.approx(0.88)

    def test_empty_when_no_metrics(self) -> None:
        agg = _make_minimal_aggregated_result()
        rows = agg.to_detailed_metrics_table()
        assert rows == []


class TestToCategoryMetricsTable:
    def test_category_rows_present(self) -> None:
        agg = _make_aggregated_with_metrics()
        rows = agg.to_category_metrics_table()
        assert len(rows) > 0
        categories = {row[correctness_types.COL_CATEGORY] for row in rows}
        assert "regular" in categories

    def test_columns_match_schema(self) -> None:
        agg = _make_aggregated_with_metrics()
        rows = agg.to_category_metrics_table()
        for row in rows:
            assert set(row.keys()) == set(correctness_types.CATEGORY_METRICS_COLUMNS)

    def test_record_and_sample_count_present(self) -> None:
        agg = _make_aggregated_with_metrics()
        rows = agg.to_category_metrics_table()
        regular_row = next(row for row in rows if row[correctness_types.COL_CATEGORY] == "regular")
        assert regular_row[correctness_types.COL_RECORD_COUNT] == 400
        assert regular_row[correctness_types.COL_SAMPLE_COUNT] == 80

    def test_empty_when_no_categories(self) -> None:
        agg = _make_minimal_aggregated_result()
        rows = agg.to_category_metrics_table()
        assert rows == []

    def test_collision_raises(self) -> None:
        """Two category names that sanitize to the same key should raise ValueError."""
        empty = np.array([], dtype=np.float64)
        attempt_metrics: dict[float, correctness_types.ThresholdedMetrics] = {}
        sample_metrics: dict[float, correctness_types.SampleLevelMetrics] = {}
        cat_kwargs: dict[str, typing.Any] = {
            "record_count": 50,
            "sample_count": 10,
            "class_balance": correctness_types.ClassBalanceStats(
                overall_positive_rate=0.5,
                per_sample_positive_rates=[0.5] * 10,
                num_all_correct_samples=0,
                num_all_incorrect_samples=0,
                code_type_proportions={},
                predict_type_proportions={},
            ),
            "threshold_free": correctness_types.ThresholdFreeMetrics(
                auroc=0.85,
                average_precision=0.80,
                tpr_at_fpr={0.01: 0.60},
                fpr_grid=np.linspace(0, 1, 10),
                tpr_grid=np.linspace(0, 1, 10),
                precision_grid=np.linspace(1, 0.5, 10),
                recall_grid=np.linspace(0, 1, 10),
            ),
            "attempt_metrics": attempt_metrics,
            "sample_metrics": sample_metrics,
        }
        cat_results = {
            "a/b": correctness_types.CategoryResult(category="a/b", **cat_kwargs),
            "a_b": correctness_types.CategoryResult(category="a_b", **cat_kwargs),
        }
        run = correctness_types.SingleRunResult(
            guardrail_metadata={},
            threshold_free=correctness_types.ThresholdFreeMetrics(
                auroc=None,
                average_precision=None,
                tpr_at_fpr=None,
                fpr_grid=empty,
                tpr_grid=empty,
                precision_grid=empty,
                recall_grid=empty,
            ),
            attempt_metrics={},
            sample_metrics={},
            category_results=cat_results,
            bootstrap_cis={},
            difficulty_stats=None,
            verification_cost_stats=None,
        )
        agg = correctness_types.AggregatedResult(
            split_summary={"test_record_count": 10},
            class_balance=_make_class_balance(),
            per_run=[run],
            cross_run_mean={},
            cross_run_std={},
            cross_run_p5={},
            cross_run_num_valid={},
            hierarchical_cis={},
            difficulty_stats=None,
            verification_cost_stats=None,
        )
        with pytest.raises(ValueError, match="collision"):
            agg.to_category_metrics_table()


class TestBuildSafeCatReverseMap:
    def test_no_collision(self) -> None:
        result = correctness_types.build_safe_cat_reverse_map({"regular": typing.cast("typing.Any", None)})
        assert result == {"regular": "regular"}

    def test_slash_replaced(self) -> None:
        result = correctness_types.build_safe_cat_reverse_map({"biasing/hinted": typing.cast("typing.Any", None)})
        assert "biasing_hinted" in result
        assert result["biasing_hinted"] == "biasing/hinted"


class TestValidateCrossRunCategories:
    def test_single_run_passes(self) -> None:
        correctness_types.validate_cross_run_categories([typing.cast("typing.Any", None)])  # <= 1 is fine

    def test_mismatched_categories_raises(self) -> None:
        import types as builtin_types

        run0 = builtin_types.SimpleNamespace(category_results={"a": None})
        run1 = builtin_types.SimpleNamespace(category_results={"b": None})
        with pytest.raises(ValueError, match="category set mismatch"):
            correctness_types.validate_cross_run_categories([run0, run1])  # type: ignore[arg-type]
