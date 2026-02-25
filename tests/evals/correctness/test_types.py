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


class TestGuardrailSplits:
    def test_to_summary(self) -> None:
        splits = correctness_types.GuardrailSplits(
            guardrail_train=[],
            guardrail_valid=[],
            guardrail_test=[],
            train_problem_ids=frozenset({"a", "b"}),
            valid_problem_ids=frozenset({"c"}),
            test_problem_ids=frozenset({"d", "e"}),
        )
        summary = splits.to_summary()
        assert summary["train_problem_ids"] == ["a", "b"]
        assert summary["valid_problem_ids"] == ["c"]
        assert summary["test_problem_ids"] == ["d", "e"]
        assert summary["train_record_count"] == 0


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
