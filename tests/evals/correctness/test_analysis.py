"""Tests for pyine.evals.correctness.analysis -- W&B extraction and data model logic."""

from __future__ import annotations

import typing

import numpy as np
import pytest

import pyine.evals.correctness
import pyine.evals.correctness.analysis
import pyine.evals.correctness.types as correctness_types


class _MockSummary(dict[str, typing.Any]):
    """Mock wandb run summary that behaves like a dict with attribute access."""

    def __getattr__(self, name: str) -> typing.Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _MockRun:
    """Mock wandb Run for testing analysis extraction functions."""

    def __init__(
        self,
        run_id: str = "run-001",
        name: str = "test-run",
        group: str = "test-group",
        project: str = "test-project",
        entity: str = "test-entity",
        created_at: str = "2025-01-15T12:00:00",
        summary: dict[str, typing.Any] | None = None,
    ) -> None:
        self.id = run_id
        self.name = name
        self.group = group
        self.project = project
        self.entity = entity
        self.created_at = created_at
        self.summary = _MockSummary(summary or {})


_SUBSET = "guardrail_test"


def _make_single_type_summary(prefix: str = f"benchmark/{_SUBSET}/") -> dict[str, typing.Any]:
    """Build a W&B summary dict for a single-type correctness eval."""
    return {
        f"{prefix}auroc/mean": 0.92,
        f"{prefix}auroc/bootstrap_ci_lower": 0.88,
        f"{prefix}auroc/bootstrap_ci_upper": 0.95,
        f"{prefix}average_precision/mean": 0.87,
        f"{prefix}tpr_at_fpr_0_01/mean": 0.72,
        f"{prefix}sample_count": 100,
        f"{prefix}record_count": 500,
        f"{prefix}class_balance/overall_positive_rate": 0.6,
        f"{prefix}fpr_0_01/tpr/mean": 0.72,
        f"{prefix}fpr_0_01/fpr/mean": 0.009,
        f"{prefix}fpr_0_01/guarded_pass_rate/mean": 0.88,
        f"{prefix}fpr_0_01/unsafe_slip_rate/mean": 0.05,
        f"{prefix}category/regular/auroc/mean": 0.94,
        f"{prefix}category/regular/record_count": 400,
        f"{prefix}category/regular/sample_count": 80,
        f"{prefix}category/regular/fpr_0_01/tpr/mean": 0.75,
        f"{prefix}category/hinted/auroc/mean": 0.85,
        f"{prefix}category/hinted/record_count": 100,
        f"{prefix}category/hinted/sample_count": 20,
    }


def _make_multi_type_summary() -> dict[str, typing.Any]:
    """Build a W&B summary dict for a multi-type correctness eval."""
    base: dict[str, typing.Any] = {
        f"benchmark/{_SUBSET}/_guardrail_type_names": ["type_a", "type_b"],
    }
    for type_name in ("type_a", "type_b"):
        prefix = f"benchmark/{_SUBSET}/{type_name}/"
        base.update(
            {
                f"{prefix}auroc/mean": 0.90 if type_name == "type_a" else 0.85,
                f"{prefix}average_precision/mean": 0.82,
                f"{prefix}sample_count": 50,
                f"{prefix}record_count": 250,
                f"{prefix}class_balance/overall_positive_rate": 0.55,
                f"{prefix}fpr_0_01/tpr/mean": 0.70,
                f"{prefix}fpr_0_01/guarded_pass_rate/mean": 0.80,
            }
        )
    return base


# ---- Type Detection Tests ----


class TestDetectGuardrailTypeNames:
    """Tests for detect_guardrail_type_names."""

    def test_returns_none_for_single_type(self) -> None:
        run = _MockRun(summary=_make_single_type_summary())
        result = pyine.evals.correctness.analysis.detect_guardrail_type_names(run, _SUBSET)  # type: ignore[arg-type]
        assert result is None

    def test_returns_types_from_metadata_key(self) -> None:
        run = _MockRun(summary=_make_multi_type_summary())
        result = pyine.evals.correctness.analysis.detect_guardrail_type_names(run, _SUBSET)  # type: ignore[arg-type]
        assert result == ["type_a", "type_b"]

    def test_heuristic_fallback_finds_types(self) -> None:
        """When _guardrail_type_names metadata is absent, heuristic detects types."""
        summary = _make_multi_type_summary()
        del summary[f"benchmark/{_SUBSET}/_guardrail_type_names"]
        run = _MockRun(summary=summary)
        result = pyine.evals.correctness.analysis.detect_guardrail_type_names(run, _SUBSET)  # type: ignore[arg-type]
        assert result == ["type_a", "type_b"]

    def test_heuristic_returns_none_for_unvalidated_candidates(self) -> None:
        """Candidates below the 2-anchor threshold are ignored in non-strict mode."""
        summary = {
            f"benchmark/{_SUBSET}/mystery_type/some_odd_key": 0.5,
        }
        run = _MockRun(summary=summary)
        result = pyine.evals.correctness.analysis.detect_guardrail_type_names(run, _SUBSET)  # type: ignore[arg-type]
        assert result is None

    def test_heuristic_strict_mode_raises_for_unvalidated_candidates(self) -> None:
        """Strict mode raises ValueError when candidates fail validation."""
        summary = {
            f"benchmark/{_SUBSET}/mystery_type/some_odd_key": 0.5,
        }
        run = _MockRun(summary=summary)
        with pytest.raises(ValueError, match="none met validation"):
            pyine.evals.correctness.analysis.detect_guardrail_type_names(
                run,
                _SUBSET,
                strict=True,  # type: ignore[arg-type]
            )

    def test_structural_tokens_filtered_out(self) -> None:
        """Known structural tokens (auroc, category, etc.) are not treated as type candidates."""
        summary = _make_single_type_summary()
        # add a key under fpr_ prefix which should also be filtered
        summary[f"benchmark/{_SUBSET}/fpr_0_05/tpr/mean"] = 0.8
        run = _MockRun(summary=summary)
        result = pyine.evals.correctness.analysis.detect_guardrail_type_names(run, _SUBSET)  # type: ignore[arg-type]
        assert result is None


# ---- Resolve Type and Prefix Tests ----


class TestResolveTypeAndPrefix:
    def test_single_type_no_type_name(self) -> None:
        run = _MockRun(summary=_make_single_type_summary())
        resolved_type, prefix = pyine.evals.correctness.analysis._resolve_type_and_prefix(run, _SUBSET, None)  # type: ignore[arg-type]
        assert resolved_type is None
        assert prefix == f"benchmark/{_SUBSET}/"

    def test_multi_type_auto_select_single(self) -> None:
        """When only one type exists and type_name is None, auto-selects it."""
        summary: dict[str, typing.Any] = {
            f"benchmark/{_SUBSET}/_guardrail_type_names": ["only_type"],
        }
        prefix = f"benchmark/{_SUBSET}/only_type/"
        summary.update(
            {
                f"{prefix}auroc/mean": 0.9,
                f"{prefix}average_precision/mean": 0.85,
                f"{prefix}sample_count": 50,
            }
        )
        run = _MockRun(summary=summary)
        resolved_type, resolved_prefix = pyine.evals.correctness.analysis._resolve_type_and_prefix(run, _SUBSET, None)  # type: ignore[arg-type]
        assert resolved_type == "only_type"
        assert resolved_prefix == f"benchmark/{_SUBSET}/only_type/"

    def test_multi_type_raises_without_type_name(self) -> None:
        run = _MockRun(summary=_make_multi_type_summary())
        with pytest.raises(ValueError, match="multiple guardrail types"):
            pyine.evals.correctness.analysis._resolve_type_and_prefix(run, _SUBSET, None)  # type: ignore[arg-type]

    def test_type_name_not_found_raises(self) -> None:
        run = _MockRun(summary=_make_multi_type_summary())
        with pytest.raises(ValueError, match="not found in run"):
            pyine.evals.correctness.analysis._resolve_type_and_prefix(run, _SUBSET, "nonexistent")  # type: ignore[arg-type]

    def test_type_name_on_single_type_run_raises(self) -> None:
        run = _MockRun(summary=_make_single_type_summary())
        with pytest.raises(ValueError, match="single-type eval"):
            pyine.evals.correctness.analysis._resolve_type_and_prefix(run, _SUBSET, "some_type")  # type: ignore[arg-type]

    def test_explicit_type_name_returns_prefixed(self) -> None:
        run = _MockRun(summary=_make_multi_type_summary())
        resolved_type, prefix = pyine.evals.correctness.analysis._resolve_type_and_prefix(run, _SUBSET, "type_a")  # type: ignore[arg-type]
        assert resolved_type == "type_a"
        assert prefix == f"benchmark/{_SUBSET}/type_a/"


# ---- Metric Extraction Tests ----


class TestExtractCorrectnessMetrics:
    def test_extracts_single_type_metrics(self) -> None:
        run = _MockRun(summary=_make_single_type_summary())
        metrics = pyine.evals.correctness.analysis.extract_correctness_metrics(run, _SUBSET)  # type: ignore[arg-type]
        assert metrics.auroc.value == pytest.approx(0.92)
        assert metrics.auroc.ci_lower == pytest.approx(0.88)
        assert metrics.auroc.ci_upper == pytest.approx(0.95)
        assert metrics.average_precision.value == pytest.approx(0.87)
        assert metrics.sample_count == 100
        assert metrics.record_count == 500
        assert metrics.overall_positive_rate == pytest.approx(0.6)
        assert metrics.guardrail_type_name is None

    def test_extracts_tpr_at_fpr(self) -> None:
        run = _MockRun(summary=_make_single_type_summary())
        metrics = pyine.evals.correctness.analysis.extract_correctness_metrics(run, _SUBSET)  # type: ignore[arg-type]
        assert 0.01 in metrics.tpr_at_fpr
        assert metrics.tpr_at_fpr[0.01].value == pytest.approx(0.72)

    def test_extracts_per_fpr_metrics(self) -> None:
        run = _MockRun(summary=_make_single_type_summary())
        metrics = pyine.evals.correctness.analysis.extract_correctness_metrics(run, _SUBSET)  # type: ignore[arg-type]
        assert 0.01 in metrics.attempt_metrics
        assert metrics.attempt_metrics[0.01]["tpr"].value == pytest.approx(0.72)
        assert 0.01 in metrics.sample_metrics
        assert metrics.sample_metrics[0.01]["guarded_pass_rate"].value == pytest.approx(0.88)

    def test_extracts_multi_type_with_type_name(self) -> None:
        run = _MockRun(summary=_make_multi_type_summary())
        metrics = pyine.evals.correctness.analysis.extract_correctness_metrics(
            run,
            _SUBSET,
            type_name="type_a",  # type: ignore[arg-type]
        )
        assert metrics.auroc.value == pytest.approx(0.90)
        assert metrics.guardrail_type_name == "type_a"

    def test_resolved_param_skips_detection(self) -> None:
        """When _resolved is passed, detection is skipped."""
        run = _MockRun(summary=_make_single_type_summary())
        resolved = (None, f"benchmark/{_SUBSET}/")
        metrics = pyine.evals.correctness.analysis.extract_correctness_metrics(
            run,
            _SUBSET,
            _resolved=resolved,  # type: ignore[arg-type]
        )
        assert metrics.auroc.value == pytest.approx(0.92)

    def test_run_metadata_fields(self) -> None:
        run = _MockRun(
            run_id="r42",
            name="my-run",
            group="my-group",
            project="my-project",
            entity="my-entity",
            created_at="2025-06-01",
            summary=_make_single_type_summary(),
        )
        metrics = pyine.evals.correctness.analysis.extract_correctness_metrics(run, _SUBSET)  # type: ignore[arg-type]
        assert metrics.run_id == "r42"
        assert metrics.run_name == "my-run"
        assert metrics.run_group == "my-group"
        assert metrics.project == "my-project"
        assert metrics.entity == "my-entity"


class TestResolveMetricValidation:
    def test_unknown_metric_raises(self) -> None:
        run = _MockRun(summary=_make_single_type_summary())
        metrics = pyine.evals.correctness.analysis.extract_correctness_metrics(run, _SUBSET)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unsupported metric"):
            pyine.evals.correctness.analysis._resolve_metric(metrics, "not_a_metric", target_fpr=0.01)

    def test_fpr_conditioned_metric_requires_target_fpr(self) -> None:
        run = _MockRun(summary=_make_single_type_summary())
        metrics = pyine.evals.correctness.analysis.extract_correctness_metrics(run, _SUBSET)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="requires target_fpr"):
            pyine.evals.correctness.analysis._resolve_metric(metrics, "tpr", target_fpr=None)

    def test_missing_target_fpr_raises(self) -> None:
        run = _MockRun(summary=_make_single_type_summary())
        metrics = pyine.evals.correctness.analysis.extract_correctness_metrics(run, _SUBSET)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="target_fpr=0.05 not available"):
            pyine.evals.correctness.analysis._resolve_metric(metrics, "tpr_at_fpr", target_fpr=0.05)


# ---- Category Extraction Tests ----


class TestExtractCorrectnessCategoryMetrics:
    def test_extracts_categories(self) -> None:
        run = _MockRun(summary=_make_single_type_summary())
        categories = pyine.evals.correctness.analysis.extract_correctness_category_metrics(run, _SUBSET)  # type: ignore[arg-type]
        assert len(categories) == 2
        cat_names = [c.category for c in categories]
        assert "regular" in cat_names
        assert "hinted" in cat_names

    def test_category_auroc(self) -> None:
        run = _MockRun(summary=_make_single_type_summary())
        categories = pyine.evals.correctness.analysis.extract_correctness_category_metrics(run, _SUBSET)  # type: ignore[arg-type]
        regular = next(c for c in categories if c.category == "regular")
        assert regular.auroc.value == pytest.approx(0.94)
        assert regular.record_count == 400
        assert regular.sample_count == 80

    def test_empty_summary_returns_empty_list(self) -> None:
        run = _MockRun(summary={})
        categories = pyine.evals.correctness.analysis.extract_correctness_category_metrics(run, _SUBSET)  # type: ignore[arg-type]
        assert categories == []


# ---- Fetch Summary Tests ----


class TestFetchCorrectnessEvalSummary:
    def test_fetches_combined_summary(self) -> None:
        run = _MockRun(summary=_make_single_type_summary())
        summary = pyine.evals.correctness.analysis.fetch_correctness_eval_summary(run, _SUBSET)  # type: ignore[arg-type]
        assert summary.run_info.auroc.value == pytest.approx(0.92)
        assert len(summary.category_metrics) == 2

    def test_multi_type_with_explicit_type(self) -> None:
        run = _MockRun(summary=_make_multi_type_summary())
        summary = pyine.evals.correctness.analysis.fetch_correctness_eval_summary(
            run,
            _SUBSET,
            type_name="type_b",  # type: ignore[arg-type]
        )
        assert summary.run_info.auroc.value == pytest.approx(0.85)
        assert summary.run_info.guardrail_type_name == "type_b"

    def test_extract_metrics_contract_with_aggregated_flat_dict(self) -> None:
        """Parser stays compatible with AggregatedResult.to_flat_dict output keys."""
        import numpy as np

        import pyine.evals.correctness.types as correctness_types

        empty_grid = np.array([], dtype=np.float64)
        run_result = correctness_types.SingleRunResult(
            guardrail_metadata={"name": "contract"},
            threshold_free=correctness_types.ThresholdFreeMetrics(
                auroc=None,
                average_precision=None,
                tpr_at_fpr=None,
                fpr_grid=empty_grid,
                tpr_grid=empty_grid,
                precision_grid=empty_grid,
                recall_grid=empty_grid,
            ),
            attempt_metrics={},
            sample_metrics={},
            category_results={},
            bootstrap_cis={},
            difficulty_stats=None,
            verification_cost_stats=None,
        )
        class_balance = correctness_types.ClassBalanceStats(
            overall_positive_rate=0.6,
            per_sample_positive_rates=[0.6, 0.4],
            num_all_correct_samples=0,
            num_all_incorrect_samples=0,
            code_type_proportions={"original": 1.0},
            predict_type_proportions={"program_output": 1.0},
        )
        aggregated = correctness_types.AggregatedResult(
            split_summary={"test_record_count": 10},
            class_balance=class_balance,
            per_run=[run_result],
            cross_run_mean={
                "auroc": 0.91,
                "average_precision": 0.86,
                "tpr_at_fpr_0_01": 0.73,
                "fpr_0_01/tpr": 0.73,
                "fpr_0_01/guarded_pass_rate": 0.84,
            },
            cross_run_std={},
            cross_run_p5={},
            cross_run_num_valid={},
            hierarchical_cis={},
            difficulty_stats=None,
            verification_cost_stats=None,
        )
        prefixed_summary = {f"benchmark/{_SUBSET}/{key}": value for key, value in aggregated.to_flat_dict().items()}
        run = _MockRun(summary=prefixed_summary)
        metrics = pyine.evals.correctness.analysis.extract_correctness_metrics(run, _SUBSET)  # type: ignore[arg-type]
        assert metrics.auroc.value == pytest.approx(0.91)
        assert metrics.average_precision.value == pytest.approx(0.86)
        assert metrics.tpr_at_fpr[0.01].value == pytest.approx(0.73)
        assert metrics.sample_metrics[0.01]["guarded_pass_rate"].value == pytest.approx(0.84)


class TestPlotValidation:
    def test_plot_category_breakdown_rejects_unsupported_threshold_free_metric(self) -> None:
        run = _MockRun(summary=_make_single_type_summary())
        summary = pyine.evals.correctness.analysis.fetch_correctness_eval_summary(run, _SUBSET)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unsupported category metric"):
            pyine.evals.correctness.analysis.plot_category_breakdown(summary, "average_precision")

    def test_plot_category_breakdown_rejects_unknown_metric_at_fpr(self) -> None:
        run = _MockRun(summary=_make_single_type_summary())
        summary = pyine.evals.correctness.analysis.fetch_correctness_eval_summary(run, _SUBSET)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unsupported category metric"):
            pyine.evals.correctness.analysis.plot_category_breakdown(
                summary,
                "unknown_metric",
                target_fpr=0.01,
            )


# ---- DataFrame Summarization Tests ----


class TestSummarizeCorrectnessRunsToDataframe:
    def test_creates_dataframe_from_summaries(self) -> None:
        run = _MockRun(summary=_make_single_type_summary())
        summary = pyine.evals.correctness.analysis.fetch_correctness_eval_summary(run, _SUBSET)  # type: ignore[arg-type]
        df = pyine.evals.correctness.analysis.summarize_correctness_runs_to_dataframe([summary])
        assert len(df) == 1
        assert df.iloc[0]["auroc"] == pytest.approx(0.92)
        assert df.iloc[0]["sample_count"] == 100

    def test_empty_list_returns_empty_dataframe(self) -> None:
        df = pyine.evals.correctness.analysis.summarize_correctness_runs_to_dataframe([])
        assert len(df) == 0


# ---- FPR Parsing Tests ----


class TestEvalResultToSummary:
    """Tests for eval_result_to_summary (correctness pickle path)."""

    @staticmethod
    def _make_single_run_result(
        category_results: dict[str, correctness_types.CategoryResult] | None = None,
        fpr_values: list[float] | None = None,
    ) -> correctness_types.SingleRunResult:
        fpr_values = fpr_values or [0.01]
        attempt_metrics: dict[float, correctness_types.ThresholdedMetrics] = {}
        sample_metrics: dict[float, correctness_types.SampleLevelMetrics] = {}
        for fpr_val in fpr_values:
            attempt_metrics[fpr_val] = correctness_types.ThresholdedMetrics(
                target_fpr=fpr_val,
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
            )
            sample_metrics[fpr_val] = correctness_types.SampleLevelMetrics(
                target_fpr=fpr_val,
                base_pass_rate=0.9,
                guarded_pass_rate=0.88,
                unsafe_slip_rate=0.05,
                total_block_rate=0.02,
                best_of_k_success_rate=0.95,
                cons_pass_rate=0.7,
                cons_unsafe_slip_rate=0.03,
                cons_justified_reject_rate=0.8,
            )
        cat_results = category_results or {
            "regular": correctness_types.CategoryResult(
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
            ),
        }
        return correctness_types.SingleRunResult(
            guardrail_metadata={"name": "test"},
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
            category_results=cat_results,
            bootstrap_cis={},
            difficulty_stats=None,
            verification_cost_stats=None,
        )

    @staticmethod
    def _make_eval_result(
        per_run: list[correctness_types.SingleRunResult] | None = None,
        eval_metadata: dict[str, typing.Any] | None = None,
        cross_run_mean: dict[str, float] | None = None,
        split_summary: dict[str, typing.Any] | None = None,
    ) -> pyine.evals.correctness.CorrectnessEvalResult:
        import pyine.evals.correctness

        default_cross_run_mean: dict[str, float] = {
            "auroc": 0.92,
            "average_precision": 0.87,
            "tpr_at_fpr_0_01": 0.72,
            "fpr_0_01/tpr": 0.80,
            "fpr_0_01/fpr": 0.01,
            "fpr_0_01/fnr": 0.20,
            "fpr_0_01/precision": 40 / 42,
            "fpr_0_01/npv": 198 / 208,
            "fpr_0_01/guarded_pass_rate": 0.88,
            "fpr_0_01/unsafe_slip_rate": 0.05,
            "fpr_0_01/base_pass_rate": 0.9,
            "fpr_0_01/total_block_rate": 0.02,
            "fpr_0_01/best_of_k_success_rate": 0.95,
            "fpr_0_01/cons_pass_rate": 0.7,
            "fpr_0_01/cons_unsafe_slip_rate": 0.03,
            "fpr_0_01/cons_justified_reject_rate": 0.8,
            # category keys: no /mean suffix (matches real AggregatedResult cross_run_mean)
            "category/regular/auroc": 0.94,
            "category/regular/fpr_0_01/tpr": 0.75,
            "category/regular/fpr_0_01/fpr": 0.009,
            "category/regular/fpr_0_01/guarded_pass_rate": 0.86,
            "category/regular/fpr_0_01/unsafe_slip_rate": 0.04,
        }
        if cross_run_mean:
            default_cross_run_mean.update(cross_run_mean)
        default_metadata: dict[str, typing.Any] = {
            "eval_subset_name": "guardrail_test",
            "model_name": "test-guardrail",
        }
        if eval_metadata:
            default_metadata.update(eval_metadata)
        aggregated = correctness_types.AggregatedResult(
            split_summary=split_summary or {"test_record_count": 500},
            class_balance=correctness_types.ClassBalanceStats(
                overall_positive_rate=0.6,
                per_sample_positive_rates=[0.6] * 100,
                num_all_correct_samples=0,
                num_all_incorrect_samples=0,
                code_type_proportions={"original": 1.0},
                predict_type_proportions={},
            ),
            per_run=per_run or [TestEvalResultToSummary._make_single_run_result()],
            cross_run_mean=default_cross_run_mean,
            cross_run_std={},
            cross_run_p5={},
            cross_run_num_valid={},
            hierarchical_cis={},
            difficulty_stats=None,
            verification_cost_stats=None,
        )
        return pyine.evals.correctness.CorrectnessEvalResult(
            metrics={},
            eval_metadata=default_metadata,
            aggregated=aggregated,
        )

    def test_basic_summary(self) -> None:
        result = self._make_eval_result()
        summary = pyine.evals.correctness.analysis.eval_result_to_summary(result)
        assert summary.run_info.auroc.value == pytest.approx(0.92)
        assert summary.run_info.average_precision.value == pytest.approx(0.87)
        assert summary.run_info.sample_count == 100
        assert summary.run_info.record_count == 500
        assert summary.run_info.overall_positive_rate == pytest.approx(0.6)

    def test_tpr_at_fpr_extraction(self) -> None:
        result = self._make_eval_result()
        summary = pyine.evals.correctness.analysis.eval_result_to_summary(result)
        assert 0.01 in summary.run_info.tpr_at_fpr
        assert summary.run_info.tpr_at_fpr[0.01].value == pytest.approx(0.72)

    def test_per_fpr_metrics(self) -> None:
        result = self._make_eval_result()
        summary = pyine.evals.correctness.analysis.eval_result_to_summary(result)
        assert 0.01 in summary.run_info.attempt_metrics
        assert summary.run_info.attempt_metrics[0.01]["tpr"].value == pytest.approx(0.80)
        assert 0.01 in summary.run_info.sample_metrics
        assert summary.run_info.sample_metrics[0.01]["guarded_pass_rate"].value == pytest.approx(0.88)

    def test_category_metrics(self) -> None:
        result = self._make_eval_result()
        summary = pyine.evals.correctness.analysis.eval_result_to_summary(result)
        assert len(summary.category_metrics) == 1
        cat = summary.category_metrics[0]
        assert cat.category == "regular"
        assert cat.record_count == 400
        assert cat.sample_count == 80
        # verify extracted category metric values (not just counts)
        assert cat.auroc.value == pytest.approx(0.94)
        assert 0.01 in cat.attempt_metrics
        assert cat.attempt_metrics[0.01]["tpr"].value == pytest.approx(0.75)
        assert cat.attempt_metrics[0.01]["fpr"].value == pytest.approx(0.009)
        assert 0.01 in cat.sample_metrics
        assert cat.sample_metrics[0.01]["guarded_pass_rate"].value == pytest.approx(0.86)
        assert cat.sample_metrics[0.01]["unsafe_slip_rate"].value == pytest.approx(0.04)

    def test_guardrail_type_name_passthrough(self) -> None:
        result = self._make_eval_result()
        summary = pyine.evals.correctness.analysis.eval_result_to_summary(result, guardrail_type_name="my_type")
        assert summary.run_info.guardrail_type_name == "my_type"

    def test_guardrail_type_from_metadata(self) -> None:
        result = self._make_eval_result(eval_metadata={"guardrail_type_name": "meta_type"})
        summary = pyine.evals.correctness.analysis.eval_result_to_summary(result)
        assert summary.run_info.guardrail_type_name == "meta_type"

    def test_guardrail_type_mismatch_raises(self) -> None:
        result = self._make_eval_result(eval_metadata={"guardrail_type_name": "meta_type"})
        with pytest.raises(ValueError, match="does not match"):
            pyine.evals.correctness.analysis.eval_result_to_summary(result, guardrail_type_name="other_type")

    def test_category_with_slash_in_name(self) -> None:
        cat_results = {
            "biasing/misleading": correctness_types.CategoryResult(
                category="biasing/misleading",
                record_count=100,
                sample_count=20,
                class_balance=correctness_types.ClassBalanceStats(
                    overall_positive_rate=0.5,
                    per_sample_positive_rates=[0.5] * 20,
                    num_all_correct_samples=0,
                    num_all_incorrect_samples=0,
                    code_type_proportions={"misleading": 1.0},
                    predict_type_proportions={},
                ),
                threshold_free=correctness_types.ThresholdFreeMetrics(
                    auroc=0.85,
                    average_precision=0.80,
                    tpr_at_fpr={0.01: 0.60},
                    fpr_grid=np.linspace(0, 1, 10),
                    tpr_grid=np.linspace(0, 1, 10),
                    precision_grid=np.linspace(1, 0.5, 10),
                    recall_grid=np.linspace(0, 1, 10),
                ),
                attempt_metrics={},
                sample_metrics={},
            ),
        }
        run_result = self._make_single_run_result(category_results=cat_results)
        result = self._make_eval_result(
            per_run=[run_result],
            cross_run_mean={
                "auroc": 0.92,
                "average_precision": 0.87,
                "category/biasing_misleading/auroc": 0.85,
            },
        )
        summary = pyine.evals.correctness.analysis.eval_result_to_summary(result)
        assert len(summary.category_metrics) == 1
        assert summary.category_metrics[0].category == "biasing_misleading"
        assert summary.category_metrics[0].record_count == 100
        assert summary.category_metrics[0].auroc.value == pytest.approx(0.85)

    def test_safe_name_collision_raises(self) -> None:
        """Two original names colliding to the same sanitized key should raise."""
        cat_results = {}
        for cat_name in ("a/b_c", "a_b/c"):
            cat_results[cat_name] = correctness_types.CategoryResult(
                category=cat_name,
                record_count=50,
                sample_count=10,
                class_balance=correctness_types.ClassBalanceStats(
                    overall_positive_rate=0.5,
                    per_sample_positive_rates=[0.5] * 10,
                    num_all_correct_samples=0,
                    num_all_incorrect_samples=0,
                    code_type_proportions={},
                    predict_type_proportions={},
                ),
                threshold_free=correctness_types.ThresholdFreeMetrics(
                    auroc=0.85,
                    average_precision=0.80,
                    tpr_at_fpr={0.01: 0.60},
                    fpr_grid=np.linspace(0, 1, 10),
                    tpr_grid=np.linspace(0, 1, 10),
                    precision_grid=np.linspace(1, 0.5, 10),
                    recall_grid=np.linspace(0, 1, 10),
                ),
                attempt_metrics={},
                sample_metrics={},
            )
        run_result = self._make_single_run_result(category_results=cat_results)
        result = self._make_eval_result(per_run=[run_result])
        with pytest.raises(ValueError, match="collision"):
            pyine.evals.correctness.analysis.eval_result_to_summary(result)

    def test_cross_run_category_mismatch_raises(self) -> None:
        run1 = self._make_single_run_result()
        # run2 has different categories
        cat_results2 = {
            "other_cat": correctness_types.CategoryResult(
                category="other_cat",
                record_count=400,
                sample_count=80,
                class_balance=correctness_types.ClassBalanceStats(
                    overall_positive_rate=0.6,
                    per_sample_positive_rates=[0.6] * 80,
                    num_all_correct_samples=0,
                    num_all_incorrect_samples=0,
                    code_type_proportions={},
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
                attempt_metrics={},
                sample_metrics={},
            ),
        }
        run2 = self._make_single_run_result(category_results=cat_results2)
        result = self._make_eval_result(per_run=[run1, run2])
        with pytest.raises(ValueError, match="category set mismatch"):
            pyine.evals.correctness.analysis.eval_result_to_summary(result)

    def test_cost_keys_not_in_attempt_metrics(self) -> None:
        """fpr_*/cost_* keys should not leak into attempt/sample metrics."""
        result = self._make_eval_result(
            cross_run_mean={
                "auroc": 0.92,
                "average_precision": 0.87,
                "fpr_0_01/tpr": 0.80,
                "fpr_0_01/cost_total": 1000.0,
            },
        )
        summary = pyine.evals.correctness.analysis.eval_result_to_summary(result)
        # FPR 0.01 must be detected (otherwise the exclusion check is vacuous)
        assert 0.01 in summary.run_info.attempt_metrics
        assert "cost_total" not in summary.run_info.attempt_metrics[0.01]
        assert 0.01 in summary.run_info.sample_metrics
        assert "cost_total" not in summary.run_info.sample_metrics[0.01]

    def test_subset_name_mismatch_raises(self) -> None:
        result = self._make_eval_result()
        with pytest.raises(ValueError, match="does not match"):
            pyine.evals.correctness.analysis.eval_result_to_summary(result, subset_name="wrong_subset")

    def test_record_count_subset_sensitive(self) -> None:
        """test subset uses test_record_count, valid uses valid_record_count."""
        result = self._make_eval_result(
            eval_metadata={"eval_subset_name": "guardrail_test"},
            split_summary={"test_record_count": 500},
        )
        summary = pyine.evals.correctness.analysis.eval_result_to_summary(result)
        assert summary.run_info.record_count == 500

    def test_record_count_valid_subset(self) -> None:
        result = self._make_eval_result(
            eval_metadata={"eval_subset_name": "guardrail_valid"},
            split_summary={"valid_record_count": 200},
        )
        summary = pyine.evals.correctness.analysis.eval_result_to_summary(result)
        assert summary.run_info.record_count == 200

    def test_record_count_missing_key_raises(self) -> None:
        result = self._make_eval_result(
            eval_metadata={"eval_subset_name": "test"},
            split_summary={"other_key": 100},
        )
        with pytest.raises(ValueError, match="split_summary missing"):
            pyine.evals.correctness.analysis.eval_result_to_summary(result)

    def test_multiple_per_run_consistent(self) -> None:
        """Multiple per_run with same categories should succeed."""
        run1 = self._make_single_run_result()
        run2 = self._make_single_run_result()
        result = self._make_eval_result(per_run=[run1, run2])
        summary = pyine.evals.correctness.analysis.eval_result_to_summary(result)
        assert summary.run_info.auroc.value == pytest.approx(0.92)


class TestParseFprCapture:
    def test_decimal_format(self) -> None:
        assert pyine.evals.correctness.analysis._parse_fpr_capture("0_01") == pytest.approx(0.01)
        assert pyine.evals.correctness.analysis._parse_fpr_capture("0_001") == pytest.approx(0.001)

    def test_scientific_notation(self) -> None:
        assert pyine.evals.correctness.analysis._parse_fpr_capture("1e-05") == pytest.approx(1e-05)

    def test_scientific_with_underscore(self) -> None:
        assert pyine.evals.correctness.analysis._parse_fpr_capture("0_00001e+00") == pytest.approx(0.00001)


# ---- Detect FPR Keys Tests ----


class TestDetectFprKeys:
    def test_detects_decimal_fpr_keys(self) -> None:
        summary = _MockSummary(
            {
                "prefix/fpr_0_01/tpr/mean": 0.7,
                "prefix/fpr_0_05/tpr/mean": 0.8,
            }
        )
        result = pyine.evals.correctness.analysis._detect_fpr_keys(summary, "prefix/")
        assert result == [pytest.approx(0.01), pytest.approx(0.05)]

    def test_detects_scientific_fpr_keys(self) -> None:
        summary = _MockSummary({"prefix/fpr_1e-05/tpr/mean": 0.6})
        result = pyine.evals.correctness.analysis._detect_fpr_keys(summary, "prefix/")
        assert result == [pytest.approx(1e-05)]

    def test_empty_summary(self) -> None:
        result = pyine.evals.correctness.analysis._detect_fpr_keys(_MockSummary(), "prefix/")
        assert result == []
