"""Tests for pyine.evals.code_exec.analysis utility functions.

Note: Visualization functions (plot_*) are not tested here - only data extraction
and filtering utilities.
"""

import typing

import pytest

import pyine.evals.code_exec.analysis


class MockSummary(dict[str, typing.Any]):
    """Mock wandb run summary that behaves like dict but also supports attribute access."""

    def __getattr__(self, name: str) -> typing.Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class MockWandBRun:
    """Mock wandb Run object for testing analysis functions."""

    def __init__(
        self,
        run_id: str = "run-123",
        name: str = "test-run",
        group: str | None = "test-group",
        project: str = "test-project",
        entity: str | None = "test-entity",
        created_at: str = "2025-01-15T12:00:00",
        summary: dict[str, typing.Any] | None = None,
    ) -> None:
        self.id = run_id
        self.name = name
        self.group = group
        self.project = project
        self.entity = entity
        self.created_at = created_at
        self.summary = MockSummary(summary or {})


class TestExtractRunMetrics:
    """Tests for extract_run_metrics."""

    def test_extracts_all_accuracy_types(self) -> None:
        """extract_run_metrics extracts hard, soft, and grader accuracy."""
        summary = {
            "benchmark/test/accuracy_hard": 0.85,
            "benchmark/test/accuracy_soft": 0.90,
            "benchmark/test/accuracy_grader": 0.88,
        }
        run = MockWandBRun(summary=summary)
        metrics = pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="test")
        assert metrics.run_id == "run-123"
        assert metrics.run_name == "test-run"
        assert metrics.accuracy["hard"].value == 0.85
        assert metrics.accuracy["soft"].value == 0.90
        assert metrics.accuracy["grader"].value == 0.88

    def test_handles_missing_metrics(self) -> None:
        """extract_run_metrics returns empty dict entries for missing metrics."""
        summary = {"benchmark/test/accuracy_hard": 0.75}
        run = MockWandBRun(summary=summary)
        metrics = pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="test")
        assert metrics.accuracy["hard"].value == 0.75
        assert "soft" not in metrics.accuracy
        assert "grader" not in metrics.accuracy

    def test_uses_correct_subset_prefix(self) -> None:
        """extract_run_metrics uses subset_name in metric prefix."""
        summary = {
            "benchmark/valid/accuracy_hard": 0.70,
            "benchmark/test/accuracy_hard": 0.80,
        }
        run = MockWandBRun(summary=summary)
        val_metrics = pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="valid")
        test_metrics = pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="test")
        assert val_metrics.accuracy["hard"].value == 0.70
        assert test_metrics.accuracy["hard"].value == 0.80

    def test_extracts_accuracy_cis(self) -> None:
        """extract_run_metrics extracts CI bounds into MetricWithCI."""
        summary = {
            "benchmark/test/accuracy_hard": 0.80,
            "benchmark/test/accuracy_hard_ci_lower": 0.72,
            "benchmark/test/accuracy_hard_ci_upper": 0.87,
        }
        run = MockWandBRun(summary=summary)
        metrics = pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="test")
        hard = metrics.accuracy["hard"]
        assert hard.value == 0.80
        assert hard.ci_lower == 0.72
        assert hard.ci_upper == 0.87

    def test_extracts_pass_at_k_with_cis(self) -> None:
        """extract_run_metrics extracts Pass@K metrics as dict[int, dict[MatchType, MetricWithCI]]."""
        summary = {
            "benchmark/test/pass_at_1_hard": 0.80,
            "benchmark/test/pass_at_1_hard_ci_lower": 0.72,
            "benchmark/test/pass_at_1_hard_ci_upper": 0.87,
            "benchmark/test/pass_at_5_soft": 0.95,
        }
        run = MockWandBRun(summary=summary)
        metrics = pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="test")
        assert 1 in metrics.pass_at_k
        assert metrics.pass_at_k[1]["hard"].value == 0.80
        assert metrics.pass_at_k[1]["hard"].ci_lower == 0.72
        assert metrics.pass_at_k[1]["hard"].ci_upper == 0.87
        assert 5 in metrics.pass_at_k
        assert metrics.pass_at_k[5]["soft"].value == 0.95


class TestKeywordPresence:
    """Tests for keyword_presence extraction edge cases."""

    def test_keyword_presence_zero_when_no_keywords_found(self) -> None:
        """keyword_presence is 0.0 (not None) when has_keyword_count=0 and sample_count is known."""
        summary = {
            "benchmark/test/accuracy_hard": 0.80,
            "benchmark/test/sample_count": 100,
        }
        run = MockWandBRun(summary=summary)
        metrics = pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="test")
        assert metrics.keyword_presence == 0.0

    def test_keyword_presence_none_when_sample_count_unknown(self) -> None:
        """keyword_presence is None when sample_count is not available and has_keyword_count=0."""
        summary = {"benchmark/test/accuracy_hard": 0.80}
        run = MockWandBRun(summary=summary)
        metrics = pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="test")
        assert metrics.keyword_presence is None

    def test_keyword_presence_raises_when_positive_but_no_sample_count(self) -> None:
        """keyword_presence raises ValueError when has_keyword_count>0 but sample_count is None."""
        summary = {
            "benchmark/test/accuracy_hard": 0.80,
            "benchmark/test/has_keyword/true/sample_count": 5,
        }
        run = MockWandBRun(summary=summary)
        with pytest.raises(ValueError, match="sample_count is None"):
            pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="test")

    def test_keyword_presence_zero_when_sample_count_zero(self) -> None:
        """keyword_presence is 0.0 when sample_count=0 and has_keyword_count=0 (empty subset)."""
        summary = {
            "benchmark/test/accuracy_hard": 0.0,
            "benchmark/test/sample_count": 0,
        }
        run = MockWandBRun(summary=summary)
        metrics = pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="test")
        assert metrics.keyword_presence == 0.0

    def test_keyword_presence_raises_when_positive_but_sample_count_zero(self) -> None:
        """keyword_presence raises ValueError when has_keyword_count>0 but sample_count=0."""
        summary = {
            "benchmark/test/sample_count": 0,
            "benchmark/test/has_keyword/true/sample_count": 3,
        }
        run = MockWandBRun(summary=summary)
        with pytest.raises(ValueError, match="sample_count=0"):
            pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="test")


class TestExtractCategoryMetrics:
    """Tests for extract_category_metrics."""

    def test_extracts_category_metrics(self) -> None:
        """extract_category_metrics parses category-wise metrics from summary."""
        summary = {
            "benchmark/test/code_type/original/accuracy_hard": 0.90,
            "benchmark/test/code_type/original/accuracy_soft": 0.92,
            "benchmark/test/code_type/original/count": 50,
            "benchmark/test/predict_type/output/accuracy_hard": 0.75,
            "benchmark/test/predict_type/output/count": 30,
        }
        run = MockWandBRun(summary=summary)
        categories = pyine.evals.code_exec.analysis.extract_category_metrics(run, subset_name="test")
        assert len(categories) == 2
        code_type_cat = next(c for c in categories if c.category == "code_type/original")
        assert code_type_cat.accuracy["hard"].value == 0.90
        assert code_type_cat.accuracy["soft"].value == 0.92
        assert code_type_cat.sample_count == 50
        predict_type_cat = next(c for c in categories if c.category == "predict_type/output")
        assert predict_type_cat.accuracy["hard"].value == 0.75
        assert predict_type_cat.sample_count == 30

    def test_returns_empty_for_no_categories(self) -> None:
        """extract_category_metrics returns empty list when no category metrics found."""
        summary = {"benchmark/test/accuracy_hard": 0.80}
        run = MockWandBRun(summary=summary)
        categories = pyine.evals.code_exec.analysis.extract_category_metrics(run, subset_name="test")
        assert categories == []

    def test_handles_grader_accuracy(self) -> None:
        """extract_category_metrics includes grader accuracy when present."""
        summary = {
            "benchmark/test/tags/augment/obfuscated/accuracy_hard": 0.60,
            "benchmark/test/tags/augment/obfuscated/accuracy_grader": 0.65,
            "benchmark/test/tags/augment/obfuscated/count": 20,
        }
        run = MockWandBRun(summary=summary)
        categories = pyine.evals.code_exec.analysis.extract_category_metrics(run, subset_name="test")
        assert len(categories) == 1
        cat = categories[0]
        assert cat.category == "tags/augment/obfuscated"
        assert cat.accuracy["hard"].value == 0.60
        assert cat.accuracy["grader"].value == 0.65
        assert cat.sample_count == 20

    def test_captures_structured_accuracy_pass_at_k_and_extra_metrics(self) -> None:
        """extract_category_metrics routes accuracy, pass@k, and other metrics to correct fields."""
        summary = {
            "benchmark/test/code_type/original/accuracy_hard": 0.80,
            "benchmark/test/code_type/original/accuracy_hard_ci_lower": 0.70,
            "benchmark/test/code_type/original/accuracy_hard_ci_upper": 0.88,
            "benchmark/test/code_type/original/pass_at_1_hard": 0.80,
            "benchmark/test/code_type/original/pass_at_1_hard_ci_lower": 0.72,
            "benchmark/test/code_type/original/pass_at_1_hard_ci_upper": 0.87,
            "benchmark/test/code_type/original/majority_correct_hard": 0.75,
            "benchmark/test/code_type/original/mean_output_diversity": 0.45,
            "benchmark/test/code_type/original/sample_count": 50,
            "benchmark/test/code_type/original/attempt_count": 150,
        }
        run = MockWandBRun(summary=summary)
        categories = pyine.evals.code_exec.analysis.extract_category_metrics(run, subset_name="test")
        assert len(categories) == 1
        cat = categories[0]
        # accuracy goes into structured accuracy dict
        assert cat.accuracy["hard"].value == 0.80
        assert cat.accuracy["hard"].ci_lower == pytest.approx(0.70)
        assert cat.accuracy["hard"].ci_upper == pytest.approx(0.88)
        assert cat.sample_count == 50
        assert cat.attempt_count == 150
        # pass@k goes into structured pass_at_k dict
        assert 1 in cat.pass_at_k
        assert cat.pass_at_k[1]["hard"].value == pytest.approx(0.80)
        assert cat.pass_at_k[1]["hard"].ci_lower == pytest.approx(0.72)
        assert cat.pass_at_k[1]["hard"].ci_upper == pytest.approx(0.87)
        # remaining metrics go into extra_metrics
        assert cat.extra_metrics["majority_correct_hard"].value == pytest.approx(0.75)
        assert cat.extra_metrics["mean_output_diversity"].value == pytest.approx(0.45)
        # accuracy and pass@k should NOT appear in extra_metrics
        assert "accuracy_hard" not in cat.extra_metrics
        assert "pass_at_1_hard" not in cat.extra_metrics

    def test_excludes_token_usage_and_complexity_from_categories(self) -> None:
        """extract_category_metrics excludes token_usage and complexity sub-keys as fake categories."""
        summary = {
            # real category metric
            "benchmark/test/code_type/python/accuracy_hard": 0.80,
            "benchmark/test/code_type/python/sample_count": 50,
            # non-category: complexity sub-namespace on a real category
            "benchmark/test/code_type/python/complexity/loc_mean": 25.0,
            "benchmark/test/code_type/python/complexity/loc_median": 20.0,
            # non-category: token usage sub-namespace on a real category
            "benchmark/test/code_type/python/attempt_token_usage/mean_total_tokens_mean": 100.0,
            # non-category: top-level token usage
            "benchmark/test/attempt_token_usage/mean_total_tokens_mean": 150.0,
            # non-category: top-level complexity
            "benchmark/test/complexity/loc_mean": 30.0,
        }
        run = MockWandBRun(summary=summary)
        categories = pyine.evals.code_exec.analysis.extract_category_metrics(run, subset_name="test")
        category_names = [c.category for c in categories]
        assert "code_type/python" in category_names
        # none of the spurious categories should appear
        assert "code_type/python/complexity" not in category_names
        assert "code_type/python/attempt_token_usage" not in category_names
        assert "attempt_token_usage" not in category_names
        assert "complexity" not in category_names
        assert len(categories) == 1


class TestSummarizeRunsToDataframe:
    """Tests for summarize_runs_to_dataframe."""

    def test_creates_dataframe_with_correct_columns(self) -> None:
        """summarize_runs_to_dataframe creates DataFrame with expected columns."""
        summaries = [
            pyine.evals.code_exec.analysis.EvalRunSummary(
                run_info=pyine.evals.code_exec.analysis.RunMetrics(
                    run_id="r1",
                    run_name="run-1",
                    run_group="group",
                    project="proj",
                    entity="entity",
                    created_at="2025-01-01T00:00:00",
                    subset_name="test",
                    accuracy={
                        "hard": pyine.evals.code_exec.analysis.MetricWithCI(0.80),
                        "soft": pyine.evals.code_exec.analysis.MetricWithCI(0.85),
                        "grader": pyine.evals.code_exec.analysis.MetricWithCI(0.82),
                    },
                ),
                category_metrics=[],
            ),
            pyine.evals.code_exec.analysis.EvalRunSummary(
                run_info=pyine.evals.code_exec.analysis.RunMetrics(
                    run_id="r2",
                    run_name="run-2",
                    run_group="group",
                    project="proj",
                    entity="entity",
                    created_at="2025-01-02T00:00:00",
                    subset_name="test",
                    accuracy={"hard": pyine.evals.code_exec.analysis.MetricWithCI(0.75)},
                ),
                category_metrics=[],
            ),
        ]
        df = pyine.evals.code_exec.analysis.summarize_runs_to_dataframe(summaries)
        assert len(df) == 2
        assert "run_id" in df.columns
        assert "accuracy_hard" in df.columns
        assert df.loc[0, "accuracy_hard"] == 0.80
        assert df.loc[1, "accuracy_hard"] == 0.75

    def test_handles_empty_list(self) -> None:
        """summarize_runs_to_dataframe handles empty input."""
        df = pyine.evals.code_exec.analysis.summarize_runs_to_dataframe([])
        assert len(df) == 0


class TestFetchEvalSummary:
    """Tests for fetch_eval_summary."""

    def test_combines_all_metrics(self) -> None:
        """fetch_eval_summary combines run, category, and complexity metrics."""
        summary_dict = {
            "benchmark/test/accuracy_hard": 0.85,
            "benchmark/test/accuracy_soft": 0.90,
            "benchmark/test/code_type/original/accuracy_hard": 0.88,
            "benchmark/test/code_type/original/count": 100,
        }
        run = MockWandBRun(summary=summary_dict)
        eval_summary = pyine.evals.code_exec.analysis.fetch_eval_summary(run, subset_name="test")
        assert eval_summary.run_info.accuracy["hard"].value == 0.85
        assert eval_summary.run_info.accuracy["soft"].value == 0.90
        assert len(eval_summary.category_metrics) == 1
        assert eval_summary.category_metrics[0].category == "code_type/original"


class TestFilterSamplesDataframe:
    """Tests for filter_samples_dataframe."""

    @pytest.fixture
    def sample_df(self) -> "pyine.evals.code_exec.analysis.pd.DataFrame":
        """Create a sample DataFrame for testing."""
        import pandas as pd

        return pd.DataFrame(
            {
                "identifier": ["s1", "s2", "s3", "s4"],
                "code_type": ["original", "original", "obfuscated", "obfuscated"],
                "predict_type": ["program_output", "partial_output", "program_output", "partial_output"],
                "hard_match": [1, 0, 1, 0],
                "loc": [10, 20, 30, 40],
            }
        )

    def test_filters_by_code_type(
        self,
        sample_df: "pyine.evals.code_exec.analysis.pd.DataFrame",
    ) -> None:
        """filter_samples_dataframe filters by code_type."""
        filtered = pyine.evals.code_exec.analysis.filter_samples_dataframe(
            sample_df,
            code_type="original",
        )
        assert len(filtered) == 2
        assert all(filtered["code_type"] == "original")

    def test_filters_by_predict_type(
        self,
        sample_df: "pyine.evals.code_exec.analysis.pd.DataFrame",
    ) -> None:
        """filter_samples_dataframe filters by predict_type."""
        filtered = pyine.evals.code_exec.analysis.filter_samples_dataframe(
            sample_df,
            predict_type="program_output",
        )
        assert len(filtered) == 2
        assert all(filtered["predict_type"] == "program_output")

    def test_filters_by_both(
        self,
        sample_df: "pyine.evals.code_exec.analysis.pd.DataFrame",
    ) -> None:
        """filter_samples_dataframe filters by both code_type and predict_type."""
        filtered = pyine.evals.code_exec.analysis.filter_samples_dataframe(
            sample_df,
            code_type="original",
            predict_type="program_output",
        )
        assert len(filtered) == 1
        assert filtered.iloc[0]["identifier"] == "s1"

    def test_no_filter_returns_all(
        self,
        sample_df: "pyine.evals.code_exec.analysis.pd.DataFrame",
    ) -> None:
        """filter_samples_dataframe returns all rows when no filters specified."""
        filtered = pyine.evals.code_exec.analysis.filter_samples_dataframe(sample_df)
        assert len(filtered) == 4


class TestComputeBinnedAccuracy:
    """Tests for compute_binned_accuracy."""

    @pytest.fixture
    def sample_df(self) -> "pyine.evals.code_exec.analysis.pd.DataFrame":
        """Create a sample DataFrame for testing."""
        import pandas as pd

        return pd.DataFrame(
            {
                "loc": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                "hard_match": [1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
            }
        )

    def test_computes_binned_accuracy(
        self,
        sample_df: "pyine.evals.code_exec.analysis.pd.DataFrame",
    ) -> None:
        """compute_binned_accuracy returns bin centers, means, and counts."""
        bin_centers, accuracy_means, sample_counts = pyine.evals.code_exec.analysis.compute_binned_accuracy(
            sample_df,
            complexity_metric="loc",
            accuracy_column="hard_match",
            num_bins=5,
        )
        assert len(bin_centers) > 0
        assert len(accuracy_means) == len(bin_centers)
        assert len(sample_counts) == len(bin_centers)
        assert all(0 <= m <= 1 for m in accuracy_means)

    def test_raises_for_missing_metric(
        self,
        sample_df: "pyine.evals.code_exec.analysis.pd.DataFrame",
    ) -> None:
        """compute_binned_accuracy raises for unknown complexity metric."""
        with pytest.raises(ValueError, match="not found"):
            pyine.evals.code_exec.analysis.compute_binned_accuracy(
                sample_df,
                complexity_metric="unknown_metric",
            )

    def test_handles_empty_dataframe(self) -> None:
        """compute_binned_accuracy returns empty arrays for empty DataFrame."""
        import pandas as pd

        empty_df = pd.DataFrame({"loc": [], "hard_match": []})
        bin_centers, accuracy_means, sample_counts = pyine.evals.code_exec.analysis.compute_binned_accuracy(
            empty_df,
            complexity_metric="loc",
        )
        assert len(bin_centers) == 0
        assert len(accuracy_means) == 0
        assert len(sample_counts) == 0


class TestDifficultyExcludedFromCategories:
    """Regression: difficulty stat keys must not create false categories."""

    def test_difficulty_stat_keys_do_not_create_category(self) -> None:
        """score_* keys under difficulty/ are excluded as metric-namespace keys."""
        summary = {
            "benchmark/test/code_type/original/accuracy_hard": 0.80,
            "benchmark/test/code_type/original/sample_count": 50,
            "benchmark/test/code_type/original/difficulty/score_mean": 0.5,
            "benchmark/test/code_type/original/difficulty/score_median": 0.4,
        }
        run = MockWandBRun(summary=summary)
        categories = pyine.evals.code_exec.analysis.extract_category_metrics(run, subset_name="test")
        category_names = [cat.category for cat in categories]
        assert "code_type/original" in category_names
        assert "code_type/original/difficulty" not in category_names
        assert len(categories) == 1

    def test_real_difficulty_category_is_preserved(self) -> None:
        """A real category named 'difficulty' with actual metrics is not excluded."""
        summary = {
            "benchmark/test/tags/difficulty/accuracy_hard": 0.70,
            "benchmark/test/tags/difficulty/sample_count": 30,
        }
        run = MockWandBRun(summary=summary)
        categories = pyine.evals.code_exec.analysis.extract_category_metrics(run, subset_name="test")
        category_names = [cat.category for cat in categories]
        assert "tags/difficulty" in category_names
        assert len(categories) == 1
        assert categories[0].accuracy["hard"].value == 0.70


class TestEvalResultToSummary:
    """Tests for eval_result_to_summary (code-exec pickle path)."""

    @staticmethod
    def _make_eval_result(
        metrics: dict[str, typing.Any] | None = None,
        eval_metadata: dict[str, typing.Any] | None = None,
    ) -> pyine.evals.code_exec.analysis.pyine.evals.code_exec.utils.CodeExecEvalResult:
        import pyine.evals.code_exec.utils

        default_metrics: dict[str, typing.Any] = {
            "accuracy_hard": 0.80,
            "accuracy_hard_ci_lower": 0.72,
            "accuracy_hard_ci_upper": 0.87,
            "accuracy_soft": 0.85,
            "sample_count": 100,
            "attempt_count": 100,
        }
        if metrics:
            default_metrics.update(metrics)
        default_metadata: dict[str, typing.Any] = {
            "eval_subset_name": "test",
            "model_name": "test-model",
        }
        if eval_metadata:
            default_metadata.update(eval_metadata)
        return pyine.evals.code_exec.utils.CodeExecEvalResult(
            metrics=default_metrics,
            eval_metadata=default_metadata,
            artifacts=[],
            category_to_identifiers={},
        )

    def test_basic_summary_from_eval_result(self) -> None:
        result = self._make_eval_result()
        summary = pyine.evals.code_exec.analysis.eval_result_to_summary(result)
        assert summary.run_info.accuracy["hard"].value == 0.80
        assert summary.run_info.accuracy["hard"].ci_lower == pytest.approx(0.72)
        assert summary.run_info.accuracy["hard"].ci_upper == pytest.approx(0.87)
        assert summary.run_info.accuracy["soft"].value == 0.85
        assert summary.run_info.sample_count == 100
        assert summary.run_info.subset_name == "test"

    def test_with_category_metrics(self) -> None:
        metrics = {
            "accuracy_hard": 0.80,
            "sample_count": 100,
            "code_type/original/accuracy_hard": 0.90,
            "code_type/original/sample_count": 60,
            "code_type/obfuscated/accuracy_hard": 0.65,
            "code_type/obfuscated/sample_count": 40,
        }
        result = self._make_eval_result(metrics=metrics)
        summary = pyine.evals.code_exec.analysis.eval_result_to_summary(result)
        assert len(summary.category_metrics) == 2
        cat_names = [cat.category for cat in summary.category_metrics]
        assert "code_type/obfuscated" in cat_names
        assert "code_type/original" in cat_names

    def test_with_complexity_metrics(self) -> None:
        import pyine.evals.constants
        import pyine.utils.code.complexity_metrics

        metrics: dict[str, typing.Any] = {"accuracy_hard": 0.80, "sample_count": 100}
        for metric_name in pyine.utils.code.complexity_metrics.COMPLEXITY_METRICS:
            for stat_name in pyine.evals.constants.AGGREGATION_STAT_NAMES:
                metrics[f"complexity/{metric_name}_{stat_name}"] = 1.0
        result = self._make_eval_result(metrics=metrics)
        summary = pyine.evals.code_exec.analysis.eval_result_to_summary(result)
        assert summary.complexity_metrics is not None

    def test_label_overrides(self, tmp_path: "pyine.evals.code_exec.analysis.pathlib.Path") -> None:
        source = tmp_path / "result.pkl"
        source.write_bytes(b"dummy")
        result = self._make_eval_result()
        summary = pyine.evals.code_exec.analysis.eval_result_to_summary(
            result, source_path=source, run_name="custom", run_group="grp"
        )
        assert summary.run_info.run_name == "custom"
        assert summary.run_info.run_group == "grp"

    def test_subset_name_mismatch_raises(self) -> None:
        result = self._make_eval_result()
        with pytest.raises(ValueError, match="does not match"):
            pyine.evals.code_exec.analysis.eval_result_to_summary(result, subset_name="wrong")

    def test_missing_subset_name_raises(self) -> None:
        result = self._make_eval_result(eval_metadata={"eval_subset_name": None})
        with pytest.raises(ValueError, match="must be provided"):
            pyine.evals.code_exec.analysis.eval_result_to_summary(result)

    def test_contract_with_wandb_extractors(self) -> None:
        """Verify pickle path and prefixed-dict-via-adapter produce metric-equivalent summaries."""
        import pyine.evals.constants
        import pyine.utils.code.complexity_metrics

        metrics: dict[str, typing.Any] = {
            "accuracy_hard": 0.80,
            "accuracy_hard_ci_lower": 0.72,
            "accuracy_hard_ci_upper": 0.87,
            "accuracy_soft": 0.85,
            "sample_count": 100,
            "attempt_count": 100,
            "pass_at_1_hard": 0.79,
            "pass_at_1_hard_ci_lower": 0.70,
            "pass_at_1_hard_ci_upper": 0.86,
            "majority_correct_hard": 0.75,
            "code_type/original/accuracy_hard": 0.90,
            "code_type/original/accuracy_hard_ci_lower": 0.82,
            "code_type/original/accuracy_hard_ci_upper": 0.95,
            "code_type/original/sample_count": 60,
            "code_type/original/attempt_count": 60,
            "code_type/original/pass_at_1_hard": 0.88,
            "code_type/obfuscated/accuracy_hard": 0.65,
            "code_type/obfuscated/sample_count": 40,
        }
        # add complexity metrics
        for metric_name in pyine.utils.code.complexity_metrics.COMPLEXITY_METRICS:
            for stat_name in pyine.evals.constants.AGGREGATION_STAT_NAMES:
                metrics[f"complexity/{metric_name}_{stat_name}"] = 1.0
        result = self._make_eval_result(metrics=metrics)
        pickle_summary = pyine.evals.code_exec.analysis.eval_result_to_summary(result)
        # build the same via mock W&B run with prefixed keys
        prefixed = {f"benchmark/test/{key}": value for key, value in metrics.items()}
        mock_run = MockWandBRun(summary=prefixed)
        wandb_summary = pyine.evals.code_exec.analysis.fetch_eval_summary(mock_run, "test")
        # compare run-level accuracy (all match types, including CIs)
        for match_type in pickle_summary.run_info.accuracy:
            pickle_acc = pickle_summary.run_info.accuracy[match_type]
            wandb_acc = wandb_summary.run_info.accuracy[match_type]
            assert pickle_acc.value == pytest.approx(wandb_acc.value), f"accuracy mismatch for {match_type}"
            assert pickle_acc.ci_lower == wandb_acc.ci_lower, f"ci_lower mismatch for {match_type}"
            assert pickle_acc.ci_upper == wandb_acc.ci_upper, f"ci_upper mismatch for {match_type}"
        assert set(pickle_summary.run_info.accuracy.keys()) == set(wandb_summary.run_info.accuracy.keys())
        # compare pass@k
        assert set(pickle_summary.run_info.pass_at_k.keys()) == set(wandb_summary.run_info.pass_at_k.keys())
        for k_val in pickle_summary.run_info.pass_at_k:
            for mt in pickle_summary.run_info.pass_at_k[k_val]:
                pickle_pak = pickle_summary.run_info.pass_at_k[k_val][mt]
                wandb_pak = wandb_summary.run_info.pass_at_k[k_val][mt]
                assert pickle_pak.value == pytest.approx(wandb_pak.value), f"pass@{k_val}/{mt} mismatch"
        # compare extra metrics
        assert set(pickle_summary.run_info.extra_metrics.keys()) == set(wandb_summary.run_info.extra_metrics.keys())
        for metric_name in pickle_summary.run_info.extra_metrics:
            assert pickle_summary.run_info.extra_metrics[metric_name].value == pytest.approx(
                wandb_summary.run_info.extra_metrics[metric_name].value
            ), f"extra_metrics mismatch for {metric_name}"
        # compare sample/attempt counts
        assert pickle_summary.run_info.sample_count == wandb_summary.run_info.sample_count
        assert pickle_summary.run_info.attempt_count == wandb_summary.run_info.attempt_count
        # compare category metrics
        pickle_cats = sorted(pickle_summary.category_metrics, key=lambda c: c.category)
        wandb_cats = sorted(wandb_summary.category_metrics, key=lambda c: c.category)
        assert len(pickle_cats) == len(wandb_cats)
        for pickle_cat, wandb_cat in zip(pickle_cats, wandb_cats, strict=True):
            assert pickle_cat.category == wandb_cat.category
            assert pickle_cat.sample_count == wandb_cat.sample_count
            for mt in pickle_cat.accuracy:
                assert pickle_cat.accuracy[mt].value == pytest.approx(wandb_cat.accuracy[mt].value), (
                    f"category {pickle_cat.category} accuracy/{mt} mismatch"
                )
        # compare complexity metrics presence
        assert (pickle_summary.complexity_metrics is not None) == (wandb_summary.complexity_metrics is not None)


class MockWandBFile:
    """Mock wandb File object for testing fetch_sample_metrics_table."""

    def __init__(
        self,
        name: str,
        table_data: dict[str, typing.Any] | None = None,
    ) -> None:
        self.name = name
        self._table_data = table_data

    def download(self, root: str, replace: bool = True) -> "MockWandBFile":
        import json
        import os

        if self._table_data is not None:
            file_path = os.path.join(root, self.name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self._table_data, f)
            self.name = file_path
        return self

    def close(self) -> None:
        pass  # no-op for mock


class MockWandBRunWithFiles(MockWandBRun):
    """Mock wandb Run with file support for testing fetch_sample_metrics_table."""

    def __init__(
        self,
        files: list[MockWandBFile] | None = None,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(**kwargs)
        self._files = files or []

    def files(self) -> list[MockWandBFile]:
        return self._files


class TestFetchSampleMetricsTable:
    """Tests for fetch_sample_metrics_table."""

    def test_returns_none_when_no_files(self) -> None:
        """fetch_sample_metrics_table returns None when no table files found."""
        run = MockWandBRunWithFiles(run_id="r1", files=[])
        result = pyine.evals.code_exec.analysis.fetch_sample_metrics_table(run, subset_name="test")
        assert result is None

    def test_fetches_from_table_file(self) -> None:
        """fetch_sample_metrics_table fetches table from wandb table file."""
        table_data = {
            "columns": ["identifier", "code_type", "hard_match", "loc"],
            "data": [
                ["s1", "original", 1, 10],
                ["s2", "original", 0, 20],
            ],
        }
        file = MockWandBFile(
            name="media/table/benchmark/test/sample_metrics_abc123.table.json",
            table_data=table_data,
        )
        run = MockWandBRunWithFiles(run_id="r1", files=[file])
        result = pyine.evals.code_exec.analysis.fetch_sample_metrics_table(run, subset_name="test")
        assert result is not None
        assert len(result) == 2
        assert "identifier" in result.columns
        assert "hard_match" in result.columns
        assert result.iloc[0]["identifier"] == "s1"
