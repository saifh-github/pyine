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
        project: str = "test-project",
        entity: str | None = "test-entity",
        created_at: str = "2025-01-15T12:00:00",
        summary: dict[str, typing.Any] | None = None,
    ) -> None:
        self.id = run_id
        self.name = name
        self.project = project
        self.entity = entity
        self.created_at = created_at
        self.summary = MockSummary(summary or {})


class TestExtractRunMetrics:
    """Tests for extract_run_metrics."""

    def test_extracts_all_accuracy_types(self) -> None:
        """extract_run_metrics extracts hard, soft, and grader accuracy."""
        summary = {
            "predict/test/accuracy_hard": 0.85,
            "predict/test/accuracy_soft": 0.90,
            "predict/test/accuracy_grader": 0.88,
        }
        run = MockWandBRun(summary=summary)
        metrics = pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="test")
        assert metrics.run_id == "run-123"
        assert metrics.run_name == "test-run"
        assert metrics.accuracy_hard == 0.85
        assert metrics.accuracy_soft == 0.90
        assert metrics.accuracy_grader == 0.88

    def test_handles_missing_metrics(self) -> None:
        """extract_run_metrics returns None for missing metrics."""
        summary = {"predict/test/accuracy_hard": 0.75}
        run = MockWandBRun(summary=summary)
        metrics = pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="test")
        assert metrics.accuracy_hard == 0.75
        assert metrics.accuracy_soft is None
        assert metrics.accuracy_grader is None

    def test_uses_correct_subset_prefix(self) -> None:
        """extract_run_metrics uses subset_name in metric prefix."""
        summary = {
            "predict/val/accuracy_hard": 0.70,
            "predict/test/accuracy_hard": 0.80,
        }
        run = MockWandBRun(summary=summary)
        val_metrics = pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="val")
        test_metrics = pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="test")
        assert val_metrics.accuracy_hard == 0.70
        assert test_metrics.accuracy_hard == 0.80


class TestExtractCategoryMetrics:
    """Tests for extract_category_metrics."""

    def test_extracts_category_metrics(self) -> None:
        """extract_category_metrics parses category-wise metrics from summary."""
        summary = {
            "predict/test/code_type/original/accuracy_hard": 0.90,
            "predict/test/code_type/original/accuracy_soft": 0.92,
            "predict/test/code_type/original/count": 50,
            "predict/test/predict_type/output/accuracy_hard": 0.75,
            "predict/test/predict_type/output/count": 30,
        }
        run = MockWandBRun(summary=summary)
        categories = pyine.evals.code_exec.analysis.extract_category_metrics(run, subset_name="test")
        assert len(categories) == 2
        code_type_cat = next(c for c in categories if c.category == "code_type/original")
        assert code_type_cat.accuracy_hard == 0.90
        assert code_type_cat.accuracy_soft == 0.92
        assert code_type_cat.count == 50
        predict_type_cat = next(c for c in categories if c.category == "predict_type/output")
        assert predict_type_cat.accuracy_hard == 0.75
        assert predict_type_cat.count == 30

    def test_returns_empty_for_no_categories(self) -> None:
        """extract_category_metrics returns empty list when no category metrics found."""
        summary = {"predict/test/accuracy_hard": 0.80}
        run = MockWandBRun(summary=summary)
        categories = pyine.evals.code_exec.analysis.extract_category_metrics(run, subset_name="test")
        assert categories == []

    def test_handles_grader_accuracy(self) -> None:
        """extract_category_metrics includes grader accuracy when present."""
        summary = {
            "predict/test/tags/augment/obfuscated/accuracy_hard": 0.60,
            "predict/test/tags/augment/obfuscated/accuracy_grader": 0.65,
            "predict/test/tags/augment/obfuscated/count": 20,
        }
        run = MockWandBRun(summary=summary)
        categories = pyine.evals.code_exec.analysis.extract_category_metrics(run, subset_name="test")
        assert len(categories) == 1
        cat = categories[0]
        assert cat.category == "tags/augment/obfuscated"
        assert cat.accuracy_hard == 0.60
        assert cat.accuracy_grader == 0.65


class TestFilterRunsByDate:
    """Tests for filter_runs_by_date."""

    @pytest.fixture
    def sample_runs(self) -> list[MockWandBRun]:
        """Create sample runs with different dates."""
        return [
            MockWandBRun(run_id="r1", created_at="2025-01-01T00:00:00"),
            MockWandBRun(run_id="r2", created_at="2025-01-15T00:00:00"),
            MockWandBRun(run_id="r3", created_at="2025-02-01T00:00:00"),
        ]

    def test_filters_by_start_date(self, sample_runs: list[MockWandBRun]) -> None:
        """filter_runs_by_date excludes runs before start_date."""
        filtered = pyine.evals.code_exec.analysis.filter_runs_by_date(
            sample_runs,
            start_date="2025-01-10T00:00:00",
        )
        assert len(filtered) == 2
        assert all(r.id in ["r2", "r3"] for r in filtered)

    def test_filters_by_end_date(self, sample_runs: list[MockWandBRun]) -> None:
        """filter_runs_by_date excludes runs after end_date."""
        filtered = pyine.evals.code_exec.analysis.filter_runs_by_date(
            sample_runs,
            end_date="2025-01-20T00:00:00",
        )
        assert len(filtered) == 2
        assert all(r.id in ["r1", "r2"] for r in filtered)

    def test_filters_by_date_range(self, sample_runs: list[MockWandBRun]) -> None:
        """filter_runs_by_date filters by both start and end date."""
        filtered = pyine.evals.code_exec.analysis.filter_runs_by_date(
            sample_runs,
            start_date="2025-01-10T00:00:00",
            end_date="2025-01-20T00:00:00",
        )
        assert len(filtered) == 1
        assert filtered[0].id == "r2"

    def test_no_filter_returns_all(self, sample_runs: list[MockWandBRun]) -> None:
        """filter_runs_by_date returns all runs when no dates specified."""
        filtered = pyine.evals.code_exec.analysis.filter_runs_by_date(sample_runs)
        assert len(filtered) == 3


class TestSummarizeRunsToDataframe:
    """Tests for summarize_runs_to_dataframe."""

    def test_creates_dataframe_with_correct_columns(self) -> None:
        """summarize_runs_to_dataframe creates DataFrame with expected columns."""
        summaries = [
            pyine.evals.code_exec.analysis.EvalRunSummary(
                run_info=pyine.evals.code_exec.analysis.RunMetrics(
                    run_id="r1",
                    run_name="run-1",
                    project="proj",
                    entity="entity",
                    created_at="2025-01-01T00:00:00",
                    subset_name="test",
                    accuracy_hard=0.80,
                    accuracy_soft=0.85,
                    accuracy_grader=0.82,
                ),
                category_metrics=[],
            ),
            pyine.evals.code_exec.analysis.EvalRunSummary(
                run_info=pyine.evals.code_exec.analysis.RunMetrics(
                    run_id="r2",
                    run_name="run-2",
                    project="proj",
                    entity="entity",
                    created_at="2025-01-02T00:00:00",
                    subset_name="test",
                    accuracy_hard=0.75,
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
            "predict/test/accuracy_hard": 0.85,
            "predict/test/accuracy_soft": 0.90,
            "predict/test/code_type/original/accuracy_hard": 0.88,
            "predict/test/code_type/original/count": 100,
        }
        run = MockWandBRun(summary=summary_dict)
        eval_summary = pyine.evals.code_exec.analysis.fetch_eval_summary(run, subset_name="test")
        assert eval_summary.run_info.accuracy_hard == 0.85
        assert eval_summary.run_info.accuracy_soft == 0.90
        assert len(eval_summary.category_metrics) == 1
        assert eval_summary.category_metrics[0].category == "code_type/original"
