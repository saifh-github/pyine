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
            "predict/valid/accuracy_hard": 0.70,
            "predict/test/accuracy_hard": 0.80,
        }
        run = MockWandBRun(summary=summary)
        val_metrics = pyine.evals.code_exec.analysis.extract_run_metrics(run, subset_name="valid")
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
                    run_group="group",
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
                    run_group="group",
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
            name="media/table/predict/test/sample_metrics_abc123.table.json",
            table_data=table_data,
        )
        run = MockWandBRunWithFiles(run_id="r1", files=[file])
        result = pyine.evals.code_exec.analysis.fetch_sample_metrics_table(run, subset_name="test")
        assert result is not None
        assert len(result) == 2
        assert "identifier" in result.columns
        assert "hard_match" in result.columns
        assert result.iloc[0]["identifier"] == "s1"

    def test_uses_custom_table_key(self) -> None:
        """fetch_sample_metrics_table matches custom table_key in file name."""
        table_data = {
            "columns": ["identifier", "hard_match"],
            "data": [["s1", 1]],
        }
        file = MockWandBFile(
            name="media/table/custom_key_abc123.table.json",
            table_data=table_data,
        )
        run = MockWandBRunWithFiles(run_id="r1", files=[file])
        result = pyine.evals.code_exec.analysis.fetch_sample_metrics_table(
            run,
            subset_name="test",
            table_key="custom_key",
        )
        assert result is not None
        assert len(result) == 1
