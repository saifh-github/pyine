"""Tests for pyine.evals.analysis_common -- shared analysis utilities."""

from __future__ import annotations

import pytest

import pyine.evals.analysis_common


class MockWandBRun:
    """Minimal mock wandb Run for filter_runs_by_date tests."""

    def __init__(
        self,
        run_id: str = "run-123",
        created_at: str = "2025-01-15T12:00:00",
    ) -> None:
        self.id = run_id
        self.created_at = created_at


class TestFilterRunsByDate:
    """Tests for filter_runs_by_date."""

    @pytest.fixture
    def sample_runs(self) -> list[MockWandBRun]:
        return [
            MockWandBRun(run_id="r1", created_at="2025-01-01T00:00:00"),
            MockWandBRun(run_id="r2", created_at="2025-01-15T00:00:00"),
            MockWandBRun(run_id="r3", created_at="2025-02-01T00:00:00"),
        ]

    def test_filters_by_start_date(self, sample_runs: list[MockWandBRun]) -> None:
        filtered = pyine.evals.analysis_common.filter_runs_by_date(
            sample_runs,
            start_date="2025-01-10T00:00:00",
        )
        assert len(filtered) == 2
        assert all(r.id in ["r2", "r3"] for r in filtered)

    def test_filters_by_end_date(self, sample_runs: list[MockWandBRun]) -> None:
        filtered = pyine.evals.analysis_common.filter_runs_by_date(
            sample_runs,
            end_date="2025-01-20T00:00:00",
        )
        assert len(filtered) == 2
        assert all(r.id in ["r1", "r2"] for r in filtered)

    def test_filters_by_date_range(self, sample_runs: list[MockWandBRun]) -> None:
        filtered = pyine.evals.analysis_common.filter_runs_by_date(
            sample_runs,
            start_date="2025-01-10T00:00:00",
            end_date="2025-01-20T00:00:00",
        )
        assert len(filtered) == 1
        assert filtered[0].id == "r2"

    def test_no_filter_returns_all(self, sample_runs: list[MockWandBRun]) -> None:
        filtered = pyine.evals.analysis_common.filter_runs_by_date(sample_runs)
        assert len(filtered) == 3

    def test_date_only_end_date_includes_fractional_last_second(self) -> None:
        runs = [
            MockWandBRun(run_id="r1", created_at="2025-01-20T23:59:59.500000Z"),
            MockWandBRun(run_id="r2", created_at="2025-01-21T00:00:00.000000Z"),
        ]
        filtered = pyine.evals.analysis_common.filter_runs_by_date(
            runs,
            end_date="2025-01-20",
        )
        assert [run.id for run in filtered] == ["r1"]

    def test_datetime_end_date_not_adjusted(self) -> None:
        """Explicit datetime end_date with T is NOT adjusted to end-of-day."""
        runs = [
            MockWandBRun(run_id="r1", created_at="2025-01-20T00:00:00"),
            MockWandBRun(run_id="r2", created_at="2025-01-20T12:00:00"),
        ]
        filtered = pyine.evals.analysis_common.filter_runs_by_date(
            runs,
            end_date="2025-01-20T00:00:00",
        )
        # only r1 should match; r2 at noon is after the explicit midnight boundary
        assert [run.id for run in filtered] == ["r1"]

    def test_space_separated_datetime_end_date_not_adjusted(self) -> None:
        """Explicit datetime with space separator is NOT adjusted to end-of-day."""
        runs = [
            MockWandBRun(run_id="r1", created_at="2025-01-20T00:00:00"),
            MockWandBRun(run_id="r2", created_at="2025-01-20T12:00:00"),
        ]
        filtered = pyine.evals.analysis_common.filter_runs_by_date(
            runs,
            end_date="2025-01-20 00:00:00",
        )
        # space-separated datetime is still explicit; r2 at noon should be excluded
        assert [run.id for run in filtered] == ["r1"]


class TestMetricWithCI:
    """Tests for MetricWithCI NamedTuple."""

    def test_basic_construction(self) -> None:
        metric = pyine.evals.analysis_common.MetricWithCI(value=0.85)
        assert metric.value == 0.85
        assert metric.ci_lower is None
        assert metric.ci_upper is None

    def test_with_ci_bounds(self) -> None:
        metric = pyine.evals.analysis_common.MetricWithCI(value=0.85, ci_lower=0.80, ci_upper=0.90)
        assert metric.ci_lower == 0.80
        assert metric.ci_upper == 0.90

    def test_none_value_is_truthy(self) -> None:
        """NamedTuples are always truthy, even with None value -- callers must check .value."""
        metric = pyine.evals.analysis_common.MetricWithCI(value=None)
        assert bool(metric) is True  # NamedTuple is always truthy!
        assert metric.value is None

    def test_default_all_none(self) -> None:
        metric = pyine.evals.analysis_common.MetricWithCI()
        assert metric.value is None
        assert metric.ci_lower is None
        assert metric.ci_upper is None
