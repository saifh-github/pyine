"""Tests for pyine.evals.analysis_common -- shared analysis utilities."""

from __future__ import annotations

import datetime
import typing

import pytest

import pyine.evals.analysis_common

if typing.TYPE_CHECKING:
    import pathlib


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


class TestBuildRunInfoFromMetadata:
    """Tests for build_run_info_from_metadata."""

    def test_full_metadata_with_source_path(self, tmp_path: pathlib.Path) -> None:
        source = tmp_path / "result.pkl"
        source.write_bytes(b"dummy")
        metadata = {
            "model_name": "gpt-4o",
            "reprod_metadata": {"time_since_epoch": "1700000000"},
        }
        info = pyine.evals.analysis_common.build_run_info_from_metadata(metadata, "test", source_path=source)
        assert info.run_name == "gpt-4o"
        assert info.run_group == tmp_path.name
        assert info.project == "local"
        assert info.entity is None
        assert info.subset_name == "test"
        # created_at should be from time_since_epoch, parseable as ISO 8601
        parsed = datetime.datetime.fromisoformat(info.created_at)
        assert parsed.year == 2023  # 1700000000 is Nov 2023

    def test_created_at_is_iso_format(self) -> None:
        metadata: dict[str, typing.Any] = {"reprod_metadata": {"time_since_epoch": "1700000000"}}
        info = pyine.evals.analysis_common.build_run_info_from_metadata(metadata, "test")
        parsed = datetime.datetime.fromisoformat(info.created_at)
        assert parsed.tzinfo is not None  # must have timezone

    def test_created_at_priority_time_since_epoch(self, tmp_path: pathlib.Path) -> None:
        source = tmp_path / "result.pkl"
        source.write_bytes(b"dummy")
        metadata: dict[str, typing.Any] = {"reprod_metadata": {"time_since_epoch": "1700000000"}}
        info = pyine.evals.analysis_common.build_run_info_from_metadata(metadata, "test", source_path=source)
        parsed = datetime.datetime.fromisoformat(info.created_at)
        assert abs(parsed.timestamp() - 1700000000) < 1.0

    def test_created_at_falls_back_to_mtime(self, tmp_path: pathlib.Path) -> None:
        source = tmp_path / "result.pkl"
        source.write_bytes(b"dummy")
        metadata: dict[str, typing.Any] = {}  # no reprod_metadata
        info = pyine.evals.analysis_common.build_run_info_from_metadata(metadata, "test", source_path=source)
        parsed = datetime.datetime.fromisoformat(info.created_at)
        assert abs(parsed.timestamp() - source.stat().st_mtime) < 1.0

    def test_created_at_falls_back_to_now(self) -> None:
        metadata: dict[str, typing.Any] = {}
        before = datetime.datetime.now(tz=datetime.UTC)
        info = pyine.evals.analysis_common.build_run_info_from_metadata(metadata, "test")
        after = datetime.datetime.now(tz=datetime.UTC)
        parsed = datetime.datetime.fromisoformat(info.created_at)
        assert before <= parsed <= after

    def test_minimal_metadata_falls_back_to_local(self) -> None:
        metadata: dict[str, typing.Any] = {}
        info = pyine.evals.analysis_common.build_run_info_from_metadata(metadata, "test")
        assert info.run_name == "local"
        assert info.run_group == "local"
        assert info.run_id == "local"  # no source_path -> run_id == run_name

    def test_run_id_unique_with_source_path(self, tmp_path: pathlib.Path) -> None:
        """run_id is a deterministic hash when source_path is provided."""
        source = tmp_path / "result.pkl"
        source.write_bytes(b"dummy")
        metadata = {"model_name": "gpt-4o"}
        info = pyine.evals.analysis_common.build_run_info_from_metadata(metadata, "test", source_path=source)
        assert info.run_id != info.run_name  # hash differs from display name
        assert len(info.run_id) == 12  # 12-char hex hash

    def test_run_id_differs_for_different_files(self, tmp_path: pathlib.Path) -> None:
        """Different pickle files produce different run_ids."""
        source_a = tmp_path / "a.pkl"
        source_b = tmp_path / "b.pkl"
        source_a.write_bytes(b"aa")
        source_b.write_bytes(b"bb")
        metadata: dict[str, typing.Any] = {}
        info_a = pyine.evals.analysis_common.build_run_info_from_metadata(metadata, "test", source_path=source_a)
        info_b = pyine.evals.analysis_common.build_run_info_from_metadata(metadata, "test", source_path=source_b)
        assert info_a.run_id != info_b.run_id

    def test_overrides_take_precedence(self, tmp_path: pathlib.Path) -> None:
        source = tmp_path / "result.pkl"
        source.write_bytes(b"dummy")
        metadata = {"model_name": "gpt-4o"}
        info = pyine.evals.analysis_common.build_run_info_from_metadata(
            metadata, "test", source_path=source, run_name="custom-name", run_group="custom-group"
        )
        assert info.run_name == "custom-name"
        assert info.run_group == "custom-group"

    def test_nonexistent_source_path_raises(self, tmp_path: pathlib.Path) -> None:
        metadata: dict[str, typing.Any] = {}
        with pytest.raises(FileNotFoundError, match="does not exist"):
            pyine.evals.analysis_common.build_run_info_from_metadata(
                metadata, "test", source_path=tmp_path / "nonexistent" / "path.pkl"
            )

    def test_empty_subset_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            pyine.evals.analysis_common.build_run_info_from_metadata({}, "")


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
