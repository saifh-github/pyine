"""Tests for debug dataset correctness eval compatibility.

Verifies that the debug LMDB and split file work with both loading paths:
- Probe/classifier loader (load_probe_dataset_from_lmdb)
- Correctness eval loader (load_records_from_lmdb)
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

import pytest

from pyine.guardrails.data.debug_dataset import (
    DebugDatasetResult,
    create_debug_probe_dataset,
    create_debug_probe_lmdb,
    create_debug_split_file,
)


class TestDebugLmdbCorrectnessFields:
    """Test that generated LMDB records have all fields needed by the correctness eval loader."""

    @pytest.fixture
    def lmdb_path(self, tmp_path: Path) -> Path:
        lmdb_path = tmp_path / "debug.lmdb"
        create_debug_probe_lmdb(lmdb_path, n_train=10, n_eval_families=5, n_test_families=3)
        return lmdb_path

    def test_records_have_correctness_fields(self, lmdb_path: Path) -> None:
        from pyine.data.utils.lmdb_io import LMDBReader

        with LMDBReader(lmdb_path) as reader:
            for key in reader.key_map:
                record = reader.get(key)
                assert "sample_id" in record
                assert "attempt_index" in record
                assert "hard_match" in record
                assert "soft_match" in record
                assert isinstance(record["hard_match"], bool)
                assert isinstance(record["soft_match"], bool)

    def test_sample_ids_are_trace_identifier_format(self, lmdb_path: Path) -> None:
        from pyine.data.utils.lmdb_io import LMDBReader

        with LMDBReader(lmdb_path) as reader:
            for key in reader.key_map:
                record = reader.get(key)
                sid = record["sample_id"]
                # TraceIdentifier format: DATASET/SUBSET/pNNNNNN/sNNNN/tNNNN
                parts = sid.split("/")
                assert parts[0] == "DEBUG"
                assert parts[1] in ("TRAIN", "VALID", "TEST")
                assert parts[2].startswith("p")

    def test_metadata_has_record_type_benchmark(self, lmdb_path: Path) -> None:
        from pyine.data.utils.lmdb_io import LMDBReader

        with LMDBReader(lmdb_path) as reader:
            metadata = reader.get_metadata()
            assert metadata["record_type"] == "benchmark"

    def test_test_families_created(self, lmdb_path: Path) -> None:
        from pyine.data.utils.lmdb_io import LMDBReader

        with LMDBReader(lmdb_path) as reader:
            test_keys = [k for k in reader.key_map if "DEBUG/TEST/" in k]
            # 3 test families x 3 code types = 9 records
            assert len(test_keys) == 9

    def test_total_record_count(self, lmdb_path: Path) -> None:
        from pyine.data.utils.lmdb_io import LMDBReader

        with LMDBReader(lmdb_path) as reader:
            # 10 train + 5*3 eval(valid) + 3*3 test = 10 + 15 + 9 = 34
            assert len(reader.key_map) == 34


class TestDebugLmdbBothLoadingPaths:
    """Verify that the debug LMDB loads via both the probe and correctness loaders."""

    @pytest.fixture
    def lmdb_path(self, tmp_path: Path) -> Path:
        lmdb_path = tmp_path / "debug.lmdb"
        create_debug_probe_lmdb(lmdb_path, n_train=20, n_eval_families=10, n_test_families=5)
        return lmdb_path

    def test_probe_loader_still_works(self, lmdb_path: Path) -> None:
        """Backward compatibility: probe loader works with updated LMDB."""
        from pyine.guardrails.data.datamodule_configs import ProbeDataModuleConfig
        from pyine.guardrails.data.lmdb_dataset import load_probe_dataset_from_lmdb

        config = ProbeDataModuleConfig(lmdb_path=str(lmdb_path))
        ds = load_probe_dataset_from_lmdb(config)
        assert "train" in ds
        assert "valid" in ds

    def test_correctness_loader_works(self, lmdb_path: Path) -> None:
        """Correctness eval loader loads records from the updated LMDB."""
        from pyine.evals.correctness.data_loading import load_records_from_lmdb
        from pyine.evals.correctness.types import LabelType

        records = load_records_from_lmdb([lmdb_path], label_type=LabelType.HARD_MATCH)
        assert len(records) > 0
        for rec in records:
            assert hasattr(rec, "sample_id")
            assert hasattr(rec, "model_output")
            assert hasattr(rec, "expected_output")
            assert hasattr(rec, "label")


class TestDebugSplitFile:
    """Tests for create_debug_split_file()."""

    def test_split_file_creation(self, tmp_path: Path) -> None:
        split_path = tmp_path / "split.json"
        result = create_debug_split_file(
            split_path,
            n_train=20,
            n_eval_families=10,
            n_test_families=5,
        )
        assert result.exists()
        assert result == split_path

    def test_split_file_loadable(self, tmp_path: Path) -> None:
        from pyine.data.utils.splits import SplitResult

        split_path = tmp_path / "split.json"
        create_debug_split_file(
            split_path,
            n_train=20,
            n_eval_families=10,
            n_test_families=5,
        )
        split_result = SplitResult.from_file(split_path)
        assert len(split_result.identifiers) == 35  # 20 + 10 + 5
        assert set(split_result.subset_assignments.values()) == {"train", "valid", "test"}

    def test_split_file_subset_counts(self, tmp_path: Path) -> None:
        from pyine.data.utils.splits import SplitResult

        split_path = tmp_path / "split.json"
        create_debug_split_file(
            split_path,
            n_train=20,
            n_eval_families=10,
            n_test_families=5,
        )
        split_result = SplitResult.from_file(split_path)
        train_count = sum(1 for v in split_result.subset_assignments.values() if v == "train")
        valid_count = sum(1 for v in split_result.subset_assignments.values() if v == "valid")
        test_count = sum(1 for v in split_result.subset_assignments.values() if v == "test")
        assert train_count == 20
        assert valid_count == 10
        assert test_count == 5


class TestBuildGuardrailSplitsWithDebugData:
    """Test that build_guardrail_splits works with debug LMDB + split file."""

    def test_build_guardrail_splits(self, tmp_path: Path) -> None:
        from pyine.evals.correctness.data_loading import load_records_from_lmdb
        from pyine.evals.correctness.splits import build_guardrail_splits
        from pyine.evals.correctness.types import GuardrailSplitConfig, LabelType

        lmdb_path = tmp_path / "debug.lmdb"
        split_path = tmp_path / "split.json"

        create_debug_probe_lmdb(lmdb_path, n_train=20, n_eval_families=10, n_test_families=5)
        create_debug_split_file(
            split_path,
            n_train=20,
            n_eval_families=10,
            n_test_families=5,
        )

        records = load_records_from_lmdb([lmdb_path], label_type=LabelType.HARD_MATCH)
        split_config = GuardrailSplitConfig(split_source=str(split_path))
        splits = build_guardrail_splits(records, split_config)

        assert len(splits.guardrail_valid) > 0
        assert len(splits.guardrail_test) > 0
        # Check that problem sets are disjoint
        assert splits.valid_problem_ids.isdisjoint(splits.test_problem_ids)


class TestCreateDebugProbeDatasetResult:
    """Test the DebugDatasetResult wrapper with include_split_file."""

    def test_returns_debug_dataset_result(self) -> None:
        result = create_debug_probe_dataset(n_train=10, n_eval_families=5, n_test_families=3)
        assert isinstance(result, DebugDatasetResult)
        assert result.dataset is not None
        assert result.split_file_path is None

    def test_include_split_file(self, tmp_path: Path) -> None:
        lmdb_path = tmp_path / "debug.lmdb"
        result = create_debug_probe_dataset(
            output_path=lmdb_path,
            n_train=10,
            n_eval_families=5,
            n_test_families=3,
            include_split_file=True,
        )
        assert result.split_file_path is not None
        assert result.split_file_path.exists()
