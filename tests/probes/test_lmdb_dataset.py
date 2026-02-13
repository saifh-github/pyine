"""Tests for the LMDB-to-HuggingFace probe dataset loader."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

import pytest

from pyine.probes.debug_dataset import create_debug_probe_lmdb
from pyine.probes.lmdb_dataset import (
    _parse_lmdb_key,
    load_probe_dataset_from_lmdb,
)

# ---------------------------------------------------------------------------
# _parse_lmdb_key tests
# ---------------------------------------------------------------------------


class TestParseLmdbKey:
    """Tests for LMDB key parsing."""

    def test_simple_key(self) -> None:
        sample_id, gen_count = _parse_lmdb_key("train/sample_0001/3", "train/")
        assert sample_id == "sample_0001"
        assert gen_count == 3

    def test_nested_sample_id(self) -> None:
        """Multi-segment sample IDs (e.g., TACO/train/p000001/s0000) are preserved."""
        sample_id, gen_count = _parse_lmdb_key("train/TACO/train/p000001/s0000/5", "train/")
        assert sample_id == "TACO/train/p000001/s0000"
        assert gen_count == 5

    def test_generation_count_none(self) -> None:
        """'none' generation count is parsed as 0."""
        sample_id, gen_count = _parse_lmdb_key("eval/sample_0001/none", "eval/")
        assert sample_id == "sample_0001"
        assert gen_count == 0

    def test_no_separator_raises(self) -> None:
        """Key with no '/' after prefix raises ValueError."""
        with pytest.raises(ValueError, match="no '/' separator"):
            _parse_lmdb_key("train/flat_key", "train/")

    def test_non_numeric_gen_count_raises(self) -> None:
        """Non-numeric, non-'none' generation count raises ValueError."""
        with pytest.raises(ValueError, match="non-numeric"):
            _parse_lmdb_key("train/sample_001/abc", "train/")


# ---------------------------------------------------------------------------
# load_probe_dataset_from_lmdb tests
# ---------------------------------------------------------------------------


class TestLoadProbeDatasetFromLmdb:
    """Integration tests using debug LMDB."""

    @pytest.fixture
    def debug_lmdb(self, tmp_path: Path) -> Path:
        """Create a debug LMDB for testing."""
        lmdb_path = tmp_path / "debug.lmdb"
        create_debug_probe_lmdb(lmdb_path, n_train=50, n_valid=20, seed=42)
        return lmdb_path

    def test_returns_train_and_valid_splits(self, debug_lmdb: Path) -> None:
        ds = load_probe_dataset_from_lmdb(debug_lmdb)
        assert "train" in ds
        assert "valid" in ds

    def test_columns_present(self, debug_lmdb: Path) -> None:
        ds = load_probe_dataset_from_lmdb(debug_lmdb)
        for split in ("train", "valid"):
            cols = ds[split].column_names
            assert "text" in cols
            assert "label" in cols
            assert "sample_id" in cols

    def test_labels_binary(self, debug_lmdb: Path) -> None:
        ds = load_probe_dataset_from_lmdb(debug_lmdb)
        for split in ("train", "valid"):
            labels = set(ds[split]["label"])
            assert labels.issubset({0, 1})

    def test_split_sizes(self, debug_lmdb: Path) -> None:
        ds = load_probe_dataset_from_lmdb(debug_lmdb)
        assert len(ds["train"]) == 50
        assert len(ds["valid"]) == 20

    def test_max_samples_per_split(self, debug_lmdb: Path) -> None:
        ds = load_probe_dataset_from_lmdb(debug_lmdb, max_samples_per_split=10)
        assert len(ds["train"]) == 10
        assert len(ds["valid"]) == 10

    def test_text_is_nonempty_string(self, debug_lmdb: Path) -> None:
        ds = load_probe_dataset_from_lmdb(debug_lmdb)
        for text in ds["train"]["text"]:
            assert isinstance(text, str)
            assert len(text) > 0

    def test_selection_strategy_latest(self, debug_lmdb: Path) -> None:
        """'latest' strategy works (default) — no error."""
        ds = load_probe_dataset_from_lmdb(debug_lmdb, selection_strategy="latest")
        assert len(ds["train"]) > 0

    def test_selection_strategy_best_reward(self, debug_lmdb: Path) -> None:
        """'best_reward' strategy works with debug data (all have reward_total)."""
        ds = load_probe_dataset_from_lmdb(debug_lmdb, selection_strategy="best_reward")
        assert len(ds["train"]) > 0

    def test_invalid_selection_strategy_raises(self, debug_lmdb: Path) -> None:
        with pytest.raises(ValueError, match="unknown selection_strategy"):
            load_probe_dataset_from_lmdb(debug_lmdb, selection_strategy="invalid")

    def test_wrong_prefix_raises(self, debug_lmdb: Path) -> None:
        """Non-matching prefix results in no records error."""
        with pytest.raises(ValueError, match="no records match"):
            load_probe_dataset_from_lmdb(
                debug_lmdb,
                train_key_prefix="nonexistent/",
            )

    def test_recompute_labels_soft_match(self, debug_lmdb: Path) -> None:
        """recompute_labels=True with soft_match works."""
        ds = load_probe_dataset_from_lmdb(
            debug_lmdb,
            label_metric_key="soft_match/is_match",
            recompute_labels=True,
        )
        labels = set(ds["train"]["label"])
        assert labels.issubset({0, 1})

    def test_recompute_labels_hard_match(self, debug_lmdb: Path) -> None:
        """recompute_labels=True with hard_match works."""
        ds = load_probe_dataset_from_lmdb(
            debug_lmdb,
            label_metric_key="hard_match/is_match",
            recompute_labels=True,
        )
        labels = set(ds["train"]["label"])
        assert labels.issubset({0, 1})

    def test_recompute_labels_unsupported_metric_raises(self, debug_lmdb: Path) -> None:
        """recompute_labels=True with unsupported metric raises ValueError."""
        with pytest.raises(ValueError, match="recompute_labels"):
            load_probe_dataset_from_lmdb(
                debug_lmdb,
                label_metric_key="custom/metric",
                recompute_labels=True,
            )


class TestSkipMalformedRecords:
    """Tests for skip_malformed_records behavior."""

    def test_malformed_record_raises_by_default(self, tmp_path: Path) -> None:
        """Missing required fields raise ValueError when skip_malformed=False."""
        from pyine.data.utils.lmdb_io import LMDBWriter, SerializationConfig, SerializationMethod

        lmdb_path = tmp_path / "bad.lmdb"
        ser = SerializationConfig(method=SerializationMethod.JSON_ZSTD)
        with LMDBWriter(lmdb_path, serialization_config=ser) as writer:
            # Missing prompt and model_output
            writer.put("train/bad_sample/1", {"reward_metrics": {"soft_match/is_match": 1}})
            # Valid record
            writer.put(
                "eval/good_sample/1",
                {
                    "prompt": "p",
                    "model_output": "o",
                    "reward_metrics": {"soft_match/is_match": 1},
                },
            )

        with pytest.raises(ValueError, match="missing prompt or model_output"):
            load_probe_dataset_from_lmdb(lmdb_path, train_key_prefix="train/", valid_key_prefix="eval/")

    def test_malformed_record_skipped_when_enabled(self, tmp_path: Path) -> None:
        """skip_malformed_records=True skips bad records and returns valid ones."""
        from pyine.data.utils.lmdb_io import LMDBWriter, SerializationConfig, SerializationMethod

        lmdb_path = tmp_path / "mixed.lmdb"
        ser = SerializationConfig(method=SerializationMethod.JSON_ZSTD)
        with LMDBWriter(lmdb_path, serialization_config=ser) as writer:
            # Bad record (missing prompt)
            writer.put("train/bad/1", {"model_output": "o", "reward_metrics": {"soft_match/is_match": 0}})
            # Good records
            for i in range(5):
                writer.put(
                    f"train/good_{i}/1",
                    {
                        "prompt": f"prompt {i}",
                        "model_output": f"output {i}",
                        "reward_metrics": {"soft_match/is_match": i % 2},
                    },
                )
            # Valid eval records
            for i in range(3):
                writer.put(
                    f"eval/eval_{i}/1",
                    {
                        "prompt": f"eval prompt {i}",
                        "model_output": f"eval output {i}",
                        "reward_metrics": {"soft_match/is_match": i % 2},
                    },
                )

        ds = load_probe_dataset_from_lmdb(
            lmdb_path,
            skip_malformed_records=True,
        )
        assert len(ds["train"]) == 5  # bad record skipped
