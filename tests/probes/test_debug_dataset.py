"""Tests for the synthetic LMDB debug dataset factory."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

from pyine.probes.debug_dataset import create_debug_probe_dataset, create_debug_probe_lmdb


class TestCreateDebugProbeLmdb:
    """Tests for the LMDB creation function."""

    def test_creates_lmdb_directory(self, tmp_path: Path) -> None:
        """create_debug_probe_lmdb() creates an LMDB at the given path."""
        lmdb_path = tmp_path / "debug.lmdb"
        result = create_debug_probe_lmdb(lmdb_path, n_train=10, n_valid=5)
        assert result.exists()
        assert result == lmdb_path

    def test_lmdb_readable(self, tmp_path: Path) -> None:
        """Created LMDB can be opened with LMDBReader."""
        from pyine.data.utils.lmdb_io import LMDBReader

        lmdb_path = tmp_path / "debug.lmdb"
        create_debug_probe_lmdb(lmdb_path, n_train=10, n_valid=5)
        with LMDBReader(lmdb_path) as reader:
            assert len(reader.key_map) == 15  # 10 train + 5 valid

    def test_records_have_required_fields(self, tmp_path: Path) -> None:
        """Each record has prompt, model_output, reward_metrics, etc."""
        from pyine.data.utils.lmdb_io import LMDBReader

        lmdb_path = tmp_path / "debug.lmdb"
        create_debug_probe_lmdb(lmdb_path, n_train=5, n_valid=3)
        with LMDBReader(lmdb_path) as reader:
            for key in reader.key_map:
                record = reader.get(key)
                assert "prompt" in record
                assert "model_output" in record
                assert "expected_output" in record
                assert "reward_metrics" in record
                assert "soft_match/is_match" in record["reward_metrics"]
                assert "hard_match/is_match" in record["reward_metrics"]

    def test_deterministic_with_seed(self, tmp_path: Path) -> None:
        """Same seed produces identical LMDB content."""
        from pyine.data.utils.lmdb_io import LMDBReader

        lmdb1 = tmp_path / "lmdb1"
        lmdb2 = tmp_path / "lmdb2"
        create_debug_probe_lmdb(lmdb1, n_train=10, n_valid=5, seed=123)
        create_debug_probe_lmdb(lmdb2, n_train=10, n_valid=5, seed=123)
        with LMDBReader(lmdb1) as r1, LMDBReader(lmdb2) as r2:
            assert set(r1.key_map.keys()) == set(r2.key_map.keys())
            for key in r1.key_map:
                assert r1.get(key) == r2.get(key)


class TestCreateDebugProbeDataset:
    """Tests for the convenience wrapper returning a DatasetDict."""

    def test_dataset_has_required_splits(self) -> None:
        """Output has 'train' and 'valid' splits."""
        ds = create_debug_probe_dataset(n_train=20, n_valid=10)
        assert "train" in ds
        assert "valid" in ds

    def test_dataset_has_required_columns(self) -> None:
        """Each split has 'text', 'label', and 'sample_id' columns."""
        ds = create_debug_probe_dataset(n_train=20, n_valid=10)
        for split_name in ("train", "valid"):
            cols = ds[split_name].column_names
            assert "text" in cols, f"Missing 'text' in {split_name}"
            assert "label" in cols, f"Missing 'label' in {split_name}"
            assert "sample_id" in cols, f"Missing 'sample_id' in {split_name}"

    def test_labels_are_binary(self) -> None:
        """All labels are 0 or 1."""
        ds = create_debug_probe_dataset(n_train=100, n_valid=30)
        for split_name in ("train", "valid"):
            labels = set(ds[split_name]["label"])
            assert labels.issubset({0, 1}), f"Non-binary labels in {split_name}: {labels}"

    def test_both_labels_present_in_train(self) -> None:
        """Train split contains both label=0 and label=1 samples."""
        ds = create_debug_probe_dataset(n_train=100, n_valid=30)
        labels = set(ds["train"]["label"])
        assert labels == {0, 1}, f"Expected {{0, 1}}, got {labels}"

    def test_text_is_prompt_plus_output(self) -> None:
        """Text field is plain string (prompt + model_output), not chat format."""
        ds = create_debug_probe_dataset(n_train=20, n_valid=10)
        for sample in ds["train"]:
            text = sample["text"]
            assert isinstance(text, str)
            assert len(text) > 0

    def test_sample_counts(self) -> None:
        """n_train and n_valid control split sizes."""
        ds = create_debug_probe_dataset(n_train=42, n_valid=13)
        assert len(ds["train"]) == 42
        assert len(ds["valid"]) == 13

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces identical datasets."""
        ds1 = create_debug_probe_dataset(n_train=20, n_valid=10, seed=123)
        ds2 = create_debug_probe_dataset(n_train=20, n_valid=10, seed=123)

        for split in ("train", "valid"):
            for i in range(len(ds1[split])):
                assert ds1[split][i]["label"] == ds2[split][i]["label"]
                assert ds1[split][i]["text"] == ds2[split][i]["text"]
