"""Tests for the synthetic debug dataset factory."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

import datasets

from pyine.probes.debug_dataset import create_debug_probe_dataset


class TestDebugDataset:
    """Tests for the synthetic debug dataset factory."""

    def test_dataset_has_required_splits(self) -> None:
        """Output has 'train' and 'valid' splits."""
        ds = create_debug_probe_dataset(n_train=20, n_valid=10)
        assert "train" in ds
        assert "valid" in ds

    def test_dataset_has_required_columns(self) -> None:
        """Each split has 'messages' and 'label' columns."""
        ds = create_debug_probe_dataset(n_train=20, n_valid=10)
        for split_name in ("train", "valid"):
            cols = ds[split_name].column_names
            assert "messages" in cols, f"Missing 'messages' in {split_name}"
            assert "label" in cols, f"Missing 'label' in {split_name}"

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

    def test_messages_are_chat_format(self) -> None:
        """Each messages entry is a list of dicts with 'role' and 'content' keys."""
        ds = create_debug_probe_dataset(n_train=20, n_valid=10)
        for sample in ds["train"]:
            msgs = sample["messages"]
            assert isinstance(msgs, list)
            assert len(msgs) >= 1
            for msg in msgs:
                assert "role" in msg
                assert "content" in msg

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
                assert ds1[split][i]["messages"] == ds2[split][i]["messages"]

    def test_save_to_disk_and_reload(self, tmp_path: Path) -> None:
        """Dataset saved to disk can be reloaded with datasets.load_from_disk()."""
        output = tmp_path / "debug-ds"
        create_debug_probe_dataset(output_path=output, n_train=10, n_valid=5)

        reloaded = datasets.load_from_disk(str(output))
        assert set(reloaded.keys()) == {"train", "valid"}
        assert len(reloaded["train"]) == 10
        assert len(reloaded["valid"]) == 5
