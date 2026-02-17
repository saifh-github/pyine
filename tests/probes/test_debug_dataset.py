"""Tests for the synthetic LMDB debug dataset factory."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

import pytest

from pyine.probes.debug_dataset import create_debug_probe_dataset, create_debug_probe_lmdb
from pyine.probes.lmdb_dataset import _extract_family_id


class TestCreateDebugProbeLmdb:
    """Tests for the LMDB creation function."""

    def test_creates_lmdb_directory(self, tmp_path: Path) -> None:
        """create_debug_probe_lmdb() creates an LMDB at the given path."""
        lmdb_path = tmp_path / "debug.lmdb"
        result = create_debug_probe_lmdb(lmdb_path, n_train=10, n_eval_families=5)
        assert result.exists()
        assert result == lmdb_path

    def test_lmdb_readable(self, tmp_path: Path) -> None:
        """Created LMDB can be opened with LMDBReader."""
        from pyine.data.utils.lmdb_io import LMDBReader

        lmdb_path = tmp_path / "debug.lmdb"
        create_debug_probe_lmdb(lmdb_path, n_train=10, n_eval_families=5)
        with LMDBReader(lmdb_path) as reader:
            assert len(reader.key_map) == 25  # 10 train + 5*3 eval

    def test_records_have_required_fields(self, tmp_path: Path) -> None:
        """Each record has prompt, model_output, reward_metrics, etc."""
        from pyine.data.utils.lmdb_io import LMDBReader

        lmdb_path = tmp_path / "debug.lmdb"
        create_debug_probe_lmdb(lmdb_path, n_train=5, n_eval_families=3)
        with LMDBReader(lmdb_path) as reader:
            for key in reader.key_map:
                record = reader.get(key)
                assert "prompt" in record
                assert "model_output" in record
                assert "expected_output" in record
                assert "reward_metrics" in record
                assert "reward/metrics/soft_match/is_match" in record["reward_metrics"]
                assert "reward/metrics/hard_match/is_match" in record["reward_metrics"]

    def test_deterministic_with_seed(self, tmp_path: Path) -> None:
        """Same seed produces identical LMDB content."""
        from pyine.data.utils.lmdb_io import LMDBReader

        lmdb1 = tmp_path / "lmdb1"
        lmdb2 = tmp_path / "lmdb2"
        create_debug_probe_lmdb(lmdb1, n_train=10, n_eval_families=5, seed=123)
        create_debug_probe_lmdb(lmdb2, n_train=10, n_eval_families=5, seed=123)
        with LMDBReader(lmdb1) as r1, LMDBReader(lmdb2) as r2:
            assert set(r1.key_map.keys()) == set(r2.key_map.keys())
            for key in r1.key_map:
                assert r1.get(key) == r2.get(key)


# ---------------------------------------------------------------------------
# Debug LMDB structure tests
# ---------------------------------------------------------------------------


class TestDebugLmdbStructure:
    """Tests for the structural properties of the generated LMDB."""

    @pytest.fixture
    def lmdb_data(self, tmp_path: Path) -> tuple[Path, dict[str, dict[str, typing.Any]]]:
        """Create a debug LMDB and read all records."""
        from pyine.data.utils.lmdb_io import LMDBReader

        lmdb_path = tmp_path / "debug.lmdb"
        create_debug_probe_lmdb(lmdb_path, n_train=10, n_eval_families=10, seed=42)
        with LMDBReader(lmdb_path) as reader:
            records = {key: reader.get(key) for key in reader.key_map}
        return lmdb_path, records

    def test_train_records_all_original(self, lmdb_data: tuple[Path, dict[str, dict[str, typing.Any]]]) -> None:
        """Train-prefix records all have code_type='original'."""
        _, records = lmdb_data
        train_records = {k: v for k, v in records.items() if k.startswith("train/")}
        assert len(train_records) == 10
        for rec in train_records.values():
            assert rec["code_type"] == "original"

    def test_train_records_flat_ids(self, lmdb_data: tuple[Path, dict[str, dict[str, typing.Any]]]) -> None:
        """Train-prefix sample IDs have no /a: augmentation suffix."""
        _, records = lmdb_data
        for key in records:
            if key.startswith("train/"):
                # Strip "train/" and "/1" to get sample_id
                sample_id = key[len("train/") : key.rfind("/")]
                assert "/a:" not in sample_id

    def test_eval_records_have_three_code_types(self, lmdb_data: tuple[Path, dict[str, dict[str, typing.Any]]]) -> None:
        """Eval-prefix records include 'original', 'hinted', and 'misleading' code_type values."""
        _, records = lmdb_data
        eval_records = {k: v for k, v in records.items() if k.startswith("eval/")}
        code_types = {rec["code_type"] for rec in eval_records.values()}
        assert code_types == {"original", "hinted", "misleading"}

    def test_eval_records_per_family_count(self, lmdb_data: tuple[Path, dict[str, dict[str, typing.Any]]]) -> None:
        """Each eval family produces exactly 3 records (one per code type)."""
        _, records = lmdb_data
        eval_records = {k: v for k, v in records.items() if k.startswith("eval/")}
        # Group by family
        families: dict[str, list[str]] = {}
        for key in eval_records:
            sample_id = key[len("eval/") : key.rfind("/")]
            family_id = _extract_family_id(sample_id)
            families.setdefault(family_id, []).append(key)
        for fid, keys in families.items():
            assert len(keys) == 3, f"Family {fid} has {len(keys)} records, expected 3"

    def test_eval_total_record_count(self, lmdb_data: tuple[Path, dict[str, dict[str, typing.Any]]]) -> None:
        """Total eval records = n_eval_families * 3."""
        _, records = lmdb_data
        eval_count = sum(1 for k in records if k.startswith("eval/"))
        assert eval_count == 30  # 10 families * 3

    def test_eval_family_ids_shared(self, lmdb_data: tuple[Path, dict[str, dict[str, typing.Any]]]) -> None:
        """All code-type variants of the same problem share the same family ID."""
        _, records = lmdb_data
        eval_records = {k: v for k, v in records.items() if k.startswith("eval/")}
        families: dict[str, set[str]] = {}
        for key, rec in eval_records.items():
            sample_id = key[len("eval/") : key.rfind("/")]
            family_id = _extract_family_id(sample_id)
            families.setdefault(family_id, set()).add(rec["code_type"])
        # Each family should have all 3 code types
        for fid, cts in families.items():
            assert cts == {"original", "hinted", "misleading"}, f"Family {fid} has code types {cts}"

    def test_eval_original_no_augment_suffix(self, lmdb_data: tuple[Path, dict[str, dict[str, typing.Any]]]) -> None:
        """Original-code eval sample_ids have no /a: suffix."""
        _, records = lmdb_data
        for key, rec in records.items():
            if key.startswith("eval/") and rec["code_type"] == "original":
                sample_id = key[len("eval/") : key.rfind("/")]
                assert "/a:" not in sample_id

    def test_eval_hinted_has_hints_docs_suffix(self, lmdb_data: tuple[Path, dict[str, dict[str, typing.Any]]]) -> None:
        """Hinted eval sample_ids have /a:hints_docs:000 suffix."""
        _, records = lmdb_data
        for key, rec in records.items():
            if key.startswith("eval/") and rec["code_type"] == "hinted":
                sample_id = key[len("eval/") : key.rfind("/")]
                assert "/a:hints_docs:000" in sample_id

    def test_eval_misleading_has_issues_docs_suffix(
        self, lmdb_data: tuple[Path, dict[str, dict[str, typing.Any]]]
    ) -> None:
        """Misleading eval sample_ids have /a:issues_docs:000 suffix."""
        _, records = lmdb_data
        for key, rec in records.items():
            if key.startswith("eval/") and rec["code_type"] == "misleading":
                sample_id = key[len("eval/") : key.rfind("/")]
                assert "/a:issues_docs:000" in sample_id


# ---------------------------------------------------------------------------
# Code type tags tests
# ---------------------------------------------------------------------------


class TestDebugLmdbCodeTypeTags:
    """Tests for code_type and tags field consistency."""

    @pytest.fixture
    def eval_records(self, tmp_path: Path) -> dict[str, dict[str, typing.Any]]:
        """Create LMDB and return eval records."""
        from pyine.data.utils.lmdb_io import LMDBReader

        lmdb_path = tmp_path / "debug.lmdb"
        create_debug_probe_lmdb(lmdb_path, n_train=5, n_eval_families=10, seed=42)
        with LMDBReader(lmdb_path) as reader:
            return {k: reader.get(k) for k in reader.key_map if k.startswith("eval/")}

    def test_original_records_no_augment_tag(self, eval_records: dict[str, dict[str, typing.Any]]) -> None:
        """Records with code_type='original' have no 'augment:' tags."""
        for rec in eval_records.values():
            if rec["code_type"] == "original":
                assert not any(t.startswith("augment:") for t in rec["tags"])

    def test_hinted_records_have_augment_hinted_tag(self, eval_records: dict[str, dict[str, typing.Any]]) -> None:
        """Records with code_type='hinted' have 'augment:hinted' in tags."""
        for rec in eval_records.values():
            if rec["code_type"] == "hinted":
                assert "augment:hinted" in rec["tags"]

    def test_misleading_records_have_augment_misleading_tag(
        self, eval_records: dict[str, dict[str, typing.Any]]
    ) -> None:
        """Records with code_type='misleading' have 'augment:misleading' in tags."""
        for rec in eval_records.values():
            if rec["code_type"] == "misleading":
                assert "augment:misleading" in rec["tags"]


# ---------------------------------------------------------------------------
# Label distribution tests
# ---------------------------------------------------------------------------


class TestDebugLmdbLabelDistribution:
    """Tests for label correlation with code type."""

    @pytest.fixture
    def eval_records(self, tmp_path: Path) -> dict[str, dict[str, typing.Any]]:
        """Create LMDB with enough families for statistical assertions."""
        from pyine.data.utils.lmdb_io import LMDBReader

        lmdb_path = tmp_path / "debug.lmdb"
        # Use more families for meaningful statistics
        create_debug_probe_lmdb(lmdb_path, n_train=10, n_eval_families=100, seed=42)
        with LMDBReader(lmdb_path) as reader:
            return {k: reader.get(k) for k in reader.key_map if k.startswith("eval/")}

    def test_hinted_biased_toward_label_1(self, eval_records: dict[str, dict[str, typing.Any]]) -> None:
        """Hinted records have label=1 in >60% of cases (biased toward correct)."""
        hinted = [rec for rec in eval_records.values() if rec["code_type"] == "hinted"]
        label_1_count = sum(1 for rec in hinted if rec["reward_metrics"]["reward/metrics/soft_match/is_match"] == 1)
        ratio = label_1_count / len(hinted)
        assert ratio > 0.6, f"Hinted label=1 ratio was {ratio}, expected > 0.6"

    def test_misleading_biased_toward_label_0(self, eval_records: dict[str, dict[str, typing.Any]]) -> None:
        """Misleading records have label=0 in >50% of cases (biased toward incorrect)."""
        misleading = [rec for rec in eval_records.values() if rec["code_type"] == "misleading"]
        label_0_count = sum(1 for rec in misleading if rec["reward_metrics"]["reward/metrics/soft_match/is_match"] == 0)
        ratio = label_0_count / len(misleading)
        assert ratio > 0.5, f"Misleading label=0 ratio was {ratio}, expected > 0.5"


# ---------------------------------------------------------------------------
# Integration with eval-only mode
# ---------------------------------------------------------------------------


class TestDebugLmdbIntegrationWithEvalOnly:
    """Integration tests: debug LMDB -> load_probe_dataset_from_lmdb(use_eval_only_split=True)."""

    @pytest.fixture
    def debug_lmdb(self, tmp_path: Path) -> Path:
        lmdb_path = tmp_path / "debug.lmdb"
        create_debug_probe_lmdb(lmdb_path, n_train=50, n_eval_families=30, seed=42)
        return lmdb_path

    def test_eval_only_loads_both_splits(self, debug_lmdb: Path) -> None:
        """load_probe_dataset_from_lmdb with eval-only mode returns train+valid from eval records."""
        from pyine.probes.lmdb_dataset import load_probe_dataset_from_lmdb

        ds = load_probe_dataset_from_lmdb(debug_lmdb, use_eval_only_split=True)
        assert "train" in ds
        assert "valid" in ds
        assert len(ds["train"]) + len(ds["valid"]) == 90

    def test_eval_only_code_type_column_present(self, debug_lmdb: Path) -> None:
        """Both splits have a 'code_type' column with non-empty values."""
        from pyine.probes.lmdb_dataset import load_probe_dataset_from_lmdb

        ds = load_probe_dataset_from_lmdb(debug_lmdb, use_eval_only_split=True)
        for split in ("train", "valid"):
            assert "code_type" in ds[split].column_names
            for ct in ds[split]["code_type"]:
                assert isinstance(ct, str)
                assert len(ct) > 0

    def test_eval_only_family_split_no_leakage(self, debug_lmdb: Path) -> None:
        """With split_by_family=True, no family ID appears in both train and valid."""
        from pyine.probes.lmdb_dataset import load_probe_dataset_from_lmdb

        ds = load_probe_dataset_from_lmdb(debug_lmdb, use_eval_only_split=True, split_by_family=True)
        train_families = {_extract_family_id(sid) for sid in ds["train"]["sample_id"]}
        valid_families = {_extract_family_id(sid) for sid in ds["valid"]["sample_id"]}
        assert train_families.isdisjoint(valid_families)

    def test_eval_only_code_type_filter(self, debug_lmdb: Path) -> None:
        """code_type_filter=['original', 'hinted'] excludes misleading records."""
        from pyine.probes.lmdb_dataset import load_probe_dataset_from_lmdb

        ds = load_probe_dataset_from_lmdb(
            debug_lmdb,
            use_eval_only_split=True,
            code_type_filter=["original", "hinted"],
        )
        for split in ("train", "valid"):
            code_types = set(ds[split]["code_type"])
            assert "misleading" not in code_types

    def test_eval_only_all_code_types_in_train_split(self, debug_lmdb: Path) -> None:
        """With enough families, the train split contains all three code types."""
        from pyine.probes.lmdb_dataset import load_probe_dataset_from_lmdb

        ds = load_probe_dataset_from_lmdb(debug_lmdb, use_eval_only_split=True)
        train_code_types = set(ds["train"]["code_type"])
        assert train_code_types == {"original", "hinted", "misleading"}


# ---------------------------------------------------------------------------
# Convenience wrapper tests
# ---------------------------------------------------------------------------


class TestCreateDebugProbeDataset:
    """Tests for the convenience wrapper returning a DatasetDict."""

    def test_dataset_has_required_splits(self) -> None:
        """Output has 'train' and 'valid' splits."""
        ds = create_debug_probe_dataset(n_train=20, n_eval_families=10)
        assert "train" in ds
        assert "valid" in ds

    def test_dataset_has_required_columns(self) -> None:
        """Each split has 'text', 'label', 'sample_id', and 'code_type' columns."""
        ds = create_debug_probe_dataset(n_train=20, n_eval_families=10)
        for split_name in ("train", "valid"):
            cols = ds[split_name].column_names
            assert "text" in cols, f"Missing 'text' in {split_name}"
            assert "label" in cols, f"Missing 'label' in {split_name}"
            assert "sample_id" in cols, f"Missing 'sample_id' in {split_name}"
            assert "code_type" in cols, f"Missing 'code_type' in {split_name}"

    def test_labels_are_binary(self) -> None:
        """All labels are 0 or 1."""
        ds = create_debug_probe_dataset(n_train=100, n_eval_families=30)
        for split_name in ("train", "valid"):
            labels = set(ds[split_name]["label"])
            assert labels.issubset({0, 1}), f"Non-binary labels in {split_name}: {labels}"

    def test_both_labels_present_in_train(self) -> None:
        """Train split contains both label=0 and label=1 samples."""
        ds = create_debug_probe_dataset(n_train=100, n_eval_families=30)
        labels = set(ds["train"]["label"])
        assert labels == {0, 1}, f"Expected {{0, 1}}, got {labels}"

    def test_text_is_prompt_plus_output(self) -> None:
        """Text field is plain string (prompt + model_output), not chat format."""
        ds = create_debug_probe_dataset(n_train=20, n_eval_families=10)
        for sample in ds["train"]:
            text = sample["text"]
            assert isinstance(text, str)
            assert len(text) > 0

    def test_sample_counts(self) -> None:
        """n_train and n_eval_families control split sizes."""
        ds = create_debug_probe_dataset(n_train=42, n_eval_families=13)
        assert len(ds["train"]) == 42
        assert len(ds["valid"]) == 39  # 13 families * 3 code types

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces identical datasets."""
        ds1 = create_debug_probe_dataset(n_train=20, n_eval_families=10, seed=123)
        ds2 = create_debug_probe_dataset(n_train=20, n_eval_families=10, seed=123)

        for split in ("train", "valid"):
            for i in range(len(ds1[split])):
                assert ds1[split][i]["label"] == ds2[split][i]["label"]
                assert ds1[split][i]["text"] == ds2[split][i]["text"]

    def test_eval_only_mode(self) -> None:
        """create_debug_probe_dataset(use_eval_only_split=True) returns valid DatasetDict."""
        ds = create_debug_probe_dataset(n_train=20, n_eval_families=15, use_eval_only_split=True)
        assert "train" in ds
        assert "valid" in ds
        assert len(ds["train"]) + len(ds["valid"]) == 45  # 15 families * 3

    def test_code_type_filter_passthrough(self) -> None:
        """code_type_filter parameter is passed through to load_probe_dataset_from_lmdb."""
        ds = create_debug_probe_dataset(
            n_train=20,
            n_eval_families=15,
            use_eval_only_split=True,
            code_type_filter=["original"],
        )
        for split in ("train", "valid"):
            for ct in ds[split]["code_type"]:
                assert ct == "original"

    def test_backward_compat_default_mode(self) -> None:
        """Default parameters produce a DatasetDict with train/valid (same as before)."""
        ds = create_debug_probe_dataset(n_train=20, n_eval_families=10)
        assert "train" in ds
        assert "valid" in ds
        # Default mode: two-prefix, train has n_train records
        assert len(ds["train"]) == 20
