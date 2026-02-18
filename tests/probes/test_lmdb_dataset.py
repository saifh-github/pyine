from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

import pytest

import pyine.probes.debug_dataset
import pyine.probes.lmdb_dataset


class TestParseLmdbKey:
    def test_simple_key(self) -> None:
        sample_id, gen_count = pyine.probes.lmdb_dataset._parse_lmdb_key("train/sample_0001/3", "train/")
        assert sample_id == "sample_0001"
        assert gen_count == 3

    def test_nested_sample_id(self) -> None:
        """Multi-segment sample IDs (e.g., TACO/train/p000001/s0000) are preserved."""
        sample_id, gen_count = pyine.probes.lmdb_dataset._parse_lmdb_key("train/TACO/train/p000001/s0000/5", "train/")
        assert sample_id == "TACO/train/p000001/s0000"
        assert gen_count == 5

    def test_generation_count_none(self) -> None:
        """'none' generation count is parsed as 0."""
        sample_id, gen_count = pyine.probes.lmdb_dataset._parse_lmdb_key("eval/sample_0001/none", "eval/")
        assert sample_id == "sample_0001"
        assert gen_count == 0

    def test_no_separator_raises(self) -> None:
        with pytest.raises(ValueError, match="no '/' separator"):
            pyine.probes.lmdb_dataset._parse_lmdb_key("train/flat_key", "train/")

    def test_non_numeric_gen_count_raises(self) -> None:
        with pytest.raises(ValueError, match="non-numeric"):
            pyine.probes.lmdb_dataset._parse_lmdb_key("train/sample_001/abc", "train/")


class TestExtractFamilyId:
    def test_original_sample_id(self) -> None:
        assert (
            pyine.probes.lmdb_dataset._extract_family_id("TACO/train/p000001/s0000/t0000")
            == "TACO/train/p000001/s0000/t0000"
        )

    def test_augmented_sample_id(self) -> None:
        assert (
            pyine.probes.lmdb_dataset._extract_family_id("TACO/train/p000001/s0000/t0000/a:hints_docs:001")
            == "TACO/train/p000001/s0000/t0000"
        )
        assert (
            pyine.probes.lmdb_dataset._extract_family_id("TACO/train/p000001/s0000/t0000/a:issues_docs:001")
            == "TACO/train/p000001/s0000/t0000"
        )

    def test_multi_segment_sample_id(self) -> None:
        assert (
            pyine.probes.lmdb_dataset._extract_family_id("TACO/train/p000001/s0000/t0000/a:obfuscated:000")
            == "TACO/train/p000001/s0000/t0000"
        )

    def test_no_false_positive_on_a_in_path(self) -> None:
        """The string '/a:' must be the augmentation marker, not part of a dataset name."""
        # A sample_id that has 'a' in a path segment but NOT as '/a:' marker
        assert (
            pyine.probes.lmdb_dataset._extract_family_id("dataset_a/train/p000001/s0000/t0000")
            == "dataset_a/train/p000001/s0000/t0000"
        )


class TestFilterByCodeType:
    def test_filters_to_specified_types(self) -> None:
        records = [
            ("s1", {"code_type": "original"}),
            ("s2", {"code_type": "hinted"}),
            ("s3", {"code_type": "misleading"}),
        ]
        filtered = pyine.probes.lmdb_dataset._filter_by_code_type(records, ["original", "hinted"])
        assert len(filtered) == 2
        assert {sid for sid, _ in filtered} == {"s1", "s2"}

    def test_empty_filter_list_returns_empty(self) -> None:
        records = [("s1", {"code_type": "original"})]
        assert pyine.probes.lmdb_dataset._filter_by_code_type(records, []) == []

    def test_null_code_type_excluded(self) -> None:
        """Records with code_type=None are excluded unless 'unknown' is in the filter list."""
        records = [
            ("s1", {"code_type": None}),
            ("s2", {"code_type": "original"}),
        ]
        # Without "unknown" in filter
        filtered = pyine.probes.lmdb_dataset._filter_by_code_type(records, ["original"])
        assert len(filtered) == 1

        # With "unknown" in filter
        filtered = pyine.probes.lmdb_dataset._filter_by_code_type(records, ["original", "unknown"])
        assert len(filtered) == 2


class TestSplitRecordsByFamily:
    def _make_family_samples(self, n_families: int = 10) -> list[dict[str, str | int]]:
        """Create samples with family structure (3 code types per family)."""
        samples = []
        for family_idx in range(n_families):
            base_id = f"prob_{family_idx:03d}/s0000/t0000"
            for code_type, suffix in [
                ("original", ""),
                ("hinted", "/a:hints_docs:000"),
                ("misleading", "/a:issues_docs:000"),
            ]:
                samples.append(
                    {
                        "text": f"text_{family_idx}_{code_type}",
                        "label": family_idx % 2,
                        "sample_id": f"{base_id}{suffix}",
                        "code_type": code_type,
                    }
                )
        return samples

    def test_all_family_members_in_same_split(self) -> None:
        samples = self._make_family_samples(n_families=20)
        train, valid = pyine.probes.lmdb_dataset._split_records_by_family(samples, train_ratio=0.8, seed=42)

        train_families = {pyine.probes.lmdb_dataset._extract_family_id(str(s["sample_id"])) for s in train}
        valid_families = {pyine.probes.lmdb_dataset._extract_family_id(str(s["sample_id"])) for s in valid}
        assert train_families.isdisjoint(valid_families), "Family ID found in both train and valid"

    def test_train_ratio_approximate(self) -> None:
        samples = self._make_family_samples(n_families=20)
        train, valid = pyine.probes.lmdb_dataset._split_records_by_family(samples, train_ratio=0.8, seed=42)

        # 20 families * 0.8 = 16 train families (48 samples), 4 valid families (12 samples)
        assert len(train) == 48
        assert len(valid) == 12

    def test_deterministic_with_seed(self) -> None:
        samples = self._make_family_samples(n_families=20)
        train1, valid1 = pyine.probes.lmdb_dataset._split_records_by_family(samples, train_ratio=0.8, seed=42)
        train2, valid2 = pyine.probes.lmdb_dataset._split_records_by_family(samples, train_ratio=0.8, seed=42)
        assert [s["sample_id"] for s in train1] == [s["sample_id"] for s in train2]
        assert [s["sample_id"] for s in valid1] == [s["sample_id"] for s in valid2]

    def test_different_seed_different_split(self) -> None:
        samples = self._make_family_samples(n_families=20)
        train1, _ = pyine.probes.lmdb_dataset._split_records_by_family(samples, train_ratio=0.8, seed=42)
        train2, _ = pyine.probes.lmdb_dataset._split_records_by_family(samples, train_ratio=0.8, seed=99)
        # Different seeds should produce different family assignments
        ids1 = {s["sample_id"] for s in train1}
        ids2 = {s["sample_id"] for s in train2}
        assert ids1 != ids2

    def test_single_member_families(self) -> None:
        samples = [
            {"text": f"text_{i}", "label": i % 2, "sample_id": f"sample_{i:03d}", "code_type": "original"}
            for i in range(10)
        ]
        train, valid = pyine.probes.lmdb_dataset._split_records_by_family(samples, train_ratio=0.8, seed=42)
        assert len(train) + len(valid) == 10

    def test_too_few_families_raises(self) -> None:
        samples = [
            {"text": "t", "label": 0, "sample_id": "fam_0/s0000/t0000", "code_type": "original"},
            {"text": "t", "label": 1, "sample_id": "fam_0/s0000/t0000/a:hints_docs:000", "code_type": "hinted"},
        ]
        with pytest.raises(ValueError, match="at least 2 families"):
            pyine.probes.lmdb_dataset._split_records_by_family(samples, train_ratio=0.8, seed=42)


class TestSplitRecordsRandom:
    def test_split_sizes_match_ratio(self) -> None:
        samples = [
            {"text": f"text_{i}", "label": i % 2, "sample_id": f"s_{i}", "code_type": "original"} for i in range(100)
        ]
        train, valid = pyine.probes.lmdb_dataset._split_records_random(samples, train_ratio=0.8, seed=42)
        assert len(train) == 80
        assert len(valid) == 20

    def test_deterministic_with_seed(self) -> None:
        samples = [
            {"text": f"text_{i}", "label": i % 2, "sample_id": f"s_{i}", "code_type": "original"} for i in range(50)
        ]
        train1, valid1 = pyine.probes.lmdb_dataset._split_records_random(samples, train_ratio=0.7, seed=42)
        train2, valid2 = pyine.probes.lmdb_dataset._split_records_random(samples, train_ratio=0.7, seed=42)
        assert [s["sample_id"] for s in train1] == [s["sample_id"] for s in train2]
        assert [s["sample_id"] for s in valid1] == [s["sample_id"] for s in valid2]


class TestLoadProbeDatasetFromLmdb:
    @pytest.fixture
    def debug_lmdb(self, tmp_path: Path) -> Path:
        """Create a debug LMDB for testing.

        n_train=50 creates 50 train records.
        n_eval_families=20 creates 60 eval records (20 families * 3 code types).
        """
        lmdb_path = tmp_path / "debug.lmdb"
        pyine.probes.debug_dataset.create_debug_probe_lmdb(lmdb_path, n_train=50, n_eval_families=20, seed=42)
        return lmdb_path

    def test_returns_train_and_valid_splits(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(debug_lmdb)
        assert "train" in ds
        assert "valid" in ds

    def test_columns_present(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(debug_lmdb)
        for split in ("train", "valid"):
            cols = ds[split].column_names
            assert "text" in cols
            assert "label" in cols
            assert "sample_id" in cols
            assert "code_type" in cols

    def test_labels_binary(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(debug_lmdb)
        for split in ("train", "valid"):
            labels = set(ds[split]["label"])
            assert labels.issubset({0, 1})

    def test_split_sizes(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(debug_lmdb)
        assert len(ds["train"]) == 50
        assert len(ds["valid"]) == 60  # 20 families * 3 code types

    def test_max_samples_per_split(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(debug_lmdb, max_samples_per_split=10)
        assert len(ds["train"]) == 10
        assert len(ds["valid"]) == 10

    def test_text_is_nonempty_string(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(debug_lmdb)
        for text in ds["train"]["text"]:
            assert isinstance(text, str)
            assert len(text) > 0

    def test_selection_strategy_latest(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(debug_lmdb, selection_strategy="latest")
        assert len(ds["train"]) > 0

    def test_selection_strategy_best_reward(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(debug_lmdb, selection_strategy="best_reward")
        assert len(ds["train"]) > 0

    def test_invalid_selection_strategy_raises(self, debug_lmdb: Path) -> None:
        with pytest.raises(ValueError, match="unknown selection_strategy"):
            pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(debug_lmdb, selection_strategy="invalid")

    def test_wrong_prefix_raises(self, debug_lmdb: Path) -> None:
        with pytest.raises(ValueError, match="no records match"):
            pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(
                debug_lmdb,
                train_key_prefix="nonexistent/",
            )

    def test_recompute_labels_soft_match(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(
            debug_lmdb,
            label_metric_key="reward/metrics/soft_match/is_match",
            recompute_labels=True,
        )
        labels = set(ds["train"]["label"])
        assert labels.issubset({0, 1})

    def test_recompute_labels_hard_match(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(
            debug_lmdb,
            label_metric_key="reward/metrics/hard_match/is_match",
            recompute_labels=True,
        )
        labels = set(ds["train"]["label"])
        assert labels.issubset({0, 1})

    def test_recompute_labels_unsupported_metric_raises(self, debug_lmdb: Path) -> None:
        with pytest.raises(ValueError, match="recompute_labels"):
            pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(
                debug_lmdb,
                label_metric_key="custom/metric",
                recompute_labels=True,
            )

    def test_code_type_column_present(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(debug_lmdb)
        for split in ("train", "valid"):
            assert "code_type" in ds[split].column_names
            for code_type in ds[split]["code_type"]:
                assert isinstance(code_type, str)
                assert len(code_type) > 0

    def test_code_type_filter_in_two_prefix_mode(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(debug_lmdb, code_type_filter=["original"])
        # Train records are all "original", so train should be unchanged
        assert len(ds["train"]) == 50
        # Valid should only have original records (1 per family)
        assert len(ds["valid"]) == 20
        for code_type in ds["valid"]["code_type"]:
            assert code_type == "original"


class TestLoadProbeDatasetEvalOnly:
    @pytest.fixture
    def debug_lmdb(self, tmp_path: Path) -> Path:
        """Create a debug LMDB with family-structured eval records."""
        lmdb_path = tmp_path / "debug.lmdb"
        pyine.probes.debug_dataset.create_debug_probe_lmdb(lmdb_path, n_train=50, n_eval_families=30, seed=42)
        return lmdb_path

    def test_returns_train_and_valid_splits(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(debug_lmdb, use_eval_only_split=True)
        assert "train" in ds
        assert "valid" in ds
        assert len(ds["train"]) + len(ds["valid"]) == 90  # 30 families * 3

    def test_code_type_column_present(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(debug_lmdb, use_eval_only_split=True)
        for split in ("train", "valid"):
            assert "code_type" in ds[split].column_names

    def test_code_type_filter_applied(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(
            debug_lmdb,
            use_eval_only_split=True,
            code_type_filter=["original", "hinted"],
        )
        for split in ("train", "valid"):
            code_types = set(ds[split]["code_type"])
            assert code_types.issubset({"original", "hinted"})
            assert "misleading" not in code_types

    def test_family_split_no_leakage(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(
            debug_lmdb,
            use_eval_only_split=True,
            split_by_family=True,
        )
        train_families = {pyine.probes.lmdb_dataset._extract_family_id(sid) for sid in ds["train"]["sample_id"]}
        valid_families = {pyine.probes.lmdb_dataset._extract_family_id(sid) for sid in ds["valid"]["sample_id"]}
        assert train_families.isdisjoint(valid_families), "Data leakage: family in both splits"

    def test_code_type_filter_all_excluded_raises(self, debug_lmdb: Path) -> None:
        with pytest.raises(ValueError, match="no records remain"):
            pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(
                debug_lmdb,
                use_eval_only_split=True,
                code_type_filter=["nonexistent_code_type"],
            )

    def test_both_labels_in_each_split(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(debug_lmdb, use_eval_only_split=True)
        for split in ("train", "valid"):
            labels = set(ds[split]["label"])
            assert labels == {0, 1}, f"Expected {{0, 1}} in {split}, got {labels}"

    def test_random_split_mode(self, debug_lmdb: Path) -> None:
        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(
            debug_lmdb,
            use_eval_only_split=True,
            split_by_family=False,
        )
        assert len(ds["train"]) + len(ds["valid"]) == 90


class TestSkipMalformedRecords:
    def test_malformed_record_raises_by_default(self, tmp_path: Path) -> None:
        from pyine.data.utils.lmdb_io import LMDBWriter, SerializationConfig, SerializationMethod

        lmdb_path = tmp_path / "bad.lmdb"
        serialization = SerializationConfig(method=SerializationMethod.JSON_ZSTD)
        with LMDBWriter(lmdb_path, serialization_config=serialization) as writer:
            # Missing prompt and model_output
            writer.put("train/bad_sample/1", {"reward_metrics": {"reward/metrics/soft_match/is_match": 1}})
            # Valid record
            writer.put(
                "eval/good_sample/1",
                {
                    "prompt": "p",
                    "model_output": "o",
                    "reward_metrics": {"reward/metrics/soft_match/is_match": 1},
                },
            )

        with pytest.raises(ValueError, match="missing prompt or model_output"):
            pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(
                lmdb_path, train_key_prefix="train/", valid_key_prefix="eval/"
            )

    def test_malformed_record_skipped_when_enabled(self, tmp_path: Path) -> None:
        from pyine.data.utils.lmdb_io import LMDBWriter, SerializationConfig, SerializationMethod

        lmdb_path = tmp_path / "mixed.lmdb"
        serialization = SerializationConfig(method=SerializationMethod.JSON_ZSTD)
        with LMDBWriter(lmdb_path, serialization_config=serialization) as writer:
            # Bad record (missing prompt)
            writer.put(
                "train/bad/1", {"model_output": "o", "reward_metrics": {"reward/metrics/soft_match/is_match": 0}}
            )
            # Good records
            for i in range(5):
                writer.put(
                    f"train/good_{i}/1",
                    {
                        "prompt": f"prompt {i}",
                        "model_output": f"output {i}",
                        "reward_metrics": {"reward/metrics/soft_match/is_match": i % 2},
                    },
                )
            # Valid eval records
            for i in range(3):
                writer.put(
                    f"eval/eval_{i}/1",
                    {
                        "prompt": f"eval prompt {i}",
                        "model_output": f"eval output {i}",
                        "reward_metrics": {"reward/metrics/soft_match/is_match": i % 2},
                    },
                )

        ds = pyine.probes.lmdb_dataset.load_probe_dataset_from_lmdb(
            lmdb_path,
            skip_malformed_records=True,
        )
        assert len(ds["train"]) == 5  # bad record skipped
