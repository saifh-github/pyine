import pathlib
import typing

import lmdb
import pytest

import pyine.data.utils.lmdb_io as lmdb_io


@pytest.fixture
def mock_lmdb_env(tmp_path: pathlib.Path) -> lmdb.Environment:
    """Fixture to set up a mock LMDB environment for testing."""
    env_path = tmp_path / "test_lmdb"
    env = lmdb.open(str(env_path), map_size=10**6, max_readers=10)
    yield env
    env.close()


class TestCreateKeysFunctions:
    """Tests the _create_metadata_key, _create_sample_key, and _decode_sample_key functions."""

    def test_create_keys(self) -> None:
        result = lmdb_io._create_metadata_key("key1")
        assert isinstance(result, bytes)
        assert result.startswith(lmdb_io.METADATA_PREFIX)
        assert result.endswith(b"key1")

        result = lmdb_io._create_sample_key(123)
        assert isinstance(result, bytes)
        assert result.startswith(lmdb_io.SAMPLE_PREFIX)
        result2 = lmdb_io._create_sample_key(124)
        assert result2 > result
        decoded_idx = lmdb_io._decode_sample_key(result)
        assert decoded_idx == 123


class TestLMDBWriteAndRead:
    """Unit tests for the LMDBWriter and LMDBReader functionality."""

    def _get_writer(
        self,
        path: typing.Any,
        map_size: int = 10**6,
        max_readers: int = 10,
        serialization: lmdb_io.SerializationConfig | None = None,
    ) -> lmdb_io.LMDBWriter:
        serialization_config = serialization or lmdb_io.SerializationConfig(
            method=lmdb_io.SerializationMethod.MSGSPEC,
        )
        return lmdb_io.LMDBWriter(
            path=path,
            map_size=map_size,
            max_readers=max_readers,
            serialization_config=serialization_config,
        )

    def test_writer_init(
        self,
        mock_lmdb_env: typing.Any,
    ) -> None:
        writer = self._get_writer(path=mock_lmdb_env.path())
        assert writer.path is not None
        assert writer.map_size == 10**6
        assert writer.env is not None
        writer.close()
        assert writer._env is None

    def test_put_simple(
        self,
        mock_lmdb_env: typing.Any,
    ) -> None:
        writer = self._get_writer(path=mock_lmdb_env.path())
        key, value = "key1", {"test": 1}
        inserted_key = writer.put(key=key, value=value)
        assert isinstance(inserted_key, bytes)
        with writer.env.begin() as txn:
            encoded_value = txn.get(inserted_key)
            assert encoded_value is not None
        writer.close()  # to make sure we write everything, including metadata
        size_bytes = writer.get_size_on_disk()
        assert size_bytes > 0

        reader = lmdb_io.LMDBReader(path=writer.path)
        metadata = reader.get_metadata()
        assert metadata is not None
        assert metadata["map_size"] == writer.map_size
        assert metadata["key_map"] == {"key1": inserted_key}
        assert metadata["sample_count"] == 1
        assert metadata["max_encoded_value_length"] == len(encoded_value)
        assert metadata["serialization"]["method"] == lmdb_io.SerializationMethod.MSGSPEC
        result = reader.get("key1")
        assert result == value
        assert reader.get_size_on_disk() == size_bytes

    @pytest.mark.parametrize(
        "serialization_method",
        list(lmdb_io.SerializationMethod),
    )
    def test_writer_with_various_serialization_methods(
        self,
        mock_lmdb_env: typing.Any,
        serialization_method: lmdb_io.SerializationMethod,
    ) -> None:
        # pickle-based methods require explicit opt-in due to security risks
        allow_insecure = serialization_method in {
            lmdb_io.SerializationMethod.PICKLE,
            lmdb_io.SerializationMethod.PICKLE_LZ4,
        }
        serialization_cfg = lmdb_io.SerializationConfig(
            method=serialization_method,
            allow_insecure_serialization=allow_insecure,
        )
        writer = self._get_writer(path=mock_lmdb_env.path(), serialization=serialization_cfg)
        key, value = "key1", {"test": 1}
        inserted_key = writer.put(key=key, value=value)
        assert isinstance(inserted_key, bytes)
        writer.close()

        reader = lmdb_io.LMDBReader(path=writer.path)
        metadata = reader.get_metadata()
        assert metadata["serialization"]["method"] == serialization_method
        result = reader.get("key1")
        assert result == value

    def test_put_batch(
        self,
        mock_lmdb_env: typing.Any,
    ) -> None:
        writer = self._get_writer(path=mock_lmdb_env.path())
        entries = {"key1": b"value1", "key2": b"value2"}
        inserted_keys = writer.put_batch(items=entries)
        assert len(inserted_keys) == 2
        assert all(isinstance(v, bytes) for v in inserted_keys.values())
        with writer.env.begin() as txn:
            encoded_value1 = txn.get(inserted_keys["key1"])
            assert encoded_value1 is not None
            encoded_value2 = txn.get(inserted_keys["key2"])
            assert encoded_value2 is not None
        some_bad_entries = {"key3": b"value3", "key2": "value2"}  # contains a duplicate
        inserted_keys2, errored_keys = writer.put_batch(items=some_bad_entries, raise_on_error=False)
        assert len(inserted_keys2) == 1 and "key3" in inserted_keys2
        assert len(errored_keys) == 1 and "key2" in errored_keys
        assert isinstance(errored_keys["key2"], ValueError)
        inserted_keys.update(inserted_keys2)
        with writer.env.begin() as txn:
            encoded_value3 = txn.get(inserted_keys["key3"])
            assert encoded_value3 is not None
        writer.close()  # to make sure we write everything, including metadata
        entries = {**entries, "key3": some_bad_entries["key3"]}

        reader = lmdb_io.LMDBReader(path=writer.path)
        metadata = reader.get_metadata()
        assert metadata is not None
        assert metadata["map_size"] == writer.map_size
        assert metadata["key_map"] == inserted_keys
        assert metadata["sample_count"] == 3
        max_encoded_val = max(len(encoded_value1), len(encoded_value2), len(encoded_value3))
        assert metadata["max_encoded_value_length"] == max_encoded_val
        found_vals = tuple(val for val in reader.iter_from())
        assert found_vals == tuple(entries.values())
        found_vals = tuple(v for vals in reader.iter_batched(batch_size=100) for v in vals)
        assert found_vals == tuple(entries.values())

    def test_get_indices(
        self,
        mock_lmdb_env: typing.Any,
    ) -> None:
        writer = self._get_writer(path=mock_lmdb_env.path())
        test_data = {
            "problem_001": {"type": "problem", "id": 1},
            "problem_002": {"type": "problem", "id": 2},
            "problem_100": {"type": "problem", "id": 100},
            "test_case_001": {"type": "test", "id": 1},
            "test_case_002": {"type": "test", "id": 2},
            "solution_a": {"type": "solution", "variant": "a"},
            "solution_b": {"type": "solution", "variant": "b"},
            "data_file_001.json": {"type": "data", "format": "json"},
            "data_file_002.xml": {"type": "data", "format": "xml"},
            "misc_item": {"type": "misc"},
        }
        writer.put_batch(test_data)
        writer.close()

        reader = lmdb_io.LMDBReader(path=writer.path)

        indices = reader.get_indices("problem_001")
        assert len(indices) == 1
        assert reader.get(indices[0]) == test_data["problem_001"]

        indices = reader.get_indices("problem_*")
        assert len(indices) == 3
        expected_keys = ["problem_001", "problem_002", "problem_100"]
        for idx in indices:
            assert reader.get(idx) in [test_data[key] for key in expected_keys]

        indices = reader.get_indices("*_001")
        assert len(indices) == 2
        expected_keys = ["problem_001", "test_case_001"]
        for idx in indices:
            assert reader.get(idx) in [test_data[key] for key in expected_keys]

        indices = reader.get_indices("solution_[ab]")
        assert len(indices) == 2
        expected_keys = ["solution_a", "solution_b"]
        for idx in indices:
            assert reader.get(idx) in [test_data[key] for key in expected_keys]

        indices = reader.get_indices("*.json")
        assert len(indices) == 1
        assert reader.get(indices[0]) == test_data["data_file_001.json"]

        indices = reader.get_indices("nonexistent_*")
        assert len(indices) == 0

        indices = reader.get_indices("*")
        assert len(indices) == len(test_data)
        assert sorted(indices) == list(range(len(test_data)))

        indices = reader.get_indices("*_00[12]")
        assert len(indices) == 4  # problem_001, problem_002, test_case_001, test_case_002
        assert indices == sorted(indices)

        indices, keys = reader.get_indices("problem_*", return_keys=True)
        assert len(indices) == 3
        assert len(keys) == 3
        assert all(k.startswith("problem_") for k in keys)
        assert all(reader.get(idx) == test_data[key] for idx, key in zip(indices, keys, strict=False))

        indices, keys = reader.get_indices("*", return_keys=True)
        assert len(indices) == len(test_data)
        assert len(keys) == len(test_data)
        assert indices == sorted(indices)
        for idx, key in zip(indices, keys, strict=False):
            assert reader.get(idx) == test_data[key]

        indices, keys = reader.get_indices("nonexistent_*", return_keys=True)
        assert len(indices) == 0
        assert len(keys) == 0

        with pytest.raises(TypeError):
            typing.cast("typing.Any", reader).get_indices(123)

        reader.close()


def _make_lmdb_dir(parent: pathlib.Path, name: str = "test.lmdb") -> pathlib.Path:
    """Create a minimal LMDB directory with a data.mdb file."""
    lmdb_dir = parent / name
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    (lmdb_dir / "data.mdb").touch()
    return lmdb_dir


class TestResolveLmdbPaths:
    def test_single_path(self, tmp_path: pathlib.Path) -> None:
        lmdb_dir = _make_lmdb_dir(tmp_path)
        result = lmdb_io.resolve_lmdb_paths((lmdb_dir,))
        assert len(result) == 1
        assert result[0] == lmdb_dir.resolve()

    def test_glob_resolves(self, tmp_path: pathlib.Path) -> None:
        for rank_idx in range(2):
            _make_lmdb_dir(tmp_path, f"rank_{rank_idx}")
        result = lmdb_io.resolve_lmdb_paths((pathlib.Path(str(tmp_path / "rank_*")),))
        assert len(result) == 2

    def test_auto_discover_rank_subdirs(self, tmp_path: pathlib.Path) -> None:
        for rank_idx in range(2):
            _make_lmdb_dir(tmp_path / "output", f"rank_{rank_idx}")
        result = lmdb_io.resolve_lmdb_paths((tmp_path / "output",))
        assert len(result) == 2

    def test_glob_no_match_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="matched zero paths"):
            lmdb_io.resolve_lmdb_paths((pathlib.Path(str(tmp_path / "nonexistent_*")),))

    def test_nonexistent_path_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            lmdb_io.resolve_lmdb_paths((tmp_path / "nonexistent",))

    def test_missing_data_mdb_raises(self, tmp_path: pathlib.Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="data.mdb"):
            lmdb_io.resolve_lmdb_paths((empty_dir,))

    def test_duplicate_paths_deduplicated(self, tmp_path: pathlib.Path) -> None:
        lmdb_dir = _make_lmdb_dir(tmp_path)
        result = lmdb_io.resolve_lmdb_paths((lmdb_dir, lmdb_dir))
        assert len(result) == 1


class TestParseLmdbSampleKey:
    """Tests for parse_lmdb_sample_key()."""

    def test_simple_key(self) -> None:
        sample_id, gen_count = lmdb_io.parse_lmdb_sample_key("train/sample_0001/3", "train/")
        assert sample_id == "sample_0001"
        assert gen_count == 3

    def test_nested_sample_id(self) -> None:
        sample_id, gen_count = lmdb_io.parse_lmdb_sample_key("train/TACO/train/p000001/s0000/5", "train/")
        assert sample_id == "TACO/train/p000001/s0000"
        assert gen_count == 5

    def test_generation_count_none(self) -> None:
        sample_id, gen_count = lmdb_io.parse_lmdb_sample_key("eval/sample_0001/none", "eval/")
        assert sample_id == "sample_0001"
        assert gen_count == 0

    def test_empty_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            lmdb_io.parse_lmdb_sample_key("train/sample_001/1", "")

    def test_no_separator_raises(self) -> None:
        with pytest.raises(ValueError, match="no '/' separator"):
            lmdb_io.parse_lmdb_sample_key("train/flat_key", "train/")

    def test_non_numeric_gen_count_raises(self) -> None:
        with pytest.raises(ValueError, match="non-numeric"):
            lmdb_io.parse_lmdb_sample_key("train/sample_001/abc", "train/")


class TestDeduplicateByStrategy:
    """Tests for deduplicate_by_strategy()."""

    def test_latest_strategy(self) -> None:
        grouped: dict[str, list[tuple[int, dict[str, typing.Any]]]] = {
            "s1": [(1, {"val": "a"}), (3, {"val": "c"}), (2, {"val": "b"})],
        }
        result = lmdb_io.deduplicate_by_strategy(grouped, "latest")
        assert result["s1"]["val"] == "c"  # gen_count=3

    def test_best_reward_strategy(self) -> None:
        grouped: dict[str, list[tuple[int, dict[str, typing.Any]]]] = {
            "s1": [
                (1, {"reward_total": 0.5}),
                (2, {"reward_total": 0.9}),
                (3, {"reward_total": 0.3}),
            ],
        }
        result = lmdb_io.deduplicate_by_strategy(grouped, "best_reward")
        assert result["s1"]["reward_total"] == 0.9

    def test_best_reward_missing_total_raises(self) -> None:
        grouped: dict[str, list[tuple[int, dict[str, typing.Any]]]] = {
            "s1": [(1, {"reward_total": None})],
        }
        with pytest.raises(ValueError, match="reward_total"):
            lmdb_io.deduplicate_by_strategy(grouped, "best_reward")

    def test_invalid_strategy_raises(self) -> None:
        grouped: dict[str, list[tuple[int, dict[str, typing.Any]]]] = {
            "s1": [(1, {"val": "a"})],
        }
        with pytest.raises(ValueError, match="unknown selection_strategy"):
            lmdb_io.deduplicate_by_strategy(grouped, "invalid")  # type: ignore[arg-type]


class TestLoadAndDeduplicateLmdbRecords:
    """Tests for load_and_deduplicate_lmdb_records() with a real LMDB."""

    def test_end_to_end(self, tmp_path: pathlib.Path) -> None:
        import pyine.probes.data.debug_dataset

        lmdb_path = tmp_path / "debug.lmdb"
        pyine.probes.data.debug_dataset.create_debug_probe_lmdb(lmdb_path, n_train=10, n_eval_families=5, seed=42)
        with lmdb_io.LMDBReader(lmdb_path) as reader:
            records = lmdb_io.load_and_deduplicate_lmdb_records(reader, "train/", "latest")
        assert len(records) == 10
        # Should be sorted by sample_id
        sample_ids = [sid for sid, _rec in records]
        assert sample_ids == sorted(sample_ids)

    def test_prefix_filters(self, tmp_path: pathlib.Path) -> None:
        import pyine.probes.data.debug_dataset

        lmdb_path = tmp_path / "debug.lmdb"
        pyine.probes.data.debug_dataset.create_debug_probe_lmdb(lmdb_path, n_train=10, n_eval_families=5, seed=42)
        with lmdb_io.LMDBReader(lmdb_path) as reader:
            train = lmdb_io.load_and_deduplicate_lmdb_records(reader, "train/", "latest")
            valid = lmdb_io.load_and_deduplicate_lmdb_records(reader, "eval/", "latest")
        assert len(train) == 10
        assert len(valid) == 15  # 5 families * 3 code types
