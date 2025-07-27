import lmdb
import pytest

import pyine.data.utils.lmdb_io as lmdb_io


@pytest.fixture
def mock_lmdb_env(tmp_path):
    """Fixture to set up a mock LMDB environment for testing."""
    env_path = tmp_path / "test_lmdb"
    env = lmdb.open(str(env_path), map_size=10**6, max_readers=10)
    yield env
    env.close()


class TestCreateKeysFunctions:
    """Tests the _create_metadata_key, _create_sample_key, and _decode_sample_key functions."""

    def test_create_keys(self):
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
        path,
        map_size=10**6,
        max_readers=10,
        serialization=lmdb_io.SerializationMethod.PICKLE,
    ):
        return lmdb_io.LMDBWriter(
            path=path,
            map_size=map_size,
            max_readers=max_readers,
            serialization=serialization,
        )

    def test_writer_init(self, mock_lmdb_env):
        writer = self._get_writer(path=mock_lmdb_env.path())
        assert writer.path is not None
        assert writer.map_size == 10**6
        assert writer.env is not None
        writer.close()
        assert writer.env is None

    def test_put_simple(self, mock_lmdb_env):
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
        assert metadata["serialization"] == lmdb_io.SerializationMethod.PICKLE
        result = reader.get("key1")
        assert result == value
        assert reader.get_size_on_disk() == size_bytes

    @pytest.mark.parametrize(
        "serialization_method",
        [
            lmdb_io.SerializationMethod.PICKLE,
            lmdb_io.SerializationMethod.PICKLE_LZ4,
            lmdb_io.SerializationMethod.JSON,
            lmdb_io.SerializationMethod.JSON_LZ4,
        ],
    )
    def test_writer_with_various_serialization_methods(self, mock_lmdb_env, serialization_method):
        writer = self._get_writer(path=mock_lmdb_env.path(), serialization=serialization_method)
        key, value = "key1", {"test": 1}
        inserted_key = writer.put(key=key, value=value)
        assert isinstance(inserted_key, bytes)
        writer.close()

        reader = lmdb_io.LMDBReader(path=writer.path)
        metadata = reader.get_metadata()
        assert metadata["serialization"] == serialization_method
        result = reader.get("key1")
        assert result == value

    def test_put_batch(self, mock_lmdb_env):
        writer = self._get_writer(path=mock_lmdb_env.path())
        entries = {"key1": b"value1", "key2": b"value2"}
        inserted_keys = writer.put_batch(items=entries)
        assert len(inserted_keys) == 2
        assert all([isinstance(v, bytes) for v in inserted_keys])
        with writer.env.begin() as txn:
            encoded_value1 = txn.get(inserted_keys[0])
            assert encoded_value1 is not None
            encoded_value2 = txn.get(inserted_keys[1])
            assert encoded_value2 is not None
        writer.close()  # to make sure we write everything, including metadata

        reader = lmdb_io.LMDBReader(path=writer.path)
        metadata = reader.get_metadata()
        assert metadata is not None
        assert metadata["map_size"] == writer.map_size
        assert metadata["key_map"] == {k: kin for k, kin in zip(entries.keys(), inserted_keys)}
        assert metadata["sample_count"] == 2
        assert metadata["max_encoded_value_length"] == max(len(encoded_value1), len(encoded_value2))
        assert metadata["serialization"] == lmdb_io.SerializationMethod.PICKLE
        found_vals = tuple(val for val in reader.iter_from())
        assert found_vals == tuple(entries.values())
        found_vals = tuple(v for vals in reader.iter_batched(batch_size=100) for v in vals)
        assert found_vals == tuple(entries.values())

    def test_get_indices(self, mock_lmdb_env):
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
        assert all(reader.get(idx) == test_data[key] for idx, key in zip(indices, keys))

        indices, keys = reader.get_indices("*", return_keys=True)
        assert len(indices) == len(test_data)
        assert len(keys) == len(test_data)
        assert indices == sorted(indices)
        for idx, key in zip(indices, keys):
            assert reader.get(idx) == test_data[key]

        indices, keys = reader.get_indices("nonexistent_*", return_keys=True)
        assert len(indices) == 0
        assert len(keys) == 0

        with pytest.raises(AssertionError):
            reader.get_indices(123)  # noqa; should be string

        reader.close()
