import lmdb
import pytest

import pyine.data.lmdb_io as lmdb_io


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
