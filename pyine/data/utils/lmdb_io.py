import enum
import fnmatch
import json
import pathlib
import pickle
import struct
import typing

import lmdb
import lz4.frame
import tqdm

import pyine.utils.reprod


class SerializationMethod(enum.StrEnum):
    """Supported serialization methods for LMDBWriter."""

    PICKLE = enum.auto()
    PICKLE_LZ4 = enum.auto()
    JSON = enum.auto()
    JSON_LZ4 = enum.auto()


SAMPLE_PREFIX = b"sample/"
SOLUTION_PREFIX = b"solution/"
METADATA_PREFIX = b"metadata/"
_NEXT_INTERNAL_KEY = b"_next_internal_key"


def _create_metadata_key(field_name: str) -> bytes:
    """Create a metadata key with proper prefix."""
    return METADATA_PREFIX + field_name.encode("utf-8")


def _decode_metadata_key(key: bytes) -> str:
    """Decode the metadata field name from a metadata key that possesses a prefix."""
    return key[len(METADATA_PREFIX) :].decode("utf-8")


def _create_sample_key(key_index: int) -> bytes:
    """Create a sample key for sequential access w/ 8-byte unsigned integer format."""
    key_idx_bytes = struct.pack(">Q", key_index)
    return SAMPLE_PREFIX + key_idx_bytes


def _decode_sample_key(key: bytes) -> int:
    """Decode the sample index (int) from a sample key that possesses a prefix."""
    return struct.unpack(">Q", key[len(SAMPLE_PREFIX) :])[0]


def _get_database_size(path: pathlib.Path | typing.AnyStr) -> int:
    """Calculate the total size of the LMDB dataset stored on disk (in bytes)."""
    path = pathlib.Path(path)
    if not path.is_dir():
        raise ValueError(f"Path '{path}' is not a valid directory.")
    return sum(f.stat().st_size for f in path.iterdir() if f.is_file())


class LMDBWriter:
    """A class for storing datasets in a Lightning Memory-Mapped Database (LMDB).

    LMDB provides fast, persistent key-value storage with memory-mapped files,
    making it ideal for efficiently storing and retrieving large datasets. This implementation
    focuses on arranging data in a way that maximizes sequential read performance.

    Examples:
        Basic usage to write a single value:
        >>> writer = LMDBWriter("path/to/db")
        >>> writer.put("key1", {"data": "value1"})
        >>> writer.close()

        Writing multiple values in batch:
        >>> with LMDBWriter("path/to/db") as writer:
        ...     data = {
        ...         "key1": {"data": "value1"},
        ...         "key2": {"data": "value2"}
        ...     }
        ...     writer.put_batch(data)

        Writing with different serialization methods:
        >>> writer = LMDBWriter(
        ...     "path/to/db",
        ...     serialization=SerializationMethod.JSON_LZ4
        ... )
        >>> writer.put("key1", {"data": "value1"})
        >>> writer.close()

        Writing metadata:
        >>> with LMDBWriter("path/to/db") as writer:
        ...     metadata = {
        ...         "dataset_name": "example",
        ...         "version": "1.0"
        ...     }
        ...     writer.write_metadata(metadata)

        Specifying custom map size and readers:
        >>> writer = LMDBWriter(
        ...     "path/to/db",
        ...     map_size=2 * 1024 * 1024 * 1024,  # 2GB
        ...     max_readers=256
        ... )
        >>> writer.put("key1", {"data": "value1"})
        >>> writer.close()
    """

    def __init__(
        self,
        path: pathlib.Path | typing.AnyStr,
        map_size: int = 1 * 1024 * 1024 * 1024 * 1024,  # 1TB default size (1 * 1024^4)
        max_readers: int = 126,  # Typical default max readers for LMDB
        serialization: SerializationMethod = SerializationMethod.PICKLE,
    ) -> None:
        """Initialize the LMDB database.

        Args:
            path: Path where the LMDB will be stored.
            map_size: Maximum size database may grow to; default 1TB (1 * 1024^4 bytes).
            max_readers: Maximum number of simultaneous readers (126 by default, typical for LMDB).
            serialization: Method to serialize objects (default: pickle).
        """
        self.path: pathlib.Path = pathlib.Path(path)
        self.map_size: int = map_size
        self.max_readers: int = max_readers
        self.path.mkdir(parents=True, exist_ok=True)
        self.env: lmdb.Environment = lmdb.open(
            path=str(self.path),
            map_size=self.map_size,
            max_readers=self.max_readers,
            readonly=False,
        )
        if not isinstance(serialization, SerializationMethod):
            raise ValueError(f"serialization must be an instance of: {list(SerializationMethod)}")
        self.serialization = serialization
        self._next_internal_key = 0
        self.key_map: dict[str, bytes] = {}  # external-to-internal key map
        self.max_encoded_value_length: int = 0  # in bytes
        self._reprod_metadata = pyine.utils.reprod.get_reprod_metadata()

    def __enter__(self) -> "LMDBWriter":
        """Context manager entry point; returns the LMDBWriter instance."""
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: typing.Any | None,
    ) -> None:
        """Context manager exit point.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        self.close()

    def __del__(self) -> None:
        """Destructor invoked when the object is garbage collected."""
        self.close()

    def close(self) -> None:
        """Close the LMDB environment."""
        if hasattr(self, "env") and self.env is not None:
            self._write_internal_metadata()
            self.env.close()
            self.env = None

    def _serialize(
        self,
        obj: typing.Any,
    ) -> bytes:
        """Serialize an object to bytes."""
        if self.serialization == SerializationMethod.PICKLE:
            return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if self.serialization == SerializationMethod.PICKLE_LZ4:
            pickled_data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            return lz4.frame.compress(pickled_data)
        if self.serialization == SerializationMethod.JSON:
            return json.dumps(obj).encode("utf-8")
        if self.serialization == SerializationMethod.JSON_LZ4:
            json_data = json.dumps(obj).encode("utf-8")
            return lz4.frame.compress(json_data)
        raise ValueError(f"invalid serialization method: {self.serialization}")

    def write_metadata(
        self,
        metadata: dict[str, typing.Any],
        overwrite: bool = False,
    ):
        """Write arbitrary metadata to the database, with optional overwrite protection.

        Args:
            metadata: Dictionary of metadata to write.
            overwrite: Whether to overwrite existing metadata fields.

        Returns:
            A dictionary mapping metadata fields to the internal keys used to store them.
        """
        output_keys = {}
        with self.env.begin(write=True) as txn:
            for key, value in metadata.items():
                key_bytes = _create_metadata_key(key)
                if not overwrite and txn.get(key_bytes) is not None:
                    raise ValueError(f"Metadata key '{key}' already exists. Use overwrite=True to replace it.")
                self._write_metadata_value(txn, key, value)
                output_keys[key] = key_bytes
        return output_keys

    def _write_internal_metadata(self):
        """Writes metadata to the database."""
        with self.env.begin(write=True) as txn:
            # store the next internal key for continuity (if needed)
            txn.put(_NEXT_INTERNAL_KEY, struct.pack(">Q", self._next_internal_key))
            self._write_metadata_value(txn, "map_size", self.map_size)
            self._write_metadata_value(txn, "sample_count", len(self.key_map))
            self._write_metadata_value(txn, "key_map", self.key_map)
            self._write_metadata_value(txn, "serialization", self.serialization.value)
            self._write_metadata_value(txn, "max_encoded_value_length", self.max_encoded_value_length)
            for key, val in self._reprod_metadata.items():
                self._write_metadata_value(txn, key, val)

    def _write_metadata_value(self, txn: lmdb.Transaction, field_name: str, value: typing.Any):
        """Writes a single metadata value to the database."""
        # note: for metadata, we always write data using pickle only
        key_bytes = _create_metadata_key(field_name)
        txn.put(key_bytes, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))

    def get_size_on_disk(self) -> int:
        """Calculate the total size of the LMDB dataset stored on disk (in bytes)."""
        return _get_database_size(self.path)

    def put(
        self,
        key: str,
        value: typing.Any,
    ) -> bytes:
        """Insert a key-value pair into the database.

        A value at the specified key should NOT already exist in the database.

        Note: we will use the internal key iterator to generate the REAL key to store this
        value, but will keep track of the original key for lookup purposes.

        Args:
            key: The (verbose, human-readable) key to store, which will be used for lookups.
            value: The value to store.

        Returns:
            The internal key used to store the value in the database.
        """
        if not isinstance(key, str):
            raise ValueError(f"key must be a string, but got: {type(key)}")
        if key in self.key_map:
            raise ValueError(f"key '{key}' already exists in the database")
        internal_key = _create_sample_key(self._next_internal_key)
        self.key_map[key] = internal_key
        self._next_internal_key += 1
        with self.env.begin(write=True) as txn:
            encoded_value = self._serialize(value)
            self.max_encoded_value_length = max(self.max_encoded_value_length, len(encoded_value))
            ret = txn.put(internal_key, encoded_value, overwrite=False)
            assert ret, "internal key collision"
        return internal_key

    def put_batch(
        self,
        items: dict[str, typing.Any],
        show_progress: bool = True,
    ) -> list[bytes]:
        """Insert multiple key-value pairs into the database.

        Values at the specified keys should NOT already exist in the database.

        Note: we will use the internal key iterator to generate the REAL keys to store these
        value, but will keep track of the original keys for lookup purposes.

        Args:
            items: Dictionary of key-value pairs to store.
            show_progress: Whether to display a progress bar.

        Returns:
            The list of internal keys used to store the values in the database.
        """
        generated_interal_keys = []
        items_iterator = tqdm.tqdm(items.items(), desc="Writing to database") if show_progress else items.items()
        with self.env.begin(write=True) as txn:
            for key, value in items_iterator:
                if not isinstance(key, str):
                    raise ValueError(f"key must be a string, but got: {type(key)}")
                if key in self.key_map:
                    raise ValueError(f"key '{key}' already exists in the database")
                internal_key = _create_sample_key(self._next_internal_key)
                self.key_map[key] = internal_key
                self._next_internal_key += 1
                encoded_value = self._serialize(value)
                self.max_encoded_value_length = max(self.max_encoded_value_length, len(encoded_value))
                ret = txn.put(internal_key, encoded_value, overwrite=False)
                assert ret, "internal key collision"
                generated_interal_keys.append(internal_key)
            return generated_interal_keys


class LMDBReader:
    """LMDB reader highly optimized for fast sequential reading.

    Examples:
        Basic usage to read a value by key or index:
        >>> reader = LMDBReader("path/to/db")
        >>> value = reader.get("key1")  # get by key
        >>> value = reader.get(0)       # get by index
        >>> reader.close()

        Reading metadata:
        >>> reader = LMDBReader("path/to/db")
        >>> metadata = reader.get_metadata()
        >>> print(metadata["dataset_name"])
        >>> reader.close()

        Sequential iteration through all items:
        >>> reader = LMDBReader("path/to/db")
        >>> for value in reader:  # or: for value in reader.iter_from():
        ...     print(value)
        >>> reader.close()

        Iterate through a specific range:
        >>> reader = LMDBReader("path/to/db")
        >>> for value in reader.iter_from(start_idx=10, end_idx=20):
        ...     print(value)
        >>> reader.close()

        Batch iteration for better performance:
        >>> reader = LMDBReader("path/to/db")
        >>> for batch in reader.iter_batched(batch_size=32):
        ...     print(f"Got batch of {len(batch)} items")
        >>> reader.close()

        Using context manager:
        >>> with LMDBReader("path/to/db") as reader:
        ...     print(f"Database has {len(reader)} items")
        ...     value = reader.get("key1")
    """

    def __init__(
        self,
        path: pathlib.Path | typing.AnyStr,
    ):
        """
        Initialize the optimized LMDB reader.

        Args:
            path: Path to the LMDB database
        """
        self.path: pathlib.Path = pathlib.Path(path)
        self.env = lmdb.open(
            str(self.path),
            readonly=True,  # open in read-only mode for better performance and safety
            readahead=True,  # always enable readahead for better sequential read performance
            lock=False,  # lock not needed in read-only mode
            max_readers=126,  # use default max readers
        )
        self._load_metadata()

    def _deserialize(self, data: bytes) -> typing.Any:
        """Deserialize bytes into an object."""
        if self.serialization == SerializationMethod.PICKLE:
            return pickle.loads(data)
        if self.serialization == SerializationMethod.PICKLE_LZ4:
            decompressed_data = lz4.frame.decompress(data)
            return pickle.loads(decompressed_data)
        if self.serialization == SerializationMethod.JSON:
            return json.loads(data.decode("utf-8"))
        if self.serialization == SerializationMethod.JSON_LZ4:
            decompressed_data = lz4.frame.decompress(data)
            return json.loads(decompressed_data.decode("utf-8"))
        raise ValueError(f"invalid serialization method: {self.serialization}")

    def _load_metadata(self):
        """Load metadata from the database."""
        with self.env.begin() as txn:
            next_internal_key_bytes = txn.get(_NEXT_INTERNAL_KEY)
            if next_internal_key_bytes is None:
                raise ValueError("database does not contain next_internal_key; did writing closure fail?")
            self._next_internal_key: int = struct.unpack(">Q", next_internal_key_bytes)[0]
            map_size_bytes = txn.get(_create_metadata_key("map_size"))
            self.orig_map_size: int = pickle.loads(map_size_bytes)
            sample_count_bytes = txn.get(_create_metadata_key("sample_count"))
            self.sample_count: int = pickle.loads(sample_count_bytes)
            serialization_bytes = txn.get(_create_metadata_key("serialization"))
            self.serialization = SerializationMethod(pickle.loads(serialization_bytes))
            max_encoded_value_length_bytes = txn.get(_create_metadata_key("max_encoded_value_length"))
            self.max_encoded_value_length: int = pickle.loads(max_encoded_value_length_bytes)
            self.key_map: dict[str, bytes] = pickle.loads(txn.get(_create_metadata_key("key_map")))
            assert len(self.key_map) == self.sample_count, "key_map length does not match sample_count"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self) -> None:
        self.close()

    def __len__(self):
        return len(self.key_map)

    def __iter__(self):
        return self.iter_from()

    def close(self):
        """Close the database."""
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
            self.env = None

    def get_metadata(self) -> dict[str, typing.Any]:
        """Returns a dictionary of all metadata stored in the database."""
        results = {}
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            cursor.set_range(METADATA_PREFIX)
            while cursor.key().startswith(METADATA_PREFIX):
                metadata_field_name = _decode_metadata_key(cursor.key())
                assert metadata_field_name not in results, f"duplicate metadata field: {metadata_field_name}"
                metadata_value_encoded = cursor.value()
                results[metadata_field_name] = pickle.loads(metadata_value_encoded)
                cursor.next()
        return results

    def get_size_on_disk(self) -> int:
        """Calculate the total size of the LMDB dataset stored on disk (in bytes)."""
        return _get_database_size(self.path)

    def get(self, key_or_idx: int | str) -> typing.Any:
        """Get a value by its key or dataset index."""
        if isinstance(key_or_idx, str):
            if key_or_idx not in self.key_map:
                raise ValueError(f"key '{key_or_idx}' not found in the database")
            key = self.key_map[key_or_idx]
        elif isinstance(key_or_idx, int):
            key = _create_sample_key(key_or_idx)
        else:
            raise ValueError(f"key_or_idx must be a string or integer, but got: {type(key_or_idx)}")
        with self.env.begin() as txn:
            value_bytes = txn.get(key)
            if value_bytes is not None:
                return self._deserialize(value_bytes)
            return None

    def get_indices(
        self,
        pattern: str,
        return_keys: bool = False,
    ) -> list[int] | tuple[list[int], list[str]]:
        """Get sample indices for all keys that match the provided fnmatch-compatible pattern.

        Args:
            pattern: pattern to match against keys (e.g., "problem/*", "test_data:[0-9]*")
            return_keys: if True, returns a tuple of (indices, keys) instead of just indices

        Returns:
            If `return_keys` is False: List of sample indices (integers) of matched keys, sorted in ascending order.
            If `return_keys` is True: Tuple of (indices, keys) where both lists are sorted by index order.
        """
        assert isinstance(pattern, str), "pattern must be a string"
        matched_keys = fnmatch.filter(self.key_map.keys(), pattern)
        index_key_pairs = [((_decode_sample_key(self.key_map[key])), key) for key in matched_keys]
        index_key_pairs.sort()
        if return_keys:
            indices, keys = zip(*index_key_pairs) if index_key_pairs else ([], [])
            return list(indices), list(keys)
        else:
            return [idx for idx, _ in index_key_pairs]

    def iter_from(
        self,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> typing.Iterator[typing.Any]:
        """Iterate through items starting from a specific index, yielding values sequentially."""
        assert isinstance(start_idx, int), "start_idx must be an integer"
        assert start_idx >= 0, "start_idx must be non-negative"
        if end_idx is not None:
            assert isinstance(end_idx, int), "end_idx must be an integer or None"
            assert end_idx >= start_idx, "end_idx must be greater than or equal to start_idx"
        start_key = _create_sample_key(start_idx)
        with self.env.begin() as txn:
            cursor = txn.cursor()
            found = cursor.set_range(start_key)
            while found:
                key, value = cursor.item()
                if not key.startswith(SAMPLE_PREFIX):
                    break
                current_idx = _decode_sample_key(key)
                if end_idx is not None and current_idx >= end_idx:
                    break
                yield self._deserialize(value)
                found = cursor.next()

    def iter_batched(
        self,
        batch_size: int,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> typing.Iterator[list[typing.Any]]:
        """
        Iterate through samples in batches for better performance, up to an optional end index.

        Args:
            batch_size: Number of samples to return in each batch
            start_idx: Index to start from (inclusive)
            end_idx: Index to stop at (exclusive)

        Yields:
            Batches of sample data
        """
        batch = []
        for sample in self.iter_from(start_idx, end_idx):
            batch.append(sample)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
