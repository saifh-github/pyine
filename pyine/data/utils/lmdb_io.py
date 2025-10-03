import contextlib
import enum
import fnmatch
import pathlib
import pickle
import struct
import typing

import lmdb
import lz4.frame
import msgspec
import orjson
import pydantic
import tqdm
import zstandard

import pyine.utils.reprod

__all__ = [
    "SerializationMethod",
    "SerializationConfig",
    "LMDBWriter",
    "LMDBReader",
]


class SerializationMethod(enum.StrEnum):
    """Supported serialization methods for LMDBWriter."""

    PICKLE = enum.auto()
    """Use Python's built-in pickle module (note: not recommended due to safety concerns)."""
    PICKLE_LZ4 = enum.auto()
    """Use Python's built-in pickle module with LZ4 compression (note: not recommended due to safety concerns)."""
    MSGSPEC = enum.auto()
    """Use the msgspec module for serialization."""
    JSON = enum.auto()
    """Use the orjson module for serialization (it's faster than the regular stdlib implementation)."""
    JSON_LZ4 = enum.auto()
    """Use the orjson module for serialization with LZ4 compression."""
    JSON_ZSTD = enum.auto()
    """Use the orjson module for serialization with ZSTD compression."""


class SerializationConfig(pydantic.BaseModel):
    """Serialization configuration for LMDBWriter."""

    model_config = pydantic.ConfigDict(frozen=True, use_enum_values=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    method: SerializationMethod | str = SerializationMethod.MSGSPEC
    """Serialization method to use."""
    compression_kwargs: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    """Compression arguments to pass to the compression method (unused if not compressing)."""
    allow_insecure_serialization: bool = False
    """Allow potentially unsafe serialization formats (e.g., pickle)."""


def _ensure_insecure_serialization_allowed(config: SerializationConfig) -> None:
    """Raise if insecure serialization methods are used without explicit opt-in."""
    if not config.allow_insecure_serialization:
        raise ValueError(
            "Insecure serialization method requested. Set allow_insecure_serialization=True to acknowledge the risk."
        )


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


def _get_database_size(path: pathlib.Path | str) -> int:
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
        ...     data = {"key1": {"data": "value1"}, "key2": {"data": "value2"}}
        ...     writer.put_batch(data)

        Writing with different serialization methods:
        >>> writer = LMDBWriter(
        ...     "path/to/db",
        ...     serialization_config=SerializationConfig(
        ...         SerializationConfig.JSON_ZSTD,
        ...         compression_kwargs={"level": 3},
        ...     ),
        ... )
        >>> writer.put("key1", {"data": "value1"})
        >>> writer.close()

        Writing metadata:
        >>> with LMDBWriter("path/to/db") as writer:
        ...     metadata = {"dataset_name": "example", "version": "1.0"}
        ...     writer.write_metadata(metadata)

        Specifying custom map size and readers:
        >>> writer = LMDBWriter(
        ...     "path/to/db",
        ...     map_size=2 * 1024 * 1024 * 1024,  # 2GB
        ...     max_readers=256,
        ... )
        >>> writer.put("key1", {"data": "value1"})
        >>> writer.close()
    """

    def __init__(
        self,
        path: pathlib.Path | str,
        map_size: int = 1 * (1024**4),  # 1TB default size; good for large datasets (what we want)
        max_readers: int = 126,  # typical default max readers for LMDB
        max_allowed_value_length: int = 2 * (1024**3),  # 2GB by default
        serialization_config: SerializationConfig | None = None,
    ) -> None:
        """Initialize the LMDB database.

        Args:
            path: Path where the LMDB will be stored.
            map_size: Maximum size database may grow to; defaults to 1TB (1 * 1024^4 bytes), which
                is good for large datasets, i.e. what we intend to create in this framework.
            max_readers: Maximum number of simultaneous readers (126 by default, typical for LMDB).
            serialization_config: Configuration specifying method to serialize objects.
        """
        self.path: pathlib.Path = pathlib.Path(path)
        self.map_size: int = map_size
        self.max_readers: int = max_readers
        self.path.mkdir(parents=True, exist_ok=True)
        self.serialization = serialization_config or SerializationConfig()
        self._env: lmdb.Environment | None = lmdb.open(  # type: ignore[reportUnknownMemberType]
            path=str(self.path),
            map_size=self.map_size,
            max_readers=self.max_readers,
            readonly=False,
        )
        self._next_internal_key = 0
        self.key_map: dict[str, bytes] = {}  # external-to-internal key map
        self.max_encoded_value_length: int = 0  # in bytes; will be tracked as we write the dataset
        self.max_allowed_value_length: int = max_allowed_value_length  # in bytes
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
        self.close(write_metadata=True)

    def __del__(self) -> None:
        """Destructor invoked when the object is garbage collected.

        Note: avoid performing I/O (metadata writes) here as interpreter shutdown
        order is undefined. As a safety net, we try to close the environment only.
        """
        with contextlib.suppress(Exception):
            self.close(write_metadata=False)

    def close(self, write_metadata: bool = True) -> None:
        """Close the LMDB environment.

        Args:
            write_metadata: When True (default), write internal metadata before closing.
                Set to False in contexts where writes are unsafe (e.g., __del__).
        """
        if hasattr(self, "_env") and self._env is not None:
            try:
                if write_metadata:
                    self._write_internal_metadata()
            finally:
                # always attempt to close the environment even if metadata write fails
                self._env.close()
                self._env = None

    @property
    def env(self) -> lmdb.Environment:
        """Get the underlying LMDB environment."""
        if self._env is None:
            raise RuntimeError("LMDB environment not initialized or previously closed")
        return self._env

    def _serialize(
        self,
        obj: typing.Any,
    ) -> bytes:
        """Serialize an object to bytes."""
        if self.serialization.method == SerializationMethod.PICKLE:
            _ensure_insecure_serialization_allowed(self.serialization)
            return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if self.serialization.method == SerializationMethod.PICKLE_LZ4:
            _ensure_insecure_serialization_allowed(self.serialization)
            pickled_data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            compressed_data = lz4.frame.compress(pickled_data, **self.serialization.compression_kwargs)  # type: ignore[reportUnknownMemberType]
            return typing.cast("bytes", compressed_data)
        if self.serialization.method == SerializationMethod.MSGSPEC:
            return msgspec.msgpack.encode(obj)
        if self.serialization.method == SerializationMethod.JSON:
            return orjson.dumps(obj)
        if self.serialization.method == SerializationMethod.JSON_LZ4:
            json_data = orjson.dumps(obj)
            compressed_data = lz4.frame.compress(json_data, **self.serialization.compression_kwargs)  # type: ignore[reportUnknownMemberType]
            return typing.cast("bytes", compressed_data)
        if self.serialization.method == SerializationMethod.JSON_ZSTD:
            zstd_compressor = zstandard.ZstdCompressor(**self.serialization.compression_kwargs)
            json_data = orjson.dumps(obj)
            return zstd_compressor.compress(json_data)
        raise NotImplementedError

    def write_metadata(self, metadata: dict[str, typing.Any], overwrite: bool = False) -> dict[str, bytes]:
        """Write arbitrary metadata to the database.

        Args:
            metadata: Dictionary of metadata to write.
            overwrite: Whether to overwrite existing metadata fields.

        Returns:
            A dictionary mapping metadata fields to the internal keys used to store them.
        """
        output_keys: dict[str, bytes] = {}
        with self.env.begin(write=True) as txn:  # type: ignore[reportUnknownMemberType]
            for key, value in metadata.items():
                key_bytes = _create_metadata_key(key)
                if not overwrite and txn.get(key_bytes) is not None:  # type: ignore[reportUnknownMemberType]
                    raise ValueError(f"Metadata key '{key}' already exists. Use overwrite=True to replace it.")
                self._write_metadata_value(txn, key, value)
                output_keys[key] = key_bytes
        return output_keys

    def _write_internal_metadata(self) -> None:
        """Writes fixed metadata fields as well as reproducibility tags to the database."""
        with self.env.begin(write=True) as txn:  # type: ignore[reportUnknownMemberType]
            # store the next internal key for continuity (if needed)
            txn.put(  # type: ignore[reportUnknownMemberType]
                _NEXT_INTERNAL_KEY,
                struct.pack(">Q", self._next_internal_key),
                overwrite=True,
            )
            self._write_metadata_value(txn, "map_size", self.map_size)
            self._write_metadata_value(txn, "sample_count", len(self.key_map))
            self._write_metadata_value(txn, "key_map", self.key_map)
            self._write_metadata_value(txn, "serialization", self.serialization.model_dump())
            self._write_metadata_value(txn, "max_encoded_value_length", self.max_encoded_value_length)
            for key, val in self._reprod_metadata.items():
                self._write_metadata_value(txn, key, val)

    def _write_metadata_value(
        self,
        txn: lmdb.Transaction,
        field_name: str,
        value: typing.Any,
    ) -> None:
        """Writes a single metadata value to the database."""
        # note: for metadata, we always write data using msgspec only
        key_bytes = _create_metadata_key(field_name)
        if not (0 < len(key_bytes) < self.env.max_key_size()):
            raise ValueError("metadata key length error")
        encoded_value = msgspec.msgpack.encode(value)
        if not (0 < len(encoded_value) < self.max_allowed_value_length):
            raise ValueError("metadata value length error")
        txn.put(key_bytes, encoded_value, overwrite=True)  # type: ignore[reportUnknownMemberType]

    def get_size_on_disk(self) -> int:
        """Calculate the total size of the LMDB dataset stored on disk (in bytes)."""
        return _get_database_size(self.path)

    def put(
        self,
        key: str,
        value: typing.Any,
    ) -> bytes:
        """Insert a key-value pair into the database.

        A value at the specified key should NOT exist in the database. If anything goes wrong
        when preparing the write operation, an exception will be raised, and the database will
        remain intact.

        Note: we will use the internal key iterator to generate the REAL key to store this
        value, but will keep track of the original key for lookup purposes.

        Args:
            key: The (verbose, human-readable) key to store, which will be used for lookups.
            value: The value to store.

        Returns:
            The internal key used to store the value in the database.
        """
        if key in self.key_map:
            raise ValueError(f"key '{key}' already exists in the database")
        internal_key = _create_sample_key(self._next_internal_key)
        if not (0 < len(internal_key) < self.env.max_key_size()):
            raise RuntimeError("internal key length error")
        with self.env.begin(write=True) as txn:  # type: ignore[reportUnknownMemberType]
            try:
                encoded_value = self._serialize(value)
            except Exception as e:
                raise RuntimeError(f"failed to serialize value for key: {key}") from e
            if not (0 < len(encoded_value) < self.max_allowed_value_length):
                raise ValueError("encoded value length error")
            ret = txn.put(internal_key, encoded_value, overwrite=True)  # type: ignore[reportUnknownMemberType]
            if not ret:
                raise RuntimeError("internal key collision")
        self.max_encoded_value_length = max(self.max_encoded_value_length, len(encoded_value))
        self.key_map[key] = internal_key
        self._next_internal_key += 1
        return internal_key

    def put_batch(
        self,
        items: dict[str, typing.Any],
        show_progress: bool = True,
        raise_on_error: bool = True,
    ) -> dict[str, bytes] | tuple[dict[str, bytes], dict[str, Exception]]:
        """Insert multiple key-value pairs into the database.

        Values at the specified keys should NOT exist in the database. If anything goes wrong
        when preparing the write operations, an exception will be raised or the operation will be
        skipped (depending on `raise_on_error`), keeping the database intact from that operation.
        If `raise_on_error` is False, the operation will be skipped and the exceptions will be
        returned alongside the generated internal keys.

        Note: we will use the internal key iterator to generate the REAL keys to store these
        value, but will keep track of the original keys for lookup purposes.

        Args:
            items: Dictionary of key-value pairs to store.
            show_progress: Whether to display a progress bar.
            raise_on_error: Whether to raise an exception if an error occurs during writing. If
                True, the return value should only be a dictionary mapping all successfully
                inserted keys. If False, the return value will be a tuple of maps containing keys
                that were successfully inserted and exceptions that occurred during writing.

        Returns:
            A map containing successfully inserted keys, and if `raise_on_error` is True, another
            map containing exceptions that occurred during writing.
        """
        generated_internal_keys: dict[str, bytes] = {}  # input key to internal key mapping
        encountered_errors: dict[str, Exception] = {}  # (failed) input key to exception mapping
        items_iterator = tqdm.tqdm(items.items(), desc="Writing to database") if show_progress else items.items()
        with self.env.begin(write=True) as txn:  # type: ignore[reportUnknownMemberType]
            for key, value in items_iterator:
                try:
                    internal_key = _create_sample_key(self._next_internal_key)
                    if not (0 < len(internal_key) < self.env.max_key_size()):
                        raise RuntimeError("internal key length error")
                    try:
                        encoded_value = self._serialize(value)
                    except Exception as e:
                        raise RuntimeError(f"failed to serialize value for key: {key}") from e
                    if not (0 < len(encoded_value) < self.max_allowed_value_length):
                        raise ValueError("encoded value length error")
                    ret = txn.put(internal_key, encoded_value, overwrite=True)  # type: ignore[reportUnknownMemberType]
                    if not ret:
                        raise RuntimeError("internal key collision")
                except Exception as e:
                    if raise_on_error:
                        raise RuntimeError(f"failed to write value for key: {key}") from e
                    encountered_errors[key] = e
                    continue
                self.key_map[key] = internal_key
                self._next_internal_key += 1
                self.max_encoded_value_length = max(self.max_encoded_value_length, len(encoded_value))
                generated_internal_keys[key] = internal_key
            if raise_on_error:
                return generated_internal_keys
            return generated_internal_keys, encountered_errors


class LMDBReader:
    """LMDB reader highly optimized for fast sequential reading.

    Examples:
        Basic usage to read a value by key or index:
        >>> reader = LMDBReader("path/to/db")
        >>> value_by_key = reader.get("key1")  # get by key
        >>> value_by_index = reader.get(0)  # get by index
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
        path: pathlib.Path | str,
    ) -> None:
        """
        Initialize the optimized LMDB reader.

        Args:
            path: Path to the LMDB database
        """
        self.path: pathlib.Path = pathlib.Path(path)
        self._env: lmdb.Environment | None = lmdb.open(  # type: ignore[reportUnknownMemberType]
            str(self.path),
            readonly=True,  # open in read-only mode for better performance and safety
            readahead=True,  # always enable readahead for better sequential read performance
            lock=False,  # lock not needed in read-only mode
            max_readers=126,  # use default max readers
        )
        self._load_metadata()

    def close(self) -> None:
        """Close the database."""
        if hasattr(self, "_env") and self._env is not None:
            self._env.close()
            self._env = None

    @property
    def env(self) -> lmdb.Environment:
        """Get the underlying LMDB environment."""
        if self._env is None:
            raise RuntimeError("LMDB environment not initialized or previously closed")
        return self._env

    def _deserialize(self, data: bytes) -> typing.Any:
        """Deserialize bytes into an object."""
        if self.serialization.method == SerializationMethod.PICKLE:
            _ensure_insecure_serialization_allowed(self.serialization)
            return pickle.loads(data)  # noqa: S301 - gated by allow_insecure_serialization
        if self.serialization.method == SerializationMethod.PICKLE_LZ4:
            _ensure_insecure_serialization_allowed(self.serialization)
            decompressed_data = typing.cast("bytes", lz4.frame.decompress(data))  # type: ignore[reportUnknownMemberType]
            return pickle.loads(decompressed_data)  # noqa: S301 - gated by allow_insecure_serialization
        if self.serialization.method == SerializationMethod.MSGSPEC:
            return msgspec.msgpack.decode(data)
        if self.serialization.method == SerializationMethod.JSON:
            return orjson.loads(data)
        if self.serialization.method == SerializationMethod.JSON_LZ4:
            decompressed_data = typing.cast("bytes", lz4.frame.decompress(data))  # type: ignore[reportUnknownMemberType]
            return orjson.loads(decompressed_data)
        if self.serialization.method == SerializationMethod.JSON_ZSTD:
            zstd_decompressor = zstandard.ZstdDecompressor()
            decompressed_data: bytes = zstd_decompressor.decompress(data)
            return orjson.loads(decompressed_data)
        raise NotImplementedError

    def _load_metadata(self) -> None:
        """Load metadata from the database."""
        # note: for metadata, we always store stuff with msgspec
        with self.env.begin() as txn:  # type: ignore[reportUnknownMemberType]

            def _get_and_decode_metadata(key: str) -> typing.Any:
                metadata_key_bytes: bytes = _create_metadata_key(key)
                metadata_val_bytes: bytes | None = txn.get(metadata_key_bytes)  # type: ignore[reportUnknownMemberType]
                assert metadata_val_bytes is not None
                return msgspec.msgpack.decode(metadata_val_bytes)

            next_internal_key_bytes: bytes | None = txn.get(_NEXT_INTERNAL_KEY)  # type: ignore[reportUnknownMemberType]
            if next_internal_key_bytes is None:
                raise ValueError("database does not contain next_internal_key; did writing closure fail?")
            self._next_internal_key: int = struct.unpack(">Q", next_internal_key_bytes)[0]  # type: ignore[reportUnknownArgumentType]
            self.orig_map_size: int = _get_and_decode_metadata("map_size")
            self.sample_count: int = _get_and_decode_metadata("sample_count")
            serialization_raw: dict[str, typing.Any] = _get_and_decode_metadata("serialization")
            self.serialization = SerializationConfig(**serialization_raw)
            self.max_encoded_value_length: int = _get_and_decode_metadata("max_encoded_value_length")
            self.key_map: dict[str, bytes] = _get_and_decode_metadata("key_map")
            if len(self.key_map) != self.sample_count:
                raise RuntimeError("key_map length does not match sample_count")

    def __enter__(self) -> "LMDBReader":
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: typing.Any | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.key_map)

    def __iter__(self) -> typing.Iterator[typing.Any]:
        return self.iter_from()

    def get_metadata(self) -> dict[str, typing.Any]:
        """Returns a dictionary of all metadata stored in the database."""
        # note: for metadata, we always store stuff with msgspec
        results: dict[str, typing.Any] = {}
        with self.env.begin(write=False) as txn:  # type: ignore[reportUnknownMemberType]
            cursor = txn.cursor()  # type: ignore[reportUnknownMemberType]
            found = cursor.set_range(METADATA_PREFIX)  # type: ignore[reportUnknownMemberType]
            while found:
                key = typing.cast("bytes | None", cursor.key())
                if key is None or not key.startswith(METADATA_PREFIX):
                    break
                metadata_field_name = _decode_metadata_key(key)
                if metadata_field_name in results:
                    raise RuntimeError(f"duplicate metadata field: {metadata_field_name}")
                metadata_value_encoded = typing.cast("bytes", cursor.value())
                results[metadata_field_name] = msgspec.msgpack.decode(metadata_value_encoded)
                found = cursor.next()
        return results

    def get_size_on_disk(self) -> int:
        """Calculate the total size of the LMDB dataset stored on disk (in bytes)."""
        return _get_database_size(self.path)

    def get(self, key_or_idx: int | str) -> typing.Any:
        """Get a value by its key or dataset index."""
        if isinstance(key_or_idx, str):
            if key_or_idx not in self.key_map:
                raise KeyError(f"key '{key_or_idx}' not found in the database")
            key = self.key_map[key_or_idx]
        else:
            if not (0 <= key_or_idx < len(self)):
                raise IndexError(f"index {key_or_idx} out of range (0 <= index < {len(self)})")
            key = _create_sample_key(key_or_idx)
        with self.env.begin() as txn:  # type: ignore[reportUnknownMemberType]
            output: bytes | None = txn.get(key)  # type: ignore[reportUnknownMemberType]
            if output is not None:
                output = self._deserialize(output)
            return output

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
        matched_keys = fnmatch.filter(self.key_map.keys(), pattern)
        index_key_pairs: list[tuple[int, str]] = [(_decode_sample_key(self.key_map[key]), key) for key in matched_keys]
        index_key_pairs.sort()
        if return_keys:
            indices, keys = zip(*index_key_pairs, strict=False) if index_key_pairs else ([], [])
            return list(indices), list(keys)
        return [idx for idx, _ in index_key_pairs]

    def iter_from(
        self,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> typing.Iterator[typing.Any]:
        """Iterate through items starting from a specific index, yielding values sequentially."""
        if not isinstance(start_idx, int):
            raise TypeError("start_idx must be an integer")
        if start_idx < 0:
            raise ValueError("start_idx must be non-negative")
        if end_idx is not None:
            if not isinstance(end_idx, int):
                raise TypeError("end_idx must be an integer or None")
            if end_idx < start_idx:
                raise ValueError("end_idx must be greater than or equal to start_idx")
        start_key = _create_sample_key(start_idx)
        with self.env.begin() as txn:  # type: ignore[reportUnknownMemberType]
            cursor = txn.cursor()  # type: ignore[reportUnknownMemberType]
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
