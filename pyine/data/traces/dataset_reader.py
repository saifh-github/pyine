"""
This module contains a reader for the PyINE dataset of 'raw' code execution traces.

See the corresponding dataset writer and utility modules for more information on this
dataset format. See also the demo notebook (in the project's root `notebooks` directory)
for an example of how to use this dataset reader.
"""

import collections
import fnmatch
import functools
import logging
import pathlib
import typing

import msgspec
import torch.utils.data

import pyine.data.traces.dataset_utils
import pyine.data.utils.filter_rules
import pyine.data.utils.lmdb_io
import pyine.utils.code.execution
import pyine.utils.reprod

logger = logging.getLogger(__name__)


class DatasetReader(torch.utils.data.Dataset):
    """PyINE raw trace dataset reader.

    This reader provides execution traces stored in an LMDB dataset. Each trace corresponds to a
    successful code execution made for a specific solution to a coding problem, using a specific
    set of input arguments that are paired with an expected output value. These input/output pairs
    form a 'test', and the execution attempt is considered successful if the output value matches
    the expected value.

    Upon initialization, the reader attempts to load the prepared trace metadata from the dataset's
    LMDB directory. If the metadata is not found, it will prepare the metadata and save it there.

    Note: this readers does NOT attempt to structure the trace steps into deltas, so they will be
    quite verbose, likely too much so for most applications with reasoning models.

    Args:
        lmdb_path: Path to the LMDB database containing code traces.
    """

    def __init__(
        self,
        lmdb_path: pathlib.Path | typing.AnyStr,
    ) -> None:
        super().__init__()
        self.path = lmdb_path
        self.reader = pyine.data.utils.lmdb_io.LMDBReader(lmdb_path)
        self._problem_data_cache: collections.OrderedDict[
            int,
            pyine.data.traces.dataset_utils.CodingProblem,
        ] = collections.OrderedDict()
        self._problem_cache_max_size = 512
        self._init_trace_metadata()

    def _is_metadata_prepared(self) -> bool:
        """Returns True if the trace maps and metadata are prepared and ready to be used."""
        return self._get_prepared_metadata_path().is_file()

    def _save_prepared_metadata(self) -> None:
        """Saves the prepared trace maps and metadata to the lmdb directory."""
        trace_maps_and_metadata = {
            "problem_indices": self._problem_indices,
            "problem_keys": self.problem_keys,
            "trace_indices": self._trace_indices,
            "trace_keys": self.trace_keys,
            "trace_idx_to_problem_idx": self._trace_idx_to_problem_idx,
            "trace_key_to_problem_key": self.trace_key_to_problem_key,
            "augment_idx_to_parent_trace_idx": self._augment_idx_to_parent_trace_idx,
            "augment_key_to_parent_trace_key": self.augment_key_to_parent_trace_key,
            "trace_metadata": self.trace_metadata,
        }
        encoded_data = msgspec.msgpack.encode(trace_maps_and_metadata)
        with open(self._get_prepared_metadata_path(), "wb") as fd:
            fd.write(encoded_data)

    def _load_prepared_metadata(self) -> None:
        """Loads the prepared trace maps and metadata from the lmdb directory."""
        with open(self._get_prepared_metadata_path(), "rb") as fd:
            encoded_data = msgspec.msgpack.decode(fd.read())
        # key lists are public attributes, as there is little chance of confusion about their contents
        self.problem_keys: list[str] = encoded_data["problem_keys"]
        self.trace_keys: list[str] = encoded_data["trace_keys"]
        self.trace_key_to_problem_key: dict[str, str] = encoded_data["trace_key_to_problem_key"]
        self.augment_key_to_parent_trace_key: dict[str, str] = encoded_data["augment_key_to_parent_trace_key"]
        self.trace_metadata: list[pyine.data.traces.dataset_utils.TraceMetadata] = [
            pyine.data.traces.dataset_utils.TraceMetadata(**trace_meta) for trace_meta in encoded_data["trace_metadata"]
        ]
        # indices lists are private attributes, as they correspond to indices from the internal database
        self._trace_indices: list[int] = encoded_data["trace_indices"]
        self._trace_idx_to_problem_idx: dict[int, int] = encoded_data["trace_idx_to_problem_idx"]
        self._augment_idx_to_parent_trace_idx: dict[int, int] = encoded_data["augment_idx_to_parent_trace_idx"]

    def _clear_prepared_metadata(self) -> None:
        """Clears the prepared trace metadata from the lmdb directory."""
        if self._is_metadata_prepared():
            self._get_prepared_metadata_path().unlink()

    def _get_prepared_metadata_path(self) -> pathlib.Path:
        """Returns the file path used to store prepared trace metadata in the lmdb directory."""
        assert self.path.is_dir(), f"unexpected non-directory lmdb path: {self.path}"
        return self.path / "trace_metadata.msgspec"

    def _init_trace_metadata(self, force: bool = False) -> None:
        """Initializes the trace index maps and metadata potentially using already-cached data."""
        # note: the indices kept in the lists below are INTERNAL ones that map to the lmdb content
        if not force and self._is_metadata_prepared():
            self._load_prepared_metadata()
            return
        self._clear_prepared_metadata()
        self._problem_indices, self.problem_keys = self.reader.get_indices(
            pattern=pyine.data.traces.dataset_utils.PROBLEM_DATA_PATTERN,
            return_keys=True,
        )
        if len(self._problem_indices) == 0:
            raise ValueError("no problem data found in the dataset")
        if len(self._problem_indices) != len(self.problem_keys):
            raise RuntimeError("problem indices/keys length mismatch")
        self._trace_indices: list[int] = []
        self.trace_keys: list[str] = []
        self._trace_idx_to_problem_idx: dict[int, int] = {}
        self.trace_key_to_problem_key: dict[str, str] = {}
        self._augment_idx_to_parent_trace_idx: dict[int, int] = {}
        self.augment_key_to_parent_trace_key: dict[str, str] = {}
        self.trace_metadata: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
        for prob_iter_idx, (problem_idx, problem_key) in enumerate(
            zip(self._problem_indices, self.problem_keys, strict=False)
        ):
            if not problem_key.endswith(pyine.data.traces.dataset_utils.PROBLEM_DATA_SUFFIX):
                raise ValueError(f"malformed problem key: {problem_key}")
            # fix problem key by removing the problem metadata suffix
            problem_key = problem_key[: -len(pyine.data.traces.dataset_utils.PROBLEM_DATA_SUFFIX)]
            self.problem_keys[prob_iter_idx] = problem_key
            problem_data = self.reader.get(problem_idx)
            problem_data = pyine.data.traces.dataset_utils.CodingProblem.model_validate(problem_data)
            curr_trace_data_pattern = problem_key + pyine.data.traces.dataset_utils.TRACE_DATA_SUFFIX
            curr_augm_trace_data_pattern = problem_key + pyine.data.traces.dataset_utils.AUGM_TRACE_DATA_SUFFIX
            curr_trace_indices, curr_trace_keys = self.reader.get_indices(
                pattern=curr_trace_data_pattern,
                return_keys=True,
            )
            assert len(curr_trace_indices) == len(curr_trace_keys)
            # for 'forward-compatibility' with deltas datasets, remove any elements with the deltas suffix
            curr_trace_indices, curr_trace_keys = zip(
                *[
                    (idx, key)
                    for idx, key in zip(curr_trace_indices, curr_trace_keys, strict=False)
                    if not key.endswith(pyine.data.traces.dataset_utils.DELTAS_SUFFIX)
                ],
                strict=False,
            )
            if len(curr_trace_indices) == 0:
                raise ValueError(f"no trace data found for problem: {problem_key}")
            if len(curr_trace_indices) != len(curr_trace_keys):
                raise RuntimeError("trace indices/keys length mismatch")
            self._trace_indices.extend(curr_trace_indices)
            self.trace_keys.extend(curr_trace_keys)
            curr_augm_key_to_parent_key: dict[str, str] = {}
            for trace_idx, trace_key in zip(curr_trace_indices, curr_trace_keys, strict=False):
                self._trace_idx_to_problem_idx[trace_idx] = problem_idx
                self.trace_key_to_problem_key[trace_key] = problem_key
                if fnmatch.fnmatch(trace_key, curr_augm_trace_data_pattern):
                    augm_trace_id = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(trace_key)
                    parent_trace_id = augm_trace_id.get_augmentless_identifier()
                    curr_augm_key_to_parent_key[trace_key] = str(parent_trace_id)
                # note: we combine problem tags, trace (exec) tags, and augmentation tags into a single list
                trace_data = self.reader.get(trace_idx)
                trace_data = pyine.utils.code.execution.TraceResult.model_validate(trace_data)
                assert trace_data.identifier is not None, "trace identifier is required"
                assert trace_data.identifier == trace_key, "trace identifier mismatch"
                trace_id = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(trace_data.identifier)
                curr_trace_tags = []
                curr_trace_tags.extend(problem_data.problem_tags)
                curr_trace_tags.extend(trace_data.tags)
                if trace_id.is_augmented:
                    curr_trace_tags.append(f"augment:{trace_id.augment_category}")
                self.trace_metadata.append(
                    pyine.data.traces.dataset_utils.TraceMetadata(
                        identifier=trace_data.identifier,
                        parent_dataset_hash=self.hash,
                        index=len(self.trace_metadata),  # increments as we append new traces to this list
                        internal_index=trace_idx,
                        step_count=trace_data.valid_step_count,
                        code_string=trace_data.code_string,
                        inputs=trace_data.inputs,
                        expected_output=trace_data.expected_output,
                        return_value=trace_data.return_value,
                        exception=trace_data.exception,
                        stdout=trace_data.stdout,
                        stderr=trace_data.stderr,
                        metadata=trace_data.metadata,
                        tags=curr_trace_tags,
                    )
                )
            for augm_key, parent_key in curr_augm_key_to_parent_key.items():
                if parent_key not in self.trace_keys:
                    raise KeyError(f"augmentation parent trace key {parent_key} not found in dataset")
                augm_idx = self._trace_indices[self.trace_keys.index(augm_key)]
                parent_idx = self._trace_indices[self.trace_keys.index(parent_key)]
                self._augment_idx_to_parent_trace_idx[augm_idx] = parent_idx
            self.augment_key_to_parent_trace_key.update(curr_augm_key_to_parent_key)
        self._save_prepared_metadata()

    def __len__(self) -> int:
        """Returns the total number of traces in the dataset accessible via ``__getitem__``"""
        return len(self.trace_keys)

    @functools.cached_property
    def metadata(self) -> dict[str, typing.Any]:
        """Returns a dictionary of all metadata stored in the database."""
        return self.reader.get_metadata()

    @functools.cached_property
    def size_on_disk(self) -> int:
        """Returns the total size of the LMDB dataset stored on disk (in bytes)."""
        return self.reader.get_size_on_disk()

    @functools.cached_property
    def hash(self) -> str:
        """Returns the hash of this dataset (computed from relevant lmdb files on disk)."""
        # note: we don't compute the hash over the entire lmdb dir, just over the file that matters
        # (that folder will likely contain other stuff such as processing logs and metadata caches)
        expected_data_mdb_file = self.reader.path / "data.mdb"
        assert expected_data_mdb_file.is_file(), f"missing data.mdb file: {expected_data_mdb_file}"
        return pyine.utils.reprod.compute_hash(expected_data_mdb_file)

    @functools.cached_property
    def parent_dataset_name(self) -> str:
        """Returns the name of the parent dataset used to create this dataset."""
        return self.metadata["parent_dataset"]["dataset_name"]

    def _resolve_trace_index_or_key(
        self,
        index_or_key: int | str,
    ) -> int:
        """Returns the dataset position for the provided user-facing index or key."""
        if isinstance(index_or_key, int):
            if 0 <= index_or_key < len(self):
                return index_or_key
            raise IndexError(f"index {index_or_key} out of range")
        return self._resolve_trace_key(index_or_key)

    def _resolve_trace_key(
        self,
        trace_key: str,
    ) -> int:
        """Returns the dataset position for the provided user-facing key."""
        if trace_key in self.trace_keys:
            return self.trace_keys.index(trace_key)
        raise KeyError(f"key {trace_key} not found in dataset")

    def __getitem__(self, index_or_key: int | str) -> pyine.utils.code.execution.TraceResult:
        """Fetches an individual trace data object from the LMDB database by external index or key.

        Args:
            index_or_key: index or key of the trace to retrieve.

        Returns:
            A `TraceResult` object containing the requested trace data.
        """
        trace_idx = self._resolve_trace_index_or_key(index_or_key)
        internal_trace_idx = self._trace_indices[trace_idx]
        return pyine.utils.code.execution.TraceResult.model_validate(self.reader.get(internal_trace_idx))

    def get_problem_data(
        self,
        index_or_key: int | str,
    ) -> pyine.data.traces.dataset_utils.CodingProblem:
        """Fetches the problem data associated with a trace by external index or key.

        Args:
            index_or_key: Index or key of the trace for which to retrieve parent problem data.

        Returns:
            A `CodingProblem` object containing the problem data associated with the trace.
        """
        trace_idx = self._resolve_trace_index_or_key(index_or_key)
        internal_trace_idx = self._trace_indices[trace_idx]
        cached = self._problem_data_cache.get(internal_trace_idx)
        if cached is not None:
            self._problem_data_cache.move_to_end(internal_trace_idx)
            return cached
        problem_idx = self._trace_idx_to_problem_idx[internal_trace_idx]
        problem_data = pyine.data.traces.dataset_utils.CodingProblem.model_validate(self.reader.get(problem_idx))
        self._problem_data_cache[internal_trace_idx] = problem_data
        if len(self._problem_data_cache) > self._problem_cache_max_size:
            self._problem_data_cache.popitem(last=False)
        return problem_data

    def get_trace_metadata(self, index_or_key: int | str) -> pyine.data.traces.dataset_utils.TraceMetadata:
        """Returns the metadata associated with a trace by external index or key."""
        trace_idx = self._resolve_trace_index_or_key(index_or_key)
        return self.trace_metadata[trace_idx]

    def get_tags(self, index_or_key: int | str) -> list[str]:
        """Returns a list of tags for a given trace so that we can decide whether to filter it."""
        trace_idx = self._resolve_trace_index_or_key(index_or_key)
        return self.trace_metadata[trace_idx].tags.copy()

    def __str__(self) -> str:
        """Returns a string representation of the dataset reader (for debugging purposes)."""
        return f"{self.__class__.__name__}({self.path}) with {len(self)} instances"


DatasetOrDatasetPath = pathlib.Path | typing.AnyStr | DatasetReader


def get_traces_metadata(
    datasets: list[DatasetOrDatasetPath] | DatasetOrDatasetPath,
    base_filter: pyine.data.utils.filter_rules.FilterType | None = None,
) -> list[pyine.data.traces.dataset_utils.TraceMetadata]:
    """Returns a list of TraceMetadata objects for all traces in the provided dataset(s).

    Args:
        datasets: the path or dataset object (or list of) from which to get trace metadata.
        base_filter: filter to apply to the dataset reader(s) to get target traces. If None, then
            no filter will be applied, and metadata for all available traces will be returned.

    Returns:
        A list of TraceMetadata objects for all targeted traces in the provided dataset.
    """
    if not isinstance(datasets, list):
        datasets = [datasets]
    readers = [d if isinstance(d, torch.utils.data.Dataset) else DatasetReader(d) for d in datasets]
    logger.info(f"preparing traces metadata for {len(readers)} dataset reader(s)...")
    all_trace_keys = []
    for reader in readers:
        all_trace_keys.extend(reader.trace_keys)
    if len(set(all_trace_keys)) != len(all_trace_keys):
        raise ValueError("there should be no duplicates in the list of trace keys across all datasets")
    base_traces_meta: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
    for reader in readers:
        for trace_idx in range(len(reader)):
            tags = reader.get_tags(trace_idx)
            is_banned = base_filter(tags) if base_filter is not None else False
            if not is_banned:
                trace_metadata = reader.get_trace_metadata(trace_idx)
                assert reader.trace_keys[trace_idx] == trace_metadata.identifier
                assert trace_idx == trace_metadata.index
                assert reader.hash == trace_metadata.parent_dataset_hash
                base_traces_meta.append(trace_metadata)
    return base_traces_meta
