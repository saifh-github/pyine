"""
This module contains a reader for the PyINE dataset of 'raw' code execution traces.

See the corresponding dataset writer and utility modules for more information on this
dataset format. See also the demo notebook (in the project's root `notebooks` directory)
for an example of how to use this dataset reader.
"""

import fnmatch
import functools
import pathlib
import typing

import msgspec
import torch.utils.data

import pyine.data.traces.dataset_utils
import pyine.data.utils.lmdb_io
import pyine.utils.code.execution
import pyine.utils.reprod


class DatasetReader(torch.utils.data.Dataset):
    """PyINE raw trace dataset reader.

    This reader provides execution traces stored in an LMDB dataset. Each trace corresponds to a
    successful code execution made for a specific solution to a coding problem, using a specific
    set of input arguments that are paired with an expected output value. These input/output pairs
    form a 'test', and the execution attempt is considered successful if the output value matches
    the expected value.

    Note: this readers does NOT attempt to structure the traces into deltas, so they will be quite
    verbose, likely too much so for most applications with reasoning models.

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
        self._init_trace_maps()

    def _are_trace_maps_prepared(self) -> bool:
        """Returns True if the trace maps are prepared and ready to be used."""
        return self._get_prepared_trace_maps_file_path().is_file()

    def _save_prepared_trace_maps(self) -> None:
        """Saves the prepared trace maps to the lmdb directory."""
        trace_maps = dict(
            problem_indices=self._problem_indices,
            problem_keys=self.problem_keys,
            trace_indices=self._trace_indices,
            trace_keys=self.trace_keys,
            trace_idx_to_problem_idx=self._trace_idx_to_problem_idx,
            trace_key_to_problem_key=self.trace_key_to_problem_key,
            augment_idx_to_parent_trace_idx=self._augment_idx_to_parent_trace_idx,
            augment_key_to_parent_trace_key=self.augment_key_to_parent_trace_key,
            trace_tag_lists=self.trace_tag_lists,
        )
        encoded_data = msgspec.msgpack.encode(trace_maps)
        with open(self._get_prepared_trace_maps_file_path(), "wb") as fd:
            fd.write(encoded_data)

    def _load_prepared_trace_maps(self):
        """Loads the prepared trace maps from the lmdb directory."""
        with open(self._get_prepared_trace_maps_file_path(), "rb") as fd:
            encoded_data = msgspec.msgpack.decode(fd.read())
        # key lists are public attributes, as there is little chance of confusion about their contents
        self.problem_keys: list[str] = encoded_data["problem_keys"]
        self.trace_keys: list[str] = encoded_data["trace_keys"]
        self.trace_key_to_problem_key: dict[str, str] = encoded_data["trace_key_to_problem_key"]
        self.augment_key_to_parent_trace_key: dict[str, str] = encoded_data["augment_key_to_parent_trace_key"]
        self.trace_tag_lists: list[list[str]] = encoded_data["trace_tag_lists"]
        # indices lists are private attributes, as they correspond to indices from the internal database
        self._trace_indices: list[int] = encoded_data["trace_indices"]
        self._trace_idx_to_problem_idx: dict[int, int] = encoded_data["trace_idx_to_problem_idx"]
        self._augment_idx_to_parent_trace_idx: dict[int, int] = encoded_data["augment_idx_to_parent_trace_idx"]

    def _clear_prepared_trace_maps(self) -> None:
        """Clears the prepared trace maps from the lmdb directory."""
        if self._are_trace_maps_prepared():
            self._get_prepared_trace_maps_file_path().unlink()

    def _get_prepared_trace_maps_file_path(self) -> pathlib.Path:
        """Returns the file path used to store prepared trace maps in the lmdb directory."""
        # note: the file name that will be created contains a hash that depends on input params
        assert self.path.is_dir(), f"unexpected non-directory lmdb path: {self.path}"
        return self.path / "trace_maps.msgspec"

    def _init_trace_maps(self, force: bool = False) -> None:
        """Initializes the trace index maps potentially using already-cached data."""
        # note: the indices kept in the lists below are INTERNAL ones that map to the lmdb content
        if not force and self._are_trace_maps_prepared():
            self._load_prepared_trace_maps()
            return
        self._clear_prepared_trace_maps()
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
        self.trace_tag_lists: list[list[str]] = []
        for iter_idx, (problem_idx, problem_key) in enumerate(zip(self._problem_indices, self.problem_keys)):
            if not problem_key.endswith(pyine.data.traces.dataset_utils.PROBLEM_DATA_SUFFIX):
                raise ValueError(f"malformed problem key: {problem_key}")
            # fix problem key by removing the problem metadata suffix
            problem_key = problem_key[: -len(pyine.data.traces.dataset_utils.PROBLEM_DATA_SUFFIX)]
            self.problem_keys[iter_idx] = problem_key
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
                    for idx, key in zip(curr_trace_indices, curr_trace_keys)
                    if not key.endswith(pyine.data.traces.dataset_utils.DELTAS_SUFFIX)
                ]
            )
            if len(curr_trace_indices) == 0:
                raise ValueError(f"no trace data found for problem: {problem_key}")
            if len(curr_trace_indices) != len(curr_trace_keys):
                raise RuntimeError("trace indices/keys length mismatch")
            self._trace_indices.extend(curr_trace_indices)
            self.trace_keys.extend(curr_trace_keys)
            curr_augm_key_to_parent_key: dict[str, str] = {}
            for trace_idx, trace_key in zip(curr_trace_indices, curr_trace_keys):
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
                self.trace_tag_lists.append(curr_trace_tags)
            for augm_key, parent_key in curr_augm_key_to_parent_key.items():
                if parent_key not in self.trace_keys:
                    raise KeyError(f"augmentation parent trace key {parent_key} not found in dataset")
                augm_idx = self._trace_indices[self.trace_keys.index(augm_key)]
                parent_idx = self._trace_indices[self.trace_keys.index(parent_key)]
                self._augment_idx_to_parent_trace_idx[augm_idx] = parent_idx
            self.augment_key_to_parent_trace_key.update(curr_augm_key_to_parent_key)
        self._save_prepared_trace_maps()

    def __len__(self) -> int:
        """Returns the total number of traces in the dataset accessible via ``__getitem__``"""
        return len(self.trace_keys)

    def get_metadata(self) -> dict[str, typing.Any]:
        """Returns a dictionary of all metadata stored in the database."""
        return self.reader.get_metadata()

    def get_size_on_disk(self) -> int:
        """Calculate the total size of the LMDB dataset stored on disk (in bytes)."""
        return self.reader.get_size_on_disk()

    def get_hash(self) -> str:
        """Returns the hash of this dataset (computed from relevant lmdb files on disk)."""
        # note: we don't compute the hash over the entire lmdb dir, just over the file that matters
        # (that folder will likely contain other stuff such as processing logs and metadata caches)
        expected_data_mdb_file = self.reader.path / "data.mdb"
        assert expected_data_mdb_file.is_file(), f"missing data.mdb file: {expected_data_mdb_file}"
        return pyine.utils.reprod.compute_hash(expected_data_mdb_file)

    def get_parent_dataset_name(self) -> str:
        """Returns the name of the parent dataset used to create this dataset."""
        return self.reader.get_metadata()["parent_dataset"]["dataset_name"]

    def _get_trace_idx_from_idx_or_key(self, index_or_key: int | str) -> int:
        """Returns an external trace index from an external index or key."""
        if isinstance(index_or_key, int):
            if not (0 <= index_or_key < len(self)):
                raise IndexError(f"index {index_or_key} out of range")
            return index_or_key
        elif isinstance(index_or_key, str):
            if index_or_key not in self.trace_keys:
                raise KeyError(f"key {index_or_key} not found in dataset")
            return self.trace_keys.index(index_or_key)
        raise ValueError(f"invalid index_or_key type: {type(index_or_key)}")

    def __getitem__(self, index_or_key: int | str) -> pyine.utils.code.execution.TraceResult:
        """Fetches an individual trace data object from the LMDB database by external index or key.

        Args:
            index_or_key: index or key of the trace to retrieve.

        Returns:
            A `TraceResult` object containing the requested trace data.
        """
        trace_idx = self._get_trace_idx_from_idx_or_key(index_or_key)
        internal_trace_idx = self._trace_indices[trace_idx]
        trace_data = self.reader.get(internal_trace_idx)
        trace_data = pyine.utils.code.execution.TraceResult.model_validate(trace_data)
        return trace_data

    @functools.lru_cache(maxsize=512)
    def get_problem_data(self, index_or_key: int | str) -> pyine.data.traces.dataset_utils.CodingProblem:
        """Fetches the problem data associated with a trace by external index or key.

        Args:
            index_or_key: Index or key of the trace for which to retrieve parent problem data.

        Returns:
            A `CodingProblem` object containing the problem data associated with the trace.
        """
        trace_idx = self._get_trace_idx_from_idx_or_key(index_or_key)
        internal_trace_idx = self._trace_indices[trace_idx]
        problem_idx = self._trace_idx_to_problem_idx[internal_trace_idx]
        problem_data = self.reader.get(problem_idx)
        problem_data = pyine.data.traces.dataset_utils.CodingProblem.model_validate(problem_data)
        return problem_data

    def get_tags(self, index_or_key: int | str) -> list[str]:
        """Returns a list of tags for a given trace so that we can decide whether to filter it."""
        trace_idx = self._get_trace_idx_from_idx_or_key(index_or_key)
        output_tags = self.trace_tag_lists[trace_idx].copy()
        return output_tags

    def __str__(self):
        """Returns a string representation of the dataset reader (for debugging purposes)."""
        return f"{self.__class__.__name__}({self.path}) with {len(self)} instances"
