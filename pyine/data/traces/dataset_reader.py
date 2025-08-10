"""
This module contains a reader for the PyINE dataset of 'raw' code execution traces.

See the corresponding dataset writer and utility modules for more information on this
dataset format. See also the demo notebook (in the project's root `notebooks` directory)
for an example of how to use this dataset reader.
"""

import functools
import pathlib
import typing

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
        self.reader = pyine.data.utils.lmdb_io.LMDBReader(lmdb_path)
        self.problem_indices, self.problem_keys = self.reader.get_indices(
            pattern=pyine.data.traces.dataset_utils.PROBLEM_DATA_PATTERN,
            return_keys=True,
        )
        assert len(self.problem_indices) > 0, "no problem data found in the dataset"
        assert len(self.problem_indices) == len(self.problem_keys)
        self._init_trace_maps()

    def _init_trace_maps(self) -> None:
        """Initializes the trace index maps according to requested filtering rules."""
        # note: the indices kept in the lists below are INTERNAL ones that map to the lmdb content
        self.trace_indices: list[int] = []
        self.trace_keys: list[str] = []
        self.trace_idx_to_problem_idx: dict[int, int] = {}
        for problem_idx, problem_key in zip(self.problem_indices, self.problem_keys):
            assert problem_key.endswith(pyine.data.traces.dataset_utils.PROBLEM_DATA_SUFFIX)
            problem_prefix = problem_key[: -len(pyine.data.traces.dataset_utils.PROBLEM_DATA_SUFFIX)]
            curr_trace_data_pattern = problem_prefix + pyine.data.traces.dataset_utils.TRACE_DATA_SUFFIX
            curr_trace_indices, curr_trace_keys = self.reader.get_indices(
                pattern=curr_trace_data_pattern,
                return_keys=True,
            )
            assert len(curr_trace_indices) > 0, f"no trace data found for problem: {problem_key}"
            assert len(curr_trace_indices) == len(curr_trace_keys)
            self.trace_indices.extend(curr_trace_indices)
            self.trace_keys.extend(curr_trace_keys)
            for trace_idx in curr_trace_indices:
                self.trace_idx_to_problem_idx[trace_idx] = problem_idx

    def __len__(self) -> int:
        """
        Returns the total number of traces in the dataset.

        Returns:
            int: Total number of traces in the LMDB database.
        """
        # defines the upper bound of the range of EXTERNAL trace indices (that don't map to lmdb)
        return len(self.trace_indices)

    def get_metadata(self) -> dict[str, typing.Any]:
        """Returns a dictionary of all metadata stored in the database."""
        return self.reader.get_metadata()

    def get_size_on_disk(self) -> int:
        """Calculate the total size of the LMDB dataset stored on disk (in bytes)."""
        return self.reader.get_size_on_disk()

    def get_source_dataset_name(self) -> str:
        """Returns the name of the source dataset used to create this dataset."""
        return self.reader.get_metadata()["source_dataset"]["source_dataset_name"]

    def __getitem__(self, index_or_key: int | str) -> pyine.utils.code.execution.TraceResult:
        """
        Fetches an individual trace data object from the LMDB database by external index or key.

        Args:
            index_or_key: index or key of the trace to retrieve.
        """
        if isinstance(index_or_key, int):
            if not (0 <= index_or_key < len(self)):
                raise IndexError(f"index {index_or_key} out of range")
        elif isinstance(index_or_key, str):
            assert index_or_key in self.trace_keys, f"key {index_or_key} not found in dataset"
            index_or_key = self.trace_keys.index(index_or_key)
        else:
            raise ValueError(f"invalid index_or_key type: {type(index_or_key)}")
        trace_data = self.reader.get(self.trace_indices[index_or_key])
        trace = pyine.utils.code.execution.TraceResult.model_validate(trace_data)
        return trace

    @functools.lru_cache(maxsize=512)
    def get_problem_data(self, index_or_key: int | str) -> pyine.data.traces.dataset_utils.CodingProblem:
        """
        Fetches the problem data associated with a trace by external index or key.

        Args:
            index_or_key: Index or key of the trace for which to retrieve parent problem data.
        """
        if isinstance(index_or_key, int):
            assert 0 <= index_or_key < len(self), f"index {index_or_key} out of range"
            problem_idx = self.trace_idx_to_problem_idx[self.trace_indices[index_or_key]]
        elif isinstance(index_or_key, str):
            assert index_or_key in self.trace_keys, f"key {index_or_key} not found in dataset"
            internal_trace_idx = self.trace_indices[self.trace_keys.index(index_or_key)]
            problem_idx = self.trace_idx_to_problem_idx[internal_trace_idx]
        else:
            raise ValueError(f"invalid index_or_key type: {type(index_or_key)}")
        problem_data = self.reader.get(problem_idx)
        problem = pyine.data.traces.dataset_utils.CodingProblem.model_validate(problem_data)
        return problem

    def close(self) -> None:
        """
        Closes the LMDBReader instance and releases resources.
        """
        self.reader.close()


if __name__ == "__main__":
    pyine.utils.reprod.entrypoint_setup()
    _taco_traces_path = pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO")
    print(f"trying to load trace dataset at: {_taco_traces_path}")
    _dataset_reader = DatasetReader(lmdb_path=_taco_traces_path)
    print(f"traces dataset contains {len(_dataset_reader)} traces")
    _target_sample_idx = 0
    print(f"sample #{_target_sample_idx}:")
    _problem = _dataset_reader.get_problem_data(_target_sample_idx)
    print(f"problem id: {_problem.problem_id}")
    print(f"problem statement: {_problem.problem_statement}")
    print(f"problem tags: {_problem.problem_tags}")
    _trace_result = _dataset_reader[_target_sample_idx]
    print(f"trace id: {_trace_result.identifier}")
    print("trace steps:")
    for _step in _trace_result.traced_steps:
        if _step is not None:
            print(f"\t{_step}")
