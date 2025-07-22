import pathlib
import typing

import torch.utils.data

import pyine.data.utils.lmdb_io
from pyine.data.raw_dataset_writer import (
    PROBLEM_DATA_PATTERN,
    PROBLEM_DATA_SUFFIX,
    TRACE_DATA_SUFFIX,
)

TRACE_DATA_KEY = "trace_result"
TRACE_ID_KEY = "trace_id"


class DatasetParser(torch.utils.data.Dataset):
    """PyINE raw dataset reader.

    Note: this readers allows access to the RAW traces along with the original code
    and related JSON metadata. It does NOT attempt to structure the traces into anything
    useful for explanation-related experiments.

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
            pattern=PROBLEM_DATA_PATTERN,
            return_keys=True,
        )
        assert len(self.problem_indices) > 0, "no problem data found in the dataset"
        assert len(self.problem_indices) == len(self.problem_keys)
        # note: the indices kept in the lists below are INTERNAL ones that map to the lmdb content
        self.trace_indices, self.trace_keys = [], []
        self.trace_idx_to_problem_idx = {}
        for problem_idx, problem_key in zip(self.problem_indices, self.problem_keys):
            assert problem_key.endswith(PROBLEM_DATA_SUFFIX)
            problem_prefix = problem_key[: -len(PROBLEM_DATA_SUFFIX)]
            curr_trace_data_pattern = problem_prefix + TRACE_DATA_SUFFIX
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

    def __getitem__(self, index_or_key: int | str) -> dict[str, typing.Any]:
        """
        Fetches an individual trace from the LMDB database by external index or key.

        Args:
            index_or_key: index or key of the trace to retrieve.
        """
        if isinstance(index_or_key, int):
            assert 0 <= index_or_key < len(self), f"index {index_or_key} out of range"
        elif isinstance(index_or_key, str):
            assert index_or_key in self.trace_keys, f"key {index_or_key} not found in dataset"
            index_or_key = self.trace_keys.index(index_or_key)
        else:
            raise ValueError(f"invalid index_or_key type: {type(index_or_key)}")
        problem_idx = self.trace_idx_to_problem_idx[self.trace_indices[index_or_key]]
        # TODO: @@@@ might want to use a problem data cache here
        problem_data = self.reader.get(problem_idx)
        assert TRACE_DATA_KEY not in problem_data, "problem data should not contain trace result"
        trace_data = self.reader.get(self.trace_indices[index_or_key])
        assert TRACE_ID_KEY not in trace_data, "trace data should not contain trace id"
        output_data = problem_data.copy()
        output_data[TRACE_DATA_KEY] = {TRACE_ID_KEY: self.trace_keys[index_or_key], **trace_data}
        return output_data

    def get_problem_data(self, index_or_key: int | str) -> dict[str, typing.Any]:
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
        # TODO: @@@@ might want to use a problem data cache here
        return self.reader.get(problem_idx)

    def close(self) -> None:
        """
        Closes the LMDBReader instance and releases resources.
        """
        self.reader.close()


if __name__ == "__main__":
    _dataset_reader = DatasetParser(lmdb_path="data/2025-03-31-v01.raw.lmdb")
    print(f"dataset contains {len(_dataset_reader)} trace samples")
    _sample = _dataset_reader[0]
    print(f"sample: {_sample}")
    _dataset_reader.close()
