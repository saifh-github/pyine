"""
This module contains a reader for the PyINE dataset of code execution trace DELTAS.

See the corresponding dataset writer and utility modules for more information on this
dataset format. See also the demo notebook (in the project's root `notebooks` directory)
for an example of how to use this dataset reader.
"""

import pathlib
import typing

import pyine.data.deltas.dataset_utils
import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.utils.lmdb_io
import pyine.utils.code.execution
import pyine.utils.reprod


class DatasetReader(pyine.data.traces.dataset_reader.DatasetReader):
    """PyINE deltas dataset reader.

    This reader provides execution deltas stored in an LMDB dataset. Each delta describes the
    relationship between two consecutive trace events and contains information on any local
    variables updated during the step.

    Args:
        lmdb_path: Path to the LMDB database containing code traces.
    """

    def __init__(
        self,
        lmdb_path: pathlib.Path | typing.AnyStr,
    ) -> None:
        super().__init__(lmdb_path=lmdb_path)
        # first, make sure this is actually a deltas dataset, and not just a regular traces dataset
        if "trace_count" not in self.reader.get_metadata()["parent_dataset"]:
            raise RuntimeError(f"this is not a deltas dataset: {lmdb_path}")
        # split the trace indices/keys into trace results and deltas
        assert len(self.trace_indices) == len(self.trace_keys)
        deltas_pattern = f"*{pyine.data.deltas.dataset_utils.DELTAS_SUFFIX}"
        self.deltas_indices, self.deltas_keys = self.reader.get_indices(
            pattern=deltas_pattern,
            return_keys=True,
        )
        for dkey, didx in zip(self.deltas_keys, self.deltas_indices):
            assert dkey not in self.trace_keys
            assert didx not in self.trace_indices
        if len(self.trace_indices) != len(self.deltas_indices):
            raise RuntimeError("all execution traces should have a corresponding deltas entry")

    def __getitem__(self, index_or_key: int | str) -> pyine.data.deltas.dataset_utils.TraceDeltaList:
        """Fetches a list of trace deltas from the LMDB database by external trace index or key.

        Args:
            index_or_key: index or key of the trace for which to retrieve deltas.
        """
        if isinstance(index_or_key, int):
            if not (0 <= index_or_key < len(self)):
                raise IndexError(f"index {index_or_key} out of range")
        elif isinstance(index_or_key, str):
            assert (
                index_or_key in self.trace_keys or index_or_key in self.deltas_keys
            ), f"key {index_or_key} not found in dataset"
            if index_or_key in self.deltas_keys:
                index_or_key = self.deltas_keys.index(index_or_key)
            else:
                index_or_key = self.trace_keys.index(index_or_key)
        else:  # pragma: no cover
            raise ValueError(f"invalid index_or_key type: {type(index_or_key)}")
        deltas = self.reader.get(self.deltas_indices[index_or_key])
        deltas = pyine.data.deltas.dataset_utils.TraceDeltaList.model_validate(deltas)
        return deltas

    def get_trace_data(self, index_or_key: int | str) -> pyine.data.traces.dataset_utils.CodingProblem:
        """
        Fetches trace data by external index or key.

        Args:
            index_or_key: Index or key of the trace for which to retrieve data.
        """
        return pyine.data.traces.dataset_reader.DatasetReader.__getitem__(self, index_or_key)


if __name__ == "__main__":
    pyine.utils.reprod.entrypoint_setup()
    _taco_traces_path = pyine.data.deltas.dataset_utils.get_latest_dataset_path("TACO")
    print(f"trying to load deltas dataset at: {_taco_traces_path}")
    _dataset_reader = DatasetReader(lmdb_path=_taco_traces_path)
    print(f"deltas dataset contains {len(_dataset_reader)} traces")
    _target_sample_idx = 0
    print(f"sample #{_target_sample_idx}:")
    _problem = _dataset_reader.get_problem_data(_target_sample_idx)
    print(f"problem id: {_problem.problem_id}")
    print(f"problem statement: {_problem.problem_statement}")
    print(f"problem tags: {_problem.problem_tags}")
    _trace_result = _dataset_reader.get_trace_data(_target_sample_idx)
    print(f"trace id: {_trace_result.identifier}")
    _trace_deltas = _dataset_reader[_target_sample_idx]
    print("trace deltas:")
    for _delta in _trace_deltas:
        if _delta is not None:
            print(f"\t{_delta}")
