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
        if "parent_dataset" not in self.metadata or "trace_count" not in self.metadata["parent_dataset"]:
            raise RuntimeError(f"this is not a deltas dataset: {lmdb_path}")
        # split the trace indices/keys into trace results and deltas
        assert len(self._trace_indices) == len(self.trace_keys)
        deltas_pattern = f"*{pyine.data.deltas.dataset_utils.DELTAS_SUFFIX}"
        self._deltas_indices, self.deltas_keys = self.reader.get_indices(
            pattern=deltas_pattern,
            return_keys=True,
        )
        for dkey, didx in zip(self.deltas_keys, self._deltas_indices, strict=False):
            assert dkey not in self.trace_keys
            assert didx not in self._trace_indices
        if len(self._trace_indices) != len(self._deltas_indices):
            raise RuntimeError("all execution traces should have a corresponding deltas entry")

    def __getitem__(self, index_or_key: int | str) -> pyine.data.deltas.dataset_utils.TraceDeltaList:
        """Fetches a list of trace deltas from the LMDB database by external trace index or key.

        Args:
            index_or_key: index or key of the trace for which to retrieve deltas.

        Returns:
            A `TraceDeltaList` object containing the requested trace deltas.
        """
        if isinstance(index_or_key, int):
            if not (0 <= index_or_key < len(self)):
                raise IndexError(f"index {index_or_key} out of range")
        elif isinstance(index_or_key, str):
            if index_or_key not in self.trace_keys and index_or_key not in self.deltas_keys:
                raise KeyError(f"key {index_or_key} not found in dataset")
            if index_or_key in self.deltas_keys:
                index_or_key = self.deltas_keys.index(index_or_key)
            else:
                index_or_key = self.trace_keys.index(index_or_key)
        else:  # pragma: no cover
            raise ValueError(f"invalid index_or_key type: {type(index_or_key)}")
        internal_deltas_idx = self._deltas_indices[index_or_key]
        return pyine.data.deltas.dataset_utils.TraceDeltaList.model_validate(self.reader.get(internal_deltas_idx))

    def get_trace_data(self, index_or_key: int | str) -> pyine.data.traces.dataset_utils.CodingProblem:
        """Fetches an individual trace data object from the LMDB database by external index or key.

        Args:
            index_or_key: index or key of the trace to retrieve.

        Returns:
            A `TraceResult` object containing the requested trace data.
        """
        return pyine.data.traces.dataset_reader.DatasetReader.__getitem__(self, index_or_key)
