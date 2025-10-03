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
import pyine.utils.code.execution


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
        lmdb_path: pathlib.Path | str,
    ) -> None:
        super().__init__(lmdb_path=lmdb_path)
        # first, make sure this is actually a deltas dataset, and not just a regular traces dataset
        if "parent_dataset" not in self.metadata or "trace_count" not in self.metadata["parent_dataset"]:
            raise RuntimeError(f"this is not a deltas dataset: {lmdb_path}")
        # split the trace indices/keys into trace results and deltas
        assert len(self._trace_indices) == len(self.trace_keys)
        deltas_pattern = f"*{pyine.data.deltas.dataset_utils.DELTAS_SUFFIX}"
        self._deltas_indices: list[int]
        self.deltas_keys: list[str]
        self._deltas_indices, self.deltas_keys = typing.cast(
            "tuple[list[int], list[str]]",
            self.reader.get_indices(
                pattern=deltas_pattern,
                return_keys=True,
            ),
        )
        for dkey, didx in zip(self.deltas_keys, self._deltas_indices, strict=False):
            assert dkey not in self.trace_keys
            assert didx not in self._trace_indices
        if len(self._trace_indices) != len(self._deltas_indices):
            raise RuntimeError("all execution traces should have a corresponding deltas entry")

    @typing.override
    def __getitem__(
        self,
        index_or_key: int | str,
    ) -> pyine.data.deltas.dataset_utils.TraceResultWithDeltas:
        """Fetches a list of trace deltas from the LMDB database by external trace index or key.

        Args:
            index_or_key: index or key of the trace for which to retrieve deltas.

        Returns:
            A `TraceResultWithDeltas` object containing the requested trace result and deltas.
        """
        resolved_index = self._resolve_trace_index_or_key(index_or_key)
        internal_deltas_idx = self._deltas_indices[resolved_index]
        deltas_payload = self.reader.get(internal_deltas_idx)
        return pyine.data.deltas.dataset_utils.TraceResultWithDeltas.model_validate(deltas_payload)

    @typing.override
    def _resolve_trace_key(
        self,
        trace_key: str,
    ) -> int:
        """Returns the dataset position for the provided user-facing key."""
        if trace_key in self.deltas_keys:
            return self.deltas_keys.index(trace_key)
        if trace_key in self.trace_keys:
            return self.trace_keys.index(trace_key)
        raise KeyError(f"key {trace_key} not found in dataset")

    def get_trace_data(
        self,
        index_or_key: int | str,
    ) -> pyine.utils.code.execution.TraceResult:
        """Fetches an individual trace data object from the LMDB database by external index or key.

        Args:
            index_or_key: index or key of the trace to retrieve.

        Returns:
            A `TraceResult` object containing the requested trace data.
        """
        return pyine.data.traces.dataset_reader.DatasetReader.__getitem__(self, index_or_key)
