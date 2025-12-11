"""DataModule for shortcut-bias experiments using code execution trace datasets."""

from __future__ import annotations

import typing

import pyine.data.datamodule
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.base
import pyine.utils.reprod
from pyine.organisms.datamodules.shortcuts_configs import (
    ShortcutBiasDataModuleConfig,
)


class ShortcutBiasDataModule(
    pyine.organisms.datamodules.base.BiasDataModuleBase[ShortcutBiasDataModuleConfig],
):
    """DataModule wrapping one or multiple PyINE code trace datasets for shortcut-bias experiments.

    This module loads one or more LMDB trace datasets, optionally filters available traces
    using flexible rules, validates that there are no duplicate trace identifiers across
    all selected samples, and finally creates simple random train/valid/test splits and loaders.

    Filtering is performed prior to concatenation and splitting. When multiple datasets are
    provided, they are concatenated in the provided order.
    """

    @typing.override
    def _get_metadata_model_class(
        self,
    ) -> type[pyine.data.traces.dataset_utils.TraceDatasetMetadata]:
        """Return TraceDatasetMetadata for shortcut bias experiments."""
        return pyine.data.traces.dataset_utils.TraceDatasetMetadata

    @typing.override
    def _get_subset_suffixes(self) -> tuple[str, ...]:
        """Returns the suffixes that this datamodule may expect to see appended to subset names."""
        from pyine.organisms.datamodules.samples import get_all_supported_code_type_sets_suffixes

        return tuple(get_all_supported_code_type_sets_suffixes())

    @typing.override
    def _prepare_bias_specific_metadata(
        self,
        base_traces_meta: list[pyine.data.traces.dataset_utils.TraceMetadata],
        split_data: pyine.data.utils.splits.SplitResult,
    ) -> pyine.data.traces.dataset_utils.TraceDatasetMetadata:
        """Prepare shortcut-specific metadata by assigning traces to subsets based on problem splits.

        Args:
            base_traces_meta: Pre-filtered traces based on base_filter config.
            split_data: Problem split assignments loaded from split file.

        Returns:
            Complete TraceDatasetMetadata with subset assignments.
        """
        split_hash = pyine.utils.reprod.compute_hash(self.config.split_file_path)
        subset_traces_meta: dict[
            pyine.data.datamodule.SubsetNameType,
            list[pyine.data.traces.dataset_utils.TraceMetadata],
        ] = {subset_name: [] for subset_name in split_data.config.subset_names}
        unassigned_traces_meta: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
        for trace_meta in base_traces_meta:
            if str(trace_meta.problem_id) in split_data.subset_assignments:
                subset_traces_meta[split_data.subset_assignments[str(trace_meta.problem_id)]].append(trace_meta)
            else:
                unassigned_traces_meta.append(trace_meta)
        self._apply_max_solution_count_cap(subset_traces_meta, unassigned_traces_meta)
        return pyine.data.traces.dataset_utils.TraceDatasetMetadata(
            base_traces=base_traces_meta,
            subset_traces=subset_traces_meta,
            leftover_traces=unassigned_traces_meta,
            problem_assignments=split_data.subset_assignments,
            split_hash=split_hash,
        )
