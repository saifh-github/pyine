"""DataModule for shortcut-bias experiments using code execution trace datasets."""

from __future__ import annotations

import collections
import logging
import typing

import pyine.data.datamodule
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.base
import pyine.utils.reprod
from pyine.organisms.datamodules.shortcuts_configs import (
    EvaluationStrategy,
    HintType,
    ShortcutBiasDataModuleConfig,
)

logger = logging.getLogger(__name__)

# hint-related augment category patterns (see pyine.data.traces.dataset_utils.AugmentPatterns)
_HINT_CATEGORY_PATTERNS = ("hints_", "hinted", "misleading", "issues_docs")


class ShortcutBiasDataModule(
    pyine.organisms.datamodules.base.BiasDataModuleBase[ShortcutBiasDataModuleConfig],
):
    """DataModule wrapping one or multiple PyINE code trace datasets for shortcut-bias experiments.

    This module loads one or more LMDB trace datasets, optionally filters available traces
    using flexible rules, validates that there are no duplicate trace identifiers across
    all selected samples, and finally creates simple random train/valid/test splits and loaders.

    The module supports two evaluation strategies for measuring shortcut bias:
    - `hint_presence_split`: Partitions eval traces based on hint availability;
    - `counterfactual`: Creates paired subsets with matching base augments +/- hints.
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
        return "_with_hints", "_without_hints"

    @typing.override
    def _log_setup_summary(self) -> None:
        """Log a summary of shortcuts datamodule configuration after setup."""
        assert self._metadata is not None, "metadata should be loaded before logging summary"
        subset_info_parts: list[str] = []
        for subset_name in self.config.subset_names:
            traces = self._metadata.get_subset_traces(subset_name)
            subset_info_parts.append(f"{subset_name}={len(traces)}")
        subset_info = ", ".join(subset_info_parts)
        logger.info(
            f"shortcuts datamodule setup complete:\n"
            f"\thint_type={self.config.hint_type.value}, "
            f"\tevaluation_strategy={self.config.evaluation_strategy.value}, "
            f"\tsubsets=[{subset_info}]"
        )

    @typing.override
    def _prepare_bias_specific_metadata(
        self,
        base_traces_meta: list[pyine.data.traces.dataset_utils.TraceMetadata],
        split_data: pyine.data.utils.splits.SplitResult,
    ) -> pyine.data.traces.dataset_utils.TraceDatasetMetadata:
        """Prepare shortcut-specific metadata with hint-based derived evaluation subsets.

        Args:
            base_traces_meta: Pre-filtered traces based on base_filter config.
            split_data: Problem split assignments loaded from split file.

        Returns:
            Complete TraceDatasetMetadata with subset assignments and derived hint-split subsets.
        """
        split_hash = pyine.utils.reprod.compute_hash(self.config.split_file_path)
        # assign traces to primary subsets based on problem splits
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
        # build trace family pairing map for hint-based evaluation
        family_pairing_map = self._build_trace_family_pairing_map(base_traces_meta)
        # create derived subsets for hint-based evaluation
        derived_subsets = self._create_hint_split_derived_subsets(subset_traces_meta, family_pairing_map)
        self._validate_sample_counts(subset_traces_meta, derived_subsets)
        return pyine.data.traces.dataset_utils.TraceDatasetMetadata(
            base_traces=base_traces_meta,
            subset_traces=subset_traces_meta,
            derived_subsets=derived_subsets,
            leftover_traces=unassigned_traces_meta,
            problem_assignments=split_data.subset_assignments,
            split_hash=split_hash,
        )

    def _is_hint_category(self, category: str) -> bool:
        """Check if an augment category is hint-related.

        Args:
            category: The augment category string to check.

        Returns:
            True if the category is related to hints (helpful or misleading).
        """
        category_lower = category.lower()
        return any(pattern in category_lower for pattern in _HINT_CATEGORY_PATTERNS)

    def _get_base_augments(
        self,
        trace_id: pyine.data.traces.dataset_utils.TraceIdentifier,
    ) -> frozenset[str]:
        """Get augment categories excluding hint-related ones.

        The "base augments" are all augmentation categories applied to a trace, excluding any
        hint-related categories. This is used to pair traces that share the same base augments
        but differ only in hint presence.

        Args:
            trace_id: The trace identifier to analyze.

        Returns:
            Frozenset of non-hint augment category strings.
        """
        if not trace_id.is_augmented:
            return frozenset()
        categories = set(trace_id.split_augment_categories)
        # remove hint-related categories
        categories = {cat for cat in categories if not self._is_hint_category(cat)}
        return frozenset(categories)

    def _has_target_hint(
        self,
        trace_id: pyine.data.traces.dataset_utils.TraceIdentifier,
    ) -> bool:
        """Check if trace has the target hint type based on config.

        Args:
            trace_id: The trace identifier to check.

        Returns:
            True if the trace has the configured target hint type.
        """
        if self.config.hint_type == HintType.helpful:
            return trace_id.is_hinted
        return trace_id.is_misleading

    def _build_trace_family_pairing_map(
        self,
        traces: list[pyine.data.traces.dataset_utils.TraceMetadata],
    ) -> dict[str, dict[frozenset[str], dict[str, pyine.data.traces.dataset_utils.TraceMetadata]]]:
        """Map trace families to their augment-based pairs.

        This builds a nested mapping structure that groups traces by:
        1. Their augmentless family identifier (same solution + test);
        2. Their base augments (all augments except hints); and
        3. Whether they have the target hint type or not.

        Args:
            traces: List of trace metadata to analyze.

        Returns:
            Nested dict: {family_id: {base_augments: {"with_hint": trace, "without_hint": trace}}}
        """
        family_map: dict[str, dict[frozenset[str], dict[str, pyine.data.traces.dataset_utils.TraceMetadata]]] = (
            collections.defaultdict(lambda: collections.defaultdict(dict))
        )
        for trace in traces:
            family_id = str(trace.trace_id.get_augmentless_identifier())
            base_augments = self._get_base_augments(trace.trace_id)
            has_target_hint = self._has_target_hint(trace.trace_id)
            key = "with_hint" if has_target_hint else "without_hint"
            # note: if multiple traces have same family/base_augments/hint_status, last one wins
            # this is acceptable since they should be functionally equivalent for evaluation
            family_map[family_id][base_augments][key] = trace
        return dict(family_map)

    def _create_hint_split_derived_subsets(
        self,
        subset_traces_meta: dict[
            pyine.data.datamodule.SubsetNameType,
            list[pyine.data.traces.dataset_utils.TraceMetadata],
        ],
        family_pairing_map: dict[str, dict[frozenset[str], dict[str, pyine.data.traces.dataset_utils.TraceMetadata]]],
    ) -> dict[str, pyine.data.traces.dataset_utils.DerivedSubsetInfo[pyine.data.traces.dataset_utils.TraceMetadata]]:
        """Create derived subsets for hint-based evaluation splits.

        Args:
            subset_traces_meta: Dict mapping primary subset names to their trace metadata lists.
            family_pairing_map: The trace family pairing map from `_build_trace_family_pairing_map`.

        Returns:
            Dictionary of derived subset names to DerivedSubsetInfo objects.
        """
        derived_subsets: dict[
            str, pyine.data.traces.dataset_utils.DerivedSubsetInfo[pyine.data.traces.dataset_utils.TraceMetadata]
        ] = {}
        derivation_type = self.config.evaluation_strategy.value
        for eval_subset_name in self.config.eval_subset_names:
            if eval_subset_name not in subset_traces_meta:
                continue
            traces = subset_traces_meta[eval_subset_name]
            with_hints, without_hints = self._partition_traces_by_hint_strategy(traces, family_pairing_map)
            derived_subsets[f"{eval_subset_name}_with_hints"] = pyine.data.traces.dataset_utils.DerivedSubsetInfo(
                parent_subset=eval_subset_name,
                traces=with_hints,
                derivation_type=derivation_type,
            )
            derived_subsets[f"{eval_subset_name}_without_hints"] = pyine.data.traces.dataset_utils.DerivedSubsetInfo(
                parent_subset=eval_subset_name,
                traces=without_hints,
                derivation_type=derivation_type,
            )
            logger.info(
                f"created {self.config.evaluation_strategy.value} subsets for {eval_subset_name}: "
                f"{len(with_hints)} traces with hints, {len(without_hints)} without"
            )
        return derived_subsets

    def _partition_traces_by_hint_strategy(
        self,
        traces: list[pyine.data.traces.dataset_utils.TraceMetadata],
        family_pairing_map: dict[str, dict[frozenset[str], dict[str, pyine.data.traces.dataset_utils.TraceMetadata]]],
    ) -> tuple[
        list[pyine.data.traces.dataset_utils.TraceMetadata],
        list[pyine.data.traces.dataset_utils.TraceMetadata],
    ]:
        """Partition traces into with_hints and without_hints lists based on evaluation strategy.

        For both strategies, the partitioning is based on whether each trace has the target hint:
        - `_with_hints`: traces that have the target hint type;
        - `_without_hints`: traces that don't have the target hint type.

        The difference between strategies is in WHICH traces are included:
        - `hint_presence_split`: all traces are included (simple partition by hint presence);
        - `counterfactual`: only traces with a matching pair (same base augments +/- hint).

        Args:
            traces: List of traces to partition.
            family_pairing_map: The trace family pairing map.

        Returns:
            Tuple of (with_hints, without_hints) trace lists.
        """
        with_hints: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
        without_hints: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
        for trace in traces:
            trace_has_hint = self._has_target_hint(trace.trace_id)
            if self.config.evaluation_strategy == EvaluationStrategy.counterfactual:
                # counterfactual: only include traces that have a complete pair (same base augments)
                family_id = str(trace.trace_id.get_augmentless_identifier())
                base_augments = self._get_base_augments(trace.trace_id)
                family_info = family_pairing_map.get(family_id, {})
                pair_info = family_info.get(base_augments, {})
                has_complete_pair = "with_hint" in pair_info and "without_hint" in pair_info
                if has_complete_pair:
                    if trace_has_hint:
                        with_hints.append(trace)
                    else:
                        without_hints.append(trace)
                # else: drop trace from counterfactual evaluation (no matching pair)
            elif self.config.evaluation_strategy == EvaluationStrategy.hint_presence_split:
                # hint_presence_split: simple partition based on whether trace has the target hint
                if trace_has_hint:
                    with_hints.append(trace)
                else:
                    without_hints.append(trace)
        return with_hints, without_hints

    def _validate_sample_counts(
        self,
        subset_traces_meta: dict[
            pyine.data.datamodule.SubsetNameType,
            list[pyine.data.traces.dataset_utils.TraceMetadata],
        ],
        derived_subsets: dict[
            str, pyine.data.traces.dataset_utils.DerivedSubsetInfo[pyine.data.traces.dataset_utils.TraceMetadata]
        ],
    ) -> None:
        """Validate that derived subsets have sufficient samples for evaluation.

        Args:
            subset_traces_meta: Dict mapping primary subset names to their trace metadata lists.
            derived_subsets: Dict mapping derived subset names to DerivedSubsetInfo objects.

        Raises:
            ValueError: If any derived subset has fewer samples than the configured minimum.
        """
        for eval_subset_name in self.config.eval_subset_names:
            if eval_subset_name not in subset_traces_meta:
                continue
            with_hints_key = f"{eval_subset_name}_with_hints"
            without_hints_key = f"{eval_subset_name}_without_hints"
            with_hints_info = derived_subsets.get(with_hints_key)
            without_hints_info = derived_subsets.get(without_hints_key)
            with_hints_count = len(with_hints_info.traces) if with_hints_info else 0
            without_hints_count = len(without_hints_info.traces) if without_hints_info else 0
            if with_hints_count < self.config.min_samples_with_hints:
                raise ValueError(
                    f"eval subset '{with_hints_key}' has only {with_hints_count} samples "
                    f"(minimum required: {self.config.min_samples_with_hints})"
                )
            if without_hints_count < self.config.min_samples_without_hints:
                raise ValueError(
                    f"eval subset '{without_hints_key}' has only {without_hints_count} samples "
                    f"(minimum required: {self.config.min_samples_without_hints})"
                )
