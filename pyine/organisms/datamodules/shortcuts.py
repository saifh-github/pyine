"""DataModule for shortcut-bias experiments using code execution trace datasets."""

from __future__ import annotations

import collections
import collections.abc
import logging
import typing

import pydantic

import pyine.data.datamodule
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.base
import pyine.organisms.datamodules.samples
import pyine.organisms.datamodules.samples.common
import pyine.prompts
import pyine.prompts.names
import pyine.utils.reprod
from pyine.organisms.datamodules.shortcuts_configs import (
    EvaluationStrategy,
    HintType,
    ShortcutBiasDataModuleConfig,
)

if typing.TYPE_CHECKING:
    import torch.utils.data

logger = logging.getLogger(__name__)


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
    def _log_setup_summary(self) -> None:
        """Log a summary of shortcuts datamodule configuration after setup."""
        assert self._metadata is not None, "metadata should be loaded before logging summary"
        subset_info_parts: list[str] = []
        for subset_name in self.config.subset_names:
            traces = self._metadata.get_subset_traces(subset_name)
            subset_info_parts.append(f"{subset_name}={len(traces)}")
        subset_info = ", ".join(subset_info_parts)
        logger.info(
            f"shortcuts datamodule setup complete:"
            f"\n\tevaluation_strategy={self.config.evaluation_strategy.value} + {self.config.hint_type.value}"
            f"\n\tsubsets=[{subset_info}]"
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
        # get prompt DB for hint detection if enabled by config
        prompt_db: pyine.prompts.PromptResultDB | None = None
        if self._should_use_prompt_db_for_hints():
            prompt_db = pyine.prompts.get_framework_db()
            logger.debug("using prompt result DB for hint detection in derived subsets")
        # build trace family pairing map for hint-based evaluation
        family_pairing_map = self._build_trace_family_pairing_map(base_traces_meta, prompt_db)
        # create derived subsets for hint-based evaluation
        derived_subsets = self._create_hint_split_derived_subsets(subset_traces_meta, family_pairing_map, prompt_db)
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

        Delegates to the centralized AugmentPatterns.is_hint_category() for pattern matching.

        Args:
            category: The augment category string to check.

        Returns:
            True if the category is related to hints (helpful or misleading).
        """
        return pyine.data.traces.dataset_utils.AugmentPatterns.is_hint_category(category)

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

    def _check_prompt_db_for_hint(
        self,
        trace_id: pyine.data.traces.dataset_utils.TraceIdentifier,
        prompt_db: pyine.prompts.PromptResultDB,
    ) -> bool:
        """Checks if the prompt result DB has hinted code for this trace."""
        # determine which prompt names to check based on hint type
        if self.config.hint_type == HintType.helpful:
            hint_prompt_names: list[pyine.prompts.PromptNameType] = [
                pyine.prompts.names.PromptNames.HINTS_DOCS,
                pyine.prompts.names.PromptNames.HINTS_TESTS,
            ]
        else:  # misleading hints
            hint_prompt_names = [
                pyine.prompts.names.PromptNames.ISSUES_DOCS,
            ]
        # use count_entries for fast existence check (avoids fetching full records)
        # query with exact trace ID since hints are stored for specific test cases
        count = prompt_db.count_entries(
            identifier=str(trace_id),
            prompt_name=hint_prompt_names,
        )
        return count > 0

    def _should_use_prompt_db_for_hints(self) -> bool:
        """Checks if prompt DB should be used based on the default dataparser config."""
        parser_config = self.config.default_dataparser_config
        assert hasattr(parser_config, "params")
        params = parser_config.params
        assert params is not None
        if isinstance(params, pydantic.BaseModel):
            selection_config = getattr(params, "selection_config", None)
            if selection_config is not None:
                return bool(getattr(selection_config, "allow_db_lookups", False))
        else:
            assert isinstance(params, dict)
            params = typing.cast("collections.abc.Mapping[str, typing.Any]", params)
            selection_config = params.get("selection_config", {})
            assert isinstance(selection_config, dict)
            selection_config = typing.cast("collections.abc.Mapping[str, typing.Any]", selection_config)
            return bool(selection_config.get("allow_db_lookups", False))
        raise TypeError("invalid default dataparser config params type")

    def _build_trace_family_pairing_map(
        self,
        traces: list[pyine.data.traces.dataset_utils.TraceMetadata],
        prompt_db: pyine.prompts.PromptResultDB | None = None,
    ) -> dict[str, dict[frozenset[str], dict[str, pyine.data.traces.dataset_utils.TraceMetadata]]]:
        """Map trace families to their augment-based pairs.

        This builds a nested mapping structure that groups traces by:
        1. Their augmentless family identifier (same solution + test);
        2. Their base augments (all augments except hints); and
        3. Whether they have the target hint type or not.

        For traces with hints in the prompt DB (but no LMDB trace with hint augmentation), we
        register them under BOTH "with_hint" and "without_hint" keys since the same trace can be
        rendered either way at sample generation time. This creates "virtual pairs" for
        counterfactual evaluation.

        Args:
            traces: List of trace metadata to analyze.
            prompt_db: Optional prompt result database to check for hint availability.
                Used as fallback when the LMDB augmented trace is not present.

        Returns:
            Nested dict: {family_id: {base_augments: {"with_hint": trace, "without_hint": trace}}}
        """
        family_map: dict[str, dict[frozenset[str], dict[str, pyine.data.traces.dataset_utils.TraceMetadata]]] = (
            collections.defaultdict(lambda: collections.defaultdict(dict))
        )
        for trace in traces:
            family_id = str(trace.trace_id.get_augmentless_identifier())
            base_augments = self._get_base_augments(trace.trace_id)
            # check LMDB hint first
            has_lmdb_hint = self._has_lmdb_hint(trace.trace_id)
            if has_lmdb_hint:
                # trace has hint augmentation in LMDB - register under single key
                family_map[family_id][base_augments]["with_hint"] = trace
            elif prompt_db is not None and self._check_prompt_db_for_hint(trace.trace_id, prompt_db):
                # trace has hints in prompt DB but no LMDB hint - register under BOTH keys
                # since the same trace can be rendered with or without hints at sample time
                family_map[family_id][base_augments]["with_hint"] = trace
                family_map[family_id][base_augments]["without_hint"] = trace
            else:
                # trace has no hints anywhere - register as without_hint only
                family_map[family_id][base_augments]["without_hint"] = trace
        return dict(family_map)

    def _has_lmdb_hint(
        self,
        trace_id: pyine.data.traces.dataset_utils.TraceIdentifier,
    ) -> bool:
        """Check if trace has the target hint type based on LMDB augmentation only.

        Args:
            trace_id: The trace identifier to check.

        Returns:
            True if the trace has the configured target hint type in LMDB augmentation.
        """
        if self.config.hint_type == HintType.helpful:
            return trace_id.is_hinted
        return trace_id.is_misleading

    def _create_hint_split_derived_subsets(
        self,
        subset_traces_meta: dict[
            pyine.data.datamodule.SubsetNameType,
            list[pyine.data.traces.dataset_utils.TraceMetadata],
        ],
        family_pairing_map: dict[str, dict[frozenset[str], dict[str, pyine.data.traces.dataset_utils.TraceMetadata]]],
        prompt_db: pyine.prompts.PromptResultDB | None = None,
    ) -> dict[str, pyine.data.traces.dataset_utils.DerivedSubsetInfo[pyine.data.traces.dataset_utils.TraceMetadata]]:
        """Create derived subsets for hint-based evaluation splits.

        Args:
            subset_traces_meta: Dict mapping primary subset names to their trace metadata lists.
            family_pairing_map: The trace family pairing map from `_build_trace_family_pairing_map`.
            prompt_db: Optional prompt result database to check for hint availability.
                Used as fallback when LMDB trace augmentation is not present.

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
            with_hints, without_hints = self._partition_traces_by_hint_strategy(traces, family_pairing_map, prompt_db)
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
        prompt_db: pyine.prompts.PromptResultDB | None = None,
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

        For traces with hints in the prompt DB (but no LMDB hint), the same trace can appear
        in BOTH subsets since it can be rendered with or without hints at sample time.

        Note on subset overlap semantics:
            When using prompt DB hints, the SAME trace object may appear in BOTH derived subsets.
            This is intentional; at sample generation time, the trace will be rendered with hints
            in `_with_hints` (via code_type_prob_map={"hinted": 1.0}) and without hints in
            `_without_hints` (via default code selection). This enables true counterfactual
            evaluation where the only difference is hint presence.

            For evaluation metrics, be aware that:
            - The underlying trace is identical in both subsets (same problem, solution, test)
            - Only the code rendering differs (hinted vs original code from prompt DB)
            - This is NOT double-counting in the traditional sense; it's paired evaluation

        Args:
            traces: List of traces to partition.
            family_pairing_map: The trace family pairing map.
            prompt_db: Optional prompt result database to check for hint availability.
                Used as fallback when LMDB trace augmentation is not present.

        Returns:
            Tuple of (with_hints, without_hints) trace lists.
        """
        with_hints: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
        without_hints: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
        for trace in traces:
            if self.config.evaluation_strategy == EvaluationStrategy.counterfactual:
                # counterfactual: only include traces that have a complete pair (same base augments)
                family_id = str(trace.trace_id.get_augmentless_identifier())
                base_augments = self._get_base_augments(trace.trace_id)
                family_info = family_pairing_map.get(family_id, {})
                pair_info = family_info.get(base_augments, {})
                has_complete_pair = "with_hint" in pair_info and "without_hint" in pair_info
                if has_complete_pair:
                    has_lmdb_hint = self._has_lmdb_hint(trace.trace_id)
                    has_prompt_db_hint = (
                        not has_lmdb_hint
                        and prompt_db is not None
                        and self._check_prompt_db_for_hint(trace.trace_id, prompt_db)
                    )
                    if has_lmdb_hint:
                        # LMDB hinted; trace goes in with_hints only
                        with_hints.append(trace)
                    elif has_prompt_db_hint:
                        # hintless, with prompt DB hint; trace goes in BOTH (same trace, rendered differently)
                        with_hints.append(trace)
                        without_hints.append(trace)
                    else:
                        # LMDB hintless; trace goes in without_hints only
                        without_hints.append(trace)
                # else: drop trace from counterfactual evaluation (no matching pair)
            elif self.config.evaluation_strategy == EvaluationStrategy.hint_presence_split:
                # hint_presence_split: simple partition based on whether trace has the target hint
                has_lmdb_hint = self._has_lmdb_hint(trace.trace_id)
                has_prompt_db_hint = (
                    not has_lmdb_hint
                    and prompt_db is not None
                    and self._check_prompt_db_for_hint(trace.trace_id, prompt_db)
                )
                if has_lmdb_hint:
                    with_hints.append(trace)
                elif has_prompt_db_hint:
                    # prompt DB hint - trace goes in BOTH (same trace, rendered differently)
                    with_hints.append(trace)
                    without_hints.append(trace)
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
            parent_trace_count = len(subset_traces_meta[eval_subset_name])
            with_hints_key = f"{eval_subset_name}_with_hints"
            without_hints_key = f"{eval_subset_name}_without_hints"
            with_hints_info = derived_subsets.get(with_hints_key)
            without_hints_info = derived_subsets.get(without_hints_key)
            with_hints_count = len(with_hints_info.traces) if with_hints_info else 0
            without_hints_count = len(without_hints_info.traces) if without_hints_info else 0
            # check minimum counts (hard error)
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
            # warn if derived subset counts are significantly lower than parent (soft warning)
            # this can indicate that counterfactual pairing dropped many traces, or that
            # trace filtering (max_trace_steps, etc.) will further reduce usable samples
            derived_total = with_hints_count + without_hints_count
            if parent_trace_count > 0 and derived_total < 0.5 * parent_trace_count:
                logger.warning(
                    f"derived subsets for '{eval_subset_name}' have only {derived_total} traces "
                    f"(parent has {parent_trace_count}); this may indicate aggressive filtering "
                    "or missing hint pairs for counterfactual evaluation"
                )

    def _get_overlapping_trace_ids(
        self,
        eval_subset_name: str,
    ) -> frozenset[str]:
        """Get trace IDs that appear in both _with_hints and _without_hints derived subsets.

        These are traces with prompt DB hints (but no LMDB hint augmentation) that will be
        rendered differently depending on which derived subset is accessed. Sample identifiers
        for these traces need modification to ensure uniqueness.

        Args:
            eval_subset_name: Base eval subset name (e.g., "valid").

        Returns:
            Frozenset of trace identifiers that appear in both derived subsets.
        """
        assert self._metadata is not None
        with_hints_key = f"{eval_subset_name}_with_hints"
        without_hints_key = f"{eval_subset_name}_without_hints"
        if with_hints_key not in self._metadata.derived_subsets:
            return frozenset()
        if without_hints_key not in self._metadata.derived_subsets:
            return frozenset()
        with_hints_ids = frozenset(t.identifier for t in self._metadata.derived_subsets[with_hints_key].traces)
        without_hints_ids = frozenset(t.identifier for t in self._metadata.derived_subsets[without_hints_key].traces)
        return with_hints_ids & without_hints_ids

    @typing.override
    def get_parser(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.organisms.datamodules.samples.SampleBuilder:
        """Returns a data parser, wrapped with identifier modification if needed.

        For derived hint subsets (`_with_hints`, `_without_hints`), traces that appear in BOTH
        subsets (prompt DB hint traces) have their sample identifiers modified with suffixes
        to ensure uniqueness:

        - `::with_hint` suffix for samples from `_with_hints` subset
        - `::without_hint` suffix for samples from `_without_hints` subset

        This is necessary because the same trace can be rendered with or without hints depending
        on the subset, and downstream consumers (caches, evaluators, data stores) require unique
        identifiers.

        Args:
            subset_name: Name of the subset to get parser for.

        Returns:
            Sample builder, optionally wrapped with identifier modification.
        """
        base_parser = super().get_parser(subset_name)
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        # check if this is a derived hints subset that needs identifier modification
        if subset_name.endswith("_with_hints"):
            parent_subset = subset_name[: -len("_with_hints")]
            overlapping_ids = self._get_overlapping_trace_ids(parent_subset)
            if overlapping_ids:
                return SampleHintIdentifierWrapper(
                    wrapped_dataset=base_parser,
                    overlapping_trace_ids=overlapping_ids,
                    hint_suffix="with_hint",
                )  # type: ignore[return-value]
        elif subset_name.endswith("_without_hints"):
            parent_subset = subset_name[: -len("_without_hints")]
            overlapping_ids = self._get_overlapping_trace_ids(parent_subset)
            if overlapping_ids:
                return SampleHintIdentifierWrapper(
                    wrapped_dataset=base_parser,
                    overlapping_trace_ids=overlapping_ids,
                    hint_suffix="without_hint",
                )  # type: ignore[return-value]
        return base_parser


class SampleHintIdentifierWrapper:
    """Wrapper that modifies sample identifiers for traces appearing in multiple hint subsets.

    This wrapper ensures unique sample identifiers when the same trace appears in both
    `_with_hints` and `_without_hints` derived subsets. For overlapping traces, the identifier
    is modified by appending a suffix (e.g., `::with_hint` or `::without_hint`).

    This is necessary because traces with prompt DB hints (but no LMDB hint augmentation) can
    be rendered with or without hints depending on the derived subset accessed. Without identifier
    modification, the same identifier would map to different content, causing issues in:
    - Evaluation result tracking (results would overwrite each other);
    - Caching (wrong cached content could be returned);
    - Data stores and logging (duplicate key errors or data corruption).
    """

    def __init__(
        self,
        wrapped_dataset: torch.utils.data.Dataset[pyine.organisms.datamodules.samples.common.SampleData],
        overlapping_trace_ids: frozenset[str],
        hint_suffix: str,
    ) -> None:
        """Initialize the wrapper.

        Args:
            wrapped_dataset: The underlying dataset (typically a SampleBuilder).
            overlapping_trace_ids: Set of trace identifiers that appear in both hint subsets
                and need identifier modification.
            hint_suffix: Suffix to append to overlapping identifiers (e.g., "with_hint").
        """
        self._wrapped = wrapped_dataset
        self._overlapping_ids = overlapping_trace_ids
        self._suffix = f"::{hint_suffix}"

    def __getattr__(self, name: str) -> typing.Any:
        """Forward unknown attribute access to the wrapped dataset.

        This makes the wrapper transparent; any attribute not explicitly defined by the wrapper
        (e.g. `config`) is forwarded to the underlying dataset.
        """
        return getattr(self._wrapped, name)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._wrapped)  # type: ignore[arg-type]

    def __getitem__(
        self,
        index: int,
    ) -> pyine.organisms.datamodules.samples.common.SampleData:
        """Get a sample, modifying identifier if it's an overlapping trace."""
        sample = self._wrapped[index]
        if sample.identifier in self._overlapping_ids:
            return sample._replace(identifier=f"{sample.identifier}{self._suffix}")
        return sample

    @property
    def current_epoch(self) -> int:
        """Return the current epoch from the wrapped dataset."""
        if hasattr(self._wrapped, "current_epoch"):
            return self._wrapped.current_epoch  # type: ignore[union-attr]
        return 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch on the wrapped dataset if supported."""
        if hasattr(self._wrapped, "set_epoch"):
            self._wrapped.set_epoch(epoch)  # type: ignore[union-attr]

    @property
    def orig_traces(self) -> list[pyine.data.traces.dataset_utils.TraceMetadata]:
        """Return original traces from wrapped dataset (for compatibility with base parser)."""
        if hasattr(self._wrapped, "orig_traces"):
            return self._wrapped.orig_traces  # type: ignore[union-attr]
        return []

    def get_stats(self) -> dict[str, int | float | str]:
        """Get statistics from the wrapped dataset with additional wrapper info."""
        stats: dict[str, int | float | str] = {}
        if hasattr(self._wrapped, "get_stats"):
            wrapped_stats = self._wrapped.get_stats()  # type: ignore[union-attr]
            stats = typing.cast("dict[str, int | float | str]", wrapped_stats)
        stats["overlapping_trace_count"] = len(self._overlapping_ids)
        stats["hint_suffix"] = self._suffix
        return stats
