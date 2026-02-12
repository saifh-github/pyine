import collections
import collections.abc
import dataclasses
import enum
import logging

import numpy as np

import pyine.data.traces.dataset_utils
import pyine.prompts
from pyine.organisms.datamodules.samples.common import (
    SampleCodeType,
    SampleCodeTypeSet,
    TraceDatasetToSampleCodeTypeMappings,
    TraceToSampleCodeTypeMapping,
    draw_type,
    get_code_type_set_from_str,
    hint_type_to_sample_code_type,
)
from pyine.organisms.datamodules.samples.configs import SampleSelectionConfig, get_default_code_type_prob_map

logger = logging.getLogger(__name__)

__all__ = [
    "SelectedSample",
    "SampleSelectionResults",
    "SampleSelectionSource",
    "select_samples_from_trace_families",
]


class SampleSelectionSource(str, enum.Enum):
    """Describes how a sample was selected."""

    full_trace = "full_trace"
    prompt_db = "prompt_db"
    parent_fallback = "parent_fallback"


@dataclasses.dataclass(frozen=True)
class SelectedSample:
    """Result of the sample selection process inside a trace family."""

    parent_id: pyine.data.traces.dataset_utils.TraceIdentifier
    """Parent trace identifier for the trace family."""
    trace_id: pyine.data.traces.dataset_utils.TraceIdentifier
    """Target trace identifier for the sample to be generated."""
    trace_meta: pyine.data.traces.dataset_utils.TraceMetadata
    """The metadata associated with the target trace from which to generate a sample."""
    code_type: SampleCodeTypeSet
    """The type of the code snippet that will be used in the generated sample."""
    selection_source: SampleSelectionSource
    """The selection path that produced the sample."""
    code_override: str | None = None  # if None, use the target trace's code snippet directly
    """The code snippet override that will be used in the generated sample (instead of the original)."""


@dataclasses.dataclass(frozen=True)
class SampleSelectionResults:
    """Holds results and statistics about the selections done for a trace dataset."""

    orig_trace_data: TraceDatasetToSampleCodeTypeMappings
    """Trace dataset from which samples were selected."""
    selection_config: SampleSelectionConfig
    """Sample selection configuration that was used."""
    samples: list[SelectedSample]
    """List of all samples that were selected."""
    failed_selections: int = 0
    """Number of samples that were not selected due to failed attempts to find matching traces."""
    samples_with_full_trace_support: int = 0
    """Number of samples that were selected from traces that directly support the target code type."""
    samples_with_prompt_db_code: int = 0
    """Number of samples that were selected from traces that support the target code type via the prompt result db."""
    samples_with_parent_fallback: int = 0
    """Number of samples that are fallbacks-to-parent to avoid failures."""

    def __len__(self) -> int:
        """Returns the number of samples that were selected."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> SelectedSample:
        """Returns the sample associated with a specified index (where `0 <= idx < len(self)`)."""
        return self.samples[idx]

    def get_selected_traces(self) -> list[pyine.data.traces.dataset_utils.TraceMetadata]:
        """Returns all traces that were selected to generate samples."""
        target_trace_ids = {sample.trace_id for sample in self.samples}
        assert all(tid in self.orig_trace_data.trace_metadata_lut for tid in target_trace_ids)
        return [self.orig_trace_data.trace_metadata_lut[tid] for tid in target_trace_ids]

    def get_code_type_counts(self) -> dict[SampleCodeTypeSet, int]:
        """Returns the code type set counts for all selected samples."""
        code_type_counts = collections.Counter([s.code_type for s in self.samples])
        return dict(code_type_counts.items())

    def filter_samples(
        self,
        sample_filter: collections.abc.Callable[[SelectedSample], bool],
    ) -> "SampleSelectionResults":
        """Returns a copy of these results with samples filtered by the provided predicate."""
        filtered_samples = [sample for sample in self.samples if sample_filter(sample)]
        samples_with_full_trace_support = sum(
            1 for sample in filtered_samples if sample.selection_source == SampleSelectionSource.full_trace
        )
        samples_with_prompt_db_code = sum(
            1 for sample in filtered_samples if sample.selection_source == SampleSelectionSource.prompt_db
        )
        samples_with_parent_fallback = sum(
            1 for sample in filtered_samples if sample.selection_source == SampleSelectionSource.parent_fallback
        )
        return SampleSelectionResults(
            orig_trace_data=self.orig_trace_data,
            selection_config=self.selection_config,
            samples=filtered_samples,
            failed_selections=self.failed_selections,
            samples_with_full_trace_support=samples_with_full_trace_support,
            samples_with_prompt_db_code=samples_with_prompt_db_code,
            samples_with_parent_fallback=samples_with_parent_fallback,
        )

    def __post_init__(self) -> None:
        """Validates the selection stats."""
        assert self.failed_selections >= 0
        assert 0 <= self.samples_with_full_trace_support <= len(self.samples)
        assert 0 <= self.samples_with_prompt_db_code <= len(self.samples)
        assert 0 <= self.samples_with_parent_fallback <= len(self.samples)
        assert len(self.samples) == sum(
            [
                self.samples_with_full_trace_support,
                self.samples_with_prompt_db_code,
                self.samples_with_parent_fallback,
            ]
        )
        samples_with_overrides = [s for s in self.samples if s.code_override is not None]
        assert len(samples_with_overrides) == self.samples_with_prompt_db_code
        samples_with_prompt_db_source = [
            s for s in self.samples if s.selection_source == SampleSelectionSource.prompt_db
        ]
        assert len(samples_with_prompt_db_source) == self.samples_with_prompt_db_code
        samples_with_full_trace_source = [
            s for s in self.samples if s.selection_source == SampleSelectionSource.full_trace
        ]
        assert len(samples_with_full_trace_source) == self.samples_with_full_trace_support
        samples_with_parent_fallback_source = [
            s for s in self.samples if s.selection_source == SampleSelectionSource.parent_fallback
        ]
        assert len(samples_with_parent_fallback_source) == self.samples_with_parent_fallback
        assert self.selection_config.allow_db_lookups or self.samples_with_prompt_db_code == 0
        assert self.selection_config.fallback_to_orig or self.samples_with_parent_fallback == 0


def _find_trace_for_target_type(
    target_type: SampleCodeTypeSet,
    trace_data: list[TraceToSampleCodeTypeMapping],
    rng: np.random.Generator,
) -> pyine.data.traces.dataset_utils.TraceMetadata:
    """Finds a trace that matches the given target type.

    Will use the given generator to pick a trace among all traces that support the target type. If
    no compatible trace is found, raises a ValueError.
    """
    potential_choices: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
    for trace in trace_data:
        if target_type == trace.trace_sample_code_types:
            potential_choices.append(trace.target_trace_meta)
    if not potential_choices:
        raise ValueError(f"no trace found for target type {target_type}")
    picked_idx = rng.choice(len(potential_choices))
    return potential_choices[picked_idx]


def _find_db_match_for_target_type(
    target_type: SampleCodeTypeSet,
    trace_data: list[TraceToSampleCodeTypeMapping],
    prompt_result_db: pyine.prompts.PromptResultDB,
    rng: np.random.Generator,
) -> pyine.prompts.PromptResultRecord | None:
    """Finds a record that matches the given target type using the prompt result db.

    Will use the given generator to pick a record among all records that match the target type.
    Returns None if no compatible record is found.

    Uses deterministic pre-sort before random selection to ensure reproducibility across
    different DB engines or query plans.
    """
    potential_choices: list[pyine.prompts.PromptResultRecord] = []
    records_fetched_no_match: list[tuple[str, list[str]]] = []  # for debug logging
    for trace in trace_data:
        assert target_type != trace.trace_sample_code_types, "why are we looking for a db match?"
        for db_keys, code_type_sets in trace.db_supported_sample_code_types.items():
            if target_type in code_type_sets:
                # TODO: should we add a filter for max db result age here?
                records = prompt_result_db.get_by_identifier(**db_keys._asdict())
                matched_any = False
                for record in records:
                    if SampleCodeTypeSet.create_from_tags(record.tags) == target_type:
                        potential_choices.append(record)
                        matched_any = True
                if records and not matched_any:
                    # records exist but none matched; likely tag format issue
                    sample_tags = list({t for r in records for t in r.tags})
                    records_fetched_no_match.append((db_keys.identifier, sample_tags))
    if not potential_choices:
        # log debug info about records that were fetched but didn't match
        if records_fetched_no_match:
            logger.debug(
                f"prompt DB has records for target type {target_type} keys, but none matched; "
                f"sample non-matching tags: {records_fetched_no_match[:3]}"
            )
        return None
    # sort deterministically BEFORE random selection for DB-engine-stable reproducibility
    # (sorting does NOT bias uniform random selection, it just ensures stable input order)
    potential_choices.sort(key=lambda r: (r.identifier, r.creation_meta.created_at.isoformat()))
    picked_idx = rng.choice(len(potential_choices))
    return potential_choices[picked_idx]


def _filter_traces_by_base_type(
    trace_data: list[TraceToSampleCodeTypeMapping],
    code_type_prob_map: dict[SampleCodeTypeSet | str, float],
) -> list[TraceToSampleCodeTypeMapping]:
    """Filter traces to only include those matching the specified hintable base types.

    When used with require_hint_type, this allows creating counterfactual subsets with specific
    hintable base augment requirements (e.g., only original traces).

    Args:
        trace_data: List of trace-to-code-type mappings for the family.
        code_type_prob_map: Map of base type names to probs. Only types with prob > 0 are considered.
            Keys must be simple base types ("original", "obfuscated", etc.), not compound or hint types.

    Returns:
        Filtered list of traces matching at least one allowed base type.

    Raises:
        ValueError: If a hint type ("hinted", "misleading") or "stubbed" is in code_type_prob_map.
        NotImplementedError: If a multi-augment base type is encountered.
    """
    hint_type_names = {"hinted", "misleading"}
    hint_incompatible_types = {"stubbed"}  # fail fast instead of silent empty filter
    # validate and normalize allowed base keys
    allowed_base_keys: set[tuple[str, ...] | str] = set()
    for key, prob in code_type_prob_map.items():
        if prob <= 0:
            continue
        key_str = str(key) if not isinstance(key, str) else key
        # reject hint types -- they don't make sense as base type filters
        if key_str in hint_type_names:
            raise ValueError(
                f"Hint type '{key_str}' in code_type_prob_map is not valid for base type filtering. "
                f"Use only base types like 'original', 'obfuscated', 'bugged'."
            )
        # reject stubbed -- cannot produce complete groups (stubbed + hints is invalid)
        if key_str in hint_incompatible_types:
            raise ValueError(
                f"'{key_str}' in code_type_prob_map cannot be used for hint-based selection. "
                f"Stubbed traces cannot receive hints (stubbed + hints is invalid)."
            )
        # parse key to validate it's a valid code type
        try:
            parsed_set = SampleCodeTypeSet(get_code_type_set_from_str(key_str))
        except ValueError as exc:
            raise ValueError(f"Invalid code type key '{key_str}' in code_type_prob_map: {exc}") from exc
        # reject if parsed set contains hint types (e.g., "obfuscated_hinted")
        if SampleCodeType.hinted in parsed_set.types or SampleCodeType.misleading in parsed_set.types:
            raise ValueError(
                f"Code type '{key_str}' contains hint types, which is not valid for base type filtering. "
                f"Use only base types like 'original', 'obfuscated', 'bugged'."
            )
        # compute base augment key to check for multi-augment or hint types
        base_key = parsed_set.get_counterfactual_grouping_key()
        # reject multi-augment base types (tuple with len > 1)
        if isinstance(base_key, tuple) and len(base_key) > 1:
            raise NotImplementedError(
                f"Multi-augment base type '{key_str}' (base_key={base_key}) in code_type_prob_map "
                f"is not supported. Only simple base types like 'original', 'obfuscated' are allowed."
            )
        allowed_base_keys.add(base_key)
    filtered: list[TraceToSampleCodeTypeMapping] = []
    for trace in trace_data:
        trace_base_key = trace.trace_sample_code_types.get_counterfactual_grouping_key()
        if trace_base_key in allowed_base_keys:
            filtered.append(trace)
    return filtered


def _find_lmdb_trace_with_hint_type(
    hint_type: SampleCodeType,
    trace_data: list[TraceToSampleCodeTypeMapping],
    rng: np.random.Generator,
) -> tuple[pyine.data.traces.dataset_utils.TraceMetadata, SampleCodeTypeSet] | None:
    """Find a trace whose LMDB-native code type contains the specified hint type.

    Only checks ``trace_sample_code_types`` (what the trace natively is in LMDB), NOT
    ``db_supported_sample_code_types`` (prompt-DB hints). Callers should fall back to
    ``_try_prompt_db_hint_lookup`` when this returns None.

    Selection uses seeded random for reproducibility without systematic bias.

    Args:
        hint_type: The hint type to search for (hinted or misleading).
        trace_data: List of trace-to-code-type mappings for the family.
        rng: Seeded random generator for reproducible selection.

    Returns:
        (trace_meta, native_code_type) tuple if match found, None otherwise.
    """
    matching_traces: list[tuple[pyine.data.traces.dataset_utils.TraceMetadata, SampleCodeTypeSet]] = []
    for trace in trace_data:
        if trace.trace_sample_code_types.has(hint_type):
            matching_traces.append((trace.target_trace_meta, trace.trace_sample_code_types))
    if not matching_traces:
        return None
    # sort deterministically BEFORE random selection for reproducibility across different input orderings
    matching_traces.sort(key=lambda t: str(t[0].trace_id))
    picked_idx = rng.choice(len(matching_traces))
    return matching_traces[picked_idx]


def _try_prompt_db_hint_lookup(
    selection_config: SampleSelectionConfig,
    trace: TraceToSampleCodeTypeMapping,
    rng: np.random.Generator,
    prompt_result_db: pyine.prompts.PromptResultDB,
    output_selections: list["SelectedSample"],
    parent_id: pyine.data.traces.dataset_utils.TraceIdentifier,
    trace_data: TraceDatasetToSampleCodeTypeMappings,
) -> bool:
    """Try to find a hint via prompt-DB for a specific hintless trace.

    The trace must NOT already contain hints (caller enforces via ``can_receive_hint_type``).
    Determines the target hint type based on the trace's base augments:
    - for {original} traces: look up {hinted} directly;
    - for augmented traces (e.g. {obfuscated}): look up {base_augments + hint_type}.

    No cross-fallback between base types; lookup is restricted to THIS trace only, not the whole
    family. This ensures the base augment is preserved.

    IMPORTANT: handles two edge cases safely:
    1. record.identifier may be a solution ID (not trace ID) -- check and use parent_id;
    2. the resolved trace ID may have been filtered out -- skip if not in trace_metadata_lut.

    Args:
        selection_config: The selection configuration.
        trace: The specific trace to find hints for (lookup restricted to this trace).
        rng: Seeded random generator for reproducible selection.
        prompt_result_db: The prompt result database.
        output_selections: List to append successful selections to.
        parent_id: The family's parent trace identifier.
        trace_data: Full trace dataset for metadata lookup.

    Returns:
        True if selection succeeded, False otherwise.
    """
    assert selection_config.require_hint_type is not None, "require_hint_type must be set"
    hint_code_type = hint_type_to_sample_code_type(selection_config.require_hint_type)
    assert trace.trace_sample_code_types.can_receive_hint_type(hint_code_type), (
        f"trace {trace.target_trace_id} already has hints or is hint-incompatible; "
        f"caller must filter via can_receive_hint_type before calling this function"
    )
    # determine target type based on trace's base augments
    base_key = trace.trace_sample_code_types.get_counterfactual_grouping_key()
    if base_key == "original":
        # {original} traces look up {hinted} directly
        target_type = SampleCodeTypeSet(frozenset({hint_code_type}))
    elif base_key == "stubbed":
        return False  # stubbed traces cannot receive hints
    else:
        # other traces look up {base_augments + hint_type}
        base_augments = trace.trace_sample_code_types.get_hintable_base_augments()
        if not base_augments:
            return False  # shouldn't happen given base_key check above
        try:
            target_type = SampleCodeTypeSet(frozenset(base_augments | {hint_code_type}))
        except ValueError:
            return False  # invalid combination
    # lookup restricted to THIS trace only (not whole family)
    record = _find_db_match_for_target_type(
        target_type=target_type,
        trace_data=[trace],
        prompt_result_db=prompt_result_db,
        rng=rng,
    )
    if record is None:
        return False
    # safe identifier resolution
    solution_id_str = str(parent_id.get_parent_identifier())
    if record.identifier == solution_id_str:
        record_tid = parent_id
    else:
        record_tid = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(record.identifier)
    # only append if trace is in the dataset (may have been filtered out)
    if record_tid not in trace_data.trace_metadata_lut:
        logger.debug(
            f"skipping prompt-DB record {record.identifier}: resolved trace {record_tid} not in dataset (filtered out?)"
        )
        return False
    output_selections.append(
        SelectedSample(
            parent_id=parent_id,
            trace_id=record_tid,
            trace_meta=trace_data.trace_metadata_lut[record_tid],
            code_type=target_type,
            selection_source=SampleSelectionSource.prompt_db,
            code_override=record.result,
        )
    )
    return True


def select_samples_from_trace_families(
    trace_data: TraceDatasetToSampleCodeTypeMappings,
    epoch: int,
    selection_config: SampleSelectionConfig,
    prompt_result_db: pyine.prompts.PromptResultDB,
) -> SampleSelectionResults:
    """Select samples from trace families according to the configured selection mode.

    Iterates over trace families and selects one sample per family (up to ``samples_per_family``).
    Three mutually exclusive selection modes are supported:

    1. ``require_hint_type``: find traces containing a specific hint type. Tries LMDB-native
       matches first (via ``_find_lmdb_trace_with_hint_type``), then falls back to prompt-DB
       lookup (via ``_try_prompt_db_hint_lookup``) if ``allow_db_lookups`` is enabled.
    2. ``skip_code_type_selection``: accept each trace's existing code type as-is, using
       deterministic sort-then-index selection. Intended for derived subsets (e.g. ``_hintless``)
       whose traces were already pre-partitioned upstream by ``build_counterfactual_eval_subsets``
       or ``_partition_traces_by_hint_strategy`` -- no further filtering or drawing is needed.
    3. ``code_type_prob_map`` draw (default): draw a target code type from the probability map,
       then find a matching trace in LMDB or prompt-DB. Used for training.

    Args:
        trace_data: Pre-built family-level trace mappings with code type and DB availability info.
        epoch: Current training epoch (seeds the RNG for reproducible selection).
        selection_config: Configuration controlling which mode, probabilities, and fallbacks to use.
        prompt_result_db: Prompt result database for augmented code lookups.

    Returns:
        Selection results including chosen samples and statistics (failed, DB-backed, fallback counts).
    """
    rng = selection_config.get_rng(epoch)
    code_type_prob_map = selection_config.get_code_type_prob_map_resolved()
    output_selections: list[SelectedSample] = []
    failed_selections: int = 0
    samples_with_full_trace_support: int = 0
    samples_with_prompt_db_code: int = 0
    samples_with_parent_fallback: int = 0
    families_with_no_hint_compatible_traces: int = 0  # for summary warning
    for parent_id, family_mapping in trace_data.trace_families.items():
        family_trace_data = list(family_mapping.values())
        # fail fast on empty family
        if not family_trace_data:
            for _ in range(selection_config.samples_per_family):
                failed_selections += 1
            continue
        for sample_idx in range(selection_config.samples_per_family):
            got_selection = False
            # mode 1: contains-match for hint type (seeded random, preserves base augments)
            if selection_config.require_hint_type is not None:
                # apply base type filtering if code_type_prob_map specifies non-default base types
                raw_prob_map = selection_config.code_type_prob_map
                default_prob_map = get_default_code_type_prob_map()
                if raw_prob_map != default_prob_map:
                    filtered_trace_data = _filter_traces_by_base_type(
                        trace_data=family_trace_data,
                        code_type_prob_map=raw_prob_map,
                    )
                else:
                    filtered_trace_data = family_trace_data
                # sort for reproducibility: input ordering from dict iteration must not affect RNG draws
                filtered_trace_data = sorted(filtered_trace_data, key=lambda t: str(t.target_trace_id))
                # map HintType to SampleCodeType for internal use
                hint_code_type = hint_type_to_sample_code_type(selection_config.require_hint_type)
                # try LMDB match first (seeded random selection)
                match_result = _find_lmdb_trace_with_hint_type(
                    hint_type=hint_code_type,
                    trace_data=filtered_trace_data,
                    rng=rng,
                )
                if match_result is not None:
                    trace_meta, native_type = match_result
                    output_selections.append(
                        SelectedSample(
                            parent_id=parent_id,
                            trace_id=trace_meta.trace_id,
                            trace_meta=trace_meta,
                            code_type=native_type,
                            selection_source=SampleSelectionSource.full_trace,
                            code_override=None,
                        )
                    )
                    samples_with_full_trace_support += 1
                    got_selection = True
                elif selection_config.allow_db_lookups:
                    # no LMDB match, try prompt-DB lookup for filtered traces
                    # use permuted indices to avoid systematic bias, then try each until one succeeds
                    permuted_indices = rng.permutation(len(filtered_trace_data))
                    for trace_idx in permuted_indices:
                        trace: TraceToSampleCodeTypeMapping = filtered_trace_data[int(trace_idx)]
                        if not trace.trace_sample_code_types.can_receive_hint_type(hint_code_type):
                            continue  # skip hint-incompatible traces
                        got_selection = _try_prompt_db_hint_lookup(
                            selection_config=selection_config,
                            trace=trace,
                            rng=rng,
                            prompt_result_db=prompt_result_db,
                            output_selections=output_selections,
                            parent_id=parent_id,
                            trace_data=trace_data,
                        )
                        if got_selection:
                            samples_with_prompt_db_code += 1
                            break
                    # count failures (summary warning emitted at end to avoid log spam)
                    if not got_selection and sample_idx == 0:
                        families_with_no_hint_compatible_traces += 1
                if not got_selection:
                    # note: fallback_to_orig is forbidden when require_hint_type is set (enforced by validator)
                    failed_selections += 1
                continue
            # mode 2: native code type (DETERMINISTIC for counterfactual pairing)
            if selection_config.skip_code_type_selection:
                # deterministic selection: sort traces, pick by sample_idx
                # this ensures the SAME trace is selected across derived subsets
                sorted_traces = sorted(family_trace_data, key=lambda t: str(t.target_trace_id))
                trace_entry = sorted_traces[sample_idx % len(sorted_traces)]
                native_type = trace_entry.trace_sample_code_types
                output_selections.append(
                    SelectedSample(
                        parent_id=parent_id,
                        trace_id=trace_entry.target_trace_id,
                        trace_meta=trace_entry.target_trace_meta,
                        code_type=native_type,
                        selection_source=SampleSelectionSource.full_trace,
                        code_override=None,
                    )
                )
                samples_with_full_trace_support += 1
                continue
            # mode 3: old code_type_prob_map draw logic (uses RNG, useful for training)
            for _ in range(selection_config.draw_attempts):
                target_type = draw_type(code_type_prob_map, rng)
                assert isinstance(target_type, SampleCodeTypeSet)
                if (
                    target_type in trace_data.trace_family_sample_code_type_counts[parent_id]
                    and trace_data.trace_family_sample_code_type_counts[parent_id][target_type] > 0
                ):
                    # drawn target type is supported by trace directly; pick a corresponding one
                    target_trace_meta = _find_trace_for_target_type(
                        target_type=target_type,
                        trace_data=family_trace_data,
                        rng=rng,
                    )
                    output_selections.append(
                        SelectedSample(
                            parent_id=parent_id,
                            trace_id=target_trace_meta.trace_id,
                            trace_meta=target_trace_meta,
                            code_type=target_type,
                            selection_source=SampleSelectionSource.full_trace,
                            code_override=None,
                        )
                    )
                    samples_with_full_trace_support += 1
                    got_selection = True
                    break
                if (
                    selection_config.allow_db_lookups
                    and parent_id in trace_data.db_supported_family_sample_code_type_counts
                    and target_type in trace_data.db_supported_family_sample_code_type_counts[parent_id]
                    and trace_data.db_supported_family_sample_code_type_counts[parent_id][target_type] > 0
                ):
                    # drawn target type is supported by augmented code in the prompt result db
                    record = _find_db_match_for_target_type(
                        target_type=target_type,
                        trace_data=family_trace_data,
                        prompt_result_db=prompt_result_db,
                        rng=rng,
                    )
                    if record is None:
                        failed_selections += 1
                        break  # counts said it exists but DB lookup found nothing
                    solution_id_str = str(parent_id.get_parent_identifier())
                    if record.identifier == solution_id_str:
                        record_tid = parent_id  # record is tied to the solution code, so get parent trace itself
                    else:
                        record_tid = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(record.identifier)
                    # the picked record trace id might not be in the dataset if it was filtered out already
                    if record_tid in trace_data.trace_metadata_lut:
                        assert record_tid in trace_data.trace_families[parent_id], "record not in family?"
                        output_selections.append(
                            SelectedSample(
                                parent_id=parent_id,
                                trace_id=record_tid,
                                trace_meta=trace_data.trace_metadata_lut[record_tid],
                                code_type=target_type,
                                selection_source=SampleSelectionSource.prompt_db,
                                code_override=record.result,
                            )
                        )
                        samples_with_prompt_db_code += 1
                        got_selection = True
                        break
            if not got_selection and selection_config.fallback_to_orig and parent_id in trace_data.trace_metadata_lut:
                # draws all failed, but we can fallback to the orig trace, as it is in the dataset
                assert not parent_id.is_augmented
                output_selections.append(
                    SelectedSample(
                        parent_id=parent_id,
                        trace_id=parent_id,
                        trace_meta=trace_data.trace_metadata_lut[parent_id],
                        code_type=SampleCodeTypeSet.create_default(),
                        selection_source=SampleSelectionSource.parent_fallback,
                        code_override=None,
                    )
                )
                samples_with_parent_fallback += 1
                got_selection = True
            if not got_selection:
                failed_selections += 1
    # emit summary warning at end (avoids per-family log spam)
    if families_with_no_hint_compatible_traces > 0:
        logger.warning(f"selection failed for {families_with_no_hint_compatible_traces} trace families")
    return SampleSelectionResults(
        orig_trace_data=trace_data,
        selection_config=selection_config,
        samples=output_selections,
        failed_selections=failed_selections,
        samples_with_full_trace_support=samples_with_full_trace_support,
        samples_with_prompt_db_code=samples_with_prompt_db_code,
        samples_with_parent_fallback=samples_with_parent_fallback,
    )
