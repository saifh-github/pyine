"""DataModule for shortcut-bias experiments using code execution trace datasets."""

from __future__ import annotations

import collections
import collections.abc
import dataclasses
import logging
import typing

import numpy as np
import pydantic

import pyine.data.datamodule
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.base
import pyine.organisms.datamodules.samples
import pyine.organisms.datamodules.samples.common
import pyine.organisms.datamodules.samples.configs
import pyine.organisms.datamodules.samples.filtering
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


@dataclasses.dataclass
class CounterfactualGroup:
    """A group of traces that share the same (family_id, base_augment_key).

    All traces in a group:
    - share the same family_id (i.e., augmentless trace ID = same problem, solution, AND test case);
    - share the same hintable base augmentation (e.g., all have obfuscated as base, or all original);
    - differ only in hint augmentation (hinted, misleading, or none).

    This ensures true counterfactual comparison: same code, same test, different hints.

    Example for family_id="TACO/train/p000001/s0000/t0000", base_augment_key=("obfuscated",):
    - hintless_traces:   [.../t0000/a:obfuscated:001]
    - hinted_traces:     [.../t0000/a:obfuscated+hints_docs:001]
    - misleading_traces: [.../t0000/a:obfuscated+issues_docs:001]
    """

    family_id: str
    """Augmentless trace ID as a string (identifies the problem + solution + test case)."""
    base_augment_key: str | tuple[str, ...]
    """Grouping key from ``get_counterfactual_grouping_key()``, e.g. "original" or ("obfuscated",)."""
    hintless_traces: list[pyine.data.traces.dataset_utils.TraceMetadata] = dataclasses.field(
        default_factory=lambda: [],
    )
    """LMDB-backed traces with no hint augmentation (baseline for comparison)."""
    hinted_traces: list[pyine.data.traces.dataset_utils.TraceMetadata] = dataclasses.field(
        default_factory=lambda: [],
    )
    """LMDB-backed traces with helpful hint augmentation (is_hinted=True)."""
    misleading_traces: list[pyine.data.traces.dataset_utils.TraceMetadata] = dataclasses.field(
        default_factory=lambda: [],
    )
    """LMDB-backed traces with misleading hint augmentation (is_misleading=True)."""
    prompt_db_anchor: pyine.data.traces.dataset_utils.TraceMetadata | None = None
    """A single hintless trace that can receive hints via prompt-DB.

    When set, this SAME trace is used for the hintless subset AND all prompt-DB hint subsets,
    ensuring the "same trace across subsets" invariant for paired comparison.
    """
    prompt_db_has_helpful: bool = False
    """Whether the anchor trace has helpful hints available in the prompt-DB."""
    prompt_db_has_misleading: bool = False
    """Whether the anchor trace has misleading hints available in the prompt-DB."""

    def is_complete(
        self,
        eval_hint_types: tuple[HintType, ...],
    ) -> bool:
        """Check if this group can produce all required subsets.

        A group is complete if it has:
        1. at least one hintless trace (always required for baseline);
        2. for each configured hint type: either LMDB traces OR prompt-DB anchor coverage.

        Args:
            eval_hint_types: The hint types configured for evaluation.

        Returns:
            True if the group can produce traces for all required subsets.
        """
        if not self.hintless_traces:
            return False
        for hint_type in eval_hint_types:
            if hint_type == HintType.helpful:
                if not self.hinted_traces and not self.prompt_db_has_helpful:
                    return False
            elif hint_type == HintType.misleading:
                if not self.misleading_traces and not self.prompt_db_has_misleading:
                    return False
        return True


def _normalize_base_key(base_augment_key: str | tuple[str, ...]) -> str:
    """Normalize base_augment_key to string for matching against code_type_prob_map.

    Args:
        base_augment_key: Either "original", "stubbed", or a tuple like ("obfuscated",) that
            could represent a hintable base group identifier.

    Returns:
        String key for dict lookup (e.g., "original", "obfuscated").

    Raises:
        NotImplementedError: If multi-augment tuple (len > 1) is encountered.
    """
    if isinstance(base_augment_key, str):
        return base_augment_key  # "original" or "stubbed"
    assert isinstance(base_augment_key, tuple)
    if len(base_augment_key) == 1:
        return base_augment_key[0]  # ("obfuscated",) -> "obfuscated"
    raise NotImplementedError(
        f"Multi-augment base key {base_augment_key} cannot be normalized. "
        f"Only simple base types are supported for counterfactual distribution."
    )


def _sample_groups_by_distribution(
    complete_groups: list[CounterfactualGroup],
    code_type_prob_map: dict[str, float],
    rng: np.random.Generator,
) -> list[CounterfactualGroup]:
    """Sample groups according to the parent subset's code_type_prob_map.

    For counterfactual evaluation, this function ensures EQUAL COUNTS across derived subsets by
    using the minimum available across all requested base type buckets. This guarantees that
    hintless, hinted, and misleading subsets (if all enabled) have identical sizes.

    When a family has groups in multiple base-type buckets (e.g., both original and obfuscated),
    the family is assigned to exactly one bucket via a probabilistic draw weighted by
    ``code_type_prob_map``. This prevents downstream per-family selection (``samples_per_family=1``)
    from silently collapsing multiple groups into one trace.

    Multi-augment base types (e.g., ``("obfuscated", "bugged")``) are skipped with a warning, as
    they cannot be matched to a single ``code_type_prob_map`` key. @@@@ TODO: can't we have obfuscated_bugged?

    Args:
        complete_groups: Groups that can produce all required subsets.
        code_type_prob_map: Parent code type distribution, e.g. {"original": 0.5, "obfuscated": 0.5}.
        rng: Seeded random generator.

    Returns:
        Selected groups matching the target distribution (with equal counts per bucket).
    """
    # bucket groups by base_augment_key
    groups_by_base_type: dict[str, list[CounterfactualGroup]] = collections.defaultdict(list)
    skipped_multi_augment = 0
    for group in complete_groups:
        try:
            key = _normalize_base_key(group.base_augment_key)
            groups_by_base_type[key].append(group)
        except NotImplementedError:
            skipped_multi_augment += 1
            continue
    if skipped_multi_augment > 0:
        logger.warning(
            f"{skipped_multi_augment} complete groups skipped due to not-implemented multi-augment base types"
        )
    # sort each bucket deterministically for reproducible sampling
    for key in groups_by_base_type:
        groups_by_base_type[key].sort(key=lambda g: (g.family_id, str(g.base_augment_key)))
    # deduplicate families across buckets: each family contributes at most one group.
    # without this, select_samples_from_trace_families (samples_per_family=1) would silently
    # collapse multiple groups from the same family into a single trace, breaking distribution.
    # families present in multiple buckets are assigned probabilistically (weighted by
    # code_type_prob_map) so that the target distribution remains achievable.
    requested_types: list[tuple[str, float]] = [
        (base_type, prob) for base_type, prob in code_type_prob_map.items() if prob > 0
    ]
    # first pass: identify families that appear in multiple buckets
    family_to_buckets: dict[str, list[str]] = collections.defaultdict(list)
    for base_type, _ in requested_types:
        for group in groups_by_base_type.get(base_type, []):
            family_to_buckets[group.family_id].append(base_type)
    # second pass: for multi-bucket families, assign each to one bucket probabilistically
    family_assigned_bucket: dict[str, str] = {}
    deduped_families = 0
    for family_id, buckets in sorted(family_to_buckets.items()):  # sorted for determinism
        if len(buckets) <= 1:
            continue
        # draw assignment weighted by code_type_prob_map probabilities
        bucket_probs = np.array([code_type_prob_map[bt] for bt in buckets])
        bucket_probs = bucket_probs / bucket_probs.sum()
        chosen_idx = int(rng.choice(len(buckets), p=bucket_probs))
        family_assigned_bucket[family_id] = buckets[chosen_idx]
        deduped_families += 1
    # repair: prevent starved buckets when a feasible assignment exists.
    # a bucket is "starved" if shared families could go there but none were assigned,
    # AND it has no unique (single-bucket) families to keep it populated.
    # (note: the repair is heuristic and may still yield empty results in edge cases)
    if family_assigned_bucket:
        unique_per_bucket: dict[str, int] = collections.defaultdict(int)
        for _fid, buckets in family_to_buckets.items():
            if len(buckets) == 1:
                unique_per_bucket[buckets[0]] += 1
        shared_eligible: dict[str, list[str]] = collections.defaultdict(list)
        for family_id in family_assigned_bucket:
            for bucket in family_to_buckets[family_id]:
                shared_eligible[bucket].append(family_id)
        assigned_counts = collections.Counter(family_assigned_bucket.values())
        for starved_bt in sorted(shared_eligible.keys()):  # sorted for determinism
            if assigned_counts[starved_bt] > 0 or unique_per_bucket[starved_bt] > 0:
                continue  # not starved
            # steal one eligible family from a donor bucket that remains populated after removal
            # (donor keeps at least 1 family total = shared assignments + unique families)
            candidates = sorted(
                family_id
                for family_id in shared_eligible[starved_bt]
                if (
                    assigned_counts[family_assigned_bucket[family_id]]
                    + unique_per_bucket[family_assigned_bucket[family_id]]
                )
                > 1
            )
            if candidates:
                stolen_id = candidates[0]
                donor_bt = family_assigned_bucket[stolen_id]
                family_assigned_bucket[stolen_id] = starved_bt
                assigned_counts[donor_bt] -= 1
                assigned_counts[starved_bt] += 1
    # third pass: remove groups that lost the assignment draw
    if family_assigned_bucket:
        for base_type in list(groups_by_base_type.keys()):
            groups_by_base_type[base_type] = [
                group
                for group in groups_by_base_type[base_type]
                if group.family_id not in family_assigned_bucket or family_assigned_bucket[group.family_id] == base_type
            ]
    if deduped_families > 0:
        logger.debug(
            f"had {deduped_families} families in multiple base-type buckets; "
            f"each assigned probabilistically to one bucket"
        )
    # compute requested base types and their relative proportions
    if not requested_types:
        return []
    # find the bottleneck: which base type limits how many groups we can select?
    max_total_by_type: list[int] = []
    for base_type, prob in requested_types:
        available = len(groups_by_base_type.get(base_type, []))
        if prob > 0:
            max_total = int(available / prob) if available > 0 else 0
            max_total_by_type.append(max_total)
    # use minimum to ensure equal proportional counts across all buckets
    total_to_select = min(max_total_by_type) if max_total_by_type else 0
    if total_to_select == 0:
        shortage_info = {base_type: len(groups_by_base_type.get(base_type, [])) for base_type, _ in requested_types}
        logger.warning(
            f"cannot satisfy distribution {code_type_prob_map}: at least one requested "
            f"base type has no complete groups. Available per base type: {shortage_info}"
        )
        return []
    # check if we had to reduce due to shortage
    ideal_total = sum(len(groups_by_base_type.get(base_type, [])) for base_type, _ in requested_types)
    if total_to_select < ideal_total:
        shortage_info = {base_type: len(groups_by_base_type.get(base_type, [])) for base_type, _ in requested_types}
        logger.warning(
            f"distribution shortage: requested {ideal_total} groups with distribution "
            f"{code_type_prob_map}, but limited to {total_to_select} to maintain equal "
            f"counts; available per base type: {shortage_info}"
        )
    # use largest-remainder method for exact total allocation
    raw_counts = [(base_type, total_to_select * prob) for base_type, prob in requested_types]
    floor_counts = [(base_type, int(count)) for base_type, count in raw_counts]
    remainders = [(base_type, count - int(count)) for base_type, count in raw_counts]
    # distribute remainder to types with largest fractional parts
    total_floor = sum(count for _, count in floor_counts)
    remainder_to_distribute = total_to_select - total_floor
    remainders_sorted = sorted(remainders, key=lambda x: x[1], reverse=True)
    final_counts: dict[str, int] = dict(floor_counts)
    for idx in range(remainder_to_distribute):
        base_type = remainders_sorted[idx % len(remainders_sorted)][0]
        final_counts[base_type] += 1
    # now select from each bucket
    selected: list[CounterfactualGroup] = []
    for base_type, target_count in final_counts.items():
        available = groups_by_base_type.get(base_type, [])
        if target_count > 0 and len(available) >= target_count:
            indices = rng.choice(len(available), size=target_count, replace=False)
            selected.extend(available[int(idx)] for idx in indices)
    # shuffle to avoid ordering by base type (use random indices for type safety)
    shuffled_indices = rng.permutation(len(selected))
    return [selected[idx] for idx in shuffled_indices]


def _build_counterfactual_groups(
    traces: list[pyine.data.traces.dataset_utils.TraceMetadata],
    prompt_db: pyine.prompts.PromptResultDB | None,
    eval_hint_types: tuple[HintType, ...],
    check_prompt_db_fn: typing.Callable[
        [pyine.data.traces.dataset_utils.TraceIdentifier, pyine.prompts.PromptResultDB, HintType], bool
    ],
    check_lmdb_misleading_fn: typing.Callable[[pyine.data.traces.dataset_utils.TraceIdentifier], None] | None = None,
) -> dict[tuple[str, str | tuple[str, ...]], CounterfactualGroup]:
    """Build counterfactual groups keyed by (family_id, base_augment_key).

    Groups traces by their family (augmentless trace ID) and base augments, then determines hint
    availability for each group from LMDB and prompt-DB.

    Args:
        traces: List of trace metadata to group.
        prompt_db: Optional prompt result database for hint availability checks.
        eval_hint_types: The hint types configured for evaluation.
        check_prompt_db_fn: Function to check if prompt-DB has hint for a trace.
        check_lmdb_misleading_fn: Optional callback invoked when an LMDB misleading trace is
            encountered. Used to raise NotImplementedError when validated-misleading filtering
            is active (LMDB misleading traces cannot be validated yet).

    Returns:
        Dict mapping (family_id, base_augment_key) to CounterfactualGroup.
    """
    groups: dict[tuple[str, str | tuple[str, ...]], CounterfactualGroup] = {}
    for trace in traces:
        family_id = str(trace.trace_id.get_augmentless_identifier())
        # get base augment key using get_code_type_set_from_str (works with mock objects in tests)
        # note: this is equivalent to SampleCodeTypeSet.create_from_trace() but without isinstance checks
        code_types = pyine.organisms.datamodules.samples.common.get_code_type_set_from_str(
            trace.trace_id.augment_category
        )
        code_type_set = pyine.organisms.datamodules.samples.common.SampleCodeTypeSet(code_types)
        base_key = code_type_set.get_counterfactual_grouping_key()
        group_key = (family_id, base_key)
        # create group if it doesn't exist
        if group_key not in groups:
            groups[group_key] = CounterfactualGroup(family_id=family_id, base_augment_key=base_key)
        group = groups[group_key]
        # check LMDB hint status
        has_lmdb_helpful = trace.trace_id.is_hinted
        has_lmdb_misleading = trace.trace_id.is_misleading
        has_any_lmdb_hint = has_lmdb_helpful or has_lmdb_misleading
        # detect invalid LMDB state
        if has_lmdb_helpful and has_lmdb_misleading:
            raise ValueError(
                f"invalid LMDB state: trace {trace.trace_id} has both is_hinted=True and "
                f"is_misleading=True (this indicates corrupted data)"
            )
        # assign trace to appropriate list(s)
        if has_lmdb_helpful:
            group.hinted_traces.append(trace)
        elif has_lmdb_misleading:
            if check_lmdb_misleading_fn is not None:
                check_lmdb_misleading_fn(trace.trace_id)
            group.misleading_traces.append(trace)
        elif not has_any_lmdb_hint:
            group.hintless_traces.append(trace)
    # sort trace lists within each group for deterministic selection
    for group in groups.values():
        group.hintless_traces.sort(key=lambda t: str(t.trace_id))
        group.hinted_traces.sort(key=lambda t: str(t.trace_id))
        group.misleading_traces.sort(key=lambda t: str(t.trace_id))
    # check prompt-DB AFTER sort so the anchor choice is deterministic regardless of input order
    if prompt_db is not None:
        for group in groups.values():
            need_helpful = HintType.helpful in eval_hint_types and not group.hinted_traces
            need_misleading = HintType.misleading in eval_hint_types and not group.misleading_traces
            if not need_helpful and not need_misleading:
                continue
            # iterate sorted hintless traces; prefer a trace that covers ALL needed hint types
            for trace in group.hintless_traces:
                trace_code_types = pyine.organisms.datamodules.samples.common.get_code_type_set_from_str(
                    trace.trace_id.augment_category
                )
                trace_type_set = pyine.organisms.datamodules.samples.common.SampleCodeTypeSet(trace_code_types)
                if not trace_type_set.can_receive_any_hints():
                    continue
                has_helpful = need_helpful and check_prompt_db_fn(trace.trace_id, prompt_db, HintType.helpful)
                has_misleading = need_misleading and check_prompt_db_fn(trace.trace_id, prompt_db, HintType.misleading)
                if not has_helpful and not has_misleading:
                    continue
                if group.prompt_db_anchor is None:
                    # first viable trace -- use as anchor
                    group.prompt_db_anchor = trace
                    group.prompt_db_has_helpful = has_helpful
                    group.prompt_db_has_misleading = has_misleading
                elif (has_helpful and has_misleading) and not (
                    group.prompt_db_has_helpful and group.prompt_db_has_misleading
                ):
                    # upgrade: this trace covers both hint types, previous anchor only covered one
                    group.prompt_db_anchor = trace
                    group.prompt_db_has_helpful = has_helpful
                    group.prompt_db_has_misleading = has_misleading
                all_covered = (not need_helpful or group.prompt_db_has_helpful) and (
                    not need_misleading or group.prompt_db_has_misleading
                )
                if all_covered:
                    break  # anchor covers all needed hint types
    return groups


def _filter_to_complete_groups(
    groups: dict[tuple[str, str | tuple[str, ...]], CounterfactualGroup],
    eval_hint_types: tuple[HintType, ...],
) -> list[CounterfactualGroup]:
    """Filter groups to only those that can produce all required subsets.

    Args:
        groups: Dict of groups keyed by (family_id, base_augment_key).
        eval_hint_types: The hint types configured for evaluation.

    Returns:
        List of complete groups.
    """
    complete_groups: list[CounterfactualGroup] = []
    incomplete_count = 0
    no_hintless_count = 0
    for group in groups.values():
        if not group.hintless_traces:
            no_hintless_count += 1
            continue
        if group.is_complete(eval_hint_types):
            complete_groups.append(group)
        else:
            incomplete_count += 1
    if no_hintless_count > 0:
        logger.warning(
            f"{no_hintless_count} groups excluded from counterfactual analysis due to missing hintless traces"
        )
    if incomplete_count > 0:
        logger.debug(f"{incomplete_count} groups excluded due to incomplete hint coverage")
    return complete_groups


def _cap_complete_groups(
    complete_groups: list[CounterfactualGroup],
    caps_config: pyine.organisms.datamodules.samples.configs.TraceFilteringConfig,
) -> list[CounterfactualGroup]:
    """Apply cap filters to complete counterfactual groups by reusing filter_traces().

    For each group, picks a representative trace and passes all representatives through
    filter_traces() with a caps-only config. Surviving representatives are mapped back
    to their groups.

    In counterfactual mode, caps operate at the GROUP level:
    - max_traces_per_family: limits base-type variants per family (multiple groups from
      the same family produce multiple reps with the same augmentless ID);
    - max_traces_per_solution: limits counterfactual groups per solution (round-robin);
    - max_traces_per_problem: limits counterfactual groups per problem (round-robin);
    - max_trace_families: limits total distinct families (round-robin across problems).

    The caps_config's own seed (via get_rng(epoch=0)) provides determinism.

    Args:
        complete_groups: Groups that passed completeness checks.
        caps_config: Filtering config with only cap fields active.

    Returns:
        Subset of complete_groups that survived cap filtering.
    """
    if not caps_config.has_cap_filters:
        return complete_groups
    # sort for deterministic representative building (filter_traces uses dict insertion
    # order for round-robin, so input order matters for reproducibility)
    sorted_groups = sorted(
        complete_groups,
        key=lambda group: (group.family_id, str(group.base_augment_key)),
    )
    # build representative -> group mapping
    representative_to_group_key: dict[str, tuple[str, str | tuple[str, ...]]] = {}
    representatives: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
    for group in sorted_groups:
        # multi-augment base keys should not appear here (config validator rejects them
        # for counterfactual mode); treat as a data integrity error
        try:
            _normalize_base_key(group.base_augment_key)
        except NotImplementedError as e:
            raise ValueError(
                f"unexpected multi-augment base key {group.base_augment_key!r} in "
                f"counterfactual group {group.family_id!r}; this should have been "
                f"rejected by config validation"
            ) from e
        # complete groups must have hintless traces (enforced by is_complete())
        if not group.hintless_traces:
            raise ValueError(
                f"complete group {group.family_id!r} has no hintless traces; this violates the completeness invariant"
            )
        # prefer prompt_db_anchor as representative when present, since that's the trace
        # actually used across derived subsets; otherwise fall back to first hintless trace
        rep = group.prompt_db_anchor if group.prompt_db_anchor is not None else group.hintless_traces[0]
        group_key = (group.family_id, group.base_augment_key)
        if rep.identifier in representative_to_group_key:
            raise ValueError(
                f"duplicate representative identifier {rep.identifier} for groups "
                f"{representative_to_group_key[rep.identifier]} and {group_key}"
            )
        representative_to_group_key[rep.identifier] = group_key
        representatives.append(rep)
    # run filter_traces with caps-only config (quality steps are all no-ops)
    results = pyine.organisms.datamodules.samples.filtering.filter_traces(
        traces=representatives,
        epoch=0,
        filtering_config=caps_config,
    )
    surviving_keys: set[tuple[str, str | tuple[str, ...]]] = {
        representative_to_group_key[trace.identifier] for trace in results.kept_traces
    }
    capped = [group for group in complete_groups if (group.family_id, group.base_augment_key) in surviving_keys]
    logger.info(
        f"capped complete groups: {len(complete_groups)} -> {len(capped)} "
        f"({len(complete_groups) - len(capped)} removed by cap filters)"
    )
    return capped


def build_counterfactual_eval_subsets(
    traces: list[pyine.data.traces.dataset_utils.TraceMetadata],
    eval_hint_types: tuple[HintType, ...],
    code_type_prob_map: dict[str, float],
    rng: np.random.Generator,
    prompt_db: pyine.prompts.PromptResultDB | None,
    check_prompt_db_fn: typing.Callable[
        [pyine.data.traces.dataset_utils.TraceIdentifier, pyine.prompts.PromptResultDB, HintType], bool
    ],
    caps_filtering_config: pyine.organisms.datamodules.samples.configs.TraceFilteringConfig | None = None,
    check_lmdb_misleading_fn: typing.Callable[[pyine.data.traces.dataset_utils.TraceIdentifier], None] | None = None,
) -> dict[str, list[pyine.data.traces.dataset_utils.TraceMetadata]]:
    """Build counterfactual evaluation subsets with distribution control.

    FLOW (parent-driven selection):
    1. group traces by (family_id, base_augment_key);
    2. filter to COMPLETE groups (i.e. ones that can produce all required subsets);
    2.5. apply group-level caps if configured (see ``_cap_complete_groups``);
    3. sample groups according to parent's code_type_prob_map;
    4. for each selected group, add one trace to each derived subset.

    NOTE: The code_type_prob_map comes from the PARENT eval subset (e.g., "valid"). Derived subsets
    (_hintless, _hinted, _misleading) inherit this selection.

    Args:
        traces: List of traces from the parent eval subset.
        eval_hint_types: The hint types configured for evaluation.
        code_type_prob_map: Parent subset's code type distribution.
        rng: Seeded random generator.
        prompt_db: Optional prompt result database.
        check_prompt_db_fn: Function to check prompt-DB hint availability.
        caps_filtering_config: Optional caps-only filtering config to apply group-level caps
            between completeness filtering and distribution sampling.
        check_lmdb_misleading_fn: Optional callback for LMDB misleading trace validation guard.

    Returns:
        Dict with keys "hinted", "misleading", "hintless" mapping to trace lists.
    """
    # step 1: build groups
    groups = _build_counterfactual_groups(
        traces,
        prompt_db,
        eval_hint_types,
        check_prompt_db_fn,
        check_lmdb_misleading_fn=check_lmdb_misleading_fn,
    )
    # step 2: filter to complete groups
    complete_groups = _filter_to_complete_groups(groups, eval_hint_types)
    if not complete_groups:
        logger.warning("no complete counterfactual groups found; returning empty eval subsets")
        return {"hinted": [], "misleading": [], "hintless": []}
    # step 2.5: apply caps to complete groups
    if caps_filtering_config is not None:
        complete_groups = _cap_complete_groups(complete_groups, caps_filtering_config)
        if not complete_groups:
            logger.warning("no complete groups survived cap filtering; returning empty eval subsets")
            return {"hinted": [], "misleading": [], "hintless": []}
    # step 3: sample groups by distribution
    selected_groups = _sample_groups_by_distribution(complete_groups, code_type_prob_map, rng)
    # step 4: generate subsets from selected groups
    subsets: dict[str, list[pyine.data.traces.dataset_utils.TraceMetadata]] = {
        "hinted": [],
        "misleading": [],
        "hintless": [],
    }
    for group in selected_groups:
        # each group contributes ONE trace to each configured subset
        # (traces are pre-sorted in _build_counterfactual_groups)
        # KEY INVARIANT: when prompt-DB is used, the SAME anchor trace appears in
        # hintless AND all prompt-DB hint subsets ("same trace across subsets").
        # LMDB hinted/misleading traces are inherently different physical traces,
        # so independent RNG selection is fine for those.
        uses_prompt_db = group.prompt_db_anchor is not None and (
            (group.prompt_db_has_helpful and not group.hinted_traces)
            or (group.prompt_db_has_misleading and not group.misleading_traces)
        )
        if uses_prompt_db:
            # anchor trace appears in hintless + all prompt-DB hint subsets
            subsets["hintless"].append(group.prompt_db_anchor)  # type: ignore[arg-type]
        elif group.hintless_traces:
            hintless_idx = int(rng.choice(len(group.hintless_traces)))
            subsets["hintless"].append(group.hintless_traces[hintless_idx])
        if HintType.helpful in eval_hint_types:
            if group.hinted_traces:
                hinted_idx = int(rng.choice(len(group.hinted_traces)))
                subsets["hinted"].append(group.hinted_traces[hinted_idx])
            elif group.prompt_db_has_helpful and group.prompt_db_anchor is not None:
                subsets["hinted"].append(group.prompt_db_anchor)
        if HintType.misleading in eval_hint_types:
            if group.misleading_traces:
                misleading_idx = int(rng.choice(len(group.misleading_traces)))
                subsets["misleading"].append(group.misleading_traces[misleading_idx])
            elif group.prompt_db_has_misleading and group.prompt_db_anchor is not None:
                subsets["misleading"].append(group.prompt_db_anchor)
    # log partition sizes and cross-partition overlap counts
    hintless_ids = {str(t.trace_id) for t in subsets["hintless"]}
    hinted_ids = {str(t.trace_id) for t in subsets["hinted"]}
    misleading_ids = {str(t.trace_id) for t in subsets["misleading"]}
    overlap_hintless_hinted = len(hintless_ids & hinted_ids)
    overlap_hintless_misleading = len(hintless_ids & misleading_ids)
    logger.debug(
        f"counterfactual subsets: {len(subsets['hintless'])} hintless, "
        f"{len(subsets['hinted'])} hinted, {len(subsets['misleading'])} misleading;\n\t"
        f"cross-partition overlaps: hintless&hinted={overlap_hintless_hinted}, "
        f"hintless&misleading={overlap_hintless_misleading}"
    )
    return subsets


class ShortcutBiasDataModule(
    pyine.organisms.datamodules.base.BiasDataModuleBase[ShortcutBiasDataModuleConfig],
):
    """DataModule wrapping one or multiple PyINE code trace datasets for shortcut-bias experiments.

    This module loads one or more LMDB trace datasets, optionally filters available traces
    using flexible rules, validates that there are no duplicate trace identifiers across
    all selected samples, and finally creates simple random train/valid/test splits and loaders.

    The module supports two evaluation strategies for measuring shortcut bias:

    - ``hint_presence_split``: Partitions ALL eval traces by hint presence into disjoint
      subsets. Base-type composition of derived subsets reflects the natural LMDB distribution
      (no ``code_type_prob_map`` filtering). Best for broad aggregate comparisons.
    - ``counterfactual``: Groups traces by (family, base augment), samples groups according
      to the parent's ``code_type_prob_map``, and produces paired subsets where the same
      trace family appears with and without hints. Best for controlled comparisons.
    """

    _validated_misleading_uids: frozenset[str] | None = None
    """Pre-computed set of annotation record UIDs validated as truly misleading.
    None when require_validated_misleading=False or before metadata prep.
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
        hint_types_str = ", ".join(hint_type.value for hint_type in self.config.eval_hint_types)
        logger.info(
            f"shortcuts datamodule setup complete:"
            f"\n\tevaluation_strategy={self.config.evaluation_strategy.value} + hint_types=[{hint_types_str}]"
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
        # pre-compute validated-misleading UIDs if validation filtering is active
        self._validated_misleading_uids: frozenset[str] | None = None
        if self.config.require_validated_misleading and prompt_db is not None:
            validation_records = prompt_db.get_by_prompt_name(
                pyine.prompts.names.PromptNames.VALIDATION_MISLEADING,
                tag_filter_rule="+verdict:misleading",
            )
            # note: rec.identifier here is the validation record's identifier, which the
            # trace_annot_validator sets to the source annotation record's record_uid
            # (see validator.py:176). This is the join key used to match annotation records
            # against their validation verdicts downstream.
            self._validated_misleading_uids = frozenset(rec.identifier for rec in validation_records)
            if not self._validated_misleading_uids:
                raise ValueError(
                    "require_validated_misleading=True but no validation records with "
                    "verdict:misleading found in the prompt result DB. Run the trace_annot_validator "
                    "first, or set require_validated_misleading=False."
                )
            logger.info(f"loaded {len(self._validated_misleading_uids)} validated-misleading record UIDs for filtering")
        elif self.config.require_validated_misleading and prompt_db is None:
            raise ValueError(
                "require_validated_misleading=True but prompt_db is None. "
                "Prompt-DB access is required for validation filtering."
            )
        # create derived subsets for hint-based evaluation
        derived_subsets, filtered_parent_counts = self._create_hint_split_derived_subsets(subset_traces_meta, prompt_db)
        self._validate_sample_counts(subset_traces_meta, derived_subsets, filtered_parent_counts)
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
        hint_type: HintType | None = None,
    ) -> bool:
        """Checks if the prompt result DB has hinted code for this trace.

        This method verifies that valid hinted code is actually available in the prompt DB,
        not just that an entry exists. This ensures consistency between trace partitioning
        (which uses this check) and sample generation (which requires matching code types).

        The method accounts for the trace's existing (non-hint) augments. For example, if a trace
        has `obfuscated` augment, we look for `{obfuscated, hinted}` code type, not just `{hinted}`.

        Args:
            trace_id: The trace identifier to check.
            prompt_db: The prompt result database to query.
            hint_type: The specific hint type to check for. If None, uses the first
                configured hint type (for backward compatibility).
        """
        # use provided hint_type or fall back to first configured type
        target_hint_type = (
            hint_type
            if hint_type is not None
            else (self.config.eval_hint_types[0] if self.config.eval_hint_types else HintType.helpful)
        )
        # determine which prompt names and hint code type to check based on hint type
        if target_hint_type == HintType.helpful:
            hint_prompt_names: list[pyine.prompts.PromptNameType] = [
                pyine.prompts.names.PromptNames.HINTS_DOCS,
                pyine.prompts.names.PromptNames.HINTS_TESTS,
            ]
            hint_code_type = pyine.organisms.datamodules.samples.common.SampleCodeType.hinted
        else:  # misleading hints
            hint_prompt_names = [
                pyine.prompts.names.PromptNames.ISSUES_DOCS,
            ]
            hint_code_type = pyine.organisms.datamodules.samples.common.SampleCodeType.misleading
        # get the trace's current code type (may include non-hint augments like obfuscated, stubbed)
        # use get_code_type_set_from_str to handle the augment_category string directly
        trace_code_types = pyine.organisms.datamodules.samples.common.get_code_type_set_from_str(
            trace_id.augment_category
        )
        if pyine.organisms.datamodules.samples.common.SampleCodeType.stubbed in trace_code_types:
            return False  # stubbed traces cannot receive hints
        is_original = trace_code_types == frozenset(
            {pyine.organisms.datamodules.samples.common.SampleCodeType.original}
        )
        # build the target code type by combining trace's base augments with the hint type
        # e.g., if trace is {obfuscated}, target is {obfuscated, hinted}
        # e.g., if trace is {original}, target is {hinted}
        if is_original:
            target_types = frozenset({hint_code_type})
        else:
            target_types = trace_code_types | {hint_code_type}
        target_code_type = pyine.organisms.datamodules.samples.common.SampleCodeTypeSet(target_types)
        # fetch actual records and verify they have the target code type in their tags
        # this ensures the check is consistent with sample generation behavior
        for prompt_name in hint_prompt_names:
            records = prompt_db.get_by_identifier(
                identifier=str(trace_id),
                prompt_name=prompt_name,
            )
            for record in records:
                record_code_type = pyine.organisms.datamodules.samples.common.SampleCodeTypeSet.create_from_tags(
                    record.tags
                )
                if record_code_type == target_code_type:
                    # when validated-misleading filtering is active, skip unvalidated records
                    if (
                        target_hint_type == HintType.misleading
                        and self._validated_misleading_uids is not None
                        and record.record_uid not in self._validated_misleading_uids
                    ):
                        continue
                    return True
        return False

    def _check_lmdb_misleading_with_validation(
        self,
        trace_id: pyine.data.traces.dataset_utils.TraceIdentifier,
    ) -> None:
        """Raises NotImplementedError if an LMDB misleading trace is encountered with validation active."""
        if self.config.require_validated_misleading:
            # TODO: check LMDB trace dataset entry tags, those should contain validation status
            #       (will need to update tracing app to include validation status directly)
            raise NotImplementedError(
                f"LMDB misleading trace {trace_id} encountered with "
                f"require_validated_misleading=True. Validation filtering for LMDB-native "
                f"misleading traces is not yet implemented; only prompt-DB-sourced misleading "
                f"hints can currently be filtered by validation status."
            )

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

    def _resolve_parent_code_type_prob_map(
        self,
        eval_subset_name: str | None,
    ) -> dict[str, float] | None:
        """Resolve code_type_prob_map from parent eval subset config.

        Checks (in order):
        1. parent eval subset's dataparser_config_overrides; or
        2. default_dataparser_config params.

        Returns None if no prob map is found in either location.
        """
        # first: check explicit overrides for the parent eval subset
        if eval_subset_name is not None and hasattr(self.config, "dataparser_config_overrides"):
            parent_override = self.config.dataparser_config_overrides.get(eval_subset_name, {})
            selection_config = parent_override.get("selection_config", {})
            parent_prob_map = selection_config.get("code_type_prob_map")
            if parent_prob_map is not None:
                return {str(key): float(val) for key, val in parent_prob_map.items()}
        # second: check default_dataparser_config (same pattern as _should_use_prompt_db_for_hints)
        if not hasattr(self.config, "default_dataparser_config"):
            return None
        parser_config = self.config.default_dataparser_config
        if not hasattr(parser_config, "params"):
            return None
        params = parser_config.params
        if isinstance(params, pydantic.BaseModel):
            selection_config_obj = getattr(params, "selection_config", None)
            if selection_config_obj is not None:
                prob_map: typing.Any = getattr(selection_config_obj, "code_type_prob_map", None)
                if prob_map is not None:
                    return {str(key): float(val) for key, val in prob_map.items()}
        else:
            assert isinstance(params, dict)
            params = typing.cast("collections.abc.Mapping[str, typing.Any]", params)
            sel_cfg = params.get("selection_config", {})
            assert isinstance(sel_cfg, dict)
            sel_cfg = typing.cast("collections.abc.Mapping[str, typing.Any]", sel_cfg)
            prob_map = sel_cfg.get("code_type_prob_map")
            if prob_map is not None:
                return {str(key): float(val) for key, val in prob_map.items()}
        return None

    def _has_lmdb_hint(
        self,
        trace_id: pyine.data.traces.dataset_utils.TraceIdentifier,
        hint_type: HintType | None = None,
    ) -> bool:
        """Check if trace has the target hint type based on LMDB augmentation only.

        Args:
            trace_id: The trace identifier to check.
            hint_type: The specific hint type to check for. If None, checks for ANY
                hint type that's configured in eval_hint_types.

        Returns:
            True if the trace has the specified (or any configured) hint type in LMDB.
        """
        if hint_type is not None:
            if hint_type == HintType.helpful:
                return trace_id.is_hinted
            return trace_id.is_misleading
        # check for any configured hint type
        for hint_type in self.config.eval_hint_types:
            if hint_type == HintType.helpful and trace_id.is_hinted:
                return True
            if hint_type == HintType.misleading and trace_id.is_misleading:
                return True
        return False

    def _create_hint_split_derived_subsets(
        self,
        subset_traces_meta: dict[
            pyine.data.datamodule.SubsetNameType,
            list[pyine.data.traces.dataset_utils.TraceMetadata],
        ],
        prompt_db: pyine.prompts.PromptResultDB | None = None,
    ) -> tuple[
        dict[str, pyine.data.traces.dataset_utils.DerivedSubsetInfo[pyine.data.traces.dataset_utils.TraceMetadata]],
        dict[str, int],
    ]:
        """Create derived subsets for hint-based evaluation splits.

        Pre-filters traces using the parent's filtering config BEFORE partitioning into
        hint-based subsets. This ensures all derived subsets see the same filtered trace
        pool, preventing counterfactual pairing breakage from independent per-subset filtering.

        Derived eval subsets are fixed at ``prepare_data()`` time and are epoch-invariant
        (always use ``epoch=0`` for pre-filtering).

        Creates `_hinted`, `_misleading`, and `_hintless` derived subsets for each eval subset.

        Args:
            subset_traces_meta: Dict mapping primary subset names to their trace metadata lists.
            prompt_db: Optional prompt result database to check for hint availability.
                Used as fallback when LMDB trace augmentation is not present.

        Returns:
            Tuple of (derived_subsets dict, filtered_parent_counts dict). The
            filtered_parent_counts maps eval subset names to the number of traces
            remaining after pre-filtering (used by ``_validate_sample_counts``).
        """
        derived_subsets: dict[
            str, pyine.data.traces.dataset_utils.DerivedSubsetInfo[pyine.data.traces.dataset_utils.TraceMetadata]
        ] = {}
        filtered_parent_counts: dict[str, int] = {}
        derivation_type = self.config.evaluation_strategy.value
        for eval_subset_name in self.config.eval_subset_names:
            if eval_subset_name not in subset_traces_meta:
                continue
            traces = subset_traces_meta[eval_subset_name]
            # resolve parent's filtering config for pre-filtering
            if hasattr(self.config, "_resolve_dataparser_config"):
                parser_config = self.config._resolve_dataparser_config(eval_subset_name)  # type: ignore[reportPrivateUsage]
                params = parser_config.get_params_dict()
                filtering_dict = params.get("filtering_config", {})
                if filtering_dict is None:
                    # None means "use defaults" in builder convention, same as {}
                    parent_filtering = pyine.organisms.datamodules.samples.configs.TraceFilteringConfig()
                elif isinstance(filtering_dict, dict):
                    filtering_dict = typing.cast("dict[str, typing.Any]", filtering_dict)
                    parent_filtering = pyine.organisms.datamodules.samples.configs.TraceFilteringConfig(
                        **filtering_dict
                    )  # {} -> defaults (active filters), explicit values -> overrides
                else:
                    parent_filtering = filtering_dict  # already a TraceFilteringConfig
            else:
                parent_filtering = pyine.organisms.datamodules.samples.configs.TraceFilteringConfig.create_disabled()
            # eval pre-filtering must be deterministic; force seed=0 if None
            if parent_filtering.any_filtering_enabled and parent_filtering.seed is None:
                logger.warning(
                    f"parent filtering config for {eval_subset_name} has seed=None "
                    f"(nondeterministic); forcing seed=0 for eval pre-filtering"
                )
                parent_filtering = parent_filtering.model_copy(update={"seed": 0})
            # run pre-filtering: counterfactual mode uses two-phase (quality first, caps
            # after completeness check via _cap_complete_groups); hint_presence_split uses
            # single-phase (all filters applied together, no completeness requirement)
            if self.config.evaluation_strategy == EvaluationStrategy.counterfactual:
                quality_config = parent_filtering.create_quality_only()
                caps_config = parent_filtering.create_caps_only() if parent_filtering.has_cap_filters else None
                if quality_config.any_filtering_enabled:
                    filtering_results = pyine.organisms.datamodules.samples.filtering.filter_traces(
                        traces=traces,
                        epoch=0,
                        filtering_config=quality_config,
                    )
                    kept_ids = {t.identifier for t in filtering_results.kept_traces}
                    traces = [t for t in traces if t.identifier in kept_ids]
                    logger.info(
                        f"quality-filtered {eval_subset_name}: {filtering_results.orig_trace_count} -> "
                        f"{len(traces)} traces ({filtering_results.filtered_trace_count} removed)"
                    )
                filtered_parent_counts[eval_subset_name] = len(traces)
                partitions = self._partition_traces_by_hint_strategy(
                    traces,
                    prompt_db,
                    eval_subset_name=eval_subset_name,
                    caps_filtering_config=caps_config,
                )
            else:
                # hint_presence_split: single-phase filtering (existing behavior)
                if parent_filtering.any_filtering_enabled:
                    filtering_results = pyine.organisms.datamodules.samples.filtering.filter_traces(
                        traces=traces,
                        epoch=0,
                        filtering_config=parent_filtering,
                    )
                    kept_ids = {t.identifier for t in filtering_results.kept_traces}
                    traces = [t for t in traces if t.identifier in kept_ids]
                    logger.info(
                        f"pre-filtered {eval_subset_name}: {filtering_results.orig_trace_count} -> "
                        f"{len(traces)} traces ({filtering_results.filtered_trace_count} removed)"
                    )
                filtered_parent_counts[eval_subset_name] = len(traces)
                partitions = self._partition_traces_by_hint_strategy(
                    traces,
                    prompt_db,
                    eval_subset_name=eval_subset_name,
                )
            # counterfactual invariant: verify family alignment across configured partitions
            if self.config.evaluation_strategy == EvaluationStrategy.counterfactual:
                configured_partitions = ["hintless"]
                if HintType.helpful in self.config.eval_hint_types:
                    configured_partitions.append("hinted")
                if HintType.misleading in self.config.eval_hint_types:
                    configured_partitions.append("misleading")
                family_sets: dict[str, set[tuple[str, str | tuple[str, ...]]]] = {}
                for partition_name in configured_partitions:
                    family_sets[partition_name] = {
                        (
                            str(trace.trace_id.get_augmentless_identifier()),
                            pyine.organisms.datamodules.samples.common.SampleCodeTypeSet(
                                pyine.organisms.datamodules.samples.common.get_code_type_set_from_str(
                                    trace.trace_id.augment_category
                                )
                            ).get_counterfactual_grouping_key(),
                        )
                        for trace in partitions[partition_name]
                    }
                all_sets = list(family_sets.values())
                all_empty = all(len(s) == 0 for s in all_sets)
                all_equal = all(s == all_sets[0] for s in all_sets[1:])
                if not all_empty and not all_equal:
                    baseline = family_sets["hintless"]
                    diffs: dict[str, dict[str, typing.Any]] = {}
                    for name, families in family_sets.items():
                        missing = baseline - families
                        extra = families - baseline
                        if missing or extra:
                            diffs[name] = {
                                "missing_count": len(missing),
                                "extra_count": len(extra),
                                "missing_examples": sorted(str(k) for k in missing)[:5],
                                "extra_examples": sorted(str(k) for k in extra)[:5],
                            }
                    sizes = {name: len(families) for name, families in family_sets.items()}
                    raise ValueError(
                        f"counterfactual pairing broken for {eval_subset_name}: "
                        f"derived subsets have different family sets. Sizes={sizes}; Diffs={diffs}"
                    )
            # create derived subset for each hint type that was configured
            if HintType.helpful in self.config.eval_hint_types:
                derived_subsets[f"{eval_subset_name}_hinted"] = pyine.data.traces.dataset_utils.DerivedSubsetInfo(
                    parent_subset=eval_subset_name,
                    traces=partitions["hinted"],
                    derivation_type=derivation_type,
                )
            if HintType.misleading in self.config.eval_hint_types:
                derived_subsets[f"{eval_subset_name}_misleading"] = pyine.data.traces.dataset_utils.DerivedSubsetInfo(
                    parent_subset=eval_subset_name,
                    traces=partitions["misleading"],
                    derivation_type=derivation_type,
                )
            # always create hintless subset for baseline comparison
            derived_subsets[f"{eval_subset_name}_hintless"] = pyine.data.traces.dataset_utils.DerivedSubsetInfo(
                parent_subset=eval_subset_name,
                traces=partitions["hintless"],
                derivation_type=derivation_type,
            )
            # log summary
            parts = [f"{len(partitions['hintless'])} hintless"]
            if HintType.helpful in self.config.eval_hint_types:
                parts.append(f"{len(partitions['hinted'])} hinted")
            if HintType.misleading in self.config.eval_hint_types:
                parts.append(f"{len(partitions['misleading'])} misleading")
            logger.info(
                f"created {self.config.evaluation_strategy.value} subsets for {eval_subset_name}: " + ", ".join(parts)
            )
        return derived_subsets, filtered_parent_counts

    def _partition_traces_by_hint_strategy(
        self,
        traces: list[pyine.data.traces.dataset_utils.TraceMetadata],
        prompt_db: pyine.prompts.PromptResultDB | None = None,
        eval_subset_name: str | None = None,
        caps_filtering_config: pyine.organisms.datamodules.samples.configs.TraceFilteringConfig | None = None,
    ) -> dict[str, list[pyine.data.traces.dataset_utils.TraceMetadata]]:
        """Partition traces into hinted, misleading, and hintless lists based on evaluation strategy.

        For both strategies, the partitioning creates three categories:
        - ``hinted``: traces that have helpful hints (if configured);
        - ``misleading``: traces that have misleading hints (if configured);
        - ``hintless``: traces that have no hints whatsoever.

        The difference between strategies:

        - ``hint_presence_split``: ALL traces are included, partitioned purely by hint
          presence. No base-type distribution control is applied -- the composition of each
          partition reflects the natural LMDB distribution. Downstream per-family selection
          (via ``require_hint_type`` or ``skip_code_type_selection``) picks one trace per family
          without base-type filtering.
        - ``counterfactual``: only traces that can form complete (family, base_augment)
          groups are included. The parent's ``code_type_prob_map`` controls the base-type
          distribution of selected groups. Each family contributes at most one group
          (deduplicated across base-type buckets).

        For traces with hints in the prompt DB (but no LMDB hint), the same trace can appear
        in multiple partitions since it can be rendered with or without hints at sample time.

        INVALID STATE DETECTION:
            Raises ValueError if a trace has both is_hinted=True AND is_misleading=True.

        Args:
            traces: List of traces to partition.
            prompt_db: Optional prompt result database to check for hint availability.
                Used as fallback when LMDB trace augmentation is not present.
            eval_subset_name: The parent eval subset name (e.g., "valid"). Used in counterfactual
                mode to read the parent's code_type_prob_map.
            caps_filtering_config: Optional caps-only filtering config forwarded to
                ``build_counterfactual_eval_subsets`` for group-level cap filtering.

        Returns:
            Dict with keys "hinted", "misleading", "hintless" mapping to trace lists.
        """
        want_helpful = HintType.helpful in self.config.eval_hint_types
        want_misleading = HintType.misleading in self.config.eval_hint_types
        if self.config.evaluation_strategy == EvaluationStrategy.counterfactual:
            # counterfactual mode: use the group-first architecture with distribution control
            rng = np.random.default_rng(self.config.split_seed)
            # read code_type_prob_map: check overrides first, then default_dataparser_config
            code_type_prob_map: dict[str, float] = {"original": 1.0}  # ultimate fallback
            resolved_prob_map = self._resolve_parent_code_type_prob_map(eval_subset_name)
            if resolved_prob_map is not None:
                code_type_prob_map = resolved_prob_map
            lmdb_misleading_fn = (
                self._check_lmdb_misleading_with_validation if self.config.require_validated_misleading else None
            )
            return build_counterfactual_eval_subsets(
                traces=traces,
                eval_hint_types=self.config.eval_hint_types,
                code_type_prob_map=code_type_prob_map,
                rng=rng,
                prompt_db=prompt_db,
                check_prompt_db_fn=self._check_prompt_db_for_hint,
                caps_filtering_config=caps_filtering_config,
                check_lmdb_misleading_fn=lmdb_misleading_fn,
            )
        # hint_presence_split: simple partition based on whether trace has each hint type
        partitions: dict[str, list[pyine.data.traces.dataset_utils.TraceMetadata]] = {
            "hinted": [],
            "misleading": [],
            "hintless": [],
        }
        excluded_due_to_non_configured_hint = 0
        for trace in traces:
            has_lmdb_helpful = trace.trace_id.is_hinted
            has_lmdb_misleading = trace.trace_id.is_misleading
            has_any_lmdb_hint = has_lmdb_helpful or has_lmdb_misleading
            # detect invalid LMDB state
            if has_lmdb_helpful and has_lmdb_misleading:
                raise ValueError(
                    f"Invalid LMDB state: trace {trace.trace_id} has both is_hinted=True and "
                    f"is_misleading=True. This indicates corrupted data."
                )
            # exclude stubbed traces from all partitions
            code_type_set = pyine.organisms.datamodules.samples.common.SampleCodeTypeSet(
                pyine.organisms.datamodules.samples.common.get_code_type_set_from_str(trace.trace_id.augment_category)
            )
            if code_type_set.is_stubbed:
                continue
            # partition by LMDB hint type (no cross-hint: each trace goes to at most one hint partition)
            if want_helpful and has_lmdb_helpful:
                partitions["hinted"].append(trace)
            elif want_misleading and has_lmdb_misleading:
                self._check_lmdb_misleading_with_validation(trace.trace_id)
                partitions["misleading"].append(trace)
            elif has_any_lmdb_hint:
                # has non-configured LMDB hint -- exclude from all partitions
                excluded_due_to_non_configured_hint += 1
            else:
                # truly hintless LMDB trace -- can go to hintless AND potentially get hints via prompt-DB
                partitions["hintless"].append(trace)
                # check prompt-DB for hint availability (only for hintless LMDB traces)
                if prompt_db is not None:
                    if want_helpful and self._check_prompt_db_for_hint(trace.trace_id, prompt_db, HintType.helpful):
                        partitions["hinted"].append(trace)
                    if want_misleading and self._check_prompt_db_for_hint(
                        trace.trace_id, prompt_db, HintType.misleading
                    ):
                        partitions["misleading"].append(trace)
        # log exclusion summary (debug level for internal bookkeeping)
        if excluded_due_to_non_configured_hint > 0:
            logger.debug(
                f"excluded {excluded_due_to_non_configured_hint} traces with non-configured LMDB hints "
                f"from all partitions (no cross-hint allowed; configured: {self.config.eval_hint_types})"
            )
        return partitions

    def _validate_sample_counts(
        self,
        subset_traces_meta: dict[
            pyine.data.datamodule.SubsetNameType,
            list[pyine.data.traces.dataset_utils.TraceMetadata],
        ],
        derived_subsets: dict[
            str, pyine.data.traces.dataset_utils.DerivedSubsetInfo[pyine.data.traces.dataset_utils.TraceMetadata]
        ],
        filtered_parent_counts: dict[str, int] | None = None,
    ) -> None:
        """Validate that derived subsets have sufficient samples for evaluation.

        Args:
            subset_traces_meta: Dict mapping primary subset names to their trace metadata lists.
            derived_subsets: Dict mapping derived subset names to DerivedSubsetInfo objects.
            filtered_parent_counts: Optional dict mapping eval subset names to the number of
                traces remaining after pre-filtering. Used for the "< 0.5 * parent" warning
                instead of the raw parent trace count, since pre-filtering reduces the
                effective parent size before partitioning.

        Raises:
            ValueError: If any derived subset has fewer samples than the configured minimum.
        """
        for eval_subset_name in self.config.eval_subset_names:
            if eval_subset_name not in subset_traces_meta:
                continue
            parent_trace_count = (
                filtered_parent_counts.get(eval_subset_name, len(subset_traces_meta[eval_subset_name]))
                if filtered_parent_counts is not None
                else len(subset_traces_meta[eval_subset_name])
            )
            hinted_key = f"{eval_subset_name}_hinted"
            misleading_key = f"{eval_subset_name}_misleading"
            hintless_key = f"{eval_subset_name}_hintless"
            hinted_info = derived_subsets.get(hinted_key)
            misleading_info = derived_subsets.get(misleading_key)
            hintless_info = derived_subsets.get(hintless_key)
            hinted_count = len(hinted_info.traces) if hinted_info else 0
            misleading_count = len(misleading_info.traces) if misleading_info else 0
            hintless_count = len(hintless_info.traces) if hintless_info else 0
            # check minimum counts (hard error)
            if hinted_count < self.config.min_samples_hinted:
                raise ValueError(
                    f"eval subset '{hinted_key}' has only {hinted_count} samples "
                    f"(minimum required: {self.config.min_samples_hinted})"
                )
            if misleading_count < self.config.min_samples_misleading:
                raise ValueError(
                    f"eval subset '{misleading_key}' has only {misleading_count} samples "
                    f"(minimum required: {self.config.min_samples_misleading})"
                )
            if hintless_count < self.config.min_samples_hintless:
                raise ValueError(
                    f"eval subset '{hintless_key}' has only {hintless_count} samples "
                    f"(minimum required: {self.config.min_samples_hintless})"
                )
            # warn if derived subset counts are significantly lower than parent (soft warning)
            derived_total = hinted_count + misleading_count + hintless_count
            if parent_trace_count > 0 and derived_total < 0.5 * parent_trace_count:
                logger.warning(
                    f"derived subsets for '{eval_subset_name}' have only {derived_total} traces "
                    f"(parent has {parent_trace_count}); this may indicate aggressive filtering "
                    "or missing hint pairs for counterfactual evaluation"
                )

    def _get_overlapping_trace_ids(
        self,
        eval_subset_name: str,
        partition_suffix: str,
    ) -> frozenset[str]:
        """Get trace IDs that appear in multiple derived subsets.

        These are traces with prompt DB hints (but no LMDB hint augmentation) that will be
        rendered differently depending on which derived subset is accessed. Sample identifiers
        for these traces need modification to ensure uniqueness.

        Args:
            eval_subset_name: Base eval subset name (e.g., "valid").
            partition_suffix: The partition suffix (e.g., "hinted", "misleading", "hintless").

        Returns:
            Frozenset of trace identifiers that appear in multiple derived subsets.
        """
        assert self._metadata is not None
        current_key = f"{eval_subset_name}_{partition_suffix}"
        if current_key not in self._metadata.derived_subsets:
            return frozenset()
        current_ids = {t.identifier for t in self._metadata.derived_subsets[current_key].traces}
        # collect trace IDs from all OTHER derived subsets for this eval subset
        other_ids: set[str] = set()
        for suffix in ["hinted", "misleading", "hintless"]:
            key = f"{eval_subset_name}_{suffix}"
            if key != current_key and key in self._metadata.derived_subsets:
                other_ids |= {t.identifier for t in self._metadata.derived_subsets[key].traces}
        return frozenset(current_ids & other_ids)

    _HINT_SUFFIX_MAP: typing.ClassVar[dict[str, str]] = {
        "_hinted": "::hinted",
        "_misleading": "::misleading",
        "_hintless": "::hintless",
    }

    @typing.override
    def _resolve_pregenerated_outputs_for_subset(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> dict[str, pyine.organisms.datamodules.samples.common.PregeneratedOutputRecord]:
        """Filter and re-key pregenerated outputs based on the subset's hint suffix context.

        For wrapped subsets (e.g. train_hinted), selects entries with the matching ``::hinted``
        suffix and strips it so keys are base trace IDs (matching ``sample.identifier`` before
        ``SampleHintIdentifierWrapper`` adds the suffix). Non-suffixed entries (for non-overlapping
        traces) are included in all subsets.
        """
        if self._pregenerated_outputs is None:
            return {}
        all_hint_suffixes = set(self._HINT_SUFFIX_MAP.values())
        # determine target suffix for this subset
        target_suffix: str | None = None
        matched_subset_suffix: str | None = None
        matched_hint_name: str | None = None  # e.g. "hinted" (without ::)
        for subset_suffix, id_suffix in self._HINT_SUFFIX_MAP.items():
            if subset_name.endswith(subset_suffix):
                target_suffix = id_suffix
                matched_subset_suffix = subset_suffix
                matched_hint_name = id_suffix.lstrip(":")
                break
        resolved: dict[str, pyine.organisms.datamodules.samples.common.PregeneratedOutputRecord] = {}
        for sample_id, record in self._pregenerated_outputs.items():
            has_any_hint_suffix = any(sample_id.endswith(suffix) for suffix in all_hint_suffixes)
            has_any_cf_suffix = sample_id.endswith("::cf_with") or sample_id.endswith("::cf_without")
            if has_any_cf_suffix:
                raise ValueError(
                    f"pregenerated output key '{sample_id}' has a keyword counterfactual suffix "
                    "(::cf_with/::cf_without); these suffixes are not handled by "
                    "ShortcutBiasDataModule, they belong to KeywordBiasDataModule's "
                    "counterfactual evaluation mode"
                )
            if target_suffix is not None and sample_id.endswith(target_suffix):
                base_id = sample_id[: -len(target_suffix)]
                if base_id in resolved:
                    existing = resolved[base_id]
                    raise ValueError(
                        f"conflicting pregenerated outputs for base trace ID '{base_id}': "
                        f"new entry '{sample_id}' (source_key='{record.source_key}') conflicts "
                        f"with existing entry (source_key='{existing.source_key}'); "
                        f"this can happen if the LMDB contains both suffixed and unsuffixed "
                        f"entries for the same trace across different export runs"
                    )
                resolved[base_id] = record
            elif not has_any_hint_suffix:
                if "::" in sample_id:
                    raise ValueError(
                        f"pregenerated output key '{sample_id}' contains unrecognized '::' "
                        f"suffix; recognized suffixes for ShortcutBiasDataModule are: "
                        f"{sorted(all_hint_suffixes)}"
                    )
                if sample_id in resolved:
                    existing = resolved[sample_id]
                    raise ValueError(
                        f"conflicting pregenerated outputs for base trace ID '{sample_id}': "
                        f"new unsuffixed entry (source_key='{record.source_key}') conflicts "
                        f"with existing entry (source_key='{existing.source_key}')"
                    )
                resolved[sample_id] = record
            # else: different hint suffix -> skip (belongs to a different hint subset)
        # reject ambiguous non-suffixed entries for overlapping traces:
        # if a trace appears in multiple hint subsets, it MUST have a suffixed LMDB entry
        # matching this subset's target suffix; an unsuffixed entry is ambiguous
        if target_suffix is not None:
            assert matched_subset_suffix is not None and matched_hint_name is not None
            parent_subset = subset_name[: -len(matched_subset_suffix)]
            overlapping_ids = self._get_overlapping_trace_ids(parent_subset, matched_hint_name)
            for base_id in list(resolved):
                if base_id in overlapping_ids:
                    # for overlapping traces in resolved, require the target suffix entry specifically
                    if f"{base_id}{target_suffix}" not in self._pregenerated_outputs:
                        raise ValueError(
                            f"pregenerated output for overlapping trace '{base_id}' is missing "
                            f"the required suffix entry '{base_id}{target_suffix}'; "
                            f"overlapping traces must have per-suffix LMDB entries to "
                            f"disambiguate which variant was generated"
                        )
            # also check overlapping IDs that have entries in other suffixes but not this one
            for base_id in overlapping_ids:
                if base_id not in resolved:
                    has_other_suffix = any(
                        f"{base_id}{suffix}" in self._pregenerated_outputs
                        for suffix in all_hint_suffixes
                        if suffix != target_suffix
                    )
                    if has_other_suffix:
                        raise ValueError(
                            f"pregenerated output for overlapping trace '{base_id}' is missing "
                            f"the required suffix entry '{base_id}{target_suffix}'; "
                            f"overlapping traces must have per-suffix LMDB entries to "
                            f"disambiguate which variant was generated"
                        )
        return resolved

    @typing.override
    def _build_parser_kwargs(
        self,
        source_data: typing.Any,
        subset_traces: list[pyine.data.traces.dataset_utils.TraceMetadata],
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> dict[str, typing.Any]:
        """Extend base parser kwargs with validated misleading UIDs when available."""
        kwargs = super()._build_parser_kwargs(source_data, subset_traces, subset_name)
        if self._validated_misleading_uids is not None:
            kwargs["validated_misleading_record_uids"] = self._validated_misleading_uids
        return kwargs

    @typing.override
    def get_parser(
        self,
        subset_name: pyine.data.datamodule.SubsetNameType,
    ) -> pyine.organisms.datamodules.samples.SampleDataParser:
        """Returns a data parser, wrapped with identifier modification if needed.

        For derived hint subsets (`_hinted`, `_misleading`, `_hintless`), traces that appear in
        multiple subsets (prompt DB hint traces) have their sample identifiers modified with
        suffixes to ensure uniqueness:

        - `::hinted` suffix for samples from `_hinted` subset;
        - `::misleading` suffix for samples from `_misleading` subset;
        - `::hintless` suffix for samples from `_hintless` subset.

        This is necessary because the same trace can be rendered with different hints depending
        on the subset, and downstream consumers (caches, evaluators, data stores) require unique
        identifiers.

        Args:
            subset_name: Name of the subset to get parser for.

        Returns:
            Data parser with subset tagging, optionally wrapped with identifier modification.
        """
        base_parser = super().get_parser(subset_name)
        if not self._is_setup_complete():
            raise RuntimeError("data parsers are not ready yet, call `setup()` first")
        # check if this is a derived hints subset that needs identifier modification
        suffix_map = {
            "_hinted": "hinted",
            "_misleading": "misleading",
            "_hintless": "hintless",
        }
        for suffix, hint_suffix in suffix_map.items():
            if subset_name.endswith(suffix):
                parent_subset = subset_name[: -len(suffix)]
                overlapping_ids = self._get_overlapping_trace_ids(parent_subset, hint_suffix)
                if overlapping_ids:
                    return SampleHintIdentifierWrapper(
                        wrapped_dataset=base_parser,
                        overlapping_trace_ids=overlapping_ids,
                        hint_suffix=hint_suffix,
                    )  # type: ignore[return-value]
                break
        return base_parser


class SampleHintIdentifierWrapper:
    """Wrapper that modifies sample identifiers for traces appearing in multiple hint subsets.

    This wrapper ensures unique sample identifiers when the same trace appears in multiple
    derived subsets (`_hinted`, `_misleading`, `_hintless`). For overlapping traces, the
    identifier is modified by appending a suffix (e.g., `::hinted`, `::misleading`, `::hintless`).

    This is necessary because traces with prompt DB hints (but no LMDB hint augmentation) can
    be rendered with different hints depending on the derived subset accessed. Without identifier
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
            overlapping_trace_ids: Set of trace identifiers that appear in multiple hint subsets
                and need identifier modification.
            hint_suffix: Suffix to append to overlapping identifiers (e.g., "hinted").
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
