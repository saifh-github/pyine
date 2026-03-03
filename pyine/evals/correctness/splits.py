"""Standalone, modular split logic for guardrail evaluation.

This module is designed to be imported by both the eval pipeline and external guardrail
training pipelines. It does not depend on ``_impl`` or ``metrics``.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import typing

import numpy as np

import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.evals.correctness.types as correctness_types
import pyine.organisms.datamodules.samples.common as samples_common

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GuardrailSplits:
    """Problem-level splits for guardrail training, calibration, and evaluation.

    All 3 splits are exposed so that training pipelines can import and use them directly via the
    standalone splits module.
    """

    guardrail_train: list[correctness_types.EvalRecord]
    """Records for guardrail training (from original validation problems)."""
    guardrail_valid: list[correctness_types.EvalRecord]
    """Records for threshold calibration (from original validation problems)."""
    guardrail_test: list[correctness_types.EvalRecord]
    """Records for final evaluation (from original test problems)."""
    train_problem_ids: frozenset[str]
    """Coding problem IDs assigned to guardrail_train."""
    valid_problem_ids: frozenset[str]
    """Coding problem IDs assigned to guardrail_valid."""
    test_problem_ids: frozenset[str]
    """Coding problem IDs assigned to guardrail_test."""

    def to_summary(self) -> dict[str, typing.Any]:
        """Return a lightweight summary suitable for AggregatedResult persistence.

        Includes problem_ids (as sorted lists for JSON serializability) and record counts
        per split. Does NOT include full EvalRecord objects.
        """
        return {
            "train_problem_ids": sorted(self.train_problem_ids),
            "valid_problem_ids": sorted(self.valid_problem_ids),
            "test_problem_ids": sorted(self.test_problem_ids),
            "train_record_count": len(self.guardrail_train),
            "valid_record_count": len(self.guardrail_valid),
            "test_record_count": len(self.guardrail_test),
        }


def extract_problem_id(
    sample_id: str,
) -> str:
    """Extract the coding-problem-level ID from a trace-level sample_id.

    Steps:
    1. Strip any ``::`` suffix via ``strip_id_suffix``
    2. Parse with ``TraceIdentifier.from_string`` (strict parser)
    3. Derive the problem ID from the parsed trace
    4. Return the canonical problem ID string (e.g. ``'TACO/TRAIN/p000001'``)

    Falls back to ``CodingProblemIdentifier.from_string`` if the ID is already
    problem-level (logged as a warning).

    Args:
        sample_id: Trace-level or problem-level identifier string.

    Returns:
        Canonical problem ID string.

    Raises:
        ValueError: If the ID cannot be parsed as either a trace or problem identifier.
    """
    cleaned = samples_common.strip_id_suffix(sample_id)
    try:
        trace_id = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(cleaned)
        problem_id = pyine.data.traces.dataset_utils.CodingProblemIdentifier(
            dataset=trace_id.dataset,
            subset=trace_id.subset,
            problem_idx=trace_id.problem_idx,
        )
        return repr(problem_id)
    except ValueError:
        pass
    # fallback: try parsing as a problem-level ID directly
    logger.warning(f"sample_id '{sample_id}' is not a trace-level ID, trying problem-level fallback")
    problem_id = pyine.data.traces.dataset_utils.CodingProblemIdentifier.from_string(cleaned)
    return repr(problem_id)


def build_guardrail_splits(
    records: list[correctness_types.EvalRecord],
    split_config: correctness_types.GuardrailSplitConfig,
) -> GuardrailSplits:
    """Build guardrail train/valid/test splits from eval records and a dataset split.

    Public entry point. Loads the SplitResult, then delegates to internal logic.

    Args:
        records: All eval records loaded from LMDB.
        split_config: Configuration specifying split source and re-split parameters.

    Returns:
        GuardrailSplits with records assigned to train, valid, and test subsets.
    """
    split_result = pyine.data.utils.splits.get_dataset_split_result(split_config.split_source)
    return _assign_guardrail_subsets(
        records=records,
        split_result=split_result,
        guardrail_valid_fraction=split_config.guardrail_valid_fraction,
        seed=split_config.seed,
        stratify_by_label=split_config.stratify_by_label,
    )


def _assign_guardrail_subsets(
    records: list[correctness_types.EvalRecord],
    split_result: pyine.data.utils.splits.SplitResult,
    guardrail_valid_fraction: float,
    seed: int,
    stratify_by_label: bool,
) -> GuardrailSplits:
    """Assign records to guardrail train/valid/test based on original dataset splits.

    Original train problems are discarded. Original test problems go to guardrail_test.
    Original valid problems are re-split into guardrail_train and guardrail_valid.

    Args:
        records: All eval records loaded from LMDB.
        split_result: The original dataset split result.
        guardrail_valid_fraction: Fraction of original valid problems for guardrail_valid.
        seed: Random seed for the re-split.
        stratify_by_label: Whether to stratify the re-split by per-problem correctness rate.

    Returns:
        GuardrailSplits with records assigned to the three subsets.
    """
    subset_to_ids = split_result.get_subset_to_ids_map()
    # build problem_id -> original_subset lookup (rejecting duplicates across subsets)
    problem_to_original_subset: dict[str, str] = {}
    for subset_name, id_list in subset_to_ids.items():
        for identifier in id_list:
            if identifier in problem_to_original_subset:
                raise ValueError(
                    f"problem_id '{identifier}' appears in both '{problem_to_original_subset[identifier]}' "
                    f"and '{subset_name}' subsets in the split file; each problem must belong to exactly one subset"
                )
            problem_to_original_subset[identifier] = subset_name
    # group records by problem_id
    problem_to_records: dict[str, list[correctness_types.EvalRecord]] = collections.defaultdict(list)
    for record in records:
        problem_to_records[record.problem_id].append(record)
    # dataset consistency check: validate problem_id.dataset matches split source
    expected_dataset = split_result.source_dataset_name
    for problem_id in problem_to_records:
        parsed_pid = pyine.data.traces.dataset_utils.CodingProblemIdentifier.from_string(problem_id)
        if parsed_pid.dataset != expected_dataset:
            raise ValueError(
                f"problem_id '{problem_id}' has dataset '{parsed_pid.dataset}' but split file "
                f"has source_dataset_name='{expected_dataset}'; ensure LMDB records come from "
                f"the same dataset as the split file"
            )
        if problem_id not in problem_to_original_subset:
            raise ValueError(
                f"problem_id '{problem_id}' from LMDB records not found in SplitResult; "
                f"ensure LMDB records come from the same dataset as the split file "
                f"(split source_dataset_name='{expected_dataset}')"
            )
    # classify problems by original subset
    original_valid_problems: list[str] = []
    test_problem_ids: set[str] = set()
    discarded_problem_ids: set[str] = set()
    for problem_id in problem_to_records:
        original_subset = problem_to_original_subset[problem_id]
        if original_subset == "train":
            discarded_problem_ids.add(problem_id)
        elif original_subset == "test":
            test_problem_ids.add(problem_id)
        elif original_subset == "valid":
            original_valid_problems.append(problem_id)
        else:
            raise ValueError(
                f"problem_id '{problem_id}' has unknown original subset '{original_subset}'; "
                f"expected one of 'train', 'valid', 'test'. Check that the split file is valid."
            )
    # re-split original valid problems into guardrail_train and guardrail_valid
    guardrail_train_ids, guardrail_valid_ids = _resplit_valid_problems(
        valid_problem_ids=original_valid_problems,
        problem_to_records=problem_to_records,
        guardrail_valid_fraction=guardrail_valid_fraction,
        seed=seed,
        stratify_by_label=stratify_by_label,
    )
    # build record lists
    train_records: list[correctness_types.EvalRecord] = []
    for pid in guardrail_train_ids:
        train_records.extend(problem_to_records[pid])
    valid_records: list[correctness_types.EvalRecord] = []
    for pid in guardrail_valid_ids:
        valid_records.extend(problem_to_records[pid])
    test_records: list[correctness_types.EvalRecord] = []
    for pid in test_problem_ids:
        test_records.extend(problem_to_records[pid])
    # validate: no problem in multiple splits
    train_valid_overlap = guardrail_train_ids & guardrail_valid_ids
    train_test_overlap = guardrail_train_ids & test_problem_ids
    valid_test_overlap = guardrail_valid_ids & test_problem_ids
    if train_valid_overlap or train_test_overlap or valid_test_overlap:
        raise ValueError(
            f"problem IDs overlap across splits: "
            f"train&valid={train_valid_overlap}, train&test={train_test_overlap}, "
            f"valid&test={valid_test_overlap}"
        )
    # validate: all non-discarded records accounted for
    total_assigned = len(train_records) + len(valid_records) + len(test_records)
    total_discarded = sum(len(problem_to_records[pid]) for pid in discarded_problem_ids)
    if total_assigned + total_discarded != len(records):
        raise ValueError(
            f"record count mismatch: {total_assigned} assigned + {total_discarded} discarded != {len(records)} total"
        )
    return GuardrailSplits(
        guardrail_train=train_records,
        guardrail_valid=valid_records,
        guardrail_test=test_records,
        train_problem_ids=frozenset(guardrail_train_ids),
        valid_problem_ids=frozenset(guardrail_valid_ids),
        test_problem_ids=frozenset(test_problem_ids),
    )


def _resplit_valid_problems(
    valid_problem_ids: list[str],
    problem_to_records: dict[str, list[correctness_types.EvalRecord]],
    guardrail_valid_fraction: float,
    seed: int,
    stratify_by_label: bool,
) -> tuple[set[str], set[str]]:
    """Re-split original valid problems into guardrail_train and guardrail_valid.

    Args:
        valid_problem_ids: Problem IDs from the original validation subset.
        problem_to_records: Mapping from problem_id to records.
        guardrail_valid_fraction: Fraction assigned to guardrail_valid.
        seed: Random seed.
        stratify_by_label: Whether to stratify by per-problem correctness rate.

    Returns:
        Tuple of (guardrail_train_ids, guardrail_valid_ids) as sets.
    """
    rng = np.random.default_rng(seed)
    if not stratify_by_label:
        shuffled = list(valid_problem_ids)
        rng.shuffle(shuffled)
        split_point = round(len(shuffled) * guardrail_valid_fraction)
        guardrail_valid_ids = set(shuffled[:split_point])
        guardrail_train_ids = set(shuffled[split_point:])
        return guardrail_train_ids, guardrail_valid_ids
    # stratify by per-problem correctness rate: all_correct, all_incorrect, and mixed
    # problems split into tercile bins by positive rate to preserve the difficulty gradient
    all_correct_pids: list[str] = []
    all_incorrect_pids: list[str] = []
    mixed_pids_with_rate: list[tuple[str, float]] = []
    for pid in valid_problem_ids:
        pid_records = problem_to_records[pid]
        num_correct = sum(1 for rec in pid_records if rec.label)
        total = len(pid_records)
        if num_correct == total:
            all_correct_pids.append(pid)
        elif num_correct == 0:
            all_incorrect_pids.append(pid)
        else:
            mixed_pids_with_rate.append((pid, num_correct / total))
    # bin mixed problems by positive rate terciles (fall back to single bin if < 3)
    mixed_strata: list[list[str]] = []
    if len(mixed_pids_with_rate) < 3:
        mixed_strata = [[pid for pid, _rate in mixed_pids_with_rate]]
    else:
        mixed_pids_with_rate.sort(key=lambda pair: pair[1])
        bin_size = len(mixed_pids_with_rate) // 3
        mixed_strata = [
            [pid for pid, _rate in mixed_pids_with_rate[:bin_size]],
            [pid for pid, _rate in mixed_pids_with_rate[bin_size : 2 * bin_size]],
            [pid for pid, _rate in mixed_pids_with_rate[2 * bin_size :]],
        ]
    all_strata: list[list[str]] = [all_correct_pids, all_incorrect_pids, *mixed_strata]
    guardrail_train_ids: set[str] = set()
    guardrail_valid_ids: set[str] = set()
    for stratum_pids in all_strata:
        rng.shuffle(stratum_pids)
        split_point = round(len(stratum_pids) * guardrail_valid_fraction)
        guardrail_valid_ids.update(stratum_pids[:split_point])
        guardrail_train_ids.update(stratum_pids[split_point:])
    return guardrail_train_ids, guardrail_valid_ids
