"""Record resampling for calibration and training data composition control."""

from __future__ import annotations

import logging
import random
import typing

import pyine.evals.correctness.types as correctness_types  # noqa: TC001

logger = logging.getLogger(__name__)


def resample_records(
    records: list[correctness_types.EvalRecord],
    config: correctness_types.RecordResamplingConfig,
) -> list[correctness_types.EvalRecord]:
    """Resample eval records according to the given config.

    Steps applied in order:
    1. Code type filtering/rebalancing (if ``code_type_proportions`` is set);
    2. Label ratio adjustment (if ``target_positive_ratio`` is set);
    3. Size cap (if ``max_records`` is set);
    4. Validation (both label classes present with minimum counts).

    Args:
        records: Input eval records to resample.
        config: Resampling configuration controlling the composition.

    Returns:
        Resampled list of eval records.

    Raises:
        ValueError: If input is empty, a requested code type has no records, or the output has
            too few records per label class.
    """
    if not records:
        raise ValueError("cannot resample an empty record list")
    if config.is_noop:
        return list(records)
    rng = random.Random(config.seed)
    result = list(records)
    # step 1: code type filtering/rebalancing
    if config.code_type_proportions is not None:
        result = _apply_code_type_rebalancing(
            result,
            config.code_type_proportions,
            config.strategy,
            rng,
        )
    # step 2: label ratio adjustment
    if config.target_positive_ratio is not None:
        result = _apply_label_ratio(
            result,
            config.target_positive_ratio,
            config.strategy,
            rng,
        )
    # step 3: size cap
    if config.max_records is not None and len(result) > config.max_records:
        if config.max_records < 2 * config.min_records_per_label:
            raise ValueError("max_records must be at least 2 * min_records_per_label to keep both label classes")
        result = _apply_size_cap(result, config.max_records, config.min_records_per_label, rng)
    # step 4: validation
    num_positive = sum(1 for rec in result if rec.label)
    num_negative = len(result) - num_positive
    if num_positive < config.min_records_per_label or num_negative < config.min_records_per_label:
        raise ValueError(
            f"resampled result has {num_positive} positive and {num_negative} negative records, "
            f"but min_records_per_label={config.min_records_per_label} requires at least that many of each"
        )
    rng.shuffle(result)
    _log_summary(records, result)
    return result


def _validate_code_type_keys(
    proportions: dict[str, float],
) -> None:
    """Validate that all keys in code_type_proportions are recognized code types.

    Uses ``get_code_type_set_from_str`` to parse each key and detects unrecognized strings
    by checking for the silent fallback to ``{original}`` (same approach as
    ``correctness_metrics._parse_code_type_set``).

    Raises:
        ValueError: If any key is not a recognized code type string.
    """
    import pyine.organisms.datamodules.samples.common as samples_common

    invalid: list[str] = []
    for key in proportions:
        parsed = samples_common.get_code_type_set_from_str(key)
        if parsed == frozenset({samples_common.SampleCodeType.original}) and key != "original":
            invalid.append(key)
    if invalid:
        valid_strings = sorted({"original"} | set(samples_common.get_all_supported_code_type_sets_suffixes()))
        raise ValueError(
            f"code_type_proportions contains unrecognized code type(s): {invalid}; "
            f"valid code types are: {valid_strings}"
        )


def _apply_code_type_rebalancing(
    records: list[correctness_types.EvalRecord],
    proportions: dict[str, float],
    strategy: typing.Literal["subsample", "oversample"],
    rng: random.Random,
) -> list[correctness_types.EvalRecord]:
    """Filter and rebalance records by code_type proportions."""
    _validate_code_type_keys(proportions)
    # partition records by code_type
    by_code_type: dict[str, list[correctness_types.EvalRecord]] = {}
    for record in records:
        by_code_type.setdefault(record.code_type, []).append(record)
    # validate every key in proportions has at least one record
    missing = [key for key in proportions if key not in by_code_type or not by_code_type[key]]
    if missing:
        raise ValueError(f"code_type_proportions references code types with no records: {missing}")
    # normalize proportions
    total_weight = sum(proportions.values())
    normalized = {key: weight / total_weight for key, weight in proportions.items()}
    # compute anchor
    if strategy == "subsample":
        anchor = min(len(by_code_type[key]) / normalized[key] for key in normalized)
    else:
        anchor = max(len(by_code_type[key]) / normalized[key] for key in normalized)
    # compute target counts and resample
    result: list[correctness_types.EvalRecord] = []
    for key in sorted(normalized.keys()):
        target_count = round(anchor * normalized[key])
        if target_count < 1:
            raise ValueError(
                "code_type_proportions target count fell below 1; adjust proportions or use a larger dataset"
            )
        if strategy == "subsample" and target_count > len(by_code_type[key]):
            raise ValueError("code_type_proportions require oversampling to meet targets; use strategy='oversample'")
        resampled = _resample_group(by_code_type[key], target_count, rng)
        result.extend(resampled)
    return result


def _apply_label_ratio(
    records: list[correctness_types.EvalRecord],
    target_ratio: float,
    strategy: typing.Literal["subsample", "oversample"],
    rng: random.Random,
) -> list[correctness_types.EvalRecord]:
    """Adjust the positive/negative label ratio."""
    positives = [rec for rec in records if rec.label]
    negatives = [rec for rec in records if not rec.label]
    num_pos = len(positives)
    num_neg = len(negatives)
    if num_pos == 0 or num_neg == 0:
        raise ValueError(
            f"cannot adjust label ratio: input has {num_pos} positive and {num_neg} negative records; "
            f"both classes must be present"
        )
    if strategy == "subsample":
        total = min(num_pos / target_ratio, num_neg / (1 - target_ratio))
    else:
        total = max(num_pos / target_ratio, num_neg / (1 - target_ratio))
    target_pos = round(total * target_ratio)
    target_neg = round(total * (1 - target_ratio))
    if target_pos < 1 or target_neg < 1:
        raise ValueError(
            "target_positive_ratio yields fewer than 1 record in a label class; adjust ratio or dataset size"
        )
    if strategy == "subsample" and (target_pos > num_pos or target_neg > num_neg):
        raise ValueError("target_positive_ratio cannot be met without oversampling; use strategy='oversample'")
    resampled_pos = _resample_group(positives, target_pos, rng)
    resampled_neg = _resample_group(negatives, target_neg, rng)
    return resampled_pos + resampled_neg


def _apply_size_cap(
    records: list[correctness_types.EvalRecord],
    max_records: int,
    min_per_label: int,
    rng: random.Random,
) -> list[correctness_types.EvalRecord]:
    """Stratified random sample down to max_records, preserving approximate label ratio.

    Clamps each class to at least ``min(min_per_label, available)`` records so that the size cap
    doesn't accidentally zero out a minority class when a feasible allocation exists.
    """
    positives = [rec for rec in records if rec.label]
    negatives = [rec for rec in records if not rec.label]
    # ratio-preserving split, then clamp for min_per_label
    pos_fraction = len(positives) / len(records)
    target_pos = round(max_records * pos_fraction)
    floor_pos = min(min_per_label, len(positives))
    floor_neg = min(min_per_label, len(negatives))
    target_pos = max(target_pos, floor_pos)
    target_pos = min(target_pos, max_records - floor_neg, len(positives))
    target_neg = max_records - target_pos
    if target_neg > len(negatives):
        target_neg = len(negatives)
        target_pos = max_records - target_neg
    if target_pos > len(positives) or target_neg > len(negatives):
        raise ValueError("size cap targets exceed available records per class")
    capped_pos = _resample_group(positives, target_pos, rng)
    capped_neg = _resample_group(negatives, target_neg, rng)
    return capped_pos + capped_neg


def _resample_group(
    group: list[correctness_types.EvalRecord],
    target_count: int,
    rng: random.Random,
) -> list[correctness_types.EvalRecord]:
    """Resample a group of records to a target count.

    Args:
        group: Records to resample.
        target_count: Desired number of records.
        rng: Random number generator for deterministic sampling.

    Returns:
        Resampled list of records.
    """
    if target_count == len(group):
        return list(group)
    if target_count < len(group):
        return rng.sample(group, target_count)
    return rng.choices(group, k=target_count)  # with replacement for oversample


def _log_summary(
    original: list[correctness_types.EvalRecord],
    result: list[correctness_types.EvalRecord],
) -> None:
    """Log a summary of the resampling result."""
    num_pos = sum(1 for rec in result if rec.label)
    num_neg = len(result) - num_pos
    code_type_counts: dict[str, int] = {}
    for rec in result:
        code_type_counts[rec.code_type] = code_type_counts.get(rec.code_type, 0) + 1
    pos_frac = num_pos / len(result) if result else 0.0
    logger.info(
        "record resampling: %d -> %d records (pos=%.1f%%), label distribution: {True: %d, False: %d}, "
        "code_type distribution: %s",
        len(original),
        len(result),
        pos_frac * 100,
        num_pos,
        num_neg,
        code_type_counts,
    )
