"""Load LMDB completion records into HuggingFace datasets for probe training.

Reads records exported by DiskRewardLogger during RL training runs and converts
them into input-label pairs: messages = structured role-attributed conversation,
label = is_match.

Supports two modes:
- **Two-prefix mode** (default): Train from ``train_key_prefix``, validate from
  ``valid_key_prefix``.
- **Eval-only mode** (``use_eval_only_split=True``): Read from a single prefix
  and split internally into train/valid, with optional family-based splitting
  and code-type filtering.
"""

from __future__ import annotations

import logging
import random
import typing

import datasets  # noqa: TC002 -- used at runtime (Dataset.from_list, DatasetDict)

if typing.TYPE_CHECKING:
    from pyine.probes.data.datamodule_configs import LabelBalanceConfig, ProbeDataModuleConfig

import pyine.data.utils.generation_record
import pyine.data.utils.lmdb_io
import pyine.utils.code.output_compare
from pyine.probes.data.reward_keys import (  # noqa: TID252 -- avoids circular import via __init__
    HARD_MATCH_KEY,
    SOFT_MATCH_KEY,
)

logger = logging.getLogger(__name__)

_RECOMPUTABLE_METRICS = frozenset({SOFT_MATCH_KEY, HARD_MATCH_KEY})


def _load_lmdb_records(
    reader: pyine.data.utils.lmdb_io.LMDBReader,
    key_prefix: str,
    selection_strategy: str,
) -> list[tuple[str, dict[str, typing.Any]]]:
    """Load and deduplicate LMDB records for a given key prefix.

    Delegates to :func:`pyine.data.utils.lmdb_io.load_and_deduplicate_lmdb_records`
    for key parsing and deduplication, then validates key_prefix field consistency.

    Args:
        reader: Open LMDBReader instance.
        key_prefix: Key prefix to filter by (e.g., "train/").
        selection_strategy: "latest" or "best_reward".

    Returns:
        List of (sample_id, record) tuples after deduplication.
    """
    records = pyine.data.utils.lmdb_io.load_and_deduplicate_lmdb_records(
        reader,
        key_prefix,
        selection_strategy,  # type: ignore[arg-type]
    )
    # validate key_prefix field consistency on deduplicated records
    for sample_id, record in records:
        record_prefix = record.get("key_prefix")
        if record_prefix is not None and record_prefix != key_prefix:
            logger.warning(
                "LMDB key prefix '%s' disagrees with record field key_prefix='%s' for sample '%s'; trusting LMDB key",
                key_prefix,
                record_prefix,
                sample_id,
            )
    return records


def _record_to_probe_sample(
    record: dict[str, typing.Any],
    sample_id: str,
    label_metric_key: str,
    recompute_labels: bool,
    compare_options: pyine.utils.code.output_compare.CompareOptions | None,
    skip_malformed: bool,
) -> dict[str, typing.Any] | None:
    """Convert a single LMDB record to a probe training sample.

    Returns:
        ``{"messages": list[dict[str, str]], "label": int, "sample_id": str, "code_type": str}``,
        or None if the record is malformed and skip_malformed is True.

    Raises:
        ValueError: If the record is malformed and skip_malformed is False.
    """
    model_output = record.get("model_output")
    if model_output is None:
        msg = f"record for sample_id '{sample_id}' is missing model_output"
        if skip_malformed:
            return None
        raise ValueError(msg)
    # build structured messages (requires prompt_messages or prompt)
    try:
        messages = pyine.data.utils.generation_record.build_messages_from_record(record, model_output)
    except ValueError:
        if skip_malformed:
            return None
        raise
    # derive label
    if recompute_labels:
        expected_output = record.get("expected_output")
        if expected_output is None:
            msg = f"record for sample_id '{sample_id}' is missing expected_output (needed for recompute)"
            if skip_malformed:
                return None
            raise ValueError(msg)
        predicted = record.get("final_answer")
        if predicted is None:
            predicted = model_output
        label = _recompute_label(label_metric_key, expected_output, predicted, compare_options)
    else:
        reward_metrics = record.get("reward_metrics")
        if reward_metrics is None or label_metric_key not in reward_metrics:
            msg = f"record for sample_id '{sample_id}' is missing reward_metrics['{label_metric_key}']"
            if skip_malformed:
                return None
            raise ValueError(msg)
        label = int(reward_metrics[label_metric_key])
    # extract code_type (always present after DiskRewardLogger export)
    code_type = record.get("code_type")
    if code_type is None:
        code_type = "unknown"
    return {"messages": messages, "label": label, "sample_id": sample_id, "code_type": code_type}


def _recompute_label(
    label_metric_key: str,
    expected: str,
    predicted: str,
    compare_options: pyine.utils.code.output_compare.CompareOptions | None,
) -> int:
    """Re-compute a binary label from expected/predicted outputs."""
    import pyine.organisms.models.rewards.terms.code_exec.utils

    if label_metric_key == SOFT_MATCH_KEY:
        result = pyine.organisms.models.rewards.terms.code_exec.utils.compute_soft_match(
            expected=expected,
            predicted=predicted,
            options=compare_options,
        )
        return int(result.equal)
    if label_metric_key == HARD_MATCH_KEY:
        return int(
            pyine.organisms.models.rewards.terms.code_exec.utils.compute_hard_match(
                expected=expected,
                predicted=predicted,
            )
        )
    raise ValueError(
        f"recompute_labels=True is only supported for label_metric_key in "
        f"{set(_RECOMPUTABLE_METRICS)}, got '{label_metric_key}'"
    )


def _extract_family_id(sample_id: str) -> str:
    """Extract the family (augmentless) identifier from a sample_id.

    Strips the ``/a:{category}:{idx}`` augmentation suffix if present.
    All code-type variants of the same problem share the same family ID.

    Examples:
        >>> _extract_family_id("TACO/train/p000001/s0000/t0000")
        'TACO/train/p000001/s0000/t0000'
        >>> _extract_family_id("TACO/train/p000001/s0000/t0000/a:hints_docs:001")
        'TACO/train/p000001/s0000/t0000'
    """
    augment_marker = "/a:"
    idx = sample_id.rfind(augment_marker)
    if idx == -1:
        return sample_id
    return sample_id[:idx]


def _filter_by_code_type(
    records: list[tuple[str, dict[str, typing.Any]]],
    code_type_filter: list[str],
) -> list[tuple[str, dict[str, typing.Any]]]:
    """Filter LMDB records to include only specified code types.

    Records with ``code_type=None`` are treated as ``"unknown"`` for filtering
    purposes.

    Args:
        records: List of (sample_id, record) tuples.
        code_type_filter: List of allowed code_type strings.

    Returns:
        Filtered list of (sample_id, record) tuples.
    """
    allowed = set(code_type_filter)
    return [(sample_id, record) for sample_id, record in records if (record.get("code_type") or "unknown") in allowed]


def _split_records_by_family(
    samples: list[dict[str, typing.Any]],
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, typing.Any]], list[dict[str, typing.Any]]]:
    """Split samples into train/valid by family ID.

    All samples sharing a family ID go to the same split, preventing data
    leakage from shared problem structure.

    Rounding rule: ``n_train = max(1, int(n_families * train_ratio))``,
    ``n_valid = n_families - n_train``. Both splits are guaranteed at least
    1 family. Raises ``ValueError`` if fewer than 2 families exist.

    Args:
        samples: List of sample dicts (must have ``"sample_id"`` key).
        train_ratio: Fraction of families assigned to train.
        seed: Random seed for deterministic shuffling.

    Returns:
        (train_samples, valid_samples) tuple.

    Raises:
        ValueError: If fewer than 2 families exist.
    """
    # group samples by family ID
    families: dict[str, list[dict[str, typing.Any]]] = {}
    for sample in samples:
        family_id = _extract_family_id(str(sample["sample_id"]))
        families.setdefault(family_id, []).append(sample)

    n_families = len(families)
    if n_families < 2:
        raise ValueError(
            f"split_by_family requires at least 2 families, got {n_families}; "
            "use split_by_family=False or provide more data"
        )

    # deterministic shuffle of family IDs
    family_ids = sorted(families.keys())
    rng = random.Random(seed)
    rng.shuffle(family_ids)

    # split families
    n_train = max(1, int(n_families * train_ratio))
    # ensure at least 1 valid family
    if n_train >= n_families:
        n_train = n_families - 1

    train_family_ids = set(family_ids[:n_train])

    train_samples: list[dict[str, typing.Any]] = []
    valid_samples: list[dict[str, typing.Any]] = []
    for family_id in family_ids:
        target = train_samples if family_id in train_family_ids else valid_samples
        target.extend(families[family_id])

    return train_samples, valid_samples


def _split_records_random(
    samples: list[dict[str, typing.Any]],
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, typing.Any]], list[dict[str, typing.Any]]]:
    """Split samples into train/valid by random shuffle.

    Args:
        samples: List of sample dicts.
        train_ratio: Fraction of samples assigned to train.
        seed: Random seed for deterministic shuffling.

    Returns:
        (train_samples, valid_samples) tuple.

    Raises:
        ValueError: If fewer than 2 samples exist.
    """
    if len(samples) < 2:
        raise ValueError(f"random split requires at least 2 samples, got {len(samples)}")

    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_train = max(1, int(len(shuffled) * train_ratio))
    if n_train >= len(shuffled):
        n_train = len(shuffled) - 1

    return shuffled[:n_train], shuffled[n_train:]


def _convert_records_to_samples(
    records: list[tuple[str, dict[str, typing.Any]]],
    label_metric_key: str,
    recompute_labels: bool,
    compare_options: pyine.utils.code.output_compare.CompareOptions | None,
    skip_malformed: bool,
) -> tuple[list[dict[str, typing.Any]], int]:
    """Convert LMDB records to probe samples, tracking skipped count.

    Returns:
        (samples, skipped_count) tuple.
    """
    samples: list[dict[str, typing.Any]] = []
    skipped = 0
    for sample_id, record in records:
        sample = _record_to_probe_sample(
            record,
            sample_id,
            label_metric_key,
            recompute_labels,
            compare_options,
            skip_malformed=skip_malformed,
        )
        if sample is None:
            skipped += 1
        else:
            samples.append(sample)
    return samples, skipped


def _apply_label_balance(
    samples: list[dict[str, typing.Any]],
    balance_config: LabelBalanceConfig,
    seed: int,
) -> list[dict[str, typing.Any]]:
    """Resample a list of probe samples to match target label/code-type proportions.

    Supports two modes (determined by which field is set on *balance_config*):

    - **Simple mode** (``target_positive_ratio``): partition by label, then
      subsample or oversample to hit the target positive fraction.
    - **Group mode** (``group_proportions``): partition by ``(code_type, label)``
      groups, then subsample or oversample each group to hit the target
      proportions.  Groups not listed in the dict are excluded.

    Args:
        samples: List of sample dicts (must have ``"label"`` and ``"code_type"`` keys).
        balance_config: Label balance configuration.
        seed: Random seed for deterministic resampling.

    Returns:
        Resampled list of sample dicts.

    Raises:
        ValueError: If a requested group has no samples, or if the target
            ratio/proportions would produce zero samples for any group.
    """
    rng = random.Random(seed)

    if balance_config.target_positive_ratio is not None:
        return _apply_simple_balance(samples, balance_config.target_positive_ratio, balance_config.strategy, rng)
    assert balance_config.group_proportions is not None
    return _apply_group_balance(samples, balance_config.group_proportions, balance_config.strategy, rng)


def _apply_simple_balance(
    samples: list[dict[str, typing.Any]],
    target_ratio: float,
    strategy: str,
    rng: random.Random,
) -> list[dict[str, typing.Any]]:
    """Simple mode: resample to hit target positive ratio."""
    positives = [s for s in samples if s["label"] == 1]
    negatives = [s for s in samples if s["label"] == 0]
    n_pos = len(positives)
    n_neg = len(negatives)

    if strategy == "subsample":
        # Total = min(n_pos / target, n_neg / (1 - target))
        total = min(n_pos / target_ratio, n_neg / (1 - target_ratio))
        n_target_pos = int(round(total * target_ratio))
        n_target_neg = int(round(total * (1 - target_ratio)))
    else:  # oversample
        # Total = max(n_pos / target, n_neg / (1 - target))
        total = max(n_pos / target_ratio, n_neg / (1 - target_ratio))
        n_target_pos = int(round(total * target_ratio))
        n_target_neg = int(round(total * (1 - target_ratio)))

    # Guard against degenerate results
    if n_target_pos == 0:
        raise ValueError(
            f"target_positive_ratio={target_ratio} with {n_pos} positive and {n_neg} negative samples "
            f"would require 0 positive samples after {strategy}"
        )
    if n_target_neg == 0:
        raise ValueError(
            f"target_positive_ratio={target_ratio} with {n_pos} positive and {n_neg} negative samples "
            f"would require 0 negative samples after {strategy}"
        )

    resampled_pos = _resample_group(positives, n_target_pos, strategy, rng)
    resampled_neg = _resample_group(negatives, n_target_neg, strategy, rng)

    result = resampled_pos + resampled_neg
    original_count = len(samples)
    pos_count = len(resampled_pos)
    neg_count = len(resampled_neg)
    logger.info(
        "label_balance (simple) applied: %d -> %d samples, distribution: {label=1: %d, label=0: %d}",
        original_count,
        len(result),
        pos_count,
        neg_count,
    )
    return result


def _apply_group_balance(
    samples: list[dict[str, typing.Any]],
    group_proportions: dict[str, float],
    strategy: str,
    rng: random.Random,
) -> list[dict[str, typing.Any]]:
    """Group mode: resample (code_type, label) groups to hit target proportions."""
    # Partition samples into (code_type, label) buckets
    buckets: dict[str, list[dict[str, typing.Any]]] = {}
    for sample in samples:
        key = f"{sample['code_type']}:{sample['label']}"
        buckets.setdefault(key, []).append(sample)

    # Validate group availability
    missing = [k for k in group_proportions if k not in buckets or len(buckets[k]) == 0]
    if missing:
        raise ValueError(f"group_proportions references groups with no samples in the data: {missing}")

    # Normalize proportions to sum to 1.0
    total_weight = sum(group_proportions.values())
    normalized = {k: v / total_weight for k, v in group_proportions.items()}

    # Compute target count per group
    if strategy == "subsample":
        # Constraining group: smallest actual_count / target_proportion
        anchor = min(len(buckets[k]) / normalized[k] for k in normalized)
    else:  # oversample
        # Constraining group: largest actual_count / target_proportion
        anchor = max(len(buckets[k]) / normalized[k] for k in normalized)

    target_counts: dict[str, int] = {}
    for k in normalized:
        target_counts[k] = int(round(anchor * normalized[k]))

    # Guard against degenerate results
    for k, count in target_counts.items():
        if count == 0:
            raise ValueError(
                f"group_proportions would produce 0 samples for group '{k}' "
                f"(proportion too small relative to constraining group size)"
            )

    # Resample each group
    result: list[dict[str, typing.Any]] = []
    distribution: dict[str, int] = {}
    for k in sorted(group_proportions.keys()):
        resampled = _resample_group(buckets[k], target_counts[k], strategy, rng)
        result.extend(resampled)
        distribution[k] = len(resampled)

    original_count = len(samples)
    logger.info(
        "label_balance (group) applied: %d -> %d samples, distribution: %s",
        original_count,
        len(result),
        distribution,
    )
    return result


def _resample_group(
    group: list[dict[str, typing.Any]],
    target_count: int,
    strategy: str,
    rng: random.Random,
) -> list[dict[str, typing.Any]]:
    """Resample a group of samples to a target count."""
    if target_count == len(group):
        return list(group)
    if target_count < len(group):
        # subsample: deterministic selection
        return rng.sample(group, target_count)
    # oversample: sampling with replacement for the extra samples
    return rng.choices(group, k=target_count)


def _validate_probe_split(
    split_name: str,
    dataset: datasets.Dataset,
) -> None:
    """Validate a single split of the probe dataset."""
    unique_labels: set[int] = set(
        typing.cast("list[int]", dataset.unique("label"))  # pyright: ignore[reportUnknownMemberType]  # datasets stubs
    )
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"split '{split_name}': expected binary labels {{0, 1}}, got {unique_labels}")

    messages_col = typing.cast("list[list[dict[str, str]]]", dataset["messages"])
    if any(not msgs for msgs in messages_col):
        raise ValueError(f"split '{split_name}': found empty messages fields")

    if len(unique_labels) < 2:
        logger.warning(
            "split '%s' has only label(s) %s (%d samples) -- probe training may be degenerate",
            split_name,
            unique_labels,
            len(dataset),
        )


def load_probe_dataset_from_lmdb(
    config: ProbeDataModuleConfig,
    compare_options: pyine.utils.code.output_compare.CompareOptions | None = None,
) -> datasets.DatasetDict:
    """Load LMDB completion records and create a probe training dataset.

    Reads records exported by DiskRewardLogger (JSON_ZSTD serialization),
    extracts prompt+completion as input text and derives binary labels
    from reward metrics.

    Supports two modes (controlled by ``config.use_eval_only_split``):

    - **Two-prefix mode** (default): Load train records from
      ``config.train_key_prefix`` and valid records from ``config.valid_key_prefix``.
    - **Eval-only mode**: Load all records from ``config.eval_only_source_prefix``
      and split internally into train/valid.

    Args:
        config: Probe data module configuration containing LMDB path, label key,
            split strategy, and all other dataset parameters.
        compare_options: CompareOptions for soft match re-computation.

    Returns:
        DatasetDict with "train" and "valid" splits, each containing:
        - "messages": list[dict[str, str]] (structured role-attributed conversation)
        - "label": int (0 or 1)
        - "sample_id": str (for provenance tracking)
        - "code_type": str (code augmentation type)

    Raises:
        ValueError: If recompute_labels=True with unsupported label_metric_key,
            if no records match a key prefix, or if code_type_filter excludes all.
    """
    # defensive runtime check (also enforced in config validator)
    if config.recompute_labels and config.label_metric_key not in _RECOMPUTABLE_METRICS:
        raise ValueError(
            f"recompute_labels=True is only supported for label_metric_key in "
            f"{set(_RECOMPUTABLE_METRICS)}, got '{config.label_metric_key}'"
        )

    if config.use_eval_only_split:
        splits = _load_eval_only(config, compare_options)
    else:
        splits = _load_two_prefix(config, compare_options)

    logger.info(
        "Loaded probe dataset: train=%d samples, valid=%d samples",
        len(splits["train"]),
        len(splits["valid"]),
    )
    return datasets.DatasetDict(splits)  # pyright: ignore[reportCallIssue, reportArgumentType]  # datasets stubs


def _load_two_prefix(
    config: ProbeDataModuleConfig,
    compare_options: pyine.utils.code.output_compare.CompareOptions | None,
) -> dict[str, datasets.Dataset]:
    """Two-prefix loading mode (original behavior)."""
    splits: dict[str, datasets.Dataset] = {}
    with pyine.data.utils.lmdb_io.LMDBReader(config.lmdb_path) as reader:
        for split_name, prefix in [("train", config.train_key_prefix), ("valid", config.valid_key_prefix)]:
            records = _load_lmdb_records(reader, prefix, config.selection_strategy)
            if not records:
                raise ValueError(
                    f"no records match key prefix '{prefix}' in LMDB at {config.lmdb_path}; "
                    "check that the prefix matches the key_prefix used during export"
                )

            if config.code_type_filter is not None:
                records = _filter_by_code_type(records, config.code_type_filter)
                if not records:
                    raise ValueError(
                        f"no records remain after code_type_filter={config.code_type_filter} for split '{split_name}'"
                    )

            samples, skipped = _convert_records_to_samples(
                records,
                config.label_metric_key,
                config.recompute_labels,
                compare_options,
                config.skip_malformed_records,
            )

            if skipped > 0:
                logger.warning(
                    "split '%s': skipped %d malformed records out of %d total",
                    split_name,
                    skipped,
                    len(records),
                )

            if not samples:
                raise ValueError(
                    f"split '{split_name}': all {len(records)} records were malformed; no valid samples to train on"
                )

            # apply label balance resampling (before capping and validation)
            if config.label_balance is not None and split_name in config.label_balance.apply_to:
                samples = _apply_label_balance(samples, config.label_balance, config.split_seed)

            ds = datasets.Dataset.from_list(samples)  # pyright: ignore[reportUnknownMemberType]  # datasets stubs

            if config.max_samples_per_split is not None and len(ds) > config.max_samples_per_split:
                ds = ds.shuffle(seed=config.split_seed).select(range(config.max_samples_per_split))  # pyright: ignore[reportUnknownMemberType]  # datasets stubs

            _validate_probe_split(split_name, ds)
            splits[split_name] = ds

    return splits


def _load_eval_only(
    config: ProbeDataModuleConfig,
    compare_options: pyine.utils.code.output_compare.CompareOptions | None,
) -> dict[str, datasets.Dataset]:
    """Eval-only loading mode: single prefix, internal train/valid split."""
    with pyine.data.utils.lmdb_io.LMDBReader(config.lmdb_path) as reader:
        records = _load_lmdb_records(reader, config.eval_only_source_prefix, config.selection_strategy)

    if not records:
        raise ValueError(
            f"no records match key prefix '{config.eval_only_source_prefix}' in LMDB at {config.lmdb_path}; "
            "check that the prefix matches the key_prefix used during export"
        )

    if config.code_type_filter is not None:
        records = _filter_by_code_type(records, config.code_type_filter)
        if not records:
            raise ValueError(f"no records remain after code_type_filter={config.code_type_filter}")

    # convert to samples
    samples, skipped = _convert_records_to_samples(
        records,
        config.label_metric_key,
        config.recompute_labels,
        compare_options,
        config.skip_malformed_records,
    )

    if skipped > 0:
        logger.warning(
            "eval-only mode: skipped %d malformed records out of %d total",
            skipped,
            len(records),
        )

    if not samples:
        raise ValueError(f"eval-only mode: all {len(records)} records were malformed; no valid samples to train on")

    # split into train/valid
    seed = config.split_seed
    if config.split_by_family:
        train_samples, valid_samples = _split_records_by_family(samples, config.train_split_ratio, seed)
    else:
        train_samples, valid_samples = _split_records_random(samples, config.train_split_ratio, seed)

    # log per-split code type distribution
    for name, split_samples in [("train", train_samples), ("valid", valid_samples)]:
        code_type_counts: dict[str, int] = {}
        for sample in split_samples:
            code_type = str(sample["code_type"])
            code_type_counts[code_type] = code_type_counts.get(code_type, 0) + 1
        logger.info(
            "eval-only %s split: %d samples, code_type distribution: %s",
            name,
            len(split_samples),
            code_type_counts,
        )

    splits: dict[str, datasets.Dataset] = {}
    for split_name, split_samples in [("train", train_samples), ("valid", valid_samples)]:
        # apply label balance resampling (before capping and validation)
        balanced = split_samples
        if config.label_balance is not None and split_name in config.label_balance.apply_to:
            balanced = _apply_label_balance(split_samples, config.label_balance, seed)

        ds = datasets.Dataset.from_list(balanced)  # pyright: ignore[reportUnknownMemberType]  # datasets stubs

        if config.max_samples_per_split is not None and len(ds) > config.max_samples_per_split:
            ds = ds.shuffle(seed=seed).select(range(config.max_samples_per_split))  # pyright: ignore[reportUnknownMemberType]  # datasets stubs

        _validate_probe_split(split_name, ds)
        splits[split_name] = ds

    return splits
