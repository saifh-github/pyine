"""Load LMDB completion records into HuggingFace datasets for probe training.

Reads records exported by DiskRewardLogger during RL training runs and converts
them into input-label pairs: text = prompt + model_output, label = is_match.

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

import datasets  # noqa: TC002 — used at runtime (Dataset.from_list, DatasetDict)

from pyine.data.utils.lmdb_io import LMDBReader

if typing.TYPE_CHECKING:
    from pathlib import Path

    import pyine.utils.code.output_compare

logger = logging.getLogger(__name__)

_RECOMPUTABLE_METRICS = frozenset({"soft_match/is_match", "hard_match/is_match"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_lmdb_key(
    key: str,
    key_prefix: str,
) -> tuple[str, int]:
    """Parse an LMDB key into (sample_id, generation_count).

    Reuses the rfind("/") convention from BiasDataModuleBase._load_pregenerated_outputs().

    Args:
        key: Full LMDB key (e.g., "train/TACO/train/p000001/s0000/3").
        key_prefix: Prefix to strip (e.g., "train/").

    Returns:
        Tuple of (sample_id, generation_count).

    Raises:
        ValueError: If the key has no "/" separator after the prefix, or if
            the generation count segment is non-numeric and not "none".
    """
    suffix = key[len(key_prefix) :]
    sep = suffix.rfind("/")
    if sep == -1:
        raise ValueError(f"unexpected key format (no '/' separator after prefix): {key}")
    sample_id = suffix[:sep]
    gen_str = suffix[sep + 1 :]
    if gen_str == "none":
        return sample_id, 0
    try:
        return sample_id, int(gen_str)
    except ValueError:
        raise ValueError(
            f"non-numeric generation count '{gen_str}' in key '{key}' (expected integer or 'none')"
        ) from None


def _load_lmdb_records(
    reader: LMDBReader,
    key_prefix: str,
    selection_strategy: str,
) -> list[tuple[str, dict[str, typing.Any]]]:
    """Load and deduplicate LMDB records for a given key prefix.

    Args:
        reader: Open LMDBReader instance.
        key_prefix: Key prefix to filter by (e.g., "train/").
        selection_strategy: "latest" or "best_reward".

    Returns:
        List of (sample_id, record) tuples after deduplication.
    """
    # Group records by sample_id
    grouped: dict[str, list[tuple[int, dict[str, typing.Any]]]] = {}
    for key in reader.key_map:
        if not key.startswith(key_prefix):
            continue
        sample_id, gen_count = _parse_lmdb_key(key, key_prefix)
        record: dict[str, typing.Any] = reader.get(key)

        # Validate key_prefix field consistency
        record_prefix = record.get("key_prefix")
        if record_prefix is not None and record_prefix != key_prefix:
            logger.warning(
                "LMDB key prefix '%s' disagrees with record field key_prefix='%s' for key '%s'; trusting LMDB key",
                key_prefix,
                record_prefix,
                key,
            )

        grouped.setdefault(sample_id, []).append((gen_count, record))

    # Deduplicate
    result: list[tuple[str, dict[str, typing.Any]]] = []
    for sample_id, entries in grouped.items():
        if selection_strategy == "latest":
            best = max(entries, key=lambda entry: entry[0])
        elif selection_strategy == "best_reward":
            for gen_count, record in entries:
                if record.get("reward_total") is None:
                    raise ValueError(
                        f"best_reward selection requires reward_total for all records, "
                        f"but sample_id '{sample_id}' (generation_count={gen_count}) has None; "
                        "ensure logging.log_total=True during export"
                    )
            best = max(entries, key=lambda entry: entry[1]["reward_total"])
        else:
            raise ValueError(f"unknown selection_strategy: {selection_strategy!r}")
        result.append((sample_id, best[1]))

    return result


def _record_to_probe_sample(
    record: dict[str, typing.Any],
    sample_id: str,
    label_metric_key: str,
    recompute_labels: bool,
    compare_options: pyine.utils.code.output_compare.CompareOptions | None,
    skip_malformed: bool,
) -> dict[str, str | int] | None:
    """Convert a single LMDB record to a probe training sample.

    Returns:
        {"text": str, "label": int, "sample_id": str}, or None if the record
        is malformed and skip_malformed is True.

    Raises:
        ValueError: If the record is malformed and skip_malformed is False.
    """
    prompt = record.get("prompt")
    model_output = record.get("model_output")

    # Check required fields: prompt and model_output are always needed
    if prompt is None or model_output is None:
        msg = f"record for sample_id '{sample_id}' is missing prompt or model_output"
        if skip_malformed:
            return None
        raise ValueError(msg)

    text = prompt + model_output

    # Derive label
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

    # Extract code_type (always present after DiskRewardLogger export)
    code_type = record.get("code_type")
    if code_type is None:
        code_type = "unknown"

    return {"text": text, "label": label, "sample_id": sample_id, "code_type": code_type}


def _recompute_label(
    label_metric_key: str,
    expected: str,
    predicted: str,
    compare_options: pyine.utils.code.output_compare.CompareOptions | None,
) -> int:
    """Re-compute a binary label from expected/predicted outputs."""
    from pyine.organisms.models.rewards.terms.code_exec.utils import compute_hard_match, compute_soft_match

    if label_metric_key == "soft_match/is_match":
        result = compute_soft_match(expected=expected, predicted=predicted, options=compare_options)
        return int(result.equal)
    if label_metric_key == "hard_match/is_match":
        return int(compute_hard_match(expected=expected, predicted=predicted))
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
    return [(sid, rec) for sid, rec in records if (rec.get("code_type") or "unknown") in allowed]


def _split_records_by_family(
    samples: list[dict[str, str | int]],
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
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
    # Group samples by family ID
    families: dict[str, list[dict[str, str | int]]] = {}
    for sample in samples:
        family_id = _extract_family_id(str(sample["sample_id"]))
        families.setdefault(family_id, []).append(sample)

    n_families = len(families)
    if n_families < 2:
        raise ValueError(
            f"split_by_family requires at least 2 families, got {n_families}; "
            "use split_by_family=False or provide more data"
        )

    # Deterministic shuffle of family IDs
    family_ids = sorted(families.keys())
    rng = random.Random(seed)
    rng.shuffle(family_ids)

    # Split families
    n_train = max(1, int(n_families * train_ratio))
    # Ensure at least 1 valid family
    if n_train >= n_families:
        n_train = n_families - 1

    train_family_ids = set(family_ids[:n_train])

    train_samples: list[dict[str, str | int]] = []
    valid_samples: list[dict[str, str | int]] = []
    for fid in family_ids:
        target = train_samples if fid in train_family_ids else valid_samples
        target.extend(families[fid])

    return train_samples, valid_samples


def _split_records_random(
    samples: list[dict[str, str | int]],
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
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
) -> tuple[list[dict[str, str | int]], int]:
    """Convert LMDB records to probe samples, tracking skipped count.

    Returns:
        (samples, skipped_count) tuple.
    """
    samples: list[dict[str, str | int]] = []
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


def _validate_probe_split(split_name: str, dataset: datasets.Dataset) -> None:
    """Validate a single split of the probe dataset."""
    unique_labels: set[int] = set(
        typing.cast("list[int]", dataset.unique("label"))  # pyright: ignore[reportUnknownMemberType]  # datasets stubs
    )
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"split '{split_name}': expected binary labels {{0, 1}}, got {unique_labels}")

    texts = typing.cast("list[str]", dataset["text"])
    if any(not t for t in texts):
        raise ValueError(f"split '{split_name}': found empty text fields")

    if len(unique_labels) < 2:
        logger.warning(
            "split '%s' has only label(s) %s (%d samples) — probe training may be degenerate",
            split_name,
            unique_labels,
            len(dataset),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_probe_dataset_from_lmdb(
    lmdb_path: str | Path,
    label_metric_key: str = "soft_match/is_match",
    train_key_prefix: str = "train/",
    valid_key_prefix: str = "eval/",
    selection_strategy: str = "latest",
    recompute_labels: bool = False,
    compare_options: pyine.utils.code.output_compare.CompareOptions | None = None,
    max_samples_per_split: int | None = None,
    skip_malformed_records: bool = False,
    seed: int = 42,
    # --- Eval-only split mode ---
    use_eval_only_split: bool = False,
    eval_only_source_prefix: str = "eval/",
    train_split_ratio: float = 0.8,
    split_by_family: bool = True,
    code_type_filter: list[str] | None = None,
) -> datasets.DatasetDict:
    """Load LMDB completion records and create a probe training dataset.

    Reads records exported by DiskRewardLogger (JSON_ZSTD serialization),
    extracts prompt+completion as input text and derives binary labels
    from reward metrics.

    Supports two modes:

    - **Two-prefix mode** (``use_eval_only_split=False``, default): Load train
      records from ``train_key_prefix`` and valid records from ``valid_key_prefix``.
    - **Eval-only mode** (``use_eval_only_split=True``): Load all records from
      ``eval_only_source_prefix`` and split internally into train/valid.

    Args:
        lmdb_path: Path to LMDB database from DiskRewardLogger.
        label_metric_key: Key in reward_metrics for binary label.
        train_key_prefix: Key prefix for training records (two-prefix mode).
        valid_key_prefix: Key prefix for validation records (two-prefix mode).
        selection_strategy: "latest" or "best_reward" for deduplication.
        recompute_labels: If True, re-compute match instead of using stored metrics.
        compare_options: CompareOptions for soft match re-computation.
        max_samples_per_split: Cap samples per split (for debugging/fast iteration).
        skip_malformed_records: If True, skip records missing required fields.
        seed: Random seed for shuffling.
        use_eval_only_split: If True, use eval-only mode (single prefix, internal split).
        eval_only_source_prefix: LMDB prefix to read when ``use_eval_only_split=True``.
        train_split_ratio: Fraction of data for training (eval-only mode).
        split_by_family: Split by family ID to prevent data leakage (eval-only mode).
        code_type_filter: If set, only include records with matching code_type.
            ``None`` includes all records. Applies in both modes.

    Returns:
        DatasetDict with "train" and "valid" splits, each containing:
        - "text": str (prompt + model_output)
        - "label": int (0 or 1)
        - "sample_id": str (for provenance tracking)
        - "code_type": str (code augmentation type)

    Raises:
        ValueError: If recompute_labels=True with unsupported label_metric_key,
            if no records match a key prefix, or if code_type_filter excludes all.
    """
    # Defensive runtime check (also enforced in config validator)
    if recompute_labels and label_metric_key not in _RECOMPUTABLE_METRICS:
        raise ValueError(
            f"recompute_labels=True is only supported for label_metric_key in "
            f"{set(_RECOMPUTABLE_METRICS)}, got '{label_metric_key}'"
        )

    if use_eval_only_split:
        splits = _load_eval_only(
            lmdb_path=lmdb_path,
            eval_only_source_prefix=eval_only_source_prefix,
            selection_strategy=selection_strategy,
            label_metric_key=label_metric_key,
            recompute_labels=recompute_labels,
            compare_options=compare_options,
            skip_malformed_records=skip_malformed_records,
            train_split_ratio=train_split_ratio,
            split_by_family=split_by_family,
            code_type_filter=code_type_filter,
            max_samples_per_split=max_samples_per_split,
            seed=seed,
        )
    else:
        splits = _load_two_prefix(
            lmdb_path=lmdb_path,
            train_key_prefix=train_key_prefix,
            valid_key_prefix=valid_key_prefix,
            selection_strategy=selection_strategy,
            label_metric_key=label_metric_key,
            recompute_labels=recompute_labels,
            compare_options=compare_options,
            skip_malformed_records=skip_malformed_records,
            code_type_filter=code_type_filter,
            max_samples_per_split=max_samples_per_split,
            seed=seed,
        )

    logger.info(
        "Loaded probe dataset: train=%d samples, valid=%d samples",
        len(splits["train"]),
        len(splits["valid"]),
    )
    return datasets.DatasetDict(splits)  # pyright: ignore[reportCallIssue, reportArgumentType]  # datasets stubs


def _load_two_prefix(
    lmdb_path: str | Path,
    train_key_prefix: str,
    valid_key_prefix: str,
    selection_strategy: str,
    label_metric_key: str,
    recompute_labels: bool,
    compare_options: pyine.utils.code.output_compare.CompareOptions | None,
    skip_malformed_records: bool,
    code_type_filter: list[str] | None,
    max_samples_per_split: int | None,
    seed: int,
) -> dict[str, datasets.Dataset]:
    """Two-prefix loading mode (original behavior)."""
    splits: dict[str, datasets.Dataset] = {}
    with LMDBReader(lmdb_path) as reader:
        for split_name, prefix in [("train", train_key_prefix), ("valid", valid_key_prefix)]:
            records = _load_lmdb_records(reader, prefix, selection_strategy)
            if not records:
                raise ValueError(
                    f"no records match key prefix '{prefix}' in LMDB at {lmdb_path}; "
                    "check that the prefix matches the key_prefix used during export"
                )

            if code_type_filter is not None:
                records = _filter_by_code_type(records, code_type_filter)
                if not records:
                    raise ValueError(
                        f"no records remain after code_type_filter={code_type_filter} for split '{split_name}'"
                    )

            samples, skipped = _convert_records_to_samples(
                records,
                label_metric_key,
                recompute_labels,
                compare_options,
                skip_malformed_records,
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

            ds = datasets.Dataset.from_list(samples)  # pyright: ignore[reportUnknownMemberType]  # datasets stubs

            if max_samples_per_split is not None and len(ds) > max_samples_per_split:
                ds = ds.shuffle(seed=seed).select(range(max_samples_per_split))  # pyright: ignore[reportUnknownMemberType]  # datasets stubs

            _validate_probe_split(split_name, ds)
            splits[split_name] = ds

    return splits


def _load_eval_only(
    lmdb_path: str | Path,
    eval_only_source_prefix: str,
    selection_strategy: str,
    label_metric_key: str,
    recompute_labels: bool,
    compare_options: pyine.utils.code.output_compare.CompareOptions | None,
    skip_malformed_records: bool,
    train_split_ratio: float,
    split_by_family: bool,
    code_type_filter: list[str] | None,
    max_samples_per_split: int | None,
    seed: int,
) -> dict[str, datasets.Dataset]:
    """Eval-only loading mode: single prefix, internal train/valid split."""
    with LMDBReader(lmdb_path) as reader:
        records = _load_lmdb_records(reader, eval_only_source_prefix, selection_strategy)

    if not records:
        raise ValueError(
            f"no records match key prefix '{eval_only_source_prefix}' in LMDB at {lmdb_path}; "
            "check that the prefix matches the key_prefix used during export"
        )

    if code_type_filter is not None:
        records = _filter_by_code_type(records, code_type_filter)
        if not records:
            raise ValueError(f"no records remain after code_type_filter={code_type_filter}")

    # Convert to samples
    samples, skipped = _convert_records_to_samples(
        records,
        label_metric_key,
        recompute_labels,
        compare_options,
        skip_malformed_records,
    )

    if skipped > 0:
        logger.warning(
            "eval-only mode: skipped %d malformed records out of %d total",
            skipped,
            len(records),
        )

    if not samples:
        raise ValueError(f"eval-only mode: all {len(records)} records were malformed; no valid samples to train on")

    # Split into train/valid
    if split_by_family:
        train_samples, valid_samples = _split_records_by_family(samples, train_split_ratio, seed)
    else:
        train_samples, valid_samples = _split_records_random(samples, train_split_ratio, seed)

    # Log per-split code type distribution
    for name, split_samples in [("train", train_samples), ("valid", valid_samples)]:
        ct_counts: dict[str, int] = {}
        for s in split_samples:
            ct = str(s["code_type"])
            ct_counts[ct] = ct_counts.get(ct, 0) + 1
        logger.info("eval-only %s split: %d samples, code_type distribution: %s", name, len(split_samples), ct_counts)

    splits: dict[str, datasets.Dataset] = {}
    for split_name, split_samples in [("train", train_samples), ("valid", valid_samples)]:
        ds = datasets.Dataset.from_list(split_samples)  # pyright: ignore[reportUnknownMemberType]  # datasets stubs

        if max_samples_per_split is not None and len(ds) > max_samples_per_split:
            ds = ds.shuffle(seed=seed).select(range(max_samples_per_split))  # pyright: ignore[reportUnknownMemberType]  # datasets stubs

        _validate_probe_split(split_name, ds)
        splits[split_name] = ds

    return splits
