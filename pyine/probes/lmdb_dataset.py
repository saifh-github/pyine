"""Load LMDB completion records into HuggingFace datasets for probe training.

Reads records exported by DiskRewardLogger during RL training runs and converts
them into input-label pairs: text = prompt + model_output, label = is_match.
"""

from __future__ import annotations

import logging
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

    return {"text": text, "label": label, "sample_id": sample_id}


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
) -> datasets.DatasetDict:
    """Load LMDB completion records and create a probe training dataset.

    Reads records exported by DiskRewardLogger (JSON_ZSTD serialization),
    extracts prompt+completion as input text and derives binary labels
    from reward metrics.

    Args:
        lmdb_path: Path to LMDB database from DiskRewardLogger.
        label_metric_key: Key in reward_metrics for binary label. Permissive — any
            string is accepted as long as the key exists in record reward_metrics.
        train_key_prefix: Key prefix for training records.
        valid_key_prefix: Key prefix for validation records.
        selection_strategy: "latest" or "best_reward" for deduplication.
        recompute_labels: If True, re-compute match instead of using stored metrics.
            Only supported for label_metric_key in {"soft_match/is_match",
            "hard_match/is_match"} — raises ValueError otherwise.
        compare_options: CompareOptions for soft match re-computation. Ignored when
            recompute_labels=False or when label_metric_key is "hard_match/is_match".
        max_samples_per_split: Cap samples per split (for debugging/fast iteration).
        skip_malformed_records: If True, skip records missing required fields and log
            count at WARNING level. If False (default), raise ValueError.
        seed: Random seed for shuffling when capping samples.

    Returns:
        DatasetDict with "train" and "valid" splits, each containing:
        - "text": str (prompt + model_output)
        - "label": int (0 or 1)
        - "sample_id": str (for provenance tracking)

    Raises:
        ValueError: If recompute_labels=True with unsupported label_metric_key,
            if selection_strategy='best_reward' and reward_total is None,
            if skip_malformed_records=False and a malformed record is encountered,
            or if no records match a key prefix.
    """
    # Defensive runtime check (also enforced in config validator)
    if recompute_labels and label_metric_key not in _RECOMPUTABLE_METRICS:
        raise ValueError(
            f"recompute_labels=True is only supported for label_metric_key in "
            f"{set(_RECOMPUTABLE_METRICS)}, got '{label_metric_key}'"
        )

    splits: dict[str, datasets.Dataset] = {}
    with LMDBReader(lmdb_path) as reader:
        for split_name, prefix in [("train", train_key_prefix), ("valid", valid_key_prefix)]:
            records = _load_lmdb_records(reader, prefix, selection_strategy)
            if not records:
                raise ValueError(
                    f"no records match key prefix '{prefix}' in LMDB at {lmdb_path}; "
                    "check that the prefix matches the key_prefix used during export"
                )

            samples: list[dict[str, str | int]] = []
            skipped = 0
            for sample_id, record in records:
                sample = _record_to_probe_sample(
                    record,
                    sample_id,
                    label_metric_key,
                    recompute_labels,
                    compare_options,
                    skip_malformed=skip_malformed_records,
                )
                if sample is None:
                    skipped += 1
                else:
                    samples.append(sample)

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

    logger.info(
        "Loaded probe dataset: train=%d samples, valid=%d samples",
        len(splits["train"]),
        len(splits["valid"]),
    )
    return datasets.DatasetDict(splits)  # pyright: ignore[reportCallIssue, reportArgumentType]  # datasets stubs
