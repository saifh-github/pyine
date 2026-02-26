"""LMDB reading for the guardrail correctness evaluation pipeline.

Loads pregenerated evaluation records from LMDB datasets (produced by ``DiskEvalLogger`` during
a code execution evaluation run) and converts them into ``EvalRecord`` objects.
"""

from __future__ import annotations

import logging
import pathlib  # noqa: TC003
import typing

import pyine.data.utils.lmdb_io
import pyine.evals.correctness.splits as correctness_splits
import pyine.evals.correctness.types as correctness_types

logger = logging.getLogger(__name__)


def load_records_from_lmdb(
    lmdb_paths: typing.Sequence[pathlib.Path],
    label_type: correctness_types.LabelType,
) -> list[correctness_types.EvalRecord]:
    """Load EvalRecords from one or more LMDB datasets.

    Follows the pattern from ``pyine.evals.code_exec.reeval``: opens each LMDB with
    ``LMDBReader``, validates ``record_type == 'benchmark'``, and converts records to
    ``EvalRecord`` objects.

    Args:
        lmdb_paths: Paths to LMDB datasets containing eval records.
        label_type: Which correctness label to use (hard_match or soft_match).

    Returns:
        Flat list of EvalRecord objects from all LMDBs.

    Raises:
        ValueError: If an LMDB has the wrong record_type or duplicate (sample_id, attempt_index).
    """
    records: list[correctness_types.EvalRecord] = []
    seen_keys: set[tuple[str, int]] = set()
    seen_difficulty_scores = False
    missing_difficulty_ids: list[str] = []
    for lmdb_path in lmdb_paths:
        reader = pyine.data.utils.lmdb_io.LMDBReader(lmdb_path)
        try:
            metadata = reader.get_metadata()
            record_type = metadata.get("record_type")
            if record_type != "benchmark":
                raise ValueError(
                    f"LMDB at '{lmdb_path}' has record_type={record_type!r}, expected 'benchmark'; "
                    "ensure this is an eval export produced by DiskEvalLogger"
                )
            for record_idx in range(reader.sample_count):
                record = reader.get(record_idx)
                sample_id: str = record["sample_id"]
                attempt_index: int = record["attempt_index"]
                record_key = (sample_id, attempt_index)
                if record_key in seen_keys:
                    raise ValueError(
                        f"duplicate (sample_id, attempt_index) = ({sample_id!r}, {attempt_index}) across LMDB datasets"
                    )
                seen_keys.add(record_key)
                # extract label based on label_type
                if label_type == correctness_types.LabelType.HARD_MATCH:
                    label = bool(record["hard_match"])
                elif label_type == correctness_types.LabelType.SOFT_MATCH:
                    label = bool(record["soft_match"])
                else:
                    raise ValueError(f"unsupported label_type: {label_type!r}")
                model_output: str = record["model_output"]
                final_answer: str | None = record.get("final_answer")
                expected_output: str = record["expected_output"]
                code_type: str | None = record.get("code_type")
                if not code_type:
                    raise ValueError(
                        f"record {sample_id!r} (attempt {attempt_index}) has missing or empty code_type; "
                        f"ensure the LMDB export includes the code_type field"
                    )
                tags: list[str] = record.get("tags") or []
                difficulty_score = record.get("difficulty_score")
                if difficulty_score is not None:
                    try:
                        difficulty_score = float(difficulty_score)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"record {sample_id!r} (attempt {attempt_index}) has invalid difficulty_score "
                            f"{difficulty_score!r}; expected a numeric value"
                        ) from exc
                    seen_difficulty_scores = True
                else:
                    missing_difficulty_ids.append(sample_id)
                problem_id = correctness_splits.extract_problem_id(sample_id)
                eval_record = correctness_types.EvalRecord(
                    sample_id=sample_id,
                    problem_id=problem_id,
                    attempt_index=attempt_index,
                    model_output=model_output,
                    final_answer=final_answer,
                    expected_output=expected_output,
                    label=label,
                    code_type=code_type,
                    tags=tags,
                    record=record,
                    difficulty_score=difficulty_score,
                )
                records.append(eval_record)
        finally:
            reader.close()
    if seen_difficulty_scores and missing_difficulty_ids:
        sample_examples = ", ".join(repr(sample_id) for sample_id in missing_difficulty_ids[:5])
        detail = f" missing examples: {sample_examples}" if sample_examples else ""
        raise ValueError(
            "LMDB records have mixed difficulty_score availability; "
            "either include difficulty_score for all records or omit it entirely. "
            f"Missing count: {len(missing_difficulty_ids)}.{detail}"
        )
    return records
