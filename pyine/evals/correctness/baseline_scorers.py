"""Baseline (sanity-check) guardrail scorers for correctness evaluation.

Provides trivial scorers that satisfy the ``GuardrailScorer`` protocol and can be run
through the full evaluation pipeline to establish reference points (e.g. AUROC = 0.5).
"""

from __future__ import annotations

import hashlib
import struct
import typing

import pyine.evals.correctness.types as correctness_types


class ConstantScorer:
    """Scorer that returns a fixed value for every record.

    Useful as a degenerate baseline: with mixed-label data, a constant score produces AUROC = 0.5
    and well-defined (but trivial) threshold calibration.
    """

    def __init__(
        self,
        score_value: float = 0.5,
    ) -> None:
        """Initializes the scorer with a given constant score."""
        self._score_value = score_value

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        """Returns the constant score for every record."""
        return correctness_types.ScoringResult(
            scores=[self._score_value] * len(records),
        )

    def get_metadata(self) -> dict[str, typing.Any]:
        """Returns scorer metadata."""
        return {"scorer_type": "constant", "score_value": self._score_value}

    def get_verification_cost_unit(self) -> str | None:
        """No verification cost for this baseline."""
        return None


class UniformRandomScorer:
    """Scorer that assigns each record a deterministic pseudo-random score in [0, 1].

    Scores are derived by hashing each record's identity (seed, sample_id, attempt_index) with MD5,
    making them **order-independent**: the same record always receives the same score regardless of
    which call to ``score_records`` it appears in or the order of records within a call.

    Different seeds produce different scores for the same record, enabling independent
    replicas when used with ``evaluate_guardrail_replicas``.
    """

    def __init__(
        self,
        seed: int,
    ) -> None:
        """Initializes the random scorer with the given seed."""
        self._seed = seed

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        """Returns a deterministic pseudo-random score for each record."""
        scores = [self._hash_record(record) for record in records]
        return correctness_types.ScoringResult(scores=scores)

    def get_metadata(self) -> dict[str, typing.Any]:
        """Returns scorer metadata."""
        return {"scorer_type": "uniform_random", "seed": self._seed}

    def get_verification_cost_unit(self) -> str | None:
        """No verification cost for this baseline."""
        return None

    def _hash_record(
        self,
        record: correctness_types.EvalRecord,
    ) -> float:
        """Derive a float in [0, 1] from the record's identity and the scorer's seed."""
        key = f"{self._seed}:{record.sample_id}:{record.attempt_index}"
        digest = hashlib.md5(key.encode("utf-8")).digest()  # noqa: S324
        # interpret first 8 bytes as unsigned 64-bit int, normalize to [0, 1]
        int_val = struct.unpack("!Q", digest[:8])[0]
        return int_val / ((1 << 64) - 1)
