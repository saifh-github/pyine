"""Fake model implementations for integration testing of evaluation pipelines.

Provides three archetypes per pipeline (oracle, inverse-oracle, random) to verify that
end-to-end metric values match expectations under controlled, deterministic conditions.
"""

from __future__ import annotations

import typing

import langchain_core.messages
import numpy as np

import pyine.evals.common
import pyine.evals.correctness.types as correctness_types

# ---------------------------------------------------------------------------
# CODE_EXEC fake chains (duck-typed LangChain Runnables)
# ---------------------------------------------------------------------------


class OracleCodeExecChain:
    """Returns the expected output verbatim (perfect predictions)."""

    # required by pipeline when num_attempts_per_sample > 1
    temperature = pyine.evals.common.PASS_AT_K_DEFAULTS.temperature

    def invoke(
        self,
        sample_dict: dict[str, typing.Any],
        **kwargs: typing.Any,
    ) -> langchain_core.messages.AIMessage:
        return langchain_core.messages.AIMessage(
            content=str(sample_dict["expected_output"]),
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


class InverseOracleCodeExecChain:
    """Returns the expected output with a ``_WRONG`` suffix (always incorrect)."""

    temperature = pyine.evals.common.PASS_AT_K_DEFAULTS.temperature

    def invoke(
        self,
        sample_dict: dict[str, typing.Any],
        **kwargs: typing.Any,
    ) -> langchain_core.messages.AIMessage:
        return langchain_core.messages.AIMessage(
            content=str(sample_dict["expected_output"]) + "_WRONG",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


class RandomCodeExecChain:
    """Returns the correct answer ~50% of the time (seeded RNG)."""

    temperature = pyine.evals.common.PASS_AT_K_DEFAULTS.temperature

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def invoke(
        self,
        sample_dict: dict[str, typing.Any],
        **kwargs: typing.Any,
    ) -> langchain_core.messages.AIMessage:
        expected = str(sample_dict["expected_output"])
        content = expected if self._rng.random() < 0.5 else expected + "_WRONG"
        return langchain_core.messages.AIMessage(
            content=content,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


# ---------------------------------------------------------------------------
# CORRECTNESS fake guardrail scorers (GuardrailScorer protocol)
# ---------------------------------------------------------------------------


class OracleGuardrailScorer:
    """Returns 1.0 for correct records, 0.0 for incorrect (perfect separation).

    Reports a fixed verification cost of 1.0 per record.
    """

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        return correctness_types.ScoringResult(
            scores=[1.0 if rec.label else 0.0 for rec in records],
            verification_costs=[1.0] * len(records),
        )

    def get_metadata(self) -> dict[str, typing.Any]:
        return {"name": "oracle"}

    def get_verification_cost_unit(self) -> str | None:
        return "tokens"


class InverseOracleGuardrailScorer:
    """Returns 0.0 for correct records, 1.0 for incorrect (perfectly anti-correlated).

    Reports a fixed verification cost of 2.0 per record.
    """

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        return correctness_types.ScoringResult(
            scores=[0.0 if rec.label else 1.0 for rec in records],
            verification_costs=[2.0] * len(records),
        )

    def get_metadata(self) -> dict[str, typing.Any]:
        return {"name": "inverse_oracle"}

    def get_verification_cost_unit(self) -> str | None:
        return "tokens"


class RandomGuardrailScorer:
    """Returns uniform random scores in [0, 1] (seeded, stateful RNG).

    The RNG is persisted across ``score_records`` calls so that calibration and eval
    receive different scores (matching how a real stochastic scorer would behave).
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        scores = [float(self._rng.random()) for _ in records]
        costs = [float(self._rng.random()) * 5.0 for _ in records]
        return correctness_types.ScoringResult(scores=scores, verification_costs=costs)

    def get_metadata(self) -> dict[str, typing.Any]:
        return {"name": "random", "seed": self._seed}

    def get_verification_cost_unit(self) -> str | None:
        return "tokens"
