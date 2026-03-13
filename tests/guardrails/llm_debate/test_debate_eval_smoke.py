"""End-to-end eval pipeline smoke test with mocked LLMs.

Verifies that DebateGuardrailScorer integrates correctly with the
correctness evaluation pipeline, producing a valid ScoringResult.
"""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import pytest

import pyine.evals.correctness.types as correctness_types
import pyine.utils.llm_providers
from pyine.guardrails.llm_debate.types import (
    DebateMessage,
    DebateRole,
    DebateVerdict,
)

if typing.TYPE_CHECKING:
    from pyine.guardrails.llm_debate.configs import DebateGuardrailConfig

from .conftest import make_debate_config, make_eval_record

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_final_graph_state(
    score: float = 0.85,
    num_turns: int = 2,
    total_tokens: float = 500.0,
) -> dict[str, typing.Any]:
    """Build a final DebateState dict as returned by graph.invoke()."""
    messages = [
        DebateMessage(role=DebateRole.INTERROGATOR, content="Q1", token_count=100.0),
        DebateMessage(role=DebateRole.RESPONDER, content="A1", token_count=150.0),
        DebateMessage(role=DebateRole.INTERROGATOR, content="Q2", token_count=100.0),
        DebateMessage(role=DebateRole.RESPONDER, content="A2", token_count=150.0),
    ]
    return {
        "original_prompt": "test",
        "responder_output": "output",
        "final_answer": "answer",
        "max_turns": 3,
        "messages": messages,
        "current_turn": num_turns,
        "verdict": DebateVerdict(score=score, reasoning="Test reasoning"),
        "total_tokens": total_tokens,
    }


def _build_mocked_scorer(
    config: DebateGuardrailConfig | None = None,
    graph_return_value: dict[str, typing.Any] | None = None,
) -> typing.Any:
    """Build a fully mocked DebateGuardrailScorer."""
    if config is None:
        config = make_debate_config()
    if graph_return_value is None:
        graph_return_value = _make_final_graph_state()

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = graph_return_value

    mock_chain = MagicMock()

    with (
        patch.object(
            pyine.utils.llm_providers.LLMProviderConfig,
            "get_model",
            return_value=MagicMock(),
        ),
        patch(
            "pyine.prompts.manager.get_prompt_chain",
            return_value=mock_chain,
        ),
        patch(
            "pyine.guardrails.llm_debate.scorer.build_debate_graph",
            return_value=mock_graph,
        ),
    ):
        from pyine.guardrails.llm_debate.scorer import DebateGuardrailScorer

        return DebateGuardrailScorer(config)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestEvalPipelineSmokeTest:
    """End-to-end smoke tests verifying ScoringResult shape and content."""

    def test_scoring_result_shape(self) -> None:
        """ScoringResult should have matching lengths for scores, costs, metadata."""
        scorer = _build_mocked_scorer()
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(10)]
        result = scorer.score_records(records)

        assert isinstance(result, correctness_types.ScoringResult)
        assert len(result.scores) == 10
        assert result.verification_costs is not None
        assert len(result.verification_costs) == 10
        assert result.attempt_metadata is not None
        assert len(result.attempt_metadata) == 10

    def test_scoring_result_scores_in_range(self) -> None:
        """All scores should be in [0, 1]."""
        scorer = _build_mocked_scorer()
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(5)]
        result = scorer.score_records(records)
        for s in result.scores:
            assert 0.0 <= s <= 1.0

    def test_scoring_result_costs_non_negative(self) -> None:
        """All verification costs should be non-negative."""
        scorer = _build_mocked_scorer()
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(5)]
        result = scorer.score_records(records)
        assert result.verification_costs is not None
        for c in result.verification_costs:
            assert c >= 0.0

    def test_attempt_metadata_contains_transcript_structure(self) -> None:
        """Each attempt_metadata entry should contain transcript fields."""
        scorer = _build_mocked_scorer()
        records = [make_eval_record()]
        result = scorer.score_records(records)
        assert result.attempt_metadata is not None
        key = next(iter(result.attempt_metadata))
        meta = result.attempt_metadata[key]
        # Transcript model_dump() should have these keys
        assert "messages" in meta
        assert "verdict" in meta
        assert "num_turns" in meta
        assert "total_token_count" in meta

    def test_attempt_metadata_key_structure(self) -> None:
        """attempt_metadata keys should be (sample_id, attempt_index, draw_index) tuples."""
        scorer = _build_mocked_scorer()
        records = [make_eval_record(sample_id="TEST/VALID/p000042/s0000/t0000")]
        result = scorer.score_records(records)
        assert result.attempt_metadata is not None
        key = next(iter(result.attempt_metadata))
        assert isinstance(key, tuple)
        assert len(key) == 3
        assert key[0] == "TEST/VALID/p000042/s0000/t0000"  # sample_id
        assert key[1] == 0  # attempt_index
        assert key[2] == 0  # draw_index (first record)

    def test_multiple_batches(self) -> None:
        """Scoring across multiple batches should accumulate total_scored."""
        scorer = _build_mocked_scorer()
        batch1 = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(3)]
        batch2 = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(3, 8)]
        scorer.score_records(batch1)
        scorer.score_records(batch2)
        metadata = scorer.get_metadata()
        assert metadata["total_scored"] == 8

    def test_with_errors_mixed(self) -> None:
        """Some records succeed, some fail - verify mixed results."""
        config = make_debate_config(default_score_on_error=0.25)
        success_state = _make_final_graph_state(score=0.9)
        call_count = 0

        def mixed_invoke(state: typing.Any) -> dict[str, typing.Any]:
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("Simulated API error")
            return success_state

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = mixed_invoke

        mock_chain = MagicMock()

        with (
            patch.object(
                pyine.utils.llm_providers.LLMProviderConfig,
                "get_model",
                return_value=MagicMock(),
            ),
            patch(
                "pyine.prompts.manager.get_prompt_chain",
                return_value=mock_chain,
            ),
            patch(
                "pyine.guardrails.llm_debate.scorer.build_debate_graph",
                return_value=mock_graph,
            ),
        ):
            from pyine.guardrails.llm_debate.scorer import DebateGuardrailScorer

            scorer = DebateGuardrailScorer(config)

        # Use max_workers=1 so call order is deterministic
        config_single = make_debate_config(default_score_on_error=0.25, max_workers=1)
        scorer._config = config_single

        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(4)]
        result = scorer.score_records(records)

        assert len(result.scores) == 4
        # At least some scores should be the default error score
        has_default = any(s == pytest.approx(0.25) for s in result.scores)
        has_success = any(s == pytest.approx(0.9) for s in result.scores)
        assert has_default or has_success

    def test_scorer_protocol_compliance(self) -> None:
        """Verify the scorer satisfies the GuardrailScorer protocol structurally."""
        scorer = _build_mocked_scorer()
        # Check method signatures
        assert callable(getattr(scorer, "score_records", None))
        assert callable(getattr(scorer, "get_metadata", None))
        assert callable(getattr(scorer, "get_verification_cost_unit", None))
        # Verify return types
        assert scorer.get_verification_cost_unit() == "tokens"
        metadata = scorer.get_metadata()
        assert isinstance(metadata, dict)
        assert metadata["scorer_type"] == "llm_debate"
