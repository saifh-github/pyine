"""Unit tests for DebateGuardrailScorer.

All LLM chain and graph calls are mocked - no real API calls are made.
"""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import pytest

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
    reasoning: str = "Good reasoning",
) -> dict[str, typing.Any]:
    """Build a final DebateState dict as returned by graph.invoke()."""
    messages = []
    for i in range(num_turns):
        messages.append(
            DebateMessage(
                role=DebateRole.INTERROGATOR,
                content=f"Question {i + 1}",
                token_count=100.0,
            )
        )
        messages.append(
            DebateMessage(
                role=DebateRole.RESPONDER,
                content=f"Answer {i + 1}",
                token_count=150.0,
            )
        )
    return {
        "original_prompt": "test prompt",
        "responder_output": "test output",
        "final_answer": "test answer",
        "max_turns": 3,
        "messages": messages,
        "current_turn": num_turns,
        "verdict": DebateVerdict(score=score, reasoning=reasoning),
        "total_tokens": total_tokens,
    }


def _build_scorer_with_mock_graph(
    config: DebateGuardrailConfig | None = None,
    graph_invoke_side_effect: typing.Any = None,
    graph_invoke_return_value: typing.Any = None,
) -> tuple[typing.Any, MagicMock]:
    """Build a DebateGuardrailScorer with mocked LLM, chain, and graph.

    Returns (scorer, mock_graph) so tests can inspect/configure the mock.
    """
    if config is None:
        config = make_debate_config()

    mock_graph = MagicMock()
    if graph_invoke_side_effect is not None:
        mock_graph.invoke.side_effect = graph_invoke_side_effect
    elif graph_invoke_return_value is not None:
        mock_graph.invoke.return_value = graph_invoke_return_value
    else:
        mock_graph.invoke.return_value = _make_final_graph_state()

    with (
        patch.object(
            pyine.utils.llm_providers.LLMProviderConfig,
            "get_model",
            return_value=MagicMock(),
        ),
        patch(
            "pyine.prompts.manager.get_prompt_chain",
            return_value=MagicMock(),
        ),
        patch(
            "pyine.guardrails.llm_debate.scorer.build_debate_graph",
            return_value=mock_graph,
        ),
    ):
        from pyine.guardrails.llm_debate.scorer import DebateGuardrailScorer

        scorer = DebateGuardrailScorer(config)
    return scorer, mock_graph


# ---------------------------------------------------------------------------
# Tests: score_records shape
# ---------------------------------------------------------------------------


class TestScoreRecordsShape:
    def test_returns_correct_shape(self) -> None:
        scorer, _ = _build_scorer_with_mock_graph()
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(5)]
        result = scorer.score_records(records)
        assert len(result.scores) == 5
        assert result.verification_costs is not None
        assert len(result.verification_costs) == 5

    def test_empty_list(self) -> None:
        scorer, _ = _build_scorer_with_mock_graph()
        result = scorer.score_records([])
        assert len(result.scores) == 0
        assert result.verification_costs is not None
        assert len(result.verification_costs) == 0

    def test_single_record(self) -> None:
        scorer, _ = _build_scorer_with_mock_graph()
        result = scorer.score_records([make_eval_record()])
        assert len(result.scores) == 1


# ---------------------------------------------------------------------------
# Tests: score values
# ---------------------------------------------------------------------------


class TestScoreRecordsValues:
    def test_scores_match_verdict(self) -> None:
        """Scores should match the verdict scores from graph invoke."""
        scores = [0.1, 0.5, 0.9]
        states = [_make_final_graph_state(score=s) for s in scores]
        scorer, _ = _build_scorer_with_mock_graph(graph_invoke_side_effect=states)
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(3)]
        result = scorer.score_records(records)
        for expected, actual in zip(scores, result.scores, strict=True):
            assert actual == pytest.approx(expected)

    def test_scores_in_range(self) -> None:
        scorer, _ = _build_scorer_with_mock_graph()
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(5)]
        result = scorer.score_records(records)
        for score in result.scores:
            assert 0.0 <= score <= 1.0

    def test_costs_from_token_count(self) -> None:
        """Verification costs should reflect total_tokens from graph state."""
        state = _make_final_graph_state(total_tokens=1234.0)
        scorer, _ = _build_scorer_with_mock_graph(graph_invoke_return_value=state)
        records = [make_eval_record()]
        result = scorer.score_records(records)
        assert result.verification_costs is not None
        assert result.verification_costs[0] == pytest.approx(1234.0)

    def test_costs_non_negative(self) -> None:
        scorer, _ = _build_scorer_with_mock_graph()
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(3)]
        result = scorer.score_records(records)
        assert result.verification_costs is not None
        for cost in result.verification_costs:
            assert cost >= 0.0


# ---------------------------------------------------------------------------
# Tests: attempt_metadata
# ---------------------------------------------------------------------------


class TestAttemptMetadata:
    def test_attempt_metadata_contains_transcript(self) -> None:
        scorer, _ = _build_scorer_with_mock_graph()
        record = make_eval_record()
        result = scorer.score_records([record])
        assert result.attempt_metadata is not None
        assert len(result.attempt_metadata) == 1
        # Metadata should contain the debate transcript as a dict
        key = next(iter(result.attempt_metadata))
        metadata = result.attempt_metadata[key]
        assert "messages" in metadata or "verdict" in metadata

    def test_attempt_metadata_length_matches_records(self) -> None:
        scorer, _ = _build_scorer_with_mock_graph()
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(4)]
        result = scorer.score_records(records)
        assert result.attempt_metadata is not None
        assert len(result.attempt_metadata) == 4


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_error_uses_default_score(self) -> None:
        default_score = 0.3
        config = make_debate_config(default_score_on_error=default_score)
        scorer, _ = _build_scorer_with_mock_graph(
            config=config,
            graph_invoke_side_effect=RuntimeError("Graph execution failed"),
        )
        records = [make_eval_record()]
        result = scorer.score_records(records)
        assert result.scores[0] == default_score

    def test_error_count_incremented(self) -> None:
        scorer, _ = _build_scorer_with_mock_graph(
            graph_invoke_side_effect=RuntimeError("API error"),
        )
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(3)]
        scorer.score_records(records)
        metadata = scorer.get_metadata()
        assert metadata["error_count"] == 3

    def test_error_metadata_contains_error_info(self) -> None:
        """Error records should return metadata with error details, not empty {}."""
        scorer, _ = _build_scorer_with_mock_graph(
            graph_invoke_side_effect=RuntimeError("Graph execution failed"),
        )
        records = [make_eval_record()]
        result = scorer.score_records(records)
        assert result.attempt_metadata is not None
        key = next(iter(result.attempt_metadata))
        metadata = result.attempt_metadata[key]
        assert "error" in metadata
        assert "error_type" in metadata
        assert metadata["error_type"] == "RuntimeError"
        assert "Graph execution failed" in metadata["error"]

    def test_error_count_accumulates_across_calls(self) -> None:
        scorer, _ = _build_scorer_with_mock_graph(
            graph_invoke_side_effect=RuntimeError("Error"),
        )
        scorer.score_records([make_eval_record(sample_id="TEST/VALID/p000000/s0000/t0000")])
        scorer.score_records([make_eval_record(sample_id="TEST/VALID/p000001/s0000/t0000")])
        metadata = scorer.get_metadata()
        assert metadata["error_count"] == 2


# ---------------------------------------------------------------------------
# Tests: Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_metadata_contains_required_fields(self) -> None:
        config = make_debate_config()
        scorer, _ = _build_scorer_with_mock_graph(config=config)
        metadata = scorer.get_metadata()
        required_keys = {
            "scorer_type",
            "max_debate_turns",
            "responder_sees_debate_history",
            "total_scored",
            "error_count",
        }
        assert required_keys.issubset(set(metadata.keys()))
        assert metadata["scorer_type"] == "llm_debate"
        assert metadata["max_debate_turns"] == 2
        assert metadata["responder_sees_debate_history"] is True
        assert metadata["total_scored"] == 0
        assert metadata["error_count"] == 0

    def test_metadata_total_scored_updates(self) -> None:
        scorer, _ = _build_scorer_with_mock_graph()
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(4)]
        scorer.score_records(records)
        assert scorer.get_metadata()["total_scored"] == 4
        scorer.score_records(records[:2])
        assert scorer.get_metadata()["total_scored"] == 6

    def test_metadata_responder_sees_debate_history_field(self) -> None:
        """The responder_sees_debate_history config value should be in metadata."""
        config_true = make_debate_config(responder_sees_debate_history=True)
        scorer_true, _ = _build_scorer_with_mock_graph(config=config_true)
        assert scorer_true.get_metadata()["responder_sees_debate_history"] is True

        config_false = make_debate_config(responder_sees_debate_history=False)
        scorer_false, _ = _build_scorer_with_mock_graph(config=config_false)
        assert scorer_false.get_metadata()["responder_sees_debate_history"] is False


# ---------------------------------------------------------------------------
# Tests: Verification cost unit
# ---------------------------------------------------------------------------


class TestVerificationCostUnit:
    def test_returns_tokens(self) -> None:
        scorer, _ = _build_scorer_with_mock_graph()
        assert scorer.get_verification_cost_unit() == "tokens"


# ---------------------------------------------------------------------------
# Tests: Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_has_required_methods(self) -> None:
        scorer, _ = _build_scorer_with_mock_graph()
        assert callable(getattr(scorer, "score_records", None))
        assert callable(getattr(scorer, "get_metadata", None))
        assert callable(getattr(scorer, "get_verification_cost_unit", None))

    def test_method_signatures(self) -> None:
        import inspect

        scorer, _ = _build_scorer_with_mock_graph()
        score_sig = inspect.signature(scorer.score_records)
        assert "records" in score_sig.parameters

        meta_sig = inspect.signature(scorer.get_metadata)
        assert len(meta_sig.parameters) == 0

        cost_sig = inspect.signature(scorer.get_verification_cost_unit)
        assert len(cost_sig.parameters) == 0
