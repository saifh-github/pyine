"""Integration test for the full debate graph execution with mocked chains.

Verifies the complete flow from initial state through multi-turn debate to
final transcript structure, including token accumulation and message ordering.
"""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import pytest

from pyine.guardrails.llm_debate.graph import build_debate_graph
from pyine.guardrails.llm_debate.types import (
    DebateRole,
    DebateTranscript,
)
from pyine.prompts.configs.guardrail.debate_interrogator import InterrogatorOutput, VerdictOutput

_EXTRACT_TOKENS_TARGET = (  # noqa: S105
    "pyine.guardrails.llm_debate.graph.pyine.utils.langchain.extract_token_count_from_handler"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_initial_state(max_turns: int = 3) -> dict[str, typing.Any]:
    """Build the initial DebateState dict."""
    return {
        "original_prompt": "What is the output of:\n```python\nfor i in range(3): print(i)\n```",
        "responder_output": "The output is:\n0\n1\n2",
        "final_answer": "0\n1\n2",
        "max_turns": max_turns,
        "messages": [],
        "current_turn": 0,
        "verdict": None,
        "total_tokens": 0.0,
    }


# ---------------------------------------------------------------------------
# Integration: Full multi-turn debate
# ---------------------------------------------------------------------------


class TestFullMultiTurnDebate:
    """Full graph execution with mocked chains verifying transcript structure."""

    def test_three_turn_debate_produces_valid_transcript(self) -> None:
        """Run a 3-turn debate and verify the final state can build a DebateTranscript."""
        interrogator_responses = [
            InterrogatorOutput(
                decision="question",
                content="How does the range(3) function work in this context?",
                score=None,
                reasoning=None,
            ),
            InterrogatorOutput(
                decision="question",
                content="What would happen if range(3) were replaced with range(5)?",
                score=None,
                reasoning=None,
            ),
            InterrogatorOutput(
                decision="question",
                content="Are you certain about the newline separation?",
                score=None,
                reasoning=None,
            ),
            InterrogatorOutput(
                decision="verdict",
                content="The model demonstrates clear understanding.",
                score=0.92,
                reasoning="Accurate reasoning about print() and range() behavior.",
            ),
        ]
        responder_responses = [
            "range(3) generates [0, 1, 2] and each value is printed on its own line.",
            "With range(5) the output would be 0 through 4, each on a new line.",
            "Yes, print() adds a newline by default after each call.",
        ]

        mock_interr = MagicMock()
        mock_interr.invoke.side_effect = interrogator_responses
        mock_resp = MagicMock()
        mock_resp.invoke.side_effect = responder_responses

        with patch(_EXTRACT_TOKENS_TARGET, return_value=100.0):
            graph = build_debate_graph(mock_interr, mock_resp, responder_sees_debate_history=True)
            final_state = graph.invoke(_make_initial_state(max_turns=3))

        # Verify final state structure
        assert final_state["verdict"] is not None
        assert final_state["verdict"].score == 0.92
        assert final_state["current_turn"] == 3

        # Build transcript from final state (same as scorer does)
        transcript = DebateTranscript(
            messages=final_state["messages"],
            verdict=final_state["verdict"],
            num_turns=final_state["current_turn"],
            total_token_count=final_state["total_tokens"],
        )

        # Verify transcript structure
        assert transcript.num_turns == 3
        # 3 questions + 3 answers + 1 verdict message = 7 messages
        assert len(transcript.messages) == 7
        assert transcript.verdict.score == 0.92
        assert transcript.verdict.reasoning is not None
        assert transcript.total_token_count == pytest.approx(7 * 100.0)

        # Verify message ordering: alternating interrogator/responder
        for i, msg in enumerate(transcript.messages[:-1]):  # exclude verdict message
            expected_role = DebateRole.INTERROGATOR if i % 2 == 0 else DebateRole.RESPONDER
            assert msg.role == expected_role, f"Message {i} has wrong role"

        # Verify the transcript can be serialized
        data = transcript.model_dump()
        assert isinstance(data, dict)
        restored = DebateTranscript.model_validate(data)
        assert restored.verdict.score == transcript.verdict.score

    def test_single_turn_debate_transcript(self) -> None:
        """Single turn debate produces valid transcript."""
        mock_interr = MagicMock()
        mock_interr.invoke.side_effect = [
            InterrogatorOutput(
                decision="question",
                content="Can you verify your answer?",
                score=None,
                reasoning=None,
            ),
            InterrogatorOutput(
                decision="verdict",
                content="Looks correct.",
                score=0.75,
                reasoning="Reasonable response.",
            ),
        ]
        mock_resp = MagicMock()
        mock_resp.invoke.side_effect = ["Yes, the output is definitely 0, 1, 2."]

        with patch(_EXTRACT_TOKENS_TARGET, return_value=50.0):
            graph = build_debate_graph(mock_interr, mock_resp)
            final_state = graph.invoke(_make_initial_state(max_turns=5))

        transcript = DebateTranscript(
            messages=final_state["messages"],
            verdict=final_state["verdict"],
            num_turns=final_state["current_turn"],
            total_token_count=final_state["total_tokens"],
        )
        assert transcript.num_turns == 1
        # 1 question + 1 answer + 1 verdict message = 3
        assert len(transcript.messages) == 3
        assert transcript.verdict.score == 0.75


class TestHistoryVisibilityIntegration:
    """Integration test comparing debate with and without history visibility."""

    def test_history_affects_responder_input(self) -> None:
        """Run the same debate with history=True and False, verify responder sees different inputs."""
        interrogator_responses = [
            InterrogatorOutput(decision="question", content="Q1", score=None, reasoning=None),
            InterrogatorOutput(decision="question", content="Q2", score=None, reasoning=None),
            InterrogatorOutput(decision="verdict", content="Done", score=0.5, reasoning="OK"),
        ]
        responder_responses = ["A1", "A2"]

        results: dict[bool, list[dict[str, typing.Any]]] = {}

        for sees_history in [True, False]:
            mock_interr = MagicMock()
            mock_interr.invoke.side_effect = list(interrogator_responses)  # copy
            mock_resp = MagicMock()
            mock_resp.invoke.side_effect = list(responder_responses)  # copy

            with patch(_EXTRACT_TOKENS_TARGET, return_value=10.0):
                graph = build_debate_graph(mock_interr, mock_resp, responder_sees_debate_history=sees_history)
                graph.invoke(_make_initial_state(max_turns=3))

            # Collect the input_vars from all responder calls
            resp_inputs = [call[0][0] for call in mock_resp.invoke.call_args_list]
            results[sees_history] = resp_inputs

        # With history=True, second responder call should have non-empty debate_history
        assert results[True][1]["debate_history"] != ""
        # With history=False, all responder calls should have empty debate_history
        for inp in results[False]:
            assert inp["debate_history"] == ""


class TestTokenAccumulationIntegration:
    """Verify token accumulation across a full debate run."""

    def test_token_total_matches_sum_of_calls(self) -> None:
        """Total tokens should equal sum of per-call token counts."""
        call_count = 0

        def varying_token_count(handler: typing.Any) -> float:
            nonlocal call_count
            call_count += 1
            return float(call_count * 100)  # 100, 200, 300, ...

        mock_interr = MagicMock()
        mock_interr.invoke.side_effect = [
            InterrogatorOutput(decision="question", content="Q1", score=None, reasoning=None),
            InterrogatorOutput(decision="verdict", content="Done", score=0.8, reasoning="OK"),
        ]
        mock_resp = MagicMock()
        mock_resp.invoke.side_effect = ["A1"]

        with patch(_EXTRACT_TOKENS_TARGET, side_effect=varying_token_count):
            graph = build_debate_graph(mock_interr, mock_resp)
            final_state = graph.invoke(_make_initial_state(max_turns=3))

        # 3 calls: 100 + 200 + 300 = 600
        assert final_state["total_tokens"] == pytest.approx(600.0)
        # Each message should have its own token count
        assert final_state["messages"][0].token_count == pytest.approx(100.0)  # interr Q1
        assert final_state["messages"][1].token_count == pytest.approx(200.0)  # resp A1
        assert final_state["messages"][2].token_count == pytest.approx(300.0)  # interr verdict


class TestVerdictChainIntegration:
    """Integration test exercising the verdict chain path in a multi-turn debate."""

    def test_multi_turn_debate_with_verdict_chain(self) -> None:
        """Run a 3-turn debate where the verdict chain is used on the final turn.

        Verifies the full flow: interrogator asks questions for max_turns rounds,
        then the verdict chain is called (not the regular interrogator), producing
        a valid transcript with the verdict chain's score.
        """
        max_turns = 3
        interrogator_responses = [
            InterrogatorOutput(
                decision="question",
                content="How does range(3) work?",
                score=None,
                reasoning=None,
            ),
            InterrogatorOutput(
                decision="question",
                content="What about the newline behavior?",
                score=None,
                reasoning=None,
            ),
            InterrogatorOutput(
                decision="question",
                content="Are you sure about the separator?",
                score=None,
                reasoning=None,
            ),
        ]
        responder_responses = [
            "range(3) generates 0, 1, 2.",
            "Each print() adds a newline.",
            "Yes, print uses newline by default.",
        ]

        verdict_output = VerdictOutput(
            score=0.88,
            reasoning="Accurate understanding demonstrated across all turns.",
            content="The model clearly understands range() and print() behavior.",
        )

        mock_interr = MagicMock()
        mock_interr.invoke.side_effect = interrogator_responses
        mock_resp = MagicMock()
        mock_resp.invoke.side_effect = responder_responses
        mock_verdict = MagicMock()
        mock_verdict.invoke.return_value = verdict_output

        with patch(_EXTRACT_TOKENS_TARGET, return_value=100.0):
            graph = build_debate_graph(
                mock_interr,
                mock_resp,
                interrogator_verdict_chain=mock_verdict,
                responder_sees_debate_history=True,
            )
            final_state = graph.invoke(_make_initial_state(max_turns=max_turns))

        # Verify final state
        assert final_state["verdict"] is not None
        assert final_state["verdict"].score == 0.88
        assert final_state["verdict"].reasoning == "Accurate understanding demonstrated across all turns."
        assert final_state["current_turn"] == max_turns

        # Build transcript from final state (same as scorer does)
        transcript = DebateTranscript(
            messages=final_state["messages"],
            verdict=final_state["verdict"],
            num_turns=final_state["current_turn"],
            total_token_count=final_state["total_tokens"],
        )

        # 3 questions + 3 answers + 1 verdict message = 7 messages
        assert len(transcript.messages) == 7
        assert transcript.num_turns == 3
        assert transcript.verdict.score == 0.88

        # Regular interrogator called 3 times (questions only), verdict chain once
        assert mock_interr.invoke.call_count == 3
        assert mock_verdict.invoke.call_count == 1
        assert mock_resp.invoke.call_count == 3

        # Last message should be from the verdict chain (interrogator role)
        assert transcript.messages[-1].role == DebateRole.INTERROGATOR
        assert transcript.messages[-1].content == "The model clearly understands range() and print() behavior."

        # Verify the transcript can be serialized
        data = transcript.model_dump()
        restored = DebateTranscript.model_validate(data)
        assert restored.verdict.score == transcript.verdict.score
