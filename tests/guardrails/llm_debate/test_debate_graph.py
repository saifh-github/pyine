"""Unit tests for the LangGraph debate state machine (graph.py).

Tests graph construction, state transitions, and debate history visibility
modes using mocked LLM chains.
"""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import pytest

from pyine.guardrails.llm_debate.types import (
    DebateMessage,
    DebateRole,
)
from pyine.prompts.configs.guardrail.debate_interrogator import InterrogatorOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Mock target for the token-extraction utility used inside graph nodes.
_EXTRACT_TOKENS_TARGET = (  # noqa: S105
    "pyine.guardrails.llm_debate.graph.pyine.utils.langchain.extract_token_count_from_handler"
)


def _make_interrogator_question(content: str = "Why did you predict that?") -> InterrogatorOutput:
    """Build an InterrogatorOutput for a probing question."""
    return InterrogatorOutput(
        decision="question",
        content=content,
        score=None,
        reasoning=None,
    )


def _make_interrogator_verdict(score: float = 0.85, reasoning: str = "Good") -> InterrogatorOutput:
    """Build an InterrogatorOutput for a final verdict."""
    return InterrogatorOutput(
        decision="verdict",
        content=reasoning,
        score=score,
        reasoning=reasoning,
    )


def _make_initial_state(max_turns: int = 3) -> dict[str, typing.Any]:
    """Build the initial DebateState dict for graph invocation."""
    return {
        "original_prompt": "Predict the output:\nprint([1,2,3])",
        "responder_output": "The output is [1, 2, 3]",
        "final_answer": "[1, 2, 3]",
        "max_turns": max_turns,
        "messages": [],
        "current_turn": 0,
        "verdict": None,
        "total_tokens": 0.0,
    }


def _build_mock_chains(
    interrogator_responses: list[InterrogatorOutput],
    responder_responses: list[str],
) -> tuple[MagicMock, MagicMock]:
    """Build mocked interrogator and responder chains.

    The mocks accept (input_vars, config=...) matching the real invoke signature.
    """
    mock_interrogator_chain = MagicMock()
    mock_interrogator_chain.invoke.side_effect = interrogator_responses

    mock_responder_chain = MagicMock()
    mock_responder_chain.invoke.side_effect = responder_responses

    return mock_interrogator_chain, mock_responder_chain


def _build_and_invoke(
    interrogator_responses: list[InterrogatorOutput],
    responder_responses: list[str],
    max_turns: int = 3,
    responder_sees_debate_history: bool = True,
    token_count_return: float = 50.0,
) -> tuple[dict[str, typing.Any], MagicMock, MagicMock]:
    """Build graph, invoke with mocked chains, return (final_state, mock_interr, mock_resp)."""
    from pyine.guardrails.llm_debate.graph import build_debate_graph

    mock_interr, mock_resp = _build_mock_chains(interrogator_responses, responder_responses)

    with patch(_EXTRACT_TOKENS_TARGET, return_value=token_count_return):
        graph = build_debate_graph(
            mock_interr,
            mock_resp,
            responder_sees_debate_history=responder_sees_debate_history,
        )
        initial_state = _make_initial_state(max_turns=max_turns)
        final_state = graph.invoke(initial_state)

    return final_state, mock_interr, mock_resp


# ---------------------------------------------------------------------------
# Tests: Graph construction
# ---------------------------------------------------------------------------


class TestGraphConstruction:
    def test_build_debate_graph_returns_compiled(self) -> None:
        from pyine.guardrails.llm_debate.graph import build_debate_graph

        mock_interr, mock_resp = _build_mock_chains([], [])
        graph = build_debate_graph(mock_interr, mock_resp)
        assert callable(getattr(graph, "invoke", None))

    def test_build_debate_graph_with_history_flag(self) -> None:
        from pyine.guardrails.llm_debate.graph import build_debate_graph

        mock_interr, mock_resp = _build_mock_chains([], [])
        graph_t = build_debate_graph(mock_interr, mock_resp, responder_sees_debate_history=True)
        graph_f = build_debate_graph(mock_interr, mock_resp, responder_sees_debate_history=False)
        assert callable(getattr(graph_t, "invoke", None))
        assert callable(getattr(graph_f, "invoke", None))


# ---------------------------------------------------------------------------
# Tests: State transitions
# ---------------------------------------------------------------------------


class TestStateTransitions:
    def test_immediate_verdict_on_first_turn(self) -> None:
        """Interrogator renders verdict immediately (no debate rounds)."""
        final_state, mock_interr, mock_resp = _build_and_invoke(
            [_make_interrogator_verdict(score=0.9, reasoning="Obviously correct")],
            [],
        )
        assert final_state["verdict"] is not None
        assert final_state["verdict"].score == 0.9
        assert final_state["current_turn"] == 0  # no responder turns
        assert mock_resp.invoke.call_count == 0

    def test_single_turn_then_verdict(self) -> None:
        """One round of Q&A followed by verdict."""
        final_state, mock_interr, mock_resp = _build_and_invoke(
            [
                _make_interrogator_question("Explain your reasoning"),
                _make_interrogator_verdict(score=0.8),
            ],
            ["I predicted [1,2,3] because the print statement..."],
        )
        assert final_state["verdict"] is not None
        assert final_state["verdict"].score == 0.8
        assert final_state["current_turn"] == 1
        assert mock_interr.invoke.call_count == 2
        assert mock_resp.invoke.call_count == 1

    def test_max_turns_forces_verdict(self) -> None:
        """Interrogator asks questions until max_turns, then must render verdict."""
        max_turns = 2
        final_state, mock_interr, mock_resp = _build_and_invoke(
            [
                _make_interrogator_question("Question 1"),
                _make_interrogator_question("Question 2"),
                _make_interrogator_verdict(score=0.6, reasoning="Forced verdict"),
            ],
            ["Answer 1", "Answer 2"],
            max_turns=max_turns,
        )
        assert final_state["verdict"] is not None
        assert final_state["current_turn"] == max_turns
        assert mock_interr.invoke.call_count == 3  # 2 questions + 1 forced verdict
        assert mock_resp.invoke.call_count == 2

    def test_forced_verdict_fallback_when_llm_disobeys(self) -> None:
        """When the interrogator returns 'question' on the forced-verdict turn,
        the graph should programmatically override with a default verdict."""
        max_turns = 2
        final_state, mock_interr, mock_resp = _build_and_invoke(
            [
                _make_interrogator_question("Q1"),
                _make_interrogator_question("Q2"),
                # LLM disobeys: returns question instead of verdict at forced-verdict turn
                _make_interrogator_question("Q3 (should be overridden)"),
            ],
            ["A1", "A2"],
            max_turns=max_turns,
        )
        # The graph should have forced a verdict instead of continuing
        assert final_state["verdict"] is not None
        assert final_state["verdict"].score == 0.5  # default forced score
        assert final_state["current_turn"] == max_turns
        # Interrogator called 3 times, responder only 2 (not called after forced verdict)
        assert mock_interr.invoke.call_count == 3
        assert mock_resp.invoke.call_count == 2

    def test_multi_turn_messages_accumulate(self) -> None:
        """Messages accumulate correctly via the operator.add reducer."""
        final_state, _, _ = _build_and_invoke(
            [
                _make_interrogator_question("Q1"),
                _make_interrogator_verdict(score=0.7),
            ],
            ["A1"],
        )
        messages = final_state["messages"]
        assert len(messages) >= 2
        # First message should be from interrogator, second from responder
        assert messages[0].role == DebateRole.INTERROGATOR
        assert messages[1].role == DebateRole.RESPONDER

    def test_messages_contain_correct_content(self) -> None:
        """Message content should match what the chains returned."""
        final_state, _, _ = _build_and_invoke(
            [
                _make_interrogator_question("My probing question"),
                _make_interrogator_verdict(score=0.7, reasoning="Final verdict"),
            ],
            ["My defense response"],
        )
        messages = final_state["messages"]
        assert messages[0].content == "My probing question"
        assert messages[1].content == "My defense response"


# ---------------------------------------------------------------------------
# Tests: Token accumulation
# ---------------------------------------------------------------------------


class TestTokenAccumulation:
    def test_tokens_accumulate_via_reducer(self) -> None:
        """total_tokens accumulates across all node invocations via operator.add."""
        token_per_call = 75.0
        final_state, _, _ = _build_and_invoke(
            [
                _make_interrogator_question("Q1"),
                _make_interrogator_verdict(score=0.7),
            ],
            ["A1"],
            token_count_return=token_per_call,
        )
        # 2 interrogator calls + 1 responder call = 3 * 75.0 = 225.0
        assert final_state["total_tokens"] == pytest.approx(3 * token_per_call)

    def test_immediate_verdict_token_count(self) -> None:
        """Immediate verdict: only one interrogator call."""
        token_per_call = 100.0
        final_state, _, _ = _build_and_invoke(
            [_make_interrogator_verdict(score=0.9)],
            [],
            token_count_return=token_per_call,
        )
        assert final_state["total_tokens"] == pytest.approx(token_per_call)

    def test_per_message_token_counts(self) -> None:
        """Each DebateMessage should have token_count from the handler."""
        token_per_call = 42.0
        final_state, _, _ = _build_and_invoke(
            [
                _make_interrogator_question("Q1"),
                _make_interrogator_verdict(score=0.5),
            ],
            ["A1"],
            token_count_return=token_per_call,
        )
        for msg in final_state["messages"]:
            assert msg.token_count == pytest.approx(token_per_call)


# ---------------------------------------------------------------------------
# Tests: Debate history visibility
# ---------------------------------------------------------------------------


class TestDebateHistoryVisibility:
    def test_responder_sees_history_when_enabled(self) -> None:
        """When responder_sees_debate_history=True, responder gets non-empty debate_history."""
        _, _, mock_resp = _build_and_invoke(
            [
                _make_interrogator_question("Q1"),
                _make_interrogator_question("Q2"),
                _make_interrogator_verdict(score=0.8),
            ],
            ["A1", "A2"],
            responder_sees_debate_history=True,
        )
        assert mock_resp.invoke.call_count == 2
        # On the second responder call, debate_history should contain prior turns
        second_call_input = mock_resp.invoke.call_args_list[1][0][0]
        assert isinstance(second_call_input, dict)
        assert second_call_input["debate_history"] != ""
        # Should contain the interrogator's question text
        assert "INTERROGATOR" in second_call_input["debate_history"]

    def test_responder_no_history_when_disabled(self) -> None:
        """When responder_sees_debate_history=False, debate_history is empty string."""
        _, _, mock_resp = _build_and_invoke(
            [
                _make_interrogator_question("Q1"),
                _make_interrogator_question("Q2"),
                _make_interrogator_verdict(score=0.8),
            ],
            ["A1", "A2"],
            responder_sees_debate_history=False,
        )
        # All responder calls should have empty debate_history
        for call_args in mock_resp.invoke.call_args_list:
            input_vars = call_args[0][0]
            assert isinstance(input_vars, dict)
            assert input_vars["debate_history"] == ""

    def test_first_responder_call_always_has_history(self) -> None:
        """Even with history=True, the first responder call has history (with the first question)."""
        _, _, mock_resp = _build_and_invoke(
            [
                _make_interrogator_question("Q1"),
                _make_interrogator_verdict(score=0.75),
            ],
            ["A1"],
            responder_sees_debate_history=True,
        )
        first_call_input = mock_resp.invoke.call_args_list[0][0][0]
        # Even first call should have debate_history with the first interrogator question
        assert first_call_input["debate_history"] != ""

    def test_both_modes_produce_valid_verdict(self) -> None:
        """Both history modes produce a valid verdict in final state."""
        for sees_history in [True, False]:
            final_state, _, _ = _build_and_invoke(
                [
                    _make_interrogator_question("Q1"),
                    _make_interrogator_verdict(score=0.75),
                ],
                ["A1"],
                responder_sees_debate_history=sees_history,
            )
            assert final_state["verdict"] is not None
            assert final_state["verdict"].score == 0.75

    def test_responder_receives_interrogator_question(self) -> None:
        """Responder input_vars should always include the latest interrogator question."""
        _, _, mock_resp = _build_and_invoke(
            [
                _make_interrogator_question("Specific question about lists"),
                _make_interrogator_verdict(score=0.8),
            ],
            ["My answer"],
        )
        input_vars = mock_resp.invoke.call_args_list[0][0][0]
        assert input_vars["interrogator_question"] == "Specific question about lists"


# ---------------------------------------------------------------------------
# Tests: Turn counting
# ---------------------------------------------------------------------------


class TestTurnCounting:
    def test_turn_count_matches_responder_calls(self) -> None:
        """current_turn should equal the number of completed responder turns."""
        final_state, _, mock_resp = _build_and_invoke(
            [
                _make_interrogator_question("Q1"),
                _make_interrogator_question("Q2"),
                _make_interrogator_verdict(score=0.5),
            ],
            ["A1", "A2"],
            max_turns=5,
        )
        assert final_state["current_turn"] == 2
        assert mock_resp.invoke.call_count == 2

    def test_early_termination_turn_count(self) -> None:
        """Early verdict keeps turn count at the correct value."""
        final_state, _, _ = _build_and_invoke(
            [
                _make_interrogator_question("Q1"),
                _make_interrogator_verdict(score=0.95),
            ],
            ["A1"],
            max_turns=5,
        )
        assert final_state["current_turn"] == 1
        assert final_state["verdict"] is not None

    def test_immediate_verdict_turn_count_zero(self) -> None:
        """Immediate verdict should have current_turn == 0."""
        final_state, _, _ = _build_and_invoke(
            [_make_interrogator_verdict(score=0.5)],
            [],
        )
        assert final_state["current_turn"] == 0


# ---------------------------------------------------------------------------
# Tests: Input vars
# ---------------------------------------------------------------------------


class TestInputVars:
    def test_interrogator_receives_context(self) -> None:
        """Interrogator chain should receive the original context fields."""
        _, mock_interr, _ = _build_and_invoke(
            [_make_interrogator_verdict(score=0.9)],
            [],
        )
        input_vars = mock_interr.invoke.call_args_list[0][0][0]
        assert input_vars["original_prompt"] == "Predict the output:\nprint([1,2,3])"
        assert input_vars["responder_output"] == "The output is [1, 2, 3]"
        assert input_vars["final_answer"] == "[1, 2, 3]"
        assert "current_turn" in input_vars
        assert "max_turns" in input_vars

    def test_interrogator_receives_one_indexed_turn(self) -> None:
        """current_turn passed to the interrogator should be 1-indexed for LLM readability."""
        _, mock_interr, _ = _build_and_invoke(
            [
                _make_interrogator_question("Q1"),
                _make_interrogator_question("Q2"),
                _make_interrogator_verdict(score=0.7),
            ],
            ["A1", "A2"],
            max_turns=5,
        )
        # Internal current_turn starts at 0, but prompt should see 1-indexed
        assert mock_interr.invoke.call_args_list[0][0][0]["current_turn"] == "1"
        assert mock_interr.invoke.call_args_list[1][0][0]["current_turn"] == "2"
        assert mock_interr.invoke.call_args_list[2][0][0]["current_turn"] == "3"

    def test_responder_receives_context(self) -> None:
        """Responder chain should receive the original context and the question."""
        _, _, mock_resp = _build_and_invoke(
            [
                _make_interrogator_question("Test question"),
                _make_interrogator_verdict(score=0.8),
            ],
            ["Answer"],
        )
        input_vars = mock_resp.invoke.call_args_list[0][0][0]
        assert input_vars["original_prompt"] == "Predict the output:\nprint([1,2,3])"
        assert input_vars["responder_output"] == "The output is [1, 2, 3]"
        assert input_vars["final_answer"] == "[1, 2, 3]"
        assert input_vars["interrogator_question"] == "Test question"
        assert "debate_history" in input_vars


# ---------------------------------------------------------------------------
# Tests: Format debate history helper
# ---------------------------------------------------------------------------


class TestFormatDebateHistory:
    def test_empty_messages(self) -> None:
        from pyine.guardrails.llm_debate.graph import _format_debate_history

        result = _format_debate_history([])
        assert "no prior debate history" in result

    def test_single_message(self) -> None:
        from pyine.guardrails.llm_debate.graph import _format_debate_history

        messages = [
            DebateMessage(role=DebateRole.INTERROGATOR, content="Hello?", token_count=10.0),
        ]
        result = _format_debate_history(messages)
        assert "[INTERROGATOR]" in result
        assert "Hello?" in result

    def test_multiple_messages(self) -> None:
        from pyine.guardrails.llm_debate.graph import _format_debate_history

        messages = [
            DebateMessage(role=DebateRole.INTERROGATOR, content="Q1", token_count=10.0),
            DebateMessage(role=DebateRole.RESPONDER, content="A1", token_count=20.0),
        ]
        result = _format_debate_history(messages)
        assert "[INTERROGATOR]" in result
        assert "[RESPONDER]" in result
        assert "Q1" in result
        assert "A1" in result
