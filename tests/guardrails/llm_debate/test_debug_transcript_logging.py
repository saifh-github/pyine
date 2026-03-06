"""Unit tests for debug transcript logging in DebateGuardrailScorer.

Tests the debug_log_transcript_every_n config flag, thread-safety of the
debug counter, formatted output structure, and no-log-on-error behavior.
"""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import pyine.utils.llm_providers
from pyine.guardrails.llm_debate.types import (
    DebateMessage,
    DebateRole,
    DebateVerdict,
)

if typing.TYPE_CHECKING:
    from pyine.guardrails.llm_debate.configs import DebateGuardrailConfig

from .conftest import make_debate_config, make_debate_transcript, make_eval_record

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_final_graph_state(
    score: float = 0.85,
    num_turns: int = 2,
    total_tokens: float = 500.0,
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
        "model_a_output": "test output",
        "final_answer": "test answer",
        "max_turns": 3,
        "messages": messages,
        "current_turn": num_turns,
        "verdict": DebateVerdict(score=score, reasoning="Good reasoning"),
        "total_tokens": total_tokens,
    }


def _build_scorer_with_mock_graph(
    config: DebateGuardrailConfig | None = None,
    graph_invoke_side_effect: typing.Any = None,
    graph_invoke_return_value: typing.Any = None,
) -> tuple[typing.Any, MagicMock]:
    """Build a DebateGuardrailScorer with mocked LLM, chain, and graph."""
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
# Tests: Disabled by default
# ---------------------------------------------------------------------------


class TestDisabledByDefault:
    def test_no_debug_log_when_disabled(self) -> None:
        """With debug_log_transcript_every_n=0, no debug transcript is logged."""
        config = make_debate_config(debug_log_transcript_every_n=0)
        scorer, _ = _build_scorer_with_mock_graph(config=config)
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(10)]

        with patch("pyine.guardrails.llm_debate.scorer.logger") as mock_logger:
            scorer.score_records(records)
            # Filter for debug transcript log calls (contain "Debug transcript")
            transcript_calls = [c for c in mock_logger.info.call_args_list if "Debug transcript" in str(c)]
            assert len(transcript_calls) == 0


# ---------------------------------------------------------------------------
# Tests: Correct frequency
# ---------------------------------------------------------------------------


class TestLogsAtCorrectFrequency:
    def test_logs_every_n_records(self) -> None:
        """With every_n=3, log on records 3, 6, 9 but not 1, 2, 4, 5, 7, 8."""
        config = make_debate_config(debug_log_transcript_every_n=3, max_workers=1)
        scorer, _ = _build_scorer_with_mock_graph(config=config)
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(9)]

        with patch("pyine.guardrails.llm_debate.scorer.logger") as mock_logger:
            scorer.score_records(records)
            transcript_calls = [c for c in mock_logger.info.call_args_list if "Debug transcript" in str(c)]
            assert len(transcript_calls) == 3  # records 3, 6, 9

    def test_logs_once_for_exact_n(self) -> None:
        """With every_n=5 and 5 records, exactly 1 debug log."""
        config = make_debate_config(debug_log_transcript_every_n=5, max_workers=1)
        scorer, _ = _build_scorer_with_mock_graph(config=config)
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(5)]

        with patch("pyine.guardrails.llm_debate.scorer.logger") as mock_logger:
            scorer.score_records(records)
            transcript_calls = [c for c in mock_logger.info.call_args_list if "Debug transcript" in str(c)]
            assert len(transcript_calls) == 1

    def test_no_log_when_fewer_than_n(self) -> None:
        """With every_n=10 and only 5 records, no debug log."""
        config = make_debate_config(debug_log_transcript_every_n=10, max_workers=1)
        scorer, _ = _build_scorer_with_mock_graph(config=config)
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(5)]

        with patch("pyine.guardrails.llm_debate.scorer.logger") as mock_logger:
            scorer.score_records(records)
            transcript_calls = [c for c in mock_logger.info.call_args_list if "Debug transcript" in str(c)]
            assert len(transcript_calls) == 0


# ---------------------------------------------------------------------------
# Tests: Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_counter_no_skips_or_duplicates(self) -> None:
        """Run scoring from multiple threads, verify counter increments correctly."""
        n_records = 30
        every_n = 5
        config = make_debate_config(debug_log_transcript_every_n=every_n, max_workers=4)
        scorer, _ = _build_scorer_with_mock_graph(config=config)
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(n_records)]

        with patch("pyine.guardrails.llm_debate.scorer.logger") as mock_logger:
            scorer.score_records(records)
            transcript_calls = [c for c in mock_logger.info.call_args_list if "Debug transcript" in str(c)]
            # With 30 records and every_n=5, expect exactly 6 logs
            assert len(transcript_calls) == n_records // every_n


# ---------------------------------------------------------------------------
# Tests: Format output
# ---------------------------------------------------------------------------


class TestFormatOutput:
    def test_format_transcript_structure(self) -> None:
        """Verify _format_transcript_for_log produces expected structure."""
        from pyine.guardrails.llm_debate.scorer import DebateGuardrailScorer

        transcript = make_debate_transcript(num_turns=2, score=0.85, reasoning="Good")
        formatted = DebateGuardrailScorer._format_transcript_for_log(transcript, "sample_123", True)

        # Should contain separator lines (Unicode box-drawing character \u2500)
        assert "\u2500" * 72 in formatted
        # Should contain sample_id
        assert "sample_123" in formatted
        # Should contain history_visible field
        assert "history_visible=True" in formatted
        # Should contain role labels
        assert "INTERROGATOR (B)" in formatted
        assert "RESPONDER (A)" in formatted
        # Should contain verdict
        assert "VERDICT:" in formatted
        assert "0.850" in formatted
        # Should contain reasoning
        assert "REASONING:" in formatted

    def test_format_transcript_history_visible_false(self) -> None:
        """Verify history_visible=False is reflected in the formatted output."""
        from pyine.guardrails.llm_debate.scorer import DebateGuardrailScorer

        transcript = make_debate_transcript(num_turns=1)
        formatted = DebateGuardrailScorer._format_transcript_for_log(transcript, "sample_456", False)
        assert "history_visible=False" in formatted

    def test_format_transcript_no_reasoning(self) -> None:
        """Verify format handles None reasoning gracefully."""
        from pyine.guardrails.llm_debate.scorer import DebateGuardrailScorer

        transcript = make_debate_transcript(num_turns=1, reasoning=None)
        formatted = DebateGuardrailScorer._format_transcript_for_log(transcript, "sample_789", True)
        assert "VERDICT:" in formatted
        assert "REASONING:" not in formatted

    def test_format_transcript_turn_info(self) -> None:
        """Verify turns and token counts are in the header."""
        from pyine.guardrails.llm_debate.scorer import DebateGuardrailScorer

        transcript = make_debate_transcript(num_turns=3)
        formatted = DebateGuardrailScorer._format_transcript_for_log(transcript, "sample_abc", True)
        assert "turns=3" in formatted
        assert "tokens=" in formatted


# ---------------------------------------------------------------------------
# Tests: No log on error
# ---------------------------------------------------------------------------


class TestNoLogOnError:
    def test_no_debug_log_when_error(self) -> None:
        """When _score_single hits an error, no debug transcript is logged."""
        config = make_debate_config(debug_log_transcript_every_n=1, max_workers=1)
        scorer, _ = _build_scorer_with_mock_graph(
            config=config,
            graph_invoke_side_effect=RuntimeError("Graph failed"),
        )
        records = [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(5)]

        with patch("pyine.guardrails.llm_debate.scorer.logger") as mock_logger:
            scorer.score_records(records)
            transcript_calls = [c for c in mock_logger.info.call_args_list if "Debug transcript" in str(c)]
            assert len(transcript_calls) == 0
