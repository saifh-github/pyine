"""Unit tests for PromptedLLMGuardrailScorer.

All LLM chain calls are mocked - no real API calls are made.
"""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import pytest

import pyine.evals.correctness.types as correctness_types
import pyine.utils.llm_providers
from pyine.guardrails.prompted_llm.configs import PromptedLLMGuardrailConfig
from pyine.prompts.configs.guardrail.correctness_judge import CorrectnessJudgementWithReasoning

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_provider() -> pyine.utils.llm_providers.LLMProviderConfig:
    return pyine.utils.llm_providers.LLMProviderConfig(
        provider="openai",
        model_kwargs={"model": "gpt-4o-mini", "temperature": 0.0},
    )


def _make_config(**overrides: typing.Any) -> PromptedLLMGuardrailConfig:
    defaults: dict[str, typing.Any] = {
        "llm_provider": _make_llm_provider(),
        "max_workers": 2,
    }
    defaults.update(overrides)
    return PromptedLLMGuardrailConfig(**defaults)


def _make_record(
    sample_id: str = "TEST/VALID/p000000/s0000/t0000",
    label: bool = True,
    model_output: str = "output",
    final_answer: str = "expected",
    expected_output: str = "expected",
    prompt: str = "Analyze the following code...",
) -> correctness_types.EvalRecord:
    return correctness_types.EvalRecord(
        sample_id=sample_id,
        problem_id=sample_id.rsplit("/", 2)[0],
        attempt_index=0,
        model_output=model_output,
        final_answer=final_answer,
        expected_output=expected_output,
        label=label,
        code_type="original",
        tags=[],
        record={"prompt": prompt},
        difficulty_score=None,
    )


def _mock_judgement(score: float = 0.85) -> CorrectnessJudgementWithReasoning:
    """Create a mock judgement result."""
    return CorrectnessJudgementWithReasoning(score=score, reasoning="Looks correct")


def _build_scorer_with_mock_chain(
    config: PromptedLLMGuardrailConfig | None = None,
    chain_invoke_side_effect: typing.Any = None,
    chain_invoke_return_value: typing.Any = None,
) -> tuple[typing.Any, MagicMock]:
    """Build a PromptedLLMGuardrailScorer with mocked LLM and chain.

    Returns (scorer, mock_chain) so tests can inspect/configure the mock.
    """
    if config is None:
        config = _make_config()

    mock_chain = MagicMock()
    if chain_invoke_side_effect is not None:
        mock_chain.invoke.side_effect = chain_invoke_side_effect
    elif chain_invoke_return_value is not None:
        mock_chain.invoke.return_value = chain_invoke_return_value
    else:
        mock_chain.invoke.return_value = _mock_judgement()

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
    ):
        from pyine.guardrails.prompted_llm.scorer import PromptedLLMGuardrailScorer

        scorer = PromptedLLMGuardrailScorer(config)
    return scorer, mock_chain


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScoreRecordsShape:
    def test_score_records_returns_correct_shape(self) -> None:
        scorer, _ = _build_scorer_with_mock_chain()
        records = [_make_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(5)]
        result = scorer.score_records(records)
        assert len(result.scores) == 5
        assert result.verification_costs is not None
        assert len(result.verification_costs) == 5

    def test_score_records_empty_list(self) -> None:
        scorer, _ = _build_scorer_with_mock_chain()
        result = scorer.score_records([])
        assert len(result.scores) == 0
        assert result.verification_costs is not None
        assert len(result.verification_costs) == 0


class TestScoreRecordsValues:
    def test_score_records_scores_in_range(self) -> None:
        scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        judgements = [_mock_judgement(s) for s in scores]
        scorer, _ = _build_scorer_with_mock_chain(chain_invoke_side_effect=judgements)
        records = [_make_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(5)]
        result = scorer.score_records(records)
        for score in result.scores:
            assert 0.0 <= score <= 1.0

    def test_score_records_costs_non_negative(self) -> None:
        scorer, _ = _build_scorer_with_mock_chain()
        records = [_make_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(3)]
        result = scorer.score_records(records)
        assert result.verification_costs is not None
        for cost in result.verification_costs:
            assert cost >= 0.0


class TestErrorHandling:
    def test_error_handling_uses_default_score(self) -> None:
        default_score = 0.3
        config = _make_config(default_score_on_error=default_score)
        scorer, mock_chain = _build_scorer_with_mock_chain(
            config=config,
            chain_invoke_side_effect=RuntimeError("API error"),
        )
        records = [_make_record()]
        result = scorer.score_records(records)
        assert result.scores[0] == default_score

    def test_error_count_incremented(self) -> None:
        config = _make_config()
        scorer, _ = _build_scorer_with_mock_chain(
            config=config,
            chain_invoke_side_effect=RuntimeError("API error"),
        )
        records = [_make_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(3)]
        scorer.score_records(records)
        metadata = scorer.get_metadata()
        assert metadata["error_count"] == 3

    def test_error_count_accumulates_across_calls(self) -> None:
        config = _make_config()
        scorer, _ = _build_scorer_with_mock_chain(
            config=config,
            chain_invoke_side_effect=RuntimeError("API error"),
        )
        scorer.score_records([_make_record(sample_id="TEST/VALID/p000000/s0000/t0000")])
        scorer.score_records([_make_record(sample_id="TEST/VALID/p000001/s0000/t0000")])
        metadata = scorer.get_metadata()
        assert metadata["error_count"] == 2


class TestMetadata:
    def test_get_metadata_contains_required_fields(self) -> None:
        config = _make_config()
        scorer, _ = _build_scorer_with_mock_chain(config=config)
        metadata = scorer.get_metadata()
        required_keys = {
            "scorer_type",
            "prompt_name",
            "provider",
            "model_kwargs",
            "max_workers",
            "default_score_on_error",
            "total_scored",
            "error_count",
        }
        assert required_keys.issubset(set(metadata.keys()))
        assert metadata["scorer_type"] == "prompted_llm"
        assert metadata["prompt_name"] == "guardrail/correctness_judge"
        assert metadata["provider"] == "openai"
        assert metadata["max_workers"] == 2
        assert metadata["default_score_on_error"] == 0.5
        assert metadata["total_scored"] == 0
        assert metadata["error_count"] == 0

    def test_get_metadata_total_scored_updates(self) -> None:
        scorer, _ = _build_scorer_with_mock_chain()
        records = [_make_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(4)]
        scorer.score_records(records)
        assert scorer.get_metadata()["total_scored"] == 4
        scorer.score_records(records[:2])
        assert scorer.get_metadata()["total_scored"] == 6


class TestVerificationCostUnit:
    def test_get_verification_cost_unit(self) -> None:
        scorer, _ = _build_scorer_with_mock_chain()
        assert scorer.get_verification_cost_unit() == "tokens"


class TestProtocolConformance:
    def test_protocol_conformance(self) -> None:
        """Verify the scorer satisfies the GuardrailScorer protocol."""
        scorer, _ = _build_scorer_with_mock_chain()
        # Structural check: all required protocol methods exist and are callable
        assert callable(getattr(scorer, "score_records", None))
        assert callable(getattr(scorer, "get_metadata", None))
        assert callable(getattr(scorer, "get_verification_cost_unit", None))

    def test_protocol_structural_match(self) -> None:
        """Verify the scorer has the correct method signatures matching GuardrailScorer."""
        import inspect

        scorer, _ = _build_scorer_with_mock_chain()
        # Verify method signatures match the protocol
        score_sig = inspect.signature(scorer.score_records)
        assert "records" in score_sig.parameters

        meta_sig = inspect.signature(scorer.get_metadata)
        assert len(meta_sig.parameters) == 0

        cost_sig = inspect.signature(scorer.get_verification_cost_unit)
        assert len(cost_sig.parameters) == 0


class TestFinalAnswerHandling:
    def test_final_answer_passed_when_present(self) -> None:
        scorer, mock_chain = _build_scorer_with_mock_chain()
        record = _make_record(final_answer="42")
        scorer.score_records([record])
        call_args = mock_chain.invoke.call_args
        input_vars = call_args[0][0]
        assert "final_answer" in input_vars
        assert input_vars["final_answer"] == "42"

    def test_final_answer_none_raises(self) -> None:
        scorer, _ = _build_scorer_with_mock_chain()
        record = correctness_types.EvalRecord(
            sample_id="TEST/VALID/p000000/s0000/t0000",
            problem_id="TEST/VALID/p000000",
            attempt_index=0,
            model_output="output",
            final_answer=None,
            expected_output="expected",
            label=True,
            code_type="original",
            tags=[],
            record={"prompt": "some prompt"},
            difficulty_score=None,
        )
        with pytest.raises(AssertionError, match="final_answer is required"):
            scorer.score_records([record])

    def test_model_output_and_prompt_always_passed(self) -> None:
        scorer, mock_chain = _build_scorer_with_mock_chain()
        record = _make_record(model_output="my output", prompt="my prompt")
        scorer.score_records([record])
        call_args = mock_chain.invoke.call_args
        input_vars = call_args[0][0]
        assert input_vars["model_output"] == "my output"
        assert input_vars["prompt"] == "my prompt"


class TestSanitizeForJsonEncoding:
    """Regression tests for lone surrogate sanitization (OpenAI 400 fix)."""

    def test_lone_surrogates_replaced(self) -> None:
        from pyine.guardrails.prompted_llm.scorer import _sanitize_for_json_encoding

        text = "before\ud800after"
        result = _sanitize_for_json_encoding(text)
        assert "\ud800" not in result
        assert "before" in result
        assert "after" in result

    def test_normal_text_unchanged(self) -> None:
        from pyine.guardrails.prompted_llm.scorer import _sanitize_for_json_encoding

        text = 'def foo():\n\treturn "hello \'world\'"\n'
        assert _sanitize_for_json_encoding(text) == text

    def test_control_chars_preserved(self) -> None:
        from pyine.guardrails.prompted_llm.scorer import _sanitize_for_json_encoding

        text = "tab\there\nnewline\r\nend"
        assert _sanitize_for_json_encoding(text) == text

    def test_valid_unicode_preserved(self) -> None:
        from pyine.guardrails.prompted_llm.scorer import _sanitize_for_json_encoding

        text = "emoji: \U0001f600 cjk: \u4e16\u754c"
        assert _sanitize_for_json_encoding(text) == text

    def test_sanitized_text_json_serializable(self) -> None:
        """The whole point: after sanitization, json.dumps must not raise."""
        import json

        from pyine.guardrails.prompted_llm.scorer import _sanitize_for_json_encoding

        text = "a\ud800b\udbffc\udc00d"
        result = _sanitize_for_json_encoding(text)
        json.dumps(result)  # should not raise

    def test_scorer_passes_sanitized_inputs_to_chain(self) -> None:
        scorer, mock_chain = _build_scorer_with_mock_chain()
        record = _make_record(
            model_output="output\ud800tail",
            final_answer="answer\udbffend",
            prompt="prompt\udc00end",
        )
        scorer.score_records([record])
        input_vars = mock_chain.invoke.call_args[0][0]
        for key in ("prompt", "model_output", "final_answer"):
            value = input_vars[key]
            assert "\ud800" not in value and "\udbff" not in value and "\udc00" not in value


class TestFormatPromptMessages:
    """Regression tests for non-string prompt_messages content."""

    def test_string_content(self) -> None:
        scorer, mock_chain = _build_scorer_with_mock_chain()
        record = correctness_types.EvalRecord(
            sample_id="TEST/VALID/p000000/s0000/t0000",
            problem_id="TEST/VALID/p000000",
            attempt_index=0,
            model_output="output",
            final_answer="answer",
            expected_output="expected",
            label=True,
            code_type="original",
            tags=[],
            record={
                "prompt_messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Analyze this code."},
                ],
            },
            difficulty_score=None,
        )
        scorer.score_records([record])
        input_vars = mock_chain.invoke.call_args[0][0]
        assert "You are helpful." in input_vars["prompt"]
        assert "Analyze this code." in input_vars["prompt"]

    def test_list_content_with_text_parts(self) -> None:
        scorer, mock_chain = _build_scorer_with_mock_chain()
        record = correctness_types.EvalRecord(
            sample_id="TEST/VALID/p000000/s0000/t0000",
            problem_id="TEST/VALID/p000000",
            attempt_index=0,
            model_output="output",
            final_answer="answer",
            expected_output="expected",
            label=True,
            code_type="original",
            tags=[],
            record={
                "prompt_messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What does this code do?"},
                            {"type": "text", "text": "def foo(): pass"},
                        ],
                    },
                ],
            },
            difficulty_score=None,
        )
        scorer.score_records([record])
        input_vars = mock_chain.invoke.call_args[0][0]
        assert "What does this code do?" in input_vars["prompt"]
        assert "def foo(): pass" in input_vars["prompt"]

    def test_non_string_content_coerced(self) -> None:
        scorer, mock_chain = _build_scorer_with_mock_chain()
        record = correctness_types.EvalRecord(
            sample_id="TEST/VALID/p000000/s0000/t0000",
            problem_id="TEST/VALID/p000000",
            attempt_index=0,
            model_output="output",
            final_answer="answer",
            expected_output="expected",
            label=True,
            code_type="original",
            tags=[],
            record={
                "prompt_messages": [
                    {"role": "user", "content": 42},
                ],
            },
            difficulty_score=None,
        )
        scorer.score_records([record])
        input_vars = mock_chain.invoke.call_args[0][0]
        assert "42" in input_vars["prompt"]
