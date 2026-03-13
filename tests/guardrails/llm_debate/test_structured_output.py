"""Unit tests for structured output chain construction and VerdictOutput schema.

Tests:
- get_prompt_chain() in debate_interrogator.py
- _unwrap_retry() helper
- VerdictOutput schema validation
- debate_interrogator_verdict.py chain construction
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pydantic
import pytest

from pyine.prompts.configs.guardrail.debate_interrogator import (
    InterrogatorOutput,
    VerdictOutput,
)

# ---------------------------------------------------------------------------
# Tests: VerdictOutput schema validation
# ---------------------------------------------------------------------------


class TestVerdictOutputSchema:
    def test_create_valid_verdict(self) -> None:
        v = VerdictOutput(score=0.85, reasoning="Good understanding", content="Summary")
        assert v.score == 0.85
        assert v.reasoning == "Good understanding"
        assert v.content == "Summary"

    def test_score_bounds_lower(self) -> None:
        v = VerdictOutput(score=0.0, reasoning="Bad", content="No")
        assert v.score == 0.0

    def test_score_bounds_upper(self) -> None:
        v = VerdictOutput(score=1.0, reasoning="Perfect", content="Yes")
        assert v.score == 1.0

    def test_score_below_lower_bound(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            VerdictOutput(score=-0.1, reasoning="Bad", content="No")

    def test_score_above_upper_bound(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            VerdictOutput(score=1.1, reasoning="Too high", content="No")

    def test_frozen(self) -> None:
        v = VerdictOutput(score=0.5, reasoning="OK", content="Summary")
        with pytest.raises(pydantic.ValidationError):
            v.score = 0.9  # type: ignore[misc]

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            VerdictOutput(
                score=0.5,
                reasoning="OK",
                content="Summary",
                extra_field="bad",  # type: ignore[call-arg]
            )

    def test_requires_all_fields(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            VerdictOutput(score=0.5)  # type: ignore[call-arg]
        with pytest.raises(pydantic.ValidationError):
            VerdictOutput(score=0.5, reasoning="OK")  # type: ignore[call-arg]

    def test_no_decision_field(self) -> None:
        """VerdictOutput should NOT have a 'decision' field (unlike InterrogatorOutput)."""
        fields = set(VerdictOutput.model_fields.keys())
        assert "decision" not in fields

    def test_serialization_roundtrip(self) -> None:
        v = VerdictOutput(score=0.75, reasoning="Solid", content="Assessment")
        data = v.model_dump()
        assert data == {"score": 0.75, "reasoning": "Solid", "content": "Assessment"}
        restored = VerdictOutput.model_validate(data)
        assert restored == v

    def test_json_roundtrip(self) -> None:
        v = VerdictOutput(score=0.42, reasoning="Uncertain", content="Mixed")
        json_str = v.model_dump_json()
        restored = VerdictOutput.model_validate_json(json_str)
        assert restored == v

    def test_distinct_from_interrogator_output(self) -> None:
        """VerdictOutput and InterrogatorOutput are distinct types with no inheritance."""
        assert not issubclass(VerdictOutput, InterrogatorOutput)
        assert not issubclass(InterrogatorOutput, VerdictOutput)
        # VerdictOutput has 3 required fields, InterrogatorOutput has 'decision'
        assert "decision" in InterrogatorOutput.model_fields
        assert "decision" not in VerdictOutput.model_fields


# ---------------------------------------------------------------------------
# Tests: _unwrap_retry helper
# ---------------------------------------------------------------------------


class TestUnwrapRetry:
    def test_unwrap_plain_model(self) -> None:
        """A plain model (not wrapped in retry) should be returned as-is."""
        from pyine.prompts.configs.guardrail.debate_interrogator import _unwrap_retry

        mock_model = MagicMock()
        result = _unwrap_retry(mock_model)
        assert result is mock_model

    def test_unwrap_retry_wrapped_model(self) -> None:
        """A RunnableRetry-wrapped model should be unwrapped to extract the inner .bound model."""
        from langchain_core.runnables.retry import RunnableRetry

        from pyine.prompts.configs.guardrail.debate_interrogator import _unwrap_retry

        inner_model = MagicMock()
        # Build a mock that passes isinstance(_, RunnableRetry) checks
        retry_wrapper = MagicMock(spec=RunnableRetry)
        retry_wrapper.bound = inner_model

        result = _unwrap_retry(retry_wrapper)
        assert result is inner_model


# ---------------------------------------------------------------------------
# Tests: get_prompt_chain (debate_interrogator.py)
# ---------------------------------------------------------------------------


class TestInterrogatorGetPromptChain:
    def test_calls_with_structured_output(self) -> None:
        """get_prompt_chain should call model.with_structured_output(InterrogatorOutput)."""
        from pyine.prompts.configs.guardrail.debate_interrogator import get_prompt_chain

        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured

        with (
            patch(
                "pyine.prompts.configs.guardrail.debate_interrogator.get_prompt_template",
                return_value=MagicMock(),
            ),
            patch(
                "pyine.prompts.configs.guardrail.debate_interrogator._unwrap_retry",
                return_value=mock_model,
            ),
        ):
            chain = get_prompt_chain(mock_model)

        mock_model.with_structured_output.assert_called_once_with(
            InterrogatorOutput,
            method="json_schema",
        )
        # The chain should be a RunnableSequence
        assert chain is not None

    def test_unwraps_retry_before_structured_output(self) -> None:
        """get_prompt_chain should call _unwrap_retry before calling with_structured_output."""
        from pyine.prompts.configs.guardrail.debate_interrogator import get_prompt_chain

        mock_model = MagicMock()
        inner_model = MagicMock()
        mock_structured = MagicMock()
        inner_model.with_structured_output.return_value = mock_structured

        with (
            patch(
                "pyine.prompts.configs.guardrail.debate_interrogator.get_prompt_template",
                return_value=MagicMock(),
            ),
            patch(
                "pyine.prompts.configs.guardrail.debate_interrogator._unwrap_retry",
                return_value=inner_model,
            ) as mock_unwrap,
        ):
            get_prompt_chain(mock_model)

        # _unwrap_retry should have been called with the model
        mock_unwrap.assert_called_once_with(mock_model)
        # with_structured_output should have been called on the unwrapped model
        inner_model.with_structured_output.assert_called_once_with(
            InterrogatorOutput,
            method="json_schema",
        )

    def test_returns_runnable_sequence(self) -> None:
        """The returned chain should be a RunnableSequence."""
        import langchain_core.runnables

        from pyine.prompts.configs.guardrail.debate_interrogator import get_prompt_chain

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = MagicMock()

        with (
            patch(
                "pyine.prompts.configs.guardrail.debate_interrogator.get_prompt_template",
                return_value=MagicMock(),
            ),
            patch(
                "pyine.prompts.configs.guardrail.debate_interrogator._unwrap_retry",
                return_value=mock_model,
            ),
        ):
            chain = get_prompt_chain(mock_model)

        assert isinstance(chain, langchain_core.runnables.RunnableSequence)


# ---------------------------------------------------------------------------
# Tests: debate_interrogator_verdict.py chain construction
# ---------------------------------------------------------------------------


class TestVerdictChainConstruction:
    def test_verdict_get_prompt_chain_calls_structured_output(self) -> None:
        """The verdict module's get_prompt_chain should use VerdictOutput with structured output."""
        from pyine.prompts.configs.guardrail.debate_interrogator_verdict import get_prompt_chain

        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured

        with (
            patch(
                "pyine.prompts.configs.guardrail.debate_interrogator_verdict.get_prompt_template",
                return_value=MagicMock(),
            ),
            patch(
                "pyine.prompts.configs.guardrail.debate_interrogator._unwrap_retry",
                return_value=mock_model,
            ),
        ):
            chain = get_prompt_chain(mock_model)

        mock_model.with_structured_output.assert_called_once_with(
            VerdictOutput,
            method="json_schema",
        )
        assert chain is not None

    def test_verdict_get_prompt_chain_returns_runnable_sequence(self) -> None:
        """The verdict chain should be a RunnableSequence."""
        import langchain_core.runnables

        from pyine.prompts.configs.guardrail.debate_interrogator_verdict import get_prompt_chain

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = MagicMock()

        with (
            patch(
                "pyine.prompts.configs.guardrail.debate_interrogator_verdict.get_prompt_template",
                return_value=MagicMock(),
            ),
            patch(
                "pyine.prompts.configs.guardrail.debate_interrogator._unwrap_retry",
                return_value=mock_model,
            ),
        ):
            chain = get_prompt_chain(mock_model)

        assert isinstance(chain, langchain_core.runnables.RunnableSequence)

    def test_verdict_get_output_parser_returns_pydantic_parser(self) -> None:
        """The verdict module's get_output_parser should return a PydanticOutputParser for VerdictOutput."""
        from pyine.prompts.configs.guardrail.debate_interrogator_verdict import get_output_parser

        parser = get_output_parser()
        assert parser is not None
        # Parser should target VerdictOutput
        assert parser.pydantic_object is VerdictOutput  # type: ignore[attr-defined]

    def test_verdict_get_output_parser_unsupported_version(self) -> None:
        """Unsupported versions should raise NotImplementedError."""
        from pyine.prompts.configs.guardrail.debate_interrogator_verdict import get_output_parser

        with pytest.raises(NotImplementedError):
            get_output_parser(version="unsupported_version")
