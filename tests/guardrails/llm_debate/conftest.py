"""Shared fixtures for LLM debate guardrail tests."""

from __future__ import annotations

import typing

import pytest

import pyine.evals.correctness.types as correctness_types
import pyine.utils.llm_providers
from pyine.guardrails.llm_debate.configs import DebateGuardrailConfig
from pyine.guardrails.llm_debate.types import (
    DebateMessage,
    DebateRole,
    DebateTranscript,
    DebateVerdict,
)

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_llm_provider(**overrides: typing.Any) -> pyine.utils.llm_providers.LLMProviderConfig:
    """Build a minimal LLMProviderConfig for testing."""
    defaults: dict[str, typing.Any] = {
        "provider": "openai",
        "model_kwargs": {"model": "gpt-4o-mini", "temperature": 0.0},
    }
    defaults.update(overrides)
    return pyine.utils.llm_providers.LLMProviderConfig(**defaults)


def make_debate_config(**overrides: typing.Any) -> DebateGuardrailConfig:
    """Build a DebateGuardrailConfig with sensible test defaults."""
    defaults: dict[str, typing.Any] = {
        "interrogator_provider": make_llm_provider(),
        "responder_provider": make_llm_provider(),
        "max_workers": 2,
        "max_debate_turns": 2,
    }
    defaults.update(overrides)
    return DebateGuardrailConfig(**defaults)


def make_eval_record(
    sample_id: str = "TEST/VALID/p000000/s0000/t0000",
    label: bool = True,
    model_output: str = "The output is [1, 2, 3]",
    final_answer: str = "[1, 2, 3]",
    expected_output: str = "[1, 2, 3]",
    prompt: str = "Predict the output of the following code:\nprint([1, 2, 3])",
) -> correctness_types.EvalRecord:
    """Build a minimal EvalRecord for testing."""
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


def make_debate_message(
    role: DebateRole = DebateRole.INTERROGATOR,
    content: str = "Why did you predict [1, 2, 3]?",
    token_count: float = 100.0,
) -> DebateMessage:
    """Build a DebateMessage for testing."""
    return DebateMessage(role=role, content=content, token_count=token_count)


def make_debate_transcript(
    num_turns: int = 2,
    score: float = 0.85,
    reasoning: str | None = "Model shows good understanding",
) -> DebateTranscript:
    """Build a DebateTranscript with alternating interrogator/responder messages."""
    messages: list[DebateMessage] = []
    for i in range(num_turns):
        messages.append(
            DebateMessage(
                role=DebateRole.INTERROGATOR,
                content=f"Question {i + 1}: Can you explain your reasoning?",
                token_count=100.0 + i * 10,
            )
        )
        messages.append(
            DebateMessage(
                role=DebateRole.RESPONDER,
                content=f"Answer {i + 1}: The code outputs [1, 2, 3] because...",
                token_count=200.0 + i * 10,
            )
        )
    total_tokens = sum(m.token_count for m in messages)
    return DebateTranscript(
        messages=messages,
        verdict=DebateVerdict(score=score, reasoning=reasoning),
        num_turns=num_turns,
        total_token_count=total_tokens,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_provider() -> pyine.utils.llm_providers.LLMProviderConfig:
    return make_llm_provider()


@pytest.fixture
def debate_config() -> DebateGuardrailConfig:
    return make_debate_config()


@pytest.fixture
def sample_records() -> list[correctness_types.EvalRecord]:
    """Build a small list of eval records for testing."""
    return [make_eval_record(sample_id=f"TEST/VALID/p{i:06d}/s0000/t0000") for i in range(5)]


@pytest.fixture
def sample_transcript() -> DebateTranscript:
    return make_debate_transcript()
