"""Data types for the LLM debate guardrail."""

from __future__ import annotations

import enum

import pydantic


class DebateRole(enum.StrEnum):
    """Role of a participant in the debate."""

    INTERROGATOR = "interrogator"  # Model B
    RESPONDER = "responder"  # Model A


class DebateMessage(pydantic.BaseModel):
    """Single message in the debate transcript."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    role: DebateRole
    content: str
    token_count: float
    """Tokens used for this message generation."""


class DebateVerdict(pydantic.BaseModel):
    """Model B's final judgement after debate."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    score: float
    """0-1 confidence (higher = more likely correct)."""
    reasoning: str | None = None


class DebateTranscript(pydantic.BaseModel):
    """Full debate transcript for one record."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    messages: list[DebateMessage]
    verdict: DebateVerdict
    num_turns: int
    """Actual turns used (may be < max if early termination)."""
    total_token_count: float
    """Sum of all message token counts."""
