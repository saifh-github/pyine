"""Unit tests for LLM debate data types (DebateMessage, DebateTranscript, DebateVerdict)."""

from __future__ import annotations

import pydantic
import pytest

from pyine.guardrails.llm_debate.types import (
    DebateMessage,
    DebateRole,
    DebateTranscript,
    DebateVerdict,
)

# ---------------------------------------------------------------------------
# DebateRole
# ---------------------------------------------------------------------------


class TestDebateRole:
    def test_role_values(self) -> None:
        assert DebateRole.INTERROGATOR == "interrogator"
        assert DebateRole.RESPONDER == "responder"

    def test_role_is_str_enum(self) -> None:
        assert isinstance(DebateRole.INTERROGATOR, str)
        assert isinstance(DebateRole.RESPONDER, str)

    def test_role_from_string(self) -> None:
        assert DebateRole("interrogator") is DebateRole.INTERROGATOR
        assert DebateRole("responder") is DebateRole.RESPONDER

    def test_role_invalid_value(self) -> None:
        with pytest.raises(ValueError):
            DebateRole("judge")


# ---------------------------------------------------------------------------
# DebateMessage
# ---------------------------------------------------------------------------


class TestDebateMessage:
    def test_create_interrogator_message(self) -> None:
        msg = DebateMessage(
            role=DebateRole.INTERROGATOR,
            content="Why did you predict [1, 2, 3]?",
            token_count=150.0,
        )
        assert msg.role == DebateRole.INTERROGATOR
        assert msg.content == "Why did you predict [1, 2, 3]?"
        assert msg.token_count == 150.0

    def test_create_responder_message(self) -> None:
        msg = DebateMessage(
            role=DebateRole.RESPONDER,
            content="The code outputs [1, 2, 3] because...",
            token_count=200.0,
        )
        assert msg.role == DebateRole.RESPONDER

    def test_frozen(self) -> None:
        msg = DebateMessage(role=DebateRole.INTERROGATOR, content="question", token_count=10.0)
        with pytest.raises(pydantic.ValidationError):
            msg.content = "changed"  # type: ignore[misc]

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            DebateMessage(
                role=DebateRole.INTERROGATOR,
                content="question",
                token_count=10.0,
                extra_field="bad",  # type: ignore[call-arg]
            )

    def test_serialization_roundtrip(self) -> None:
        msg = DebateMessage(role=DebateRole.INTERROGATOR, content="test", token_count=42.0)
        data = msg.model_dump()
        assert data == {"role": "interrogator", "content": "test", "token_count": 42.0}
        restored = DebateMessage.model_validate(data)
        assert restored == msg

    def test_json_roundtrip(self) -> None:
        msg = DebateMessage(role=DebateRole.RESPONDER, content="reply", token_count=99.5)
        json_str = msg.model_dump_json()
        restored = DebateMessage.model_validate_json(json_str)
        assert restored == msg

    def test_zero_token_count(self) -> None:
        msg = DebateMessage(role=DebateRole.INTERROGATOR, content="empty", token_count=0.0)
        assert msg.token_count == 0.0

    def test_role_from_string_value(self) -> None:
        msg = DebateMessage(role="interrogator", content="q", token_count=1.0)  # type: ignore[arg-type]
        assert msg.role == DebateRole.INTERROGATOR


# ---------------------------------------------------------------------------
# DebateVerdict
# ---------------------------------------------------------------------------


class TestDebateVerdict:
    def test_create_verdict_with_reasoning(self) -> None:
        v = DebateVerdict(score=0.85, reasoning="Good understanding")
        assert v.score == 0.85
        assert v.reasoning == "Good understanding"

    def test_create_verdict_without_reasoning(self) -> None:
        v = DebateVerdict(score=0.5)
        assert v.score == 0.5
        assert v.reasoning is None

    def test_frozen(self) -> None:
        v = DebateVerdict(score=0.5)
        with pytest.raises(pydantic.ValidationError):
            v.score = 0.9  # type: ignore[misc]

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            DebateVerdict(score=0.5, extra="bad")  # type: ignore[call-arg]

    def test_serialization_roundtrip(self) -> None:
        v = DebateVerdict(score=0.75, reasoning="Solid reasoning")
        data = v.model_dump()
        assert data == {"score": 0.75, "reasoning": "Solid reasoning"}
        restored = DebateVerdict.model_validate(data)
        assert restored == v

    def test_score_boundary_values(self) -> None:
        v_zero = DebateVerdict(score=0.0)
        assert v_zero.score == 0.0
        v_one = DebateVerdict(score=1.0)
        assert v_one.score == 1.0

    def test_json_roundtrip(self) -> None:
        v = DebateVerdict(score=0.42, reasoning="test")
        json_str = v.model_dump_json()
        restored = DebateVerdict.model_validate_json(json_str)
        assert restored == v


# ---------------------------------------------------------------------------
# DebateTranscript
# ---------------------------------------------------------------------------


class TestDebateTranscript:
    def test_create_transcript(self) -> None:
        messages = [
            DebateMessage(role=DebateRole.INTERROGATOR, content="q1", token_count=100.0),
            DebateMessage(role=DebateRole.RESPONDER, content="a1", token_count=200.0),
        ]
        verdict = DebateVerdict(score=0.9, reasoning="Clear")
        t = DebateTranscript(
            messages=messages,
            verdict=verdict,
            num_turns=1,
            total_token_count=300.0,
        )
        assert len(t.messages) == 2
        assert t.verdict.score == 0.9
        assert t.num_turns == 1
        assert t.total_token_count == 300.0

    def test_frozen(self) -> None:
        t = DebateTranscript(
            messages=[],
            verdict=DebateVerdict(score=0.5),
            num_turns=0,
            total_token_count=0.0,
        )
        with pytest.raises(pydantic.ValidationError):
            t.num_turns = 5  # type: ignore[misc]

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            DebateTranscript(
                messages=[],
                verdict=DebateVerdict(score=0.5),
                num_turns=0,
                total_token_count=0.0,
                extra="bad",  # type: ignore[call-arg]
            )

    def test_serialization_roundtrip(self) -> None:
        messages = [
            DebateMessage(role=DebateRole.INTERROGATOR, content="q1", token_count=100.0),
            DebateMessage(role=DebateRole.RESPONDER, content="a1", token_count=200.0),
        ]
        t = DebateTranscript(
            messages=messages,
            verdict=DebateVerdict(score=0.8),
            num_turns=1,
            total_token_count=300.0,
        )
        data = t.model_dump()
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) == 2
        assert data["verdict"]["score"] == 0.8
        restored = DebateTranscript.model_validate(data)
        assert restored == t

    def test_json_roundtrip(self) -> None:
        t = DebateTranscript(
            messages=[
                DebateMessage(role=DebateRole.INTERROGATOR, content="q", token_count=50.0),
            ],
            verdict=DebateVerdict(score=0.6, reasoning="OK"),
            num_turns=1,
            total_token_count=50.0,
        )
        json_str = t.model_dump_json()
        restored = DebateTranscript.model_validate_json(json_str)
        assert restored == t

    def test_empty_messages(self) -> None:
        t = DebateTranscript(
            messages=[],
            verdict=DebateVerdict(score=0.5),
            num_turns=0,
            total_token_count=0.0,
        )
        assert len(t.messages) == 0

    def test_model_dump_structure(self) -> None:
        """Verify model_dump() produces the dict structure expected by scorer metadata."""
        t = DebateTranscript(
            messages=[
                DebateMessage(role=DebateRole.INTERROGATOR, content="q", token_count=50.0),
                DebateMessage(role=DebateRole.RESPONDER, content="a", token_count=80.0),
            ],
            verdict=DebateVerdict(score=0.7, reasoning="Fine"),
            num_turns=1,
            total_token_count=130.0,
        )
        d = t.model_dump()
        assert set(d.keys()) == {"messages", "verdict", "num_turns", "total_token_count"}
        assert d["messages"][0]["role"] == "interrogator"
        assert d["messages"][1]["role"] == "responder"
        assert isinstance(d["verdict"], dict)
        assert "score" in d["verdict"]
