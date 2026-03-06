"""Unit tests for DebateGuardrailConfig."""

from __future__ import annotations

import pydantic
import pytest

from pyine.guardrails.llm_debate.configs import DebateGuardrailConfig

from .conftest import make_debate_config, make_llm_provider

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_config_defaults(self) -> None:
        config = DebateGuardrailConfig(
            interrogator_provider=make_llm_provider(),
            responder_provider=make_llm_provider(),
        )
        assert config.interrogator_prompt_name == "guardrail/debate_interrogator"
        assert config.responder_prompt_name == "guardrail/debate_responder"
        assert config.use_chat_template is True
        assert config.max_debate_turns == 3
        assert config.responder_sees_debate_history is True
        assert config.max_workers == 5
        assert config.default_score_on_error == 0.5
        assert config.debug_log_transcript_every_n == 0

    def test_config_accepts_custom_values(self) -> None:
        config = DebateGuardrailConfig(
            interrogator_provider=make_llm_provider(),
            responder_provider=make_llm_provider(),
            interrogator_prompt_name="custom/interrogator",
            responder_prompt_name="custom/responder",
            use_chat_template=False,
            max_debate_turns=5,
            responder_sees_debate_history=False,
            max_workers=3,
            default_score_on_error=0.3,
            debug_log_transcript_every_n=50,
        )
        assert config.interrogator_prompt_name == "custom/interrogator"
        assert config.responder_prompt_name == "custom/responder"
        assert config.use_chat_template is False
        assert config.max_debate_turns == 5
        assert config.responder_sees_debate_history is False
        assert config.max_workers == 3
        assert config.default_score_on_error == 0.3
        assert config.debug_log_transcript_every_n == 50


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_max_workers_below_min(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            make_debate_config(max_workers=0)

    def test_max_workers_negative(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            make_debate_config(max_workers=-1)

    def test_default_score_too_low(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            make_debate_config(default_score_on_error=-0.1)

    def test_default_score_too_high(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            make_debate_config(default_score_on_error=1.1)

    def test_default_score_bounds_exact(self) -> None:
        config_zero = make_debate_config(default_score_on_error=0.0)
        assert config_zero.default_score_on_error == 0.0
        config_one = make_debate_config(default_score_on_error=1.0)
        assert config_one.default_score_on_error == 1.0

    def test_max_debate_turns_below_min(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            make_debate_config(max_debate_turns=0)

    def test_max_debate_turns_above_max(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            make_debate_config(max_debate_turns=11)

    def test_max_debate_turns_bounds_exact(self) -> None:
        config_one = make_debate_config(max_debate_turns=1)
        assert config_one.max_debate_turns == 1
        config_ten = make_debate_config(max_debate_turns=10)
        assert config_ten.max_debate_turns == 10

    def test_debug_log_negative(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            make_debate_config(debug_log_transcript_every_n=-1)

    def test_debug_log_zero_allowed(self) -> None:
        config = make_debate_config(debug_log_transcript_every_n=0)
        assert config.debug_log_transcript_every_n == 0


# ---------------------------------------------------------------------------
# Frozen / Extra
# ---------------------------------------------------------------------------


class TestConfigFrozen:
    def test_config_frozen(self) -> None:
        config = make_debate_config()
        with pytest.raises(pydantic.ValidationError):
            config.max_workers = 10  # type: ignore[misc]

    def test_config_frozen_nested(self) -> None:
        config = make_debate_config()
        with pytest.raises(pydantic.ValidationError):
            config.max_debate_turns = 5  # type: ignore[misc]

    def test_config_forbids_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            DebateGuardrailConfig(
                interrogator_provider=make_llm_provider(),
                responder_provider=make_llm_provider(),
                unknown_field="bad",  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# Provider configs
# ---------------------------------------------------------------------------


class TestProviderConfigs:
    def test_requires_interrogator_provider(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            DebateGuardrailConfig(
                responder_provider=make_llm_provider(),  # type: ignore[call-arg]
            )

    def test_requires_responder_provider(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            DebateGuardrailConfig(
                interrogator_provider=make_llm_provider(),  # type: ignore[call-arg]
            )

    def test_different_providers(self) -> None:
        config = DebateGuardrailConfig(
            interrogator_provider=make_llm_provider(provider="openai"),
            responder_provider=make_llm_provider(provider="vllm"),
        )
        assert config.interrogator_provider.provider == "openai"
        assert config.responder_provider.provider == "vllm"
