"""Unit tests for PromptedLLMGuardrailConfig."""

from __future__ import annotations

import pydantic
import pytest

import pyine.utils.llm_providers
from pyine.guardrails.prompted_llm.configs import PromptedLLMGuardrailConfig


def _make_llm_provider(**overrides: object) -> pyine.utils.llm_providers.LLMProviderConfig:
    """Build a minimal LLMProviderConfig for testing."""
    defaults: dict[str, object] = {
        "provider": "openai",
        "model_kwargs": {"model": "gpt-4o-mini", "temperature": 0.0},
    }
    defaults.update(overrides)
    return pyine.utils.llm_providers.LLMProviderConfig(**defaults)  # type: ignore[arg-type]


class TestConfigDefaults:
    def test_config_defaults(self) -> None:
        config = PromptedLLMGuardrailConfig(llm_provider=_make_llm_provider())
        assert config.prompt_name == "guardrail/correctness_judge"
        assert config.prompt_version is None
        assert config.use_chat_template is True
        assert config.max_workers == 10
        assert config.default_score_on_error == 0.5

    def test_config_accepts_custom_values(self) -> None:
        config = PromptedLLMGuardrailConfig(
            llm_provider=_make_llm_provider(),
            prompt_name="custom/prompt",
            prompt_version="v2",
            use_chat_template=False,
            max_workers=5,
            default_score_on_error=0.3,
        )
        assert config.prompt_name == "custom/prompt"
        assert config.prompt_version == "v2"
        assert config.use_chat_template is False
        assert config.max_workers == 5
        assert config.default_score_on_error == 0.3


class TestConfigValidation:
    def test_config_validation_max_workers_below_min(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            PromptedLLMGuardrailConfig(
                llm_provider=_make_llm_provider(),
                max_workers=0,
            )

    def test_config_validation_max_workers_negative(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            PromptedLLMGuardrailConfig(
                llm_provider=_make_llm_provider(),
                max_workers=-1,
            )

    def test_config_validation_default_score_too_low(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            PromptedLLMGuardrailConfig(
                llm_provider=_make_llm_provider(),
                default_score_on_error=-0.1,
            )

    def test_config_validation_default_score_too_high(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            PromptedLLMGuardrailConfig(
                llm_provider=_make_llm_provider(),
                default_score_on_error=1.1,
            )

    def test_config_validation_default_score_bounds_exact(self) -> None:
        """Boundary values 0.0 and 1.0 should be accepted."""
        config_zero = PromptedLLMGuardrailConfig(
            llm_provider=_make_llm_provider(),
            default_score_on_error=0.0,
        )
        assert config_zero.default_score_on_error == 0.0

        config_one = PromptedLLMGuardrailConfig(
            llm_provider=_make_llm_provider(),
            default_score_on_error=1.0,
        )
        assert config_one.default_score_on_error == 1.0


class TestConfigFrozen:
    def test_config_frozen(self) -> None:
        config = PromptedLLMGuardrailConfig(llm_provider=_make_llm_provider())
        with pytest.raises(pydantic.ValidationError):
            config.max_workers = 5  # type: ignore[misc]

    def test_config_forbids_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            PromptedLLMGuardrailConfig(
                llm_provider=_make_llm_provider(),
                unknown_field="bad",  # type: ignore[call-arg]
            )
