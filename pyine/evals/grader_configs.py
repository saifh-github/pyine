"""Hydra-zen config builder for LLM-based grader evaluation configs."""

import hydra_zen
import hydra_zen.typing

import pyine.utils.llm_providers


def register_hydra_configs(llm_grader_provider_config_store: hydra_zen.ZenStore) -> hydra_zen.ZenStore:
    """Registers datamodule-specific configs in the hydra store."""
    parent_config = hydra_zen.builds(
        pyine.utils.llm_providers.LLMProviderConfig,
        provider="openai",
        rate_limiter_config=None,
        with_retry_config=None,
        hydra_convert="object",
    )
    openai_gpt_5_mini = hydra_zen.make_config(
        model_kwargs=dict(model="gpt-5-mini"),
        bases=(parent_config,),
    )
    llm_grader_provider_config_store(openai_gpt_5_mini, name="openai_gpt-5-mini")
    openai_gpt_5_nano = hydra_zen.make_config(
        model_kwargs=dict(model="gpt-5-nano"),
        bases=(parent_config,),
    )
    llm_grader_provider_config_store(openai_gpt_5_nano, name="openai_gpt-5-nano")
    return llm_grader_provider_config_store
