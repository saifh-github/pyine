"""Hydra-zen config builder for LLM-based grader evaluation configs."""

import typing

import hydra.conf
import hydra_zen
import hydra_zen.typing

import pyine.apps.trainers.openai_finetune
import pyine.configs.base
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.organisms.datamodules.shortcuts
import pyine.organisms.models.utils.openai
import pyine.utils.llm_providers
import pyine.utils.reprod


def register_hydra_configs(llm_grader_provider_config_store: hydra_zen.ZenStore) -> hydra_zen.ZenStore:
    """Registers datamodule-specific configs in the hydra store."""
    # @@@@@ TODO: update w/ base?
    llm_grader_provider_config_store(
        hydra_zen.builds(
            pyine.utils.llm_providers.LLMProviderConfig,
            provider="openai",
            rate_limiter_config=None,
            with_retry_config=None,
            model_kwargs=dict(model="gpt-5-mini"),
            hydra_convert="object",
        ),
        name="openai_gpt-5-mini",
    )
    llm_grader_provider_config_store(
        hydra_zen.builds(
            pyine.utils.llm_providers.LLMProviderConfig,
            provider="openai",
            rate_limiter_config=None,
            with_retry_config=None,
            model_kwargs=dict(model="gpt-5-nano"),
            hydra_convert="object",
        ),
        name="openai_gpt-5-nano",
    )
    return llm_grader_provider_config_store
