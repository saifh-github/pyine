"""Hydra-zen config builder for LLM-based grader evaluation configs."""

import hydra_zen
import hydra_zen.typing

import pyine.utils.llm_providers


def store_hydra_configs(store: hydra_zen.ZenStore) -> None:
    """Stores grader-specific configs in the provided hydra zen store."""
    openai_llm_provider_config = hydra_zen.builds(
        pyine.utils.llm_providers.LLMProviderConfig,
        provider="openai",
        # -------------
        populate_full_signature=True,
        hydra_convert="object",
    )
    store(
        hydra_zen.make_config(
            model_kwargs=dict(model="gpt-5-mini"),
            bases=(openai_llm_provider_config,),
        ),
        name="openai_gpt-5-mini",
    )
    store(
        hydra_zen.make_config(
            model_kwargs=dict(model="gpt-5-nano"),
            bases=(openai_llm_provider_config,),
        ),
        name="openai_gpt-5-nano",
    )
