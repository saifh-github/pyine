"""Hydra-zen config builder for LLM-based grader evaluation configs."""

import hydra_zen
import hydra_zen.typing

import pyine.configs.schemas
import pyine.utils.llm_providers


def get_configs(group: str) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns LLM-based-grader-specific configs for hydra zen storage."""
    openai_llm_provider_config = hydra_zen.builds(
        pyine.utils.llm_providers.LLMProviderConfig,
        provider="openai",
        # -------------
        populate_full_signature=True,
        hydra_convert="object",
    )
    gpt5_mini_config = pyine.configs.schemas.ConfigDescription(
        name="openai_gpt-5-mini",
        group=group,
        config=hydra_zen.make_config(
            model_kwargs=dict(model="gpt-5-mini"),
            # -------------
            bases=(openai_llm_provider_config,),
            zen_meta={
                "__description__": "LLM provider settings for grading using OpenAI's gpt-5-mini.",
            },
        ),
    )
    gpt5_nano_config = pyine.configs.schemas.ConfigDescription(
        name="openai_gpt-5-nano",
        group=group,
        config=hydra_zen.make_config(
            model_kwargs=dict(model="gpt-5-nano"),
            # -------------
            bases=(openai_llm_provider_config,),
            zen_meta={
                "__description__": "LLM provider settings for grading using OpenAI's gpt-5-nano (cheaper!).",
            },
        ),
    )
    return [gpt5_mini_config, gpt5_nano_config]
