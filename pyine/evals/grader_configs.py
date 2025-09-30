"""Hydra-zen config builder for LLM-as-a-judge (grader) evaluation configs."""

import hydra_zen
import hydra_zen.typing

import pyine.configs.schemas
import pyine.utils.llm_providers


def get_provider_configs(group: str) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns LLM-based-grader provider configs for hydra zen storage."""
    openai_llm_provider_config = hydra_zen.builds(
        pyine.utils.llm_providers.LLMProviderConfig,
        provider="openai",
        # -------------
        populate_full_signature=True,
        hydra_convert="object",
    )
    openai_gpt5_mini_config = pyine.configs.schemas.ConfigDescription(
        name="openai_gpt5mini",
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
    openai_gpt5_nano_config = pyine.configs.schemas.ConfigDescription(
        name="openai_gpt5nano",
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
    return [openai_gpt5_mini_config, openai_gpt5_nano_config]
