import hydra_zen

import pyine.configs.schemas
import pyine.configs.utils
import pyine.utils.llm_providers
from pyine.evals.common import EvalType


def get_grader_provider_configs(group: str) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns LLM-based-grader provider configs for hydra zen storage."""
    openai_llm_provider_config = hydra_zen.builds(
        pyine.utils.llm_providers.LLMProviderConfig,
        provider="openai",
        # -------------
        populate_full_signature=True,
        hydra_convert="object",
    )
    openai_gpt5_mini_config = pyine.configs.utils.make_config_description(
        name="openai_gpt5mini",
        group=group,
        description="LLM provider settings for grading using OpenAI's gpt-5-mini.",
        config={
            "model_kwargs": {"model": "gpt-5-mini"},
            # -------------
            "bases": (openai_llm_provider_config,),
        },
    )
    openai_gpt5_nano_config = pyine.configs.utils.make_config_description(
        name="openai_gpt5nano",
        group=group,
        description="LLM provider settings for grading using OpenAI's gpt-5-nano (cheaper!).",
        config={
            "model_kwargs": {"model": "gpt-5-nano"},
            # -------------
            "bases": (openai_llm_provider_config,),
        },
    )
    return [openai_gpt5_mini_config, openai_gpt5_nano_config]


def get_evals_configs(
    eval_type: EvalType,
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns evals configs for hydra zen storage."""
    import pyine.evals.code_exec.configs

    if eval_type == EvalType.CODE_EXEC:
        return pyine.evals.code_exec.configs.get_evals_configs(group=group)
    raise NotImplementedError(f"evaluation type {eval_type} not implemented")
