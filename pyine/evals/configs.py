import pyine.configs.schemas
import pyine.configs.utils
import pyine.utils.llm_providers
from pyine.evals.common import EvalType


def get_grader_provider_configs(group: str) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns LLM-based-grader provider configs for hydra zen storage."""
    base_openai_config = pyine.configs.utils.make_config_description(
        pyine.utils.llm_providers.LLMProviderConfig,
        name="openai",
        group=group,
        description="LLM provider settings for grading using an OpenAI model.",
        config={
            "provider": "openai",
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    openai_gpt5_nano_config = pyine.configs.utils.make_config_description(
        name="openai_gpt5nano",
        group=group,
        description="LLM provider settings for grading using OpenAI's gpt-5-nano.",
        config={
            "model_kwargs": {"model": "gpt-5-nano"},
            # -------------
            "bases": (base_openai_config.config,),
        },
    )
    openai_gpt5_nano_for_scoring_config = pyine.configs.utils.make_config_description(
        name="openai_gpt5nano_scoring",
        group=group,
        description=(
            "LLM provider settings for score-based grading using OpenAI's gpt-5-nano; also sets "
            "numerous settings for better robustness with tier-4 API usage assumptions."
        ),
        config={
            "model_kwargs": {
                "model": "gpt-5-nano",
                "max_tokens": 1024,
                "reasoning": {"effort": "minimal"},
                "timeout": 15,  # gpt-5-nano requests should be pretty fast
                "max_retries": 0,  # retries defined via retry config below
            },
            # note: the GPT-5 series dropped support for customizing temperature, so we don't set anything here
            "with_retry_config": pyine.utils.llm_providers.get_default_openai_provider_retry_config(10),
            "rate_limiter_config": pyine.utils.llm_providers.get_default_openai_provider_rate_limit_config(),
            # -------------
            "bases": (base_openai_config.config,),
        },
    )
    return [base_openai_config, openai_gpt5_nano_config, openai_gpt5_nano_for_scoring_config]


def get_evals_configs(
    eval_type: EvalType,
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns evals configs for hydra zen storage."""
    if eval_type == EvalType.CODE_EXEC:
        import pyine.evals.code_exec.configs

        return pyine.evals.code_exec.configs.get_evals_configs(group=group)
    if eval_type == EvalType.CORRECTNESS:
        import pyine.evals.correctness.configs

        return pyine.evals.correctness.configs.get_evals_configs(group=group)
    raise NotImplementedError(f"evaluation type {eval_type} not implemented")
