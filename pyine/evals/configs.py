import openai

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
                "text": {"verbosity": "low"},
                "timeout": 60,  # gpt-5-nano requests should be pretty fast
                "max_retries": 10,  # be generous here
            },
            # note: the GPT-5 series dropped support for customizing temperature, so we don't set anything here
            "with_retry_config": {
                "retry_if_exception_type": (
                    openai.APITimeoutError,  # stalled/timeout
                    openai.APIConnectionError,  # network flake
                    openai.RateLimitError,  # 429s
                    openai.InternalServerError,  # 5xx
                ),
                "wait_exponential_jitter": True,  # backoff + jitter
                "stop_after_attempt": 3,  # on top of max_retries above
            },
            "rate_limiter_config": {
                "requests_per_second": 50,  # tier 4, with gpt-5-nano = 10K RPM, so this is a good/safe default
                "max_bucket_size": 50,
            },
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
    import pyine.evals.code_exec.configs

    if eval_type == EvalType.CODE_EXEC:
        return pyine.evals.code_exec.configs.get_evals_configs(group=group)
    raise NotImplementedError(f"evaluation type {eval_type} not implemented")
