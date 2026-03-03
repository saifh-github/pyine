"""Configuration for the prompted LLM guardrail scorer."""

import pydantic

import pyine.utils.llm_providers


class PromptedLLMGuardrailConfig(pydantic.BaseModel):
    """Configuration for the prompted LLM guardrail scorer.

    Example::

        config = PromptedLLMGuardrailConfig(
            llm_provider=LLMProviderConfig(
                provider="openai",
                model_kwargs={"model": "gpt-4o-mini", "temperature": 0.0},
            ),
        )
        scorer = PromptedLLMGuardrailScorer(config)
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    # --- LLM provider ---
    llm_provider: pyine.utils.llm_providers.LLMProviderConfig
    """LLM provider config (provider, model, rate limiter, retry, etc.)."""

    # --- Prompt ---
    prompt_name: str = "guardrail/correctness_judge"
    """Name of the prompt template to use (resolved by PromptManager)."""
    prompt_version: str | None = None
    """Prompt version override. None uses the default version."""
    use_chat_template: bool = True
    """Whether to use a chat prompt template (system + human message)."""

    # --- Scoring ---
    max_workers: int = pydantic.Field(default=10, ge=1)
    """Maximum number of concurrent LLM calls (ThreadPoolExecutor worker threads).
    Rate limiting is also handled by the LLMProviderConfig's rate_limiter_config."""
    default_score_on_error: float = pydantic.Field(default=0.5, ge=0.0, le=1.0)
    """Score to assign when the LLM call fails after retries."""
