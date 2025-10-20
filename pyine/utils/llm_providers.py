from __future__ import annotations

import os
import typing

import langchain_core.rate_limiters
import langchain_deepseek
import langchain_openai
import langchain_openai.chat_models.base
import openai
import pydantic

SupportedProviderType = typing.Literal[
    "deepseek",
    "openai",
    # @@@@ TODO: add more here if needed, e.g. for local vLLM server?
]
"""Supported LLM providers."""


class LLMProviderConfig(pydantic.BaseModel):
    """Configuration for a LLM provider."""

    provider: SupportedProviderType
    """Provider name."""
    rate_limiter_config: dict[str, typing.Any] | None = None
    """Configuration for the LangChain in-memory rate limiter."""
    with_retry_config: dict[str, typing.Any] | None = None
    """Configuration for the LangChain retry mechanism (provider-error-specific)."""
    model_kwargs: dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: typing.cast("dict[str, typing.Any]", {}),
    )
    """Keyword arguments to be passed to the LLM constructor."""

    @classmethod
    def from_dict(cls, config: dict[str, typing.Any]) -> LLMProviderConfig:
        """Parses a configuration model from a given dictionary."""
        if "provider" not in config:
            raise ValueError("missing required 'provider' field in config")
        config_top_fields = {k: v for k, v in config.items() if k in cls.model_fields}
        config_without_top_keys = {k: v for k, v in config.items() if k not in cls.model_fields}
        return cls(**config_top_fields, model_kwargs=config_without_top_keys)

    def get_model(self) -> langchain_openai.chat_models.base.BaseChatOpenAI:
        """Returns a LangChain LLM instance based on the provider config."""
        return get_model_from_provider_config(self)


def get_model_from_provider(
    provider: SupportedProviderType,
    rate_limiter_config: dict[str, typing.Any] | None = None,
    with_retry_config: dict[str, typing.Any] | None = None,
    **model_kwargs: typing.Any,  # will be forwarded to the chat model constructor
) -> langchain_openai.chat_models.base.BaseChatOpenAI:
    """Get a LangChain language model instance from a provider following the OpenAI-style API.

    Args:
        provider: The provider name. Currently supports "deepseek" and "openai".
        rate_limiter_config: Configuration for the LangChain in-memory rate limiter (if needed).
        with_retry_config: Configuration for the LangChain retry mechanism (if needed). Note that
            this retry mechanism will only apply to errors originating from the LLM provider itself.
        model_kwargs: Keyword arguments to be passed to the LLM constructor.

    Returns:
        The LangChain LLM instance that can be used for chain invocations.
    """
    rate_limiter: langchain_core.rate_limiters.InMemoryRateLimiter | None = None
    if rate_limiter_config is not None:
        rate_limiter = langchain_core.rate_limiters.InMemoryRateLimiter(
            **rate_limiter_config,
        )
    got_client_obj = "client" in model_kwargs
    if provider == "deepseek":
        if not got_client_obj:
            if "api_key" not in model_kwargs:
                env_api_key = os.environ.get("DEEPSEEK_API_KEY")
                if env_api_key:
                    model_kwargs["api_key"] = env_api_key
            if "base_url" not in model_kwargs:
                model_kwargs.update(
                    {"base_url": os.environ.get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")}
                )
        llm = typing.cast(
            "langchain_openai.chat_models.base.BaseChatOpenAI",
            langchain_deepseek.ChatDeepSeek(rate_limiter=rate_limiter, **model_kwargs),
        )
    elif provider == "openai":
        if not got_client_obj:
            if "api_key" not in model_kwargs:
                env_api_key = os.environ.get("OPENAI_API_KEY")
                if env_api_key:
                    model_kwargs["api_key"] = env_api_key
            if "base_url" not in model_kwargs:
                model_kwargs.update({"base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")})
        llm = langchain_openai.ChatOpenAI(rate_limiter=rate_limiter, **model_kwargs)
    else:
        raise NotImplementedError(f"Invalid provider: {provider}")
    if with_retry_config is not None:
        llm = typing.cast("langchain_openai.chat_models.base.BaseChatOpenAI", llm.with_retry(**with_retry_config))
    return llm


def get_model_from_provider_config(
    provider_config: LLMProviderConfig,
) -> langchain_openai.chat_models.base.BaseChatOpenAI:
    """Get a default LLM from a provider config for quick prototyping and testing.

    See the `get_model_from_provider` function for more details.
    """
    return get_model_from_provider(
        provider=provider_config.provider,
        rate_limiter_config=provider_config.rate_limiter_config,
        with_retry_config=provider_config.with_retry_config,
        **provider_config.model_kwargs,
    )


def get_default_openai_provider_retry_config(
    max_retries: int = 3,
) -> dict[str, typing.Any]:
    """Returns a default LangChain `with_retry` configuration that can be used w/ OpenAI."""
    return {
        "retry_if_exception_type": (
            openai.APITimeoutError,  # stalled/timeout
            openai.APIConnectionError,  # network flake
            openai.RateLimitError,  # 429s
            openai.InternalServerError,  # 5xx
        ),
        "wait_exponential_jitter": True,  # backoff + jitter
        "stop_after_attempt": max_retries,  # on top of max_retries specified in model config
    }


def get_default_openai_provider_rate_limit_config(
    usage_tier: int = 4,
) -> dict[str, typing.Any]:
    """Returns a default LangChain `rate_limiter_config` configuration that can be used w/ OpenAI."""
    if usage_tier != 4:
        raise NotImplementedError(f"unsupported/unknown usage tier limitations: {usage_tier=}")
    return {
        "requests_per_second": 50,
        "max_bucket_size": 50,
    }
