import functools
import os

import langchain_core.language_models
import langchain_core.prompts
import langchain_core.rate_limiters
import langchain_core.runnables
import langchain_deepseek
import langchain_openai


@functools.wraps(langchain_openai.chat_models.base.BaseChatOpenAI)
def get_model_from_provider(
    provider: str,
    rate_limiter_config: dict | None = None,
    with_retry_config: dict | None = None,
    **model_kwargs,  # will be forwarded to the chat model constructor
) -> langchain_core.language_models.BaseLanguageModel:
    """Get a default LLM from a provider for quick prototyping and testing.

    Currently supports DeepSeek and OpenAI.
    """
    rate_limiter = None
    if rate_limiter_config is not None:
        rate_limiter = langchain_core.rate_limiters.InMemoryRateLimiter(
            **rate_limiter_config,
        )
    got_client_obj = "client" in model_kwargs
    if provider == "deepseek":
        if not got_client_obj:
            if "api_key" not in model_kwargs:
                model_kwargs.update({"api_key": os.environ.get("DEEPSEEK_API_KEY")})
            if "base_url" not in model_kwargs:
                model_kwargs.update(
                    {"base_url": os.environ.get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")}
                )
        llm = langchain_deepseek.ChatDeepSeek(rate_limiter=rate_limiter, **model_kwargs)
    elif provider == "openai":
        if not got_client_obj:
            if "api_key" not in model_kwargs:
                model_kwargs.update({"api_key": os.environ.get("OPENAI_API_KEY")})
            if "base_url" not in model_kwargs:
                model_kwargs.update({"base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")})
        llm = langchain_openai.ChatOpenAI(rate_limiter=rate_limiter, **model_kwargs)
    else:
        raise ValueError(f"Invalid provider: {provider}")
    if with_retry_config is not None:
        llm = llm.with_retries(**with_retry_config)  # if you want to e.g. customize the retry backoff
    return llm
