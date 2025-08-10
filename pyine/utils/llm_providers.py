import os

import langchain_core.prompts
import langchain_core.rate_limiters
import langchain_core.runnables
import langchain_deepseek
import langchain_openai
import pydantic

LLMType = langchain_openai.ChatOpenAI  # just to make typing easier elsewhere


def get_llm_from_provider(
    provider: str,
    rate_limiter_config: dict | None = None,
    with_retry_config: dict | None = None,
    **kwargs,
) -> LLMType:
    """Get a default LLM from a provider for quick prototyping and testing.

    Currently supports DeepSeek and OpenAI.
    """
    rate_limiter = None
    if rate_limiter_config is not None:
        rate_limiter = langchain_core.rate_limiters.InMemoryRateLimiter(
            **rate_limiter_config,
        )
    if provider == "deepseek":
        if "api_key" not in kwargs:
            kwargs.update({"api_key": os.environ.get("DEEPSEEK_API_KEY")})
        if "base_url" not in kwargs:
            kwargs.update({"base_url": os.environ.get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")})
        llm = langchain_deepseek.ChatDeepSeek(rate_limiter=rate_limiter, **kwargs)
    elif provider == "openai":
        if "api_key" not in kwargs:
            kwargs.update({"api_key": os.environ.get("OPENAI_API_KEY")})
        if "base_url" not in kwargs:
            kwargs.update({"base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")})
        llm = langchain_openai.ChatOpenAI(rate_limiter=rate_limiter, **kwargs)
    else:
        raise ValueError(f"Invalid provider: {provider}")
    if with_retry_config is not None:
        llm = llm.with_retries(**with_retry_config)  # if you want to e.g. customize the retry backoff
    return llm


def get_chain(
    prompt_template: langchain_core.prompts.PromptTemplate,
    llm: LLMType,
    pydantic_model: pydantic.BaseModel | None = None,
) -> langchain_core.runnables.Runnable:
    """Get an inference chain based on a given prompt template.

    If a pydantic model is provided, the chain will be configured to parse the LLM output.

    Args:
        prompt_template: Prompt template to use for the inference chain.
        llm: LLM to use for the inference chain.
        pydantic_model: Pydantic model to use for parsing the LLM output (if any).
    """
    if pydantic_model is not None:
        llm = llm.with_structured_output(pydantic_model)
    code_execution_chain = langchain_core.runnables.RunnableSequence(
        prompt_template,
        llm,
    )
    return code_execution_chain
