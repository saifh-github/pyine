import os

import langchain_deepseek
import langchain_openai


def get_llm_from_provider(
    **kwargs,
) -> langchain_openai.llms.base.BaseOpenAI:
    """Get a default LLM from a provider for quick prototyping and testing.

    Currently supports DeepSeek and OpenAI.
    """
    provider = kwargs.get("provider", "deepseek")
    if provider == "deepseek":
        llm = langchain_deepseek.ChatDeepSeek(
            model=kwargs.get("model", "deepseek-chat"),
            temperature=kwargs.get("temperature", 0.0),  # recommended setting for coding/math
            max_tokens=kwargs.get("max_tokens", 1024),
            timeout=kwargs.get("timeout", None),
            max_retries=kwargs.get("max_retries", 50),
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=os.environ.get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1"),
        )
    elif provider == "openai":
        llm = langchain_openai.OpenAI(
            model=kwargs.get("model", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 1024),
            timeout=kwargs.get("timeout", None),
            max_retries=kwargs.get("max_retries", 50),
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")
    return llm
