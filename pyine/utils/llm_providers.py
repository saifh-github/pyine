import os

import langchain_core.prompts
import langchain_core.runnables
import langchain_deepseek
import langchain_openai
import pydantic


def get_llm_from_provider(
    **kwargs,
) -> langchain_openai.ChatOpenAI:
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
        llm = langchain_openai.ChatOpenAI(
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


def get_chain(
    prompt_template: langchain_core.prompts.PromptTemplate,
    **llm_provider_kwargs,
) -> langchain_core.runnables.Runnable:
    """Get an inference chain based on a given prompt template.

    Args:
        prompt_template: Prompt template to use for the inference chain.
        llm_provider_kwargs: Keyword arguments to pass to the LLM provider getter.
    """
    llm = get_llm_from_provider(**llm_provider_kwargs)
    # Create a partial chain that injects the examples
    code_execution_chain = langchain_core.runnables.RunnableSequence(
        prompt_template,
        llm,
    )
    return code_execution_chain


def get_structured_output_chain(
    prompt_template: langchain_core.prompts.PromptTemplate,
    pydantic_model: pydantic.BaseModel,
    **llm_provider_kwargs,
) -> langchain_core.runnables.Runnable:
    """Get a structured output inference chain based on a given prompt template and pydantic model.

    Args:
        prompt_template: Prompt template to use for the inference chain.
        pydantic_model: Pydantic model to use for parsing the LLM output.
        llm_provider_kwargs: Keyword arguments to pass to the LLM provider getter.
    """
    llm = get_llm_from_provider(**llm_provider_kwargs)
    llm_with_structured_output = llm.with_structured_output(pydantic_model)
    chain = langchain_core.runnables.RunnableSequence(
        prompt_template,
        llm_with_structured_output,
    )
    return chain
