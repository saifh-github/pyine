import langchain_core.output_parsers
import langchain_core.prompts
import langchain_core.runnables
import pydantic

import pyine.prompts.prompt_utils
import pyine.utils.llm_providers
import pyine.utils.pydantic_loader


class CallableAnalysisResponse(pydantic.BaseModel):
    """Response model for callable entrypoint analysis.

    This model is used to describe the output of a model tasked with identifying the callable
    entrypoint of a Python program. See the corresponding template YAML file for more details.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""
    entrypoint_function_name: str = pydantic.Field(
        description=(
            "Specifies the name of the entrypoint function; if the function is inside "
            "a class, this field should ONLY be the name of the function inside that class."
        ),
    )
    entrypoint_function_arg_names: list[str] = pydantic.Field(
        description=(
            "Specifies the names of the arguments that the entrypoint function takes; "
            "if the function takes no arguments, this field should be an empty list."
        ),
    )
    parent_class_name: str = pydantic.Field(
        description=(
            "Specifies the name of the parent class of the entrypoint function; this is only "
            "relevant if the entrypoint function is inside a class; if it is not, this field "
            "should be an empty string."
        ),
    )
    parent_class_arg_names: list[str] = pydantic.Field(
        description=(
            "Specifies the names of the arguments that the parent class of the entrypoint "
            "function requires in order to be instantiated; if the class constructor takes "
            "no arguments, this field should be an empty list; this field is only relevant "
            "if the targeted entrypoint function is inside a class."
        ),
    )


output_parser = langchain_core.output_parsers.PydanticOutputParser(
    pydantic_object=CallableAnalysisResponse,
)
expected_output_format_str = output_parser.get_format_instructions()


def get_prompt_config(
    version: str | None = None,
) -> pyine.prompts.prompt_utils.PromptConfig:
    """Get the prompt configuration for the callable analysis prompt.

    Args:
        version: The version of the prompt to retrieve. If None, the default version is returned.
    """
    # first, make sure the pydantic loader has already registered this class
    pyine.utils.pydantic_loader.PydanticYAMLLoader.register_models_from_module(__name__)
    # note: this will be cached by the prompt manager
    return pyine.prompts.prompt_utils.get_prompt_config(
        prompt_name="callable_analysis",
        version=version,
    )


def get_prompt_template(
    version: str | None = None,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
) -> langchain_core.prompts.PromptTemplate:
    """Get the langchain prompt template for the callable analysis prompt.

    Args:
        version: The version of the prompt to retrieve. If None, the default version is returned.
        include_examples: Whether to include few-shot examples in the template.
        target_examples: List of examples to target when rendering the prompt. Can pass in
            a list of example indices, or an integer that specifies the number of samples to
            pick randomly. If `None` is provided instead, all examples are included.
    """
    prompt_config = get_prompt_config(version=version)
    context_vars = dict(
        expected_output_format=expected_output_format_str,
    )
    return prompt_config.create_prompt_template(
        include_examples=include_examples,
        target_examples=target_examples,
        context_variables=context_vars,
    )


def get_chain(
    version: str | None = None,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    **llm_provider_kwargs,
) -> langchain_core.runnables.Runnable:
    """Get an inference chain based on the above prompt template and parser.

    Args:
        version: The version of the prompt to retrieve. If None, the default version is returned.
        include_examples: Whether to include few-shot examples in the template.
        target_examples: List of examples to target when rendering the prompt. Can pass in
            a list of example indices, or an integer that specifies the number of samples to
            pick randomly. If `None` is provided instead, all examples are included.
        llm_provider_kwargs: Keyword arguments to pass to the LLM provider getter.
    """
    llm = pyine.utils.llm_providers.get_llm_from_provider(**llm_provider_kwargs)
    llm_with_structured_output = llm.with_structured_output(CallableAnalysisResponse)
    prompt_template = get_prompt_template(
        version=version,
        include_examples=include_examples,
        target_examples=target_examples,
    )
    chain = langchain_core.runnables.RunnableSequence(
        prompt_template,
        llm_with_structured_output,
    )
    return chain
