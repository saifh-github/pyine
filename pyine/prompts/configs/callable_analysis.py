import typing

import langchain_core.output_parsers
import langchain_core.prompts
import pydantic

import pyine.prompts.manager
import pyine.prompts.utils
import pyine.utils.pydantic_loader

python_def_pattern = r"^[A-Za-z_][A-Za-z0-9_]*$"
"""Regex pattern to use to identify any variable/function/method/class name definition."""

type PythonDefType = typing.Annotated[pydantic.StrictStr, pydantic.Field(pattern=python_def_pattern)]
"""Type annotation for a Python variable/function/method/class name definition."""


class CallableAnalysisResponse(pydantic.BaseModel):
    """Response model for callable entrypoint analysis.

    This model is used to describe the output of a model tasked with identifying the callable
    entrypoint of a Python program. See the corresponding template YAML file for more details.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""
    entrypoint_function_name: PythonDefType = pydantic.Field(
        description=(
            "Specifies the name of the entrypoint function/method; if a class is meant to be used, "
            "this field should ONLY be the name of the method inside that class."
        ),
    )
    # noinspection PyTypeHints
    entrypoint_function_arg_names: list[PythonDefType] = pydantic.Field(
        description=(
            "Specifies the names of the arguments that the entrypoint function takes; "
            "if the function takes no arguments, this field should be an empty list."
        ),
    )
    parent_class_name: PythonDefType | None = pydantic.Field(
        default=None,
        description=(
            "Specifies the name of the parent class of the entrypoint; this is only relevant "
            "if the entrypoint is a class method; if it is not, this field should be unassigned."
        ),
    )
    # noinspection PyTypeHints
    parent_class_arg_names: list[PythonDefType] | None = pydantic.Field(
        default=None,
        description=(
            "Specifies the names of the arguments that the parent class of the entrypoint method "
            "requires in order to be instantiated; if the class constructor takes no arguments, "
            "this field should be an empty list. This field is only relevant if the targeted "
            "entrypoint is inside a class; if it is not, this field should be unassigned."
        ),
    )


output_parser = langchain_core.output_parsers.PydanticOutputParser(
    pydantic_object=CallableAnalysisResponse,
)
expected_output_format_str = output_parser.get_format_instructions()


def get_prompt_config(
    version: str | None = None,
) -> pyine.prompts.utils.PromptConfig:
    """Get the prompt configuration for the callable analysis prompt.

    Args:
        version: The version of the prompt to retrieve. If None, the default version is returned.
    """
    # first, make sure the pydantic loader has already registered this class
    pyine.utils.pydantic_loader.PydanticYAMLLoader.register_models_from_module(__name__)
    # note: this will be cached by the prompt manager
    return pyine.prompts.manager.get_prompt_config(
        prompt_name="callable_analysis",
        version=version,
    )


def get_prompt_template(
    version: str | None = None,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
) -> langchain_core.prompts.PromptTemplate:
    """Return the prompt template for the callable analysis prompt.

    Note: this implementation appropriately fills in all relevant partial variables, if any.

    Args:
        version: The version of the prompt to retrieve. If None, the default version is returned.
        include_examples: Whether to include few-shot examples in the template.
        target_examples: List of examples to target when rendering the prompt. Can pass in
            a list of example indices, or an integer that specifies the number of samples to
            pick randomly. If `None` is provided instead, all examples are included.
    """
    prompt_config = get_prompt_config(version=version)
    return prompt_config.create_prompt_template(
        include_examples=include_examples,
        target_examples=target_examples,
        role_variables=None,
        context_variables=dict(expected_output_format=expected_output_format_str),
        examples_block_variables=None,
    )
