import typing

import pydantic

if typing.TYPE_CHECKING:
    import langchain_core.output_parsers
    import langchain_core.prompts

    import pyine.prompts.types

python_def_pattern = r"^[A-Za-z_][A-Za-z0-9_]*$"
"""Regex pattern to use to identify any variable/function/method/class name definition."""

type PythonDefType = typing.Annotated[pydantic.StrictStr, pydantic.Field(pattern=python_def_pattern)]
"""Type annotation for a Python variable/function/method/class name definition."""


class CallableAnalysisResponse(pydantic.BaseModel):
    """Response model for callable entrypoint analysis.

    This model is used to describe the output of a model tasked with identifying the callable
    entrypoint of a Python program. See the corresponding template YAML file for more details.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
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


def get_output_parser(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> "langchain_core.output_parsers.BaseOutputParser | None":  # noqa
    """Get the output parser for the callable analysis prompt (if one should be used)."""
    # note: we currently have a single output structure for all version
    import langchain_core.output_parsers

    return langchain_core.output_parsers.PydanticOutputParser(pydantic_object=CallableAnalysisResponse)


def get_prompt_template(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
    role_variables: dict[str, typing.Any] | None = None,
    context_variables: dict[str, typing.Any] | None = None,
    examples_block_variables: dict[str, typing.Any] | None = None,
) -> "langchain_core.prompts.BasePromptTemplate":
    """Returns the prompt template for the callable analysis prompt (manager module override).

    Note: this implementation appropriately fills in all relevant partial variables, if any.
    """
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config("callable_analysis", version=version)
    parser = get_output_parser(version)
    merged_context = dict(context_variables) if context_variables else {}
    merged_context.setdefault("expected_output_format", parser.get_format_instructions())
    template = prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        role_variables=role_variables,
        context_variables=merged_context,
        examples_block_variables=examples_block_variables,
    )
    if partial_vars:
        template = template.partial(**partial_vars)
    return template
