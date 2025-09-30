import typing

import pydantic

if typing.TYPE_CHECKING:
    import langchain_core.output_parsers
    import langchain_core.prompts

    import pyine.prompts.types


python_call_pattern = r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"


type PythonCallName = typing.Annotated[
    pydantic.StrictStr,
    pydantic.Field(pattern=python_call_pattern),
]


class InputOutputRewriteResponse(pydantic.BaseModel):
    """Structured response for input/output patch generation."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    inputs: list[pydantic.StrictStr] = pydantic.Field(
        description=("List of JSON-encoded input argument payloads; each entry must be a valid JSON string."),
    )
    outputs: list[typing.Any] = pydantic.Field(
        description=("List of outputs aligned by index with inputs. Values must be serializable via JSON."),
    )
    fn_name: PythonCallName = pydantic.Field(
        description=("Fully qualified entrypoint name to execute (e.g. 'Solution.solve' or 'countSquares')."),
    )


def get_output_parser(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> "langchain_core.output_parsers.BaseOutputParser | None":  # noqa
    import langchain_core.output_parsers

    return langchain_core.output_parsers.PydanticOutputParser(pydantic_object=InputOutputRewriteResponse)


def get_prompt_template(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
) -> "langchain_core.prompts.BasePromptTemplate":
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config("input_output_rewrite", version=version)
    template = prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        role_variables=None,
        context_variables=dict(expected_output_format=get_output_parser(version).get_format_instructions()),
        examples_block_variables=None,
    )
    if partial_vars:
        template = template.partial(**partial_vars)
    return template
