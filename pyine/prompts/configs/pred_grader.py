import typing

import pydantic

if typing.TYPE_CHECKING:
    import langchain_core.output_parsers
    import langchain_core.prompts

    import pyine.prompts.types


class GradingResult(pydantic.BaseModel):
    """Structured grading results schema containing only a score."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    score: typing.Annotated[pydantic.StrictFloat, pydantic.Field(ge=0.0, le=1.0)] = pydantic.Field(
        description="Score in [0,1], where 0 means totally different and 1 means perfect match.",
    )


class GradingResultWithReasoning(GradingResult):
    """Structured grading results schema with optional reasoning supporting the score."""

    reasoning: str | None = pydantic.Field(
        default=None,
        description="Optional brief reasoning explaining the score.",
    )


def get_output_parser(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> "langchain_core.output_parsers.BaseOutputParser | None":  # noqa
    """Return the output parser for the pred_grader prompt based on version."""
    import langchain_core.output_parsers

    if version == "score_only" or version is None:
        model = GradingResult
    elif version == "with_reasoning":
        model = GradingResultWithReasoning
    else:
        raise NotImplementedError(f"Unsupported version: {version}")
    return langchain_core.output_parsers.PydanticOutputParser(pydantic_object=model)


def get_prompt_template(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
) -> "langchain_core.prompts.BasePromptTemplate":
    """Returns the prompt template for the prediction grader prompt (manager module override).

    Note: this implementation appropriately fills in all relevant partial variables, if any.
    """
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config("pred_grader", version=version)
    parser = get_output_parser(version=version)
    template = prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        role_variables=None,
        context_variables=dict(expected_output_format=parser.get_format_instructions()),
        examples_block_variables=None,
    )
    if partial_vars:
        template = template.partial(**partial_vars)
    return template
