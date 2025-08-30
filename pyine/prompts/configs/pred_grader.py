import typing

import langchain_core.output_parsers
import langchain_core.prompts
import pydantic

import pyine.prompts.manager
import pyine.prompts.utils
import pyine.utils.pydantic


class GradingResults(pydantic.BaseModel):
    """Structured grading results schema containing only a score."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    score: typing.Annotated[pydantic.StrictFloat, pydantic.Field(ge=0.0, le=1.0)] = pydantic.Field(
        description="Score in [0,1], where 0 means totally different and 1 means perfect match.",
    )


class GradingResultsWithReasoning(GradingResults):
    """Structured grading results schema with optional reasoning supporting the score."""

    reasoning: str | None = pydantic.Field(
        default=None,
        description="Optional brief reasoning explaining the score.",
    )


def get_output_parser(version: str | None = None) -> langchain_core.output_parsers.BaseOutputParser | None:  # noqa
    """Return the output parser for the pred_grader prompt based on version."""
    if version == "score_only" or version is None:
        model = GradingResults
    elif version == "with_reasoning":
        model = GradingResultsWithReasoning
    else:
        raise NotImplementedError(f"Unsupported version: {version}")
    return langchain_core.output_parsers.PydanticOutputParser(pydantic_object=model)


def get_prompt_config(version: str | None = None) -> pyine.prompts.utils.PromptConfig:
    """Get prompt config for pred_grader, ensuring Pydantic YAML tags are registered."""
    pyine.utils.pydantic.PydanticYAMLLoader.register_models_from_module(__name__)
    return pyine.prompts.manager.get_prompt_config(
        prompt_name="pred_grader",
        version=version,
    )


def get_prompt_template(
    version: str | None = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
) -> langchain_core.prompts.BasePromptTemplate:
    """Return the prompt template for pred_grader with structured output instructions injected."""
    prompt_config = get_prompt_config(version=version)
    parser = get_output_parser(version=version)
    return prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        role_variables=None,
        context_variables=dict(expected_output_format=parser.get_format_instructions()),
        examples_block_variables=None,
    )
