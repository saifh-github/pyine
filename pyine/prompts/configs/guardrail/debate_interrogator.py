"""Prompt config module for the debate interrogator (Model B).

Provides structured output parsing for the interrogator's decision:
either ask a probing question or render a verdict with a score.
"""

import typing

import langchain_core.output_parsers
import pydantic

if typing.TYPE_CHECKING:
    import pyine.prompts.types


class InterrogatorOutput(pydantic.BaseModel):
    """Structured output for the debate interrogator."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    decision: typing.Literal["question", "verdict"] = pydantic.Field(
        description=("Whether to ask a probing question ('question') or render a final verdict ('verdict')."),
    )
    content: str = pydantic.Field(
        description=("The probing question (if decision='question') or reasoning summary (if decision='verdict')."),
    )
    score: float | None = pydantic.Field(
        default=None,
        description="Confidence score in [0,1] (required when decision='verdict', null otherwise).",
    )
    reasoning: str | None = pydantic.Field(
        default=None,
        description="Optional brief reasoning explaining the verdict (only when decision='verdict').",
    )


def get_output_parser(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> langchain_core.output_parsers.BaseOutputParser[typing.Any] | None:
    """Return the output parser for the debate interrogator prompt."""
    if version == "with_reasoning" or version is None:  # default
        return langchain_core.output_parsers.PydanticOutputParser(pydantic_object=InterrogatorOutput)
    raise NotImplementedError(f"Unsupported version: {version}")


def get_prompt_template(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
    role_variables: dict[str, typing.Any] | None = None,
    context_variables: dict[str, typing.Any] | None = None,
    examples_block_variables: dict[str, typing.Any] | None = None,
) -> "pyine.prompts.types.PromptTemplate":
    """Module override that injects format instructions into the context."""
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config("guardrail/debate_interrogator", version=version)
    parser = get_output_parser(version=version)
    merged_context: dict[str, typing.Any] = dict(context_variables) if context_variables else {}
    if parser is not None:
        merged_context.setdefault("expected_output_format", parser.get_format_instructions())
    return prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        partial_vars=partial_vars,
        role_variables=role_variables,
        context_variables=merged_context or None,
        examples_block_variables=examples_block_variables,
    )
