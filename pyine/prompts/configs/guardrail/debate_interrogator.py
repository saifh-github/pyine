"""Prompt config module for the debate interrogator.

Provides structured output parsing for the interrogator's decision:
either ask a probing question or render a verdict with a score.
"""

import typing

import langchain_core.output_parsers
import langchain_core.runnables
import pydantic

if typing.TYPE_CHECKING:
    import langchain_openai.chat_models.base

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


class InterrogatorOutputNoReasoning(pydantic.BaseModel):
    """Structured output for the debate interrogator (without reasoning field).

    Same as InterrogatorOutput but omits the reasoning field to reduce token
    usage with third-party API providers.
    """

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


class VerdictOutput(pydantic.BaseModel):
    """Structured output for the interrogator's forced verdict turn.

    Unlike InterrogatorOutput, there is no 'decision' field -- the model
    must always produce a verdict with score and reasoning.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    score: float = pydantic.Field(
        ge=0.0,
        le=1.0,
        description="Confidence score in [0,1]. 1.0 = definitely correct, 0.0 = definitely incorrect.",
    )
    reasoning: str = pydantic.Field(
        description="Brief reasoning explaining the verdict.",
    )
    content: str = pydantic.Field(
        description="Summary of the assessment (becomes the final interrogator message in the transcript).",
    )


class VerdictOutputNoReasoning(pydantic.BaseModel):
    """Structured output for the interrogator's forced verdict turn (without reasoning).

    Same as VerdictOutput but omits the reasoning field to reduce token usage.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    score: float = pydantic.Field(
        ge=0.0,
        le=1.0,
        description="Confidence score in [0,1]. 1.0 = definitely correct, 0.0 = definitely incorrect.",
    )
    content: str = pydantic.Field(
        description="Summary of the assessment (becomes the final interrogator message in the transcript).",
    )


def _get_interrogator_output_class(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> type[InterrogatorOutput] | type[InterrogatorOutputNoReasoning]:
    """Return the appropriate InterrogatorOutput class for the given version."""
    if version == "no_reasoning":
        return InterrogatorOutputNoReasoning
    return InterrogatorOutput


def get_output_parser(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> langchain_core.output_parsers.BaseOutputParser[typing.Any] | None:
    """Return the output parser for the debate interrogator prompt."""
    if version == "with_reasoning" or version == "no_reasoning" or version is None:  # default
        output_class = _get_interrogator_output_class(version)
        return langchain_core.output_parsers.PydanticOutputParser(pydantic_object=output_class)
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


def get_prompt_chain(
    model: "langchain_openai.chat_models.base.BaseChatOpenAI",
    version: "pyine.prompts.types.PromptVersionType | None" = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
    runnable_name: str | None = None,
    role_variables: dict[str, typing.Any] | None = None,
    context_variables: dict[str, typing.Any] | None = None,
    examples_block_variables: dict[str, typing.Any] | None = None,
) -> langchain_core.runnables.Runnable[typing.Any, typing.Any]:
    """Build interrogator chain with structured output decoding."""
    prompt_template = get_prompt_template(
        version=version,
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        partial_vars=partial_vars,
        role_variables=role_variables,
        context_variables=context_variables,
        examples_block_variables=examples_block_variables,
    )
    # Unwrap RunnableWithRetry if provider-level retries were applied,
    # since RunnableWithRetry does not expose with_structured_output.
    unwrapped_model = _unwrap_retry(model)
    output_class = _get_interrogator_output_class(version)
    structured_model = unwrapped_model.with_structured_output(  # type: ignore[reportUnknownVariableType,reportUnknownMemberType]
        output_class,
        method="json_schema",
    )
    return langchain_core.runnables.RunnableSequence(
        prompt_template,
        structured_model,
        name=runnable_name,
    )


def _unwrap_retry(
    model: langchain_core.runnables.Runnable[typing.Any, typing.Any],
) -> "langchain_openai.chat_models.base.BaseChatOpenAI":
    """If model is wrapped in RunnableWithRetry, extract the underlying model.

    This is needed because llm_providers.py applies with_retry() and then
    typing.casts the result back to BaseChatOpenAI. The RunnableWithRetry
    wrapper does not have with_structured_output, so we must unwrap first.
    """

    from langchain_core.runnables.retry import RunnableRetry

    if isinstance(model, RunnableRetry):
        return typing.cast(
            "langchain_openai.chat_models.base.BaseChatOpenAI",
            model.bound,  # the original model before retry wrapping
        )
    return typing.cast("langchain_openai.chat_models.base.BaseChatOpenAI", model)
