"""Prompt config module for the debate interrogator verdict-only turn."""

import typing

import langchain_core.output_parsers
import langchain_core.runnables

if typing.TYPE_CHECKING:
    import langchain_openai.chat_models.base

    import pyine.prompts.types

from pyine.prompts.configs.guardrail.debate_interrogator import VerdictOutput, VerdictOutputNoReasoning


def _get_verdict_output_class(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> type[VerdictOutput] | type[VerdictOutputNoReasoning]:
    """Return the appropriate VerdictOutput class for the given version."""
    if version == "no_reasoning":
        return VerdictOutputNoReasoning
    return VerdictOutput


def get_output_parser(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> langchain_core.output_parsers.BaseOutputParser[typing.Any] | None:
    """Return the output parser for the verdict-only prompt."""
    if version == "with_reasoning" or version == "no_reasoning" or version is None:
        output_class = _get_verdict_output_class(version)
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
    """Build the verdict-only prompt template with format instructions."""
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config(
        "guardrail/debate_interrogator_verdict",
        version=version,
    )
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
) -> "langchain_core.runnables.Runnable[typing.Any, typing.Any]":
    """Build verdict chain with structured output decoding."""
    from pyine.prompts.configs.guardrail.debate_interrogator import _unwrap_retry  # pyright: ignore[reportPrivateUsage]

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
    unwrapped_model = _unwrap_retry(model)
    output_class = _get_verdict_output_class(version)
    structured_model = unwrapped_model.with_structured_output(  # type: ignore[reportUnknownVariableType,reportUnknownMemberType]
        output_class,
        method="json_schema",
    )
    return langchain_core.runnables.RunnableSequence(
        prompt_template,
        structured_model,
        name=runnable_name,
    )
