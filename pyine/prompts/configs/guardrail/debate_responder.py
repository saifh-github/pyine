"""Prompt config module for the debate responder (Model A).

The responder produces plain text responses (no structured output needed).
"""

import typing

import langchain_core.output_parsers

if typing.TYPE_CHECKING:
    import pyine.prompts.types


def get_output_parser(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> langchain_core.output_parsers.BaseOutputParser[typing.Any] | None:
    """Return the output parser for the debate responder prompt.

    Returns StrOutputParser since the responder produces plain text.
    """
    if version == "with_reasoning" or version is None:  # default
        return langchain_core.output_parsers.StrOutputParser()
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
    """Module override for template construction (no special injections needed)."""
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config("guardrail/debate_responder", version=version)
    return prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        partial_vars=partial_vars,
        role_variables=role_variables,
        context_variables=context_variables,
        examples_block_variables=examples_block_variables,
    )
