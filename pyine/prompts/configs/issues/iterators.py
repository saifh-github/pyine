import typing

if typing.TYPE_CHECKING:
    import pyine.prompts.types

# @@@@ TODO: if we figure out that we want to work with udiffs, impl something here
# @@@@ TODO: detect+fix "soft refusals": some models (e.g. o3) always want to put spoiling comments in...
# @@@@ TODO: for response structure, add regular pydantic model + a response wrapper w/ invalid flag opt

invalid_code_token = "INVALID_CODE"  # noqa: S105 - harmless sentinel token
"""Token produced by models if they encounter invalid/unusable python code."""


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
    """Returns the prompt template for the iterator/indexing/generator-bugs prompt (manager module override).

    Note: this implementation appropriately fills in all relevant partial variables, if any.
    """
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config("issues/iterators", version=version)
    merged_context = context_variables.copy() if context_variables else {}
    merged_context.setdefault("invalid_code_token", invalid_code_token)
    merged_examples_block = examples_block_variables.copy() if examples_block_variables else {}
    merged_examples_block.setdefault("invalid_code_token", invalid_code_token)
    return prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        partial_vars=partial_vars,
        role_variables=role_variables,
        context_variables=merged_context,
        examples_block_variables=merged_examples_block,
    )
