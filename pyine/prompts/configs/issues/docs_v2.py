import typing

if typing.TYPE_CHECKING:
    import pyine.prompts.types


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
    """Returns the prompt template for the dedicated misleading-docs prompt."""
    import pyine.prompts.manager

    merged_role = dict(role_variables or {})
    prompt_config = pyine.prompts.manager.get_prompt_config("issues/docs_v2", version=version)
    return prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        partial_vars=partial_vars,
        role_variables=merged_role,
        context_variables=context_variables,
        examples_block_variables=examples_block_variables,
    )
