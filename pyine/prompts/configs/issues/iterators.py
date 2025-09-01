import typing

if typing.TYPE_CHECKING:
    import langchain_core.prompts

    import pyine.prompts.types

# @@@@ TODO: if we figure out that we want to work with udiffs, impl something here
# @@@@ TODO: detect+fix "soft refusals": some models (e.g. o3) always want to put spoiling comments in...
# @@@@ TODO: for response structure, add regular pydantic model + a response wrapper w/ invalid flag opt

invalid_code_token = "INVALID_CODE"
"""Token produced by models if they encounter invalid/unusable python code."""


def get_prompt_template(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
) -> "langchain_core.prompts.BasePromptTemplate":
    """Returns the prompt template for the iterator/indexing/generator-bugs prompt (manager module override).

    Note: this implementation appropriately fills in all relevant partial variables, if any.
    """
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config("issues/iterators", version=version)
    template = prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        role_variables=None,
        context_variables=dict(invalid_code_token=invalid_code_token),
        examples_block_variables=dict(invalid_code_token=invalid_code_token),
    )
    if partial_vars:
        template = template.partial(**partial_vars)
    return template
