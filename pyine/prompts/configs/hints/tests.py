import typing

if typing.TYPE_CHECKING:
    import langchain_core.prompts

    import pyine.prompts.types

# @@@@ TODO: if we figure out that we want to work with udiffs, impl something here


def get_prompt_template(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
) -> "langchain_core.prompts.BasePromptTemplate":
    """Returns the prompt template for the hinting-via-runtime tests prompt (manager module override).

    Note: this implementation appropriately fills in all relevant partial variables, if any.
    """
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config("hints/tests", version=version)
    template = prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        role_variables=None,
        context_variables=None,
        examples_block_variables=None,
    )
    if partial_vars:
        template = template.partial(**partial_vars)
    return template
