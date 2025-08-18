import langchain_core.prompts

import pyine.prompts.manager
import pyine.prompts.utils

# @@@@ TODO: if we figure out that we want to work with udiffs, impl something here


def get_prompt_config(
    version: str | None = None,
) -> pyine.prompts.utils.PromptConfig:
    """Return the prompt configuration for the hinting-via-documentation prompt.

    Args:
        version: Specific version identifier to retrieve. If ``None``, the default version is returned.
    """
    return pyine.prompts.manager.get_prompt_config(
        prompt_name="hints/docs",
        version=version,
    )


def get_prompt_template(
    version: str | None = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
) -> langchain_core.prompts.BasePromptTemplate:
    """Return the prompt template for the hinting-via-documentation prompt.

    Note: this implementation appropriately fills in all relevant partial variables, if any.

    Args:
        version: The version of the prompt to retrieve. If None, the default version is returned.
        use_chat_template: Whether to return a chat prompt template or a regular prompt template.
        include_examples: Whether to include few-shot examples in the template.
        target_examples: List of examples to target when rendering the prompt. Can pass in
            a list of example indices, or an integer that specifies the number of samples to
            pick randomly. If `None` is provided instead, all examples are included.
    """
    prompt_config = get_prompt_config(version=version)
    return prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        role_variables=None,
        context_variables=None,
        examples_block_variables=None,
    )
