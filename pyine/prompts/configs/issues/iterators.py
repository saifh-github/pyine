import langchain_core.prompts

import pyine.prompts.manager
import pyine.prompts.utils

# @@@@ TODO: if we figure out that we want to work with udiffs, impl something here
# @@@@ TODO: detect+fix "soft refusals": some models (e.g. o3) always want to put spoiling comments in...
# @@@@ TODO: for response structure, add regular pydantic model + a response wrapper w/ invalid flag opt

invalid_code_token = "INVALID_CODE"
"""Token produced by models if they encounter invalid/unusable python code."""


def get_prompt_config(
    version: str | None = None,
) -> pyine.prompts.utils.PromptConfig:
    """Return the prompt configuration for the iterator/indexing/generator-bugs prompt.

    Args:
        version: Specific version identifier to retrieve. If ``None``, the default version is returned.
    """
    return pyine.prompts.manager.get_prompt_config(
        prompt_name="issues/iterators",
        version=version,
    )


def get_prompt_template(
    version: str | None = None,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
) -> langchain_core.prompts.PromptTemplate:
    """Return the prompt template for the iterator/indexing/generator-bugs prompt.

    Note: this implementation appropriately fills in all relevant partial variables, if any.

    Args:
        version: The version of the prompt to retrieve. If None, the default version is returned.
        include_examples: Whether to include few-shot examples in the template.
        target_examples: List of examples to target when rendering the prompt. Can pass in
            a list of example indices, or an integer that specifies the number of samples to
            pick randomly. If `None` is provided instead, all examples are included.
    """
    prompt_config = get_prompt_config(version=version)
    return prompt_config.create_prompt_template(
        include_examples=include_examples,
        target_examples=target_examples,
        role_variables=None,
        context_variables=dict(invalid_code_token=invalid_code_token),
        examples_block_variables=dict(invalid_code_token=invalid_code_token),
    )
