import functools

import pyine.prompts.manager
import pyine.prompts.utils
import pyine.utils.pydantic_loader

# @@@@ TODO: if we figure out that we want to work with udiffs, impl something here


def get_prompt_config(
    version: str | None = None,
) -> pyine.prompts.utils.PromptConfig:
    """Return the prompt configuration for the stubs prompt.

    Args:
        version: Specific version identifier to retrieve. If ``None``, the default version is returned.
    """
    return pyine.prompts.manager.get_prompt_config(
        prompt_name="hints/stubs",
        version=version,
    )


get_prompt_template = functools.partial(
    pyine.prompts.manager.get_prompt_template,
    prompt_name="hints/stubs",
    role_variables=None,
    context_variables=None,
    examples_block_variables=None,
)
"""Specialized version of `get_prompt_template` for the stubs prompt."""
