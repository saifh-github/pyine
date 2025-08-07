import functools

import pyine.prompts.manager
import pyine.prompts.utils
import pyine.utils.pydantic_loader

# @@@@ TODO: if we figure out that we want to work with udiffs, impl something here
# @@@@ TODO: detect+fix "soft refusals": some models (e.g. o3) always want to put spoiling comments in...
# @@@@ TODO: for response structure, add regular pydantic model + a response wrapper w/ invalid flag opt

invalid_code_token = "INVALID_CODE"
"""Token produced by models if they encounter invalid/unusable python code."""


def get_prompt_config(
    version: str | None = None,
) -> pyine.prompts.utils.PromptConfig:
    """Return the prompt configuration for the iterator/indexing bugs prompt.

    Args:
        version: Specific version identifier to retrieve. If ``None``, the default version is returned.
    """
    return pyine.prompts.manager.get_prompt_config(
        prompt_name="issues/iterators",
        version=version,
    )


get_prompt_template = functools.partial(
    pyine.prompts.manager.get_prompt_template,
    prompt_name="issues/iterators",
    role_variables=None,
    context_variables=dict(invalid_code_token=invalid_code_token),
    examples_block_variables=dict(invalid_code_token=invalid_code_token),
)
"""Specialized version of `get_prompt_template` for the iterator/indexing bugs prompt."""
