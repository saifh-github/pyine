import functools

import pyine.prompts.manager
import pyine.prompts.utils
import pyine.utils.llm_providers
import pyine.utils.pydantic_loader

# @@@@ TODO: for experiments, consider pyine.utils.portability.print_code_with_numbered_lines()
# @@@@ TODO: for structured execution prediction, this is where the pydantic model will go


def get_prompt_config(
    version: str | None = None,
) -> pyine.prompts.utils.PromptConfig:
    """Get the prompt configuration for the callable analysis prompt.

    Args:
        version: The version of the prompt to retrieve. If None, the default version is returned.
    """
    return pyine.prompts.manager.get_prompt_config(
        prompt_name="code_execution",
        version=version,
    )


get_prompt_template = functools.partial(
    pyine.prompts.manager.get_prompt_template,
    prompt_name="code_execution",
    role_variables=None,
    context_variables=None,
    examples_block_variables=None,
)
"""Specialized version of `get_prompt_template` for the code execution prompt."""
