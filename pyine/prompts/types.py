import typing

import pydantic

PromptNameType = str
"""Type used to represent a prompt name (e.g. 'code_summary')."""
PromptVersionType = str
"""Type used to represent a prompt version (e.g. 'v1.0', or 'with_structured_output')."""


class PromptBuildConfig(pydantic.BaseModel):
    """Configuration settings for building prompt templates and chains.

    Note: the fields in this class should perfectly match the corresponding fields in the manager's
    `get_prompt_template` and `get_prompt_chain` functions that could have overrides defined in any
    prompt module.
    """

    prompt_name: PromptNameType
    """Name of the prompt to build a template or chain for."""
    version: PromptVersionType | None = None
    """Version of the prompt to build a template or chain for."""
    use_chat_template: bool = False
    """Whether to build a chat prompt template or a regular prompt template."""
    include_examples: bool = True
    """Whether to include few-shot examples in the template."""
    target_examples: int | list[int] | None = None
    """Number or list of examples to include in the template; if `None`, all examples are included."""
    partial_vars: dict[str, typing.Any] | None = None
    """Optional partial variables to use for prompt template substitution."""
