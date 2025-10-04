import collections.abc
import typing

import langchain_core.language_models
import langchain_core.prompts
import langchain_core.runnables
import pydantic

type PromptNameType = str
"""Type used to represent a prompt name (e.g. 'code_summary')."""
type PromptVersionType = str
"""Type used to represent a prompt version (e.g. 'v1.0', or 'with_structured_output')."""
type PromptInputMapping = collections.abc.Mapping[str, typing.Any]
"""Standard input mapping type expected by runnable prompt chains."""
type PromptRunnable = langchain_core.runnables.Runnable[PromptInputMapping, typing.Any]
"""Runnable type alias with concrete input and output typing."""
type PromptTemplate = langchain_core.prompts.BasePromptTemplate[typing.Any]
"""Prompt template alias with explicit format output typing."""
type LanguageModel = langchain_core.language_models.BaseLanguageModel[typing.Any]
"""Language model alias with explicit generics for prompt building."""


class PromptBuildConfig(pydantic.BaseModel):
    """Configuration settings for building prompt templates and chains.

    Note: the fields in this class should perfectly match the corresponding fields in the manager's
    `get_prompt_template` and `get_prompt_chain` functions that could have overrides defined in any
    prompt module.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")
    """Pydantic model configuration (freezes the dataclass)."""

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
    partial_vars: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
    """Optional partial variables to use for prompt template substitution."""
    role_variables: dict[str, typing.Any] | None = None
    """Optional variables to substitute in the role block."""
    context_variables: dict[str, typing.Any] | None = None
    """Optional variables to substitute in the context block."""
    examples_block_variables: dict[str, typing.Any] | None = None
    """Optional variables to substitute in the examples block."""

    def get_template(self) -> PromptTemplate:
        """Returns a LangChain prompt template for this prompt."""
        import pyine.prompts.manager

        return pyine.prompts.manager.get_prompt_template(
            prompt_name=self.prompt_name,
            version=self.version,
            use_chat_template=self.use_chat_template,
            include_examples=self.include_examples,
            target_examples=self.target_examples,
            partial_vars=self.partial_vars,
            role_variables=self.role_variables,
            context_variables=self.context_variables,
            examples_block_variables=self.examples_block_variables,
        )

    def get_chain(
        self,
        model: LanguageModel,
        runnable_name: str | None = None,
    ) -> PromptRunnable:
        """Returns a LangChain runnable chain for this prompt."""
        import pyine.prompts.manager

        return pyine.prompts.manager.get_prompt_chain(
            model=model,
            prompt_name=self.prompt_name,
            version=self.version,
            runnable_name=runnable_name,
            use_chat_template=self.use_chat_template,
            include_examples=self.include_examples,
            target_examples=self.target_examples,
            partial_vars=self.partial_vars,
            role_variables=self.role_variables,
            context_variables=self.context_variables,
            examples_block_variables=self.examples_block_variables,
        )
