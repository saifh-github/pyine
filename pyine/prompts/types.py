from __future__ import annotations

import collections.abc
import typing

import langchain_core.prompts
import langchain_core.prompts.chat
import langchain_core.runnables
import langchain_openai.chat_models.base
import pydantic

import pyine.utils.llm_providers

type PromptNameType = str
"""Type used to represent a prompt name (e.g. 'code_summary')."""
type PromptVersionType = str
"""Type used to represent a prompt version (e.g. 'v1.0', or 'with_structured_output')."""
type PromptInputMapping = collections.abc.Mapping[str, typing.Any]
"""Standard input mapping type expected by runnable prompt chains."""
type PromptRunnable = langchain_core.runnables.Runnable[PromptInputMapping, typing.Any]
"""Runnable type alias with concrete input and output typing."""
type PromptTemplate = langchain_core.prompts.PromptTemplate | langchain_core.prompts.chat.ChatPromptTemplate
"""Prompt template alias with explicit format output typing."""
type LanguageModel = langchain_openai.chat_models.base.BaseChatOpenAI
"""Language model alias with explicit generics for prompt building."""


class PromptBuildConfig(pydantic.BaseModel):
    """Configuration settings for building prompt templates.

    Note: the fields in this class should perfectly match the corresponding fields in the manager's
    `get_prompt_template` and `get_prompt_chain` functions that could have overrides defined in any
    prompt module.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
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
    partial_vars: dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: typing.cast("dict[str, typing.Any]", {}),
    )
    """Optional partial variables to use for prompt template substitution."""
    role_variables: dict[str, typing.Any] | None = None
    """Optional variables to substitute in the role block."""
    context_variables: dict[str, typing.Any] | None = None
    """Optional variables to substitute in the context block."""
    examples_block_variables: dict[str, typing.Any] | None = None
    """Optional variables to substitute in the examples block."""

    @property
    def template(self) -> PromptTemplate:
        """Returns the LangChain prompt template object for this prompt."""
        assert self._resolved_prompt is not None, "should have been resolved by now"
        return self._resolved_prompt

    def get_chain(
        self,
        model: LanguageModel,
        runnable_name: str | None = None,
    ) -> PromptRunnable:
        """Instantiates and returns a LangChain runnable chain for this prompt."""
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

    # cache resolved prompt so we don't re-resolve it in each getter call
    _resolved_prompt: PromptTemplate | None = pydantic.PrivateAttr(default=None)

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> PromptBuildConfig:
        """Validates and resolves the prompt builder config."""
        import pyine.prompts.manager

        self._resolved_prompt = pyine.prompts.manager.get_prompt_template(
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
        return self


class PromptChainBuildConfig(pydantic.BaseModel):
    """Configuration settings for building prompt chains."""

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")
    """Pydantic model configuration (freezes the dataclass)."""

    prompt: pydantic.SerializeAsAny[PromptBuildConfig]
    """Configuration settings used to build the prompt config/template object."""
    provider: pydantic.SerializeAsAny[pyine.utils.llm_providers.LLMProviderConfig] | LanguageModel
    """Language model (provider) config to use for the prompt chain."""
    with_retry_config: dict[str, typing.Any] | None = None
    """Configuration for the LangChain retry mechanism (chain-level)."""
    runnable_name: str | None = None
    """Name of the runnable to use for the prompt chain."""

    @property
    def template(self) -> PromptTemplate:
        """Returns the LangChain prompt template object for the underlying prompt config."""
        return self.prompt.template

    @property
    def provider_model(self) -> LanguageModel:
        """Returns the resolved language model associated with this chain config."""
        assert self._resolved_provider is not None, "should have been resolved by now"
        return self._resolved_provider

    @property
    def chain(
        self,
    ) -> PromptRunnable:
        """Returns the LangChain runnable chain for the underlying prompt+provider configs."""
        assert self._resolved_chain is not None, "should have been resolved by now"
        return self._resolved_chain

    # cache resolved objects so we don't re-resolve them in each getter call
    _resolved_provider: LanguageModel | None = pydantic.PrivateAttr(default=None)
    _resolved_chain: PromptRunnable | None = pydantic.PrivateAttr(default=None)

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> PromptChainBuildConfig:
        """Validates and resolves the prompt config, provider config, and chain."""
        if isinstance(self.provider, pyine.utils.llm_providers.LLMProviderConfig):
            self._resolved_provider = pyine.utils.llm_providers.get_model_from_provider_config(
                provider_config=self.provider,
            )
        else:
            assert isinstance(self.provider, langchain_openai.chat_models.base.BaseChatOpenAI)
            self._resolved_provider = self.provider
        prompt_chain = self.prompt.get_chain(
            model=self._resolved_provider,
            runnable_name=self.runnable_name,
        )
        if self.with_retry_config is not None:
            prompt_chain = prompt_chain.with_retry(**self.with_retry_config)
        self._resolved_chain = prompt_chain
        return self
