from __future__ import annotations

import contextlib
import importlib.resources
import importlib.resources.abc
import logging
import pathlib
import typing

import langchain_core.runnables
import pydantic
import yaml

if typing.TYPE_CHECKING:
    import langchain_openai.chat_models.base

    from pyine.prompts.types import PromptNameType, PromptTemplate, PromptVersionType
    from pyine.prompts.utils import PromptConfig, VersionedPromptConfig

logger = logging.getLogger(__name__)

_PromptCacheKeyType = str
"""Type used to represent a prompt cache key (e.g. 'code_summary:v1.0')."""

PROMPT_NAME_ALIASES: dict[PromptNameType, PromptNameType] = {
    "issues/docs": "hints/docs",
}
"""Alias map used to resolve prompt names to their canonical definitions."""


class PromptManager:
    """Manager for loading and working with YAML-based prompt configuration files.

    You should not need to instantiate this class directly. Instead, use the
    `get_framework_prompt_manager()` function to get the default prompt manager (this will also
    initialize Pydantic YAML loaders for all supported models). Better yet, you can use the
    `get_prompt_config` and `get_prompt_template` functions to get relevant objects directly.
    """

    def __init__(
        self,
        package_name: str = "pyine.prompts",
        prompts_subdir: str = "templates",
    ) -> None:
        """Initialize the prompt manager.

        Args:
            package_name: Name of the package containing prompt files.
            prompts_subdir: Subdirectory within the package containing prompts.
        """
        self.package_name = package_name
        self.prompts_subdir = prompts_subdir
        self._cache: dict[_PromptCacheKeyType, PromptConfig] = {}

    @staticmethod
    def _make_cache_key(
        prompt_name: PromptNameType,
        version: PromptVersionType | None,
    ) -> _PromptCacheKeyType:
        """Generate a cache key for the given prompt name/version pair."""
        return f"{prompt_name}:{version}" if version else prompt_name

    def _resolve_prompt_alias(
        self,
        prompt_name: PromptNameType,
    ) -> tuple[PromptNameType, bool]:
        """Resolve a prompt alias to its canonical name (if any)."""
        if prompt_name in PROMPT_NAME_ALIASES:
            return PROMPT_NAME_ALIASES[prompt_name], True
        return prompt_name, False

    def _get_prompt_file_path(
        self,
        prompt_name: PromptNameType,
    ) -> pathlib.Path:
        """Get the path to a prompt YAML file within the package resources."""
        return pathlib.Path(self.prompts_subdir) / f"{prompt_name}.yaml"

    def _get_prompt_module(
        self,
        prompt_name: PromptNameType,
    ) -> typing.Any:
        """Get the module containing specific prompt template defines by name."""
        prompt_module_path = str(prompt_name).replace("/", ".")
        return importlib.import_module(f"{self.package_name}.configs.{prompt_module_path}")

    def get_prompt_module(
        self,
        prompt_name: PromptNameType,
    ) -> typing.Any:
        """Public accessor for prompt modules that exposes overrides safely."""
        return self._get_prompt_module(prompt_name)

    def _load_prompt_config(
        self,
        prompt_name: PromptNameType,
        version: PromptVersionType | None = None,
    ) -> PromptConfig:
        """Load and parse a prompt configuration from YAML.

        Args:
            prompt_name: Name of the prompt to load (without .yaml extension)
            version: Specific version to load, or None for the default version

        Returns:
            The parsed prompt configuration for the requested version.
        """
        import pyine.prompts.utils as prompt_utils

        prompt_path = self._get_prompt_file_path(prompt_name)
        logger.debug(f"loading prompt config from: {prompt_path}")
        try:
            package_files = importlib.resources.files(self.package_name)
            prompt_file = package_files.joinpath(*prompt_path.parts)
            if not prompt_file.is_file():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
            with contextlib.ExitStack() as resource_stack:
                yaml_path = resource_stack.enter_context(importlib.resources.as_file(prompt_file))
                versioned_config = prompt_utils.VersionedPromptConfig.from_yaml(yaml_path)
            if version is not None:
                if version not in versioned_config.versions:
                    available_versions = ", ".join(versioned_config.versions.keys())
                    raise ValueError(
                        f"Version '{version}' not found for prompt '{prompt_name}'. "
                        f"Available versions: {available_versions}"
                    )
                return versioned_config.versions[version]
            return versioned_config.get_default()
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {prompt_path}: {e}") from e
        except pydantic.ValidationError as e:
            raise ValueError(f"Invalid prompt schema in {prompt_path}: {e}") from e

    def get_prompt_config(
        self,
        prompt_name: PromptNameType,
        version: PromptVersionType | None = None,
    ) -> PromptConfig:
        """Get a prompt configuration with caching.

        Args:
            prompt_name: Name of the prompt to retrieve.
            version: Specific version to retrieve, or `None` for the default version.

        Returns:
            The prompt configuration for the requested version.
        """
        resolved_name, is_alias = self._resolve_prompt_alias(prompt_name)
        cache_key = self._make_cache_key(prompt_name, version)
        if cache_key in self._cache:
            return self._cache[cache_key]
        canonical_cache_key = self._make_cache_key(resolved_name, version)
        canonical_config = self._cache.get(canonical_cache_key)
        if canonical_config is None:
            canonical_config = self._load_prompt_config(resolved_name, version)
            self._cache[canonical_cache_key] = canonical_config
        if not is_alias:
            self._cache[cache_key] = canonical_config
            return canonical_config
        alias_metadata = canonical_config.metadata.model_copy(update={"name": prompt_name})
        alias_config = canonical_config.model_copy(update={"metadata": alias_metadata})
        self._cache[cache_key] = alias_config
        return alias_config

    def _get_versioned_config(self, prompt_name: PromptNameType) -> VersionedPromptConfig:
        """Get the versioned configuration for a specific prompt."""
        import pyine.prompts.utils as prompt_utils

        resolved_name, _ = self._resolve_prompt_alias(prompt_name)
        prompt_path = self._get_prompt_file_path(resolved_name)
        package_files = importlib.resources.files(self.package_name)
        prompt_file = package_files.joinpath(*prompt_path.parts)
        if not prompt_file.is_file():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        with importlib.resources.as_file(prompt_file) as yaml_path:
            return prompt_utils.VersionedPromptConfig.from_yaml(yaml_path)

    def list_prompts(self) -> list[PromptNameType]:
        """Returns a list of all available prompts in the package (as names without extension)."""
        package_files = importlib.resources.files(self.package_name)
        prompts_dir = package_files.joinpath(self.prompts_subdir)
        if not prompts_dir.is_dir():
            return []
        prompt_names: list[PromptNameType] = []
        stack: list[tuple[importlib.resources.abc.Traversable, pathlib.PurePosixPath]] = [
            (prompts_dir, pathlib.PurePosixPath())
        ]
        while stack:
            current_dir, rel_root = stack.pop()
            for entry in current_dir.iterdir():
                if entry.is_dir():
                    stack.append((entry, rel_root / entry.name))
                    continue
                if not entry.name.lower().endswith(".yaml"):
                    continue
                prompt_name = entry.name[:-5]
                if rel_root == pathlib.PurePosixPath():
                    prompt_names.append(prompt_name)
                else:
                    prompt_names.append(f"{rel_root.as_posix()}/{prompt_name}")
        for alias_name in PROMPT_NAME_ALIASES:
            if alias_name not in prompt_names:
                prompt_names.append(alias_name)
        return sorted(prompt_names)

    def list_prompt_versions(self, prompt_name: PromptNameType) -> list[PromptVersionType]:
        """List all available versions for a specific prompt."""
        return list(self._get_versioned_config(prompt_name).versions.keys())

    def get_default_prompt_version(self, prompt_name: PromptNameType) -> PromptVersionType:
        """Returns the default version used for a specific prompt."""
        return self._get_versioned_config(prompt_name).default_version

    def clear_cache(self) -> None:
        """Clear the internal prompt cache."""
        self._cache.clear()


_default_prompt_manager: PromptManager | None = None
"""Singleton instance of the framework's default prompt manager."""


def get_framework_prompt_manager() -> PromptManager:
    """Get the default prompt manager instance for the pyine framework (singleton)."""
    global _default_prompt_manager
    if _default_prompt_manager is None:
        import pyine.utils.pydantic

        # if the default framework manager is not created, make sure models are all registered too
        logger.debug("registering pydantic models for framework prompt manager")
        pyine.utils.pydantic.PydanticYAMLLoader.register_models_from_package("pyine")
        _default_prompt_manager = PromptManager()
    return _default_prompt_manager


def list_prompts() -> list[PromptNameType]:
    """Convenience function to list all available prompts in the package."""
    manager = get_framework_prompt_manager()
    return manager.list_prompts()


def list_prompt_versions(prompt_name: PromptNameType) -> list[PromptVersionType]:
    """Convenience function to list all available versions for a specific prompt."""
    manager = get_framework_prompt_manager()
    return manager.list_prompt_versions(prompt_name)


def get_prompt_config(
    prompt_name: PromptNameType,
    version: PromptVersionType | None = None,
) -> PromptConfig:
    """Convenience function to get a prompt config using the default manager.

    Args:
        prompt_name: Name of the prompt to retrieve
        version: Specific version to retrieve, or None for the default version

    Returns:
        The prompt configuration for the requested version
    """
    manager = get_framework_prompt_manager()
    return manager.get_prompt_config(prompt_name, version)


def get_prompt_template(
    prompt_name: PromptNameType,
    version: PromptVersionType | None = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
    role_variables: dict[str, typing.Any] | None = None,
    context_variables: dict[str, typing.Any] | None = None,
    examples_block_variables: dict[str, typing.Any] | None = None,
) -> PromptTemplate:
    """Convenience function to get a prompt template using the default manager.

    Note: if the prompt is known and registered, we will check its corresponding module to see if
    it possesses an override to generate the prompt template. If so, it will be used instead in
    order for partial variables to be correctly filled in.

    Args:
        prompt_name: Name of the prompt to retrieve
        version: The version of the prompt to retrieve. If None, the default version is returned.
        use_chat_template: Whether to return a chat prompt template or a regular prompt template.
        include_examples: Whether to include few-shot examples in the template.
        target_examples: List of examples to target when rendering the prompt. Can pass in
            a list of example indices, or an integer that specifies the number of samples to
            pick randomly. If `None` is provided instead, all examples are included.
        partial_vars: Optional partial variables to use for prompt template substitution.
        role_variables: Optional variables for rendering the role block.
        context_variables: Optional variables for rendering the context block.
        examples_block_variables: Optional variables for rendering the examples block template.
    """
    manager = get_framework_prompt_manager()
    prompt_module: typing.Any | None = None
    try:
        # if a getter override exists for this specific prompt in its parent module, use it
        prompt_module = manager.get_prompt_module(prompt_name)
    except ModuleNotFoundError:
        logger.debug(f"could not find module for {prompt_name}, using default template constructor")
    if prompt_module is not None and hasattr(prompt_module, "get_prompt_template"):
        return prompt_module.get_prompt_template(
            version=version,
            use_chat_template=use_chat_template,
            include_examples=include_examples,
            target_examples=target_examples,
            partial_vars=partial_vars,
            role_variables=role_variables,
            context_variables=context_variables,
            examples_block_variables=examples_block_variables,
        )
    prompt_config = manager.get_prompt_config(prompt_name=prompt_name, version=version)
    return prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        partial_vars=partial_vars,
        role_variables=role_variables,
        context_variables=context_variables,
        examples_block_variables=examples_block_variables,
    )


def get_prompt_chain(
    model: langchain_openai.chat_models.base.BaseChatOpenAI,
    prompt_name: PromptNameType,
    version: PromptVersionType | None = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
    runnable_name: str | None = None,
    role_variables: dict[str, typing.Any] | None = None,
    context_variables: dict[str, typing.Any] | None = None,
    examples_block_variables: dict[str, typing.Any] | None = None,
) -> langchain_core.runnables.Runnable[typing.Any, typing.Any]:
    """Convenience function to get a runnable prompt chain using the default manager.

    Note: if the prompt is known and registered, we will check its corresponding module to see if
    it possesses an override to generate the prompt chain. If so, it will be used instead in
    order for structured output parsers to be properly attached.

    Args:
        model: The language model to use inside the runnable prompt chain.
        prompt_name: Name of the prompt to retrieve
        version: The version of the prompt to retrieve. If None, the default version is returned.
        use_chat_template: Whether to return a chat prompt template or a regular prompt template.
        include_examples: Whether to include few-shot examples in the template.
        target_examples: List of examples to target when rendering the prompt. Can pass in
            a list of example indices, or an integer that specifies the number of samples to
            pick randomly. If `None` is provided instead, all examples are included.
        partial_vars: Optional partial variables to use for prompt template substitution.
        runnable_name: Optional name for the runnable prompt chain (passed to its constructor).
        role_variables: Optional variables for rendering the role block.
        context_variables: Optional variables for rendering the context block.
        examples_block_variables: Optional variables for rendering the examples block template.
    """
    manager = get_framework_prompt_manager()
    prompt_module: typing.Any | None = None
    try:
        # if a getter override exists for this specific prompt in its parent module, use it
        prompt_module = manager.get_prompt_module(prompt_name)
    except ModuleNotFoundError:
        logger.debug(f"could not find module for {prompt_name}, will not use any overrides")
    if prompt_module is not None and hasattr(prompt_module, "get_prompt_chain"):
        return prompt_module.get_prompt_chain(
            model=model,
            version=version,
            use_chat_template=use_chat_template,
            include_examples=include_examples,
            target_examples=target_examples,
            partial_vars=partial_vars,
            runnable_name=runnable_name,
            role_variables=role_variables,
            context_variables=context_variables,
            examples_block_variables=examples_block_variables,
        )
    prompt_template = get_prompt_template(
        prompt_name=prompt_name,
        version=version,
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        partial_vars=partial_vars,
        role_variables=role_variables,
        context_variables=context_variables,
        examples_block_variables=examples_block_variables,
    )
    if prompt_module is not None and hasattr(prompt_module, "get_output_parser"):
        output_parser = prompt_module.get_output_parser(version=version)
        if output_parser is not None:
            return langchain_core.runnables.RunnableSequence(
                prompt_template,
                model,
                output_parser,
                name=runnable_name,
            )
    return langchain_core.runnables.RunnableSequence(
        prompt_template,
        model,
        name=runnable_name,
    )
