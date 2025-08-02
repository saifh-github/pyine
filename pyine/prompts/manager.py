import functools
import importlib.resources
import logging
import pathlib
import typing

import langchain_core.prompts
import pydantic
import yaml

import pyine.prompts.utils as prompt_utils

logger = logging.getLogger(__name__)


class PromptManager:
    """Manager for loading and working with YAML-based prompt configuration files."""

    def __init__(
        self,
        package_name: str = "pyine.prompts",
        prompts_subdir: str = "templates",
    ):
        """Initialize the prompt manager.

        Args:
            package_name: Name of the package containing prompt files.
            prompts_subdir: Subdirectory within the package containing prompts.
        """
        self.package_name = package_name
        self.prompts_subdir = prompts_subdir
        self._cache: dict[str, prompt_utils.PromptConfig] = {}

    def _get_prompt_path(
        self,
        prompt_name: str,
    ) -> pathlib.Path:
        """Get the path to a prompt file within the package resources."""
        return pathlib.Path(self.prompts_subdir) / f"{prompt_name}.yaml"

    def _load_prompt_config(
        self,
        prompt_name: str,
        version: str | None = None,
    ) -> prompt_utils.PromptConfig:
        """Load and parse a prompt configuration from YAML.

        Args:
            prompt_name: Name of the prompt to load (without .yaml extension)
            version: Specific version to load, or None for the default version

        Returns:
            The parsed prompt configuration for the requested version.
        """
        prompt_path = self._get_prompt_path(prompt_name)
        try:
            package_files = importlib.resources.files(self.package_name)
            prompt_file = package_files / str(prompt_path)
            if not prompt_file.is_file():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
            versioned_config = prompt_utils.VersionedPromptConfig.from_yaml(prompt_file)  # noqa
            if version is not None:
                if version not in versioned_config.versions:
                    available_versions = ", ".join(versioned_config.versions.keys())
                    raise ValueError(
                        f"Version '{version}' not found for prompt '{prompt_name}'. "
                        f"Available versions: {available_versions}"
                    )
                return versioned_config.versions[version]
            else:
                return versioned_config.get_default()
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {prompt_path}: {e}") from e
        except pydantic.ValidationError as e:
            raise ValueError(f"Invalid prompt schema in {prompt_path}: {e}") from e

    @functools.lru_cache(maxsize=128)
    def get_prompt_config(
        self,
        prompt_name: str,
        version: str | None = None,
    ) -> prompt_utils.PromptConfig:
        """Get a prompt configuration with caching.

        Args:
            prompt_name: Name of the prompt to retrieve.
            version: Specific version to retrieve, or `None` for the default version.

        Returns:
            The prompt configuration for the requested version.
        """
        cache_key = f"{prompt_name}:{version}" if version else prompt_name
        if cache_key not in self._cache:
            self._cache[cache_key] = self._load_prompt_config(prompt_name, version)
        return self._cache[cache_key]

    def list_prompts(self) -> list[str]:
        """Returns a list of all available prompts in the package (as names without extension)."""
        package_files = importlib.resources.files(self.package_name)
        prompts_dir = package_files / self.prompts_subdir
        if not prompts_dir.is_dir():
            return []
        prompt_names = [
            file.name[:-5]  # remove .yaml extension
            for file in prompts_dir.iterdir()
            if file.is_file() and file.name.lower().endswith(".yaml")
        ]
        return sorted(prompt_names)

    def list_prompt_versions(self, prompt_name: str) -> list[str]:
        """List all available versions for a specific prompt."""
        prompt_path = self._get_prompt_path(prompt_name)
        package_files = importlib.resources.files(self.package_name)
        prompt_file = package_files / str(prompt_path)
        if not prompt_file.is_file():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        versioned_config = prompt_utils.VersionedPromptConfig.from_yaml(prompt_file)  # noqa
        return list(versioned_config.versions.keys())

    def clear_cache(self) -> None:
        """Clear the internal prompt cache."""
        self._cache.clear()
        self.get_prompt_config.cache_clear()


_default_prompt_manager: PromptManager | None = None
"""Singleton instance of the framework's default prompt manager."""


def get_framework_prompt_manager() -> PromptManager:
    """Get the default prompt manager instance for the pyine framework (singleton)."""
    global _default_prompt_manager
    if _default_prompt_manager is None:
        _default_prompt_manager = PromptManager()
    return _default_prompt_manager


def get_prompt_config(
    prompt_name: str,
    version: str | None = None,
) -> prompt_utils.PromptConfig:
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
    prompt_name: str,
    version: str | None = None,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    role_variables: dict[str, typing.Any] | None = None,
    context_variables: dict[str, typing.Any] | None = None,
    examples_block_variables: dict[str, typing.Any] | None = None,
) -> langchain_core.prompts.PromptTemplate:
    """Get the langchain prompt template for the callable analysis prompt.

    Args:
        prompt_name: Name of the prompt to retrieve
        version: The version of the prompt to retrieve. If None, the default version is returned.
        include_examples: Whether to include few-shot examples in the template.
        target_examples: List of examples to target when rendering the prompt. Can pass in
            a list of example indices, or an integer that specifies the number of samples to
            pick randomly. If `None` is provided instead, all examples are included.
        role_variables: Variables to substitute in the role template.
        context_variables: Variables to substitute in the context template.
        examples_block_variables: Variables to substitute in the examples block template. Should
            not include the 'examples_str' variable (will be added directly).
    """
    prompt_config = get_prompt_config(prompt_name=prompt_name, version=version)
    return prompt_config.create_prompt_template(
        include_examples=include_examples,
        target_examples=target_examples,
        role_variables=role_variables,
        context_variables=context_variables,
        examples_block_variables=examples_block_variables,
    )
