import functools
import importlib.resources
import logging
import pathlib

import langchain_core.prompts
import pydantic
import yaml

import pyine.prompts.utils as prompt_utils

logger = logging.getLogger(__name__)


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
    ):
        """Initialize the prompt manager.

        Args:
            package_name: Name of the package containing prompt files.
            prompts_subdir: Subdirectory within the package containing prompts.
        """
        self.package_name = package_name
        self.prompts_subdir = prompts_subdir
        self._cache: dict[str, prompt_utils.PromptConfig] = {}

    def _get_prompt_file_path(
        self,
        prompt_name: str,
    ) -> pathlib.Path:
        """Get the path to a prompt YAML file within the package resources."""
        return pathlib.Path(self.prompts_subdir) / f"{prompt_name}.yaml"

    def _get_prompt_module(
        self,
        prompt_name: str,
    ):
        """Get the module containing specific prompt template defines by name."""
        prompt_module_path = prompt_name.replace("/", ".")
        return importlib.import_module(f"{self.package_name}.configs.{prompt_module_path}")

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
        prompt_path = self._get_prompt_file_path(prompt_name)
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
        prompts_dir = pathlib.Path(package_files) / self.prompts_subdir  # noqa
        if not prompts_dir.is_dir():
            return []
        prompt_names = []
        for root, _, files in prompts_dir.walk():
            rel_path = root.relative_to(prompts_dir)
            for file in files:
                if not file.lower().endswith(".yaml"):
                    continue
                if rel_path == pathlib.Path("."):
                    prompt_names.append(file[:-5])  # remove .yaml extension
                else:
                    prompt_names.append(f"{rel_path}/{file[:-5]}")

        return sorted(prompt_names)

    def list_prompt_versions(self, prompt_name: str) -> list[str]:
        """List all available versions for a specific prompt."""
        prompt_path = self._get_prompt_file_path(prompt_name)
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
        import pyine.utils.pydantic_loader as pydantic_loader

        # if the default framework manager is not created, make sure models are all registered too
        pydantic_loader.PydanticYAMLLoader.register_models_from_package("pyine")
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
) -> langchain_core.prompts.PromptTemplate:
    """Convenience function to get a prompt template using the default manager.

    Note: if the prompt is known and registered, we will check its corresponding module to see if
    it possesses an override to generate the prompt template. If so, it will be used instead in
    order for partial variables to be correctly filled in.

    Args:
        prompt_name: Name of the prompt to retrieve
        version: The version of the prompt to retrieve. If None, the default version is returned.
        include_examples: Whether to include few-shot examples in the template.
        target_examples: List of examples to target when rendering the prompt. Can pass in
            a list of example indices, or an integer that specifies the number of samples to
            pick randomly. If `None` is provided instead, all examples are included.
    """
    manager = get_framework_prompt_manager()
    prompt_config = manager.get_prompt_config(prompt_name=prompt_name, version=version)
    try:
        prompt_module = manager._get_prompt_module(prompt_name)  # noqa
        if hasattr(prompt_module, "get_prompt_template"):
            return prompt_module.get_prompt_template(
                include_examples=include_examples,
                target_examples=target_examples,
            )
    except ModuleNotFoundError:
        logger.info(f"could not find module for {prompt_name}, using default template constructor")
        pass
    return prompt_config.create_prompt_template(
        include_examples=include_examples,
        target_examples=target_examples,
    )
