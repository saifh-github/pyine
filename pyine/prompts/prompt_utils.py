import functools
import importlib.resources
import logging
import pathlib
import random
import typing

import langchain_core.prompts
import pydantic
import yaml

DEFAULT_PROMPT_VERSION_KEY = "__default__"
"""Key used in config files to identify the default prompt version to use if none is specified.

If the config files does NOT contain this key to define the default prompt version to use, the last
prompt version in the file will be used as the default.
"""

PromptTemplateFormat = langchain_core.prompts.string.PromptTemplateFormat
"""Supported formats for prompt templates (provided by LangChain).

Note: we suggest using f-string templates by default; jinja2 templates should only be used if you
trust whoever provided them, as parsing them can lead to arbitrary code execution. See LangChain
documentation for more details.
"""

EXAMPLE_OUTPUT_KEY = "output"
"""Key used to identify expected outputs in example templates; required to be in all templates."""
EXAMPLE_OPT_INDEX_KEY = "example_idx"
"""Key used to identify example indices in example templates; optional, provided if detected."""
EXAMPLE_OPT_COUNT_KEY = "example_count"
"""Key used to identify the total number of examples in example templates; optional, provided if detected."""

logger = logging.getLogger(__name__)

# @@@@@@@ TODO: update all past prompt stuff into yamls


class PromptTemplate(pydantic.BaseModel):
    """Model for prompt templates."""

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""

    template: str
    """The prompt template string."""
    format: PromptTemplateFormat = "f-string"
    """The format of the prompt template (f-string or jinja2; f-string is preferred for security)."""


DefaultExamplesBlockTemplate = PromptTemplate(
    template="""\
{% if examples_str -%}
Here are some examples:

{{examples_str}}
{%- endif %}
""",
    format="jinja2",
)
"""Default template used for rendering a block of examples inside a prompt template."""


class PromptExample(pydantic.BaseModel):
    """Model for input/output examples used in few-shot learning."""

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")
    """Pydantic model configuration (freezes the dataclass)."""

    input_variables: dict[str, typing.Any]
    """List of inputs for the example that will be fed into an example template."""
    output: str
    """Expected output for the example (i.e. what a model should produce)."""
    description: str | None = None
    """Optional description of what this example demonstrates."""


class PromptMetadata(pydantic.BaseModel):
    """Model for prompt metadata and configuration."""

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")
    """Pydantic model configuration (freezes the dataclass)."""

    name: str
    """Name of the prompt."""
    description: str
    """Description of what this prompt does."""
    version: str
    """Version or reference name for the prompt (e.g. "v1.0", "big-provider/target_model", ...)."""
    tags: list[str] = []
    """Tags for categorizing the prompt."""


class PromptConfig(pydantic.BaseModel):
    """Complete prompt configuration model.

    All prompts are currently split into four blocks: role, context, examples, and question.
     - The role describes to the model what it should do.
     - The context describes the context in which the model operates as well as the kind of
       task(s) it will be required to solve.
     - The examples are optional and serve to provide additional context and improve the model's
       task understanding.
     - The question is the ultimate query to be answered by the model.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")
    """Pydantic model configuration (freezes the dataclass)."""

    metadata: PromptMetadata
    """Metadata about the prompt."""
    role: PromptTemplate | None = None  # role does not always need to be specified
    """The role template with potential placeholder variables."""
    context: PromptTemplate | None = None  # can be none if we just want to ask a question w/o context
    """The context template with potential placeholder variables."""
    example_template: PromptTemplate | None = None  # can be none if this prompt does not have examples
    """The template used when rendering individual examples inside the prompt template itself."""
    examples_block_template: PromptTemplate | None = DefaultExamplesBlockTemplate  # same as above
    """The template used when rendering a set of in-context examples inside the prompt template itself."""
    question: PromptTemplate  # mandatory; otherwise, why are we prompting a model without a question?
    """The question template with placeholder variables."""
    examples: list[PromptExample] = []
    """Examples tied to this prompt that can be used for few-shot/in-context learning."""
    template_block_separator: str = "\n\n"
    """Separator to use between prompt template blocks."""

    @property
    def example_count(self) -> int:
        """Number of examples available for this prompt."""
        return len(self.examples)

    def get_examples_as_text(
        self,
        target_examples: int | list[int] | None = None,
    ) -> str:
        """Formats examples as text for inclusion in prompts.

        Args:
            target_examples: List of examples to target when rendering the prompt. Can pass in
                a list of example indices, or an integer that specifies the number of samples to
                pick randomly. If `None` is provided instead, all examples are included.

        Returns:
            A string containing the formatted examples (with no more processing needed).
        """
        if not self.examples:
            return ""
        if target_examples is not None:
            if isinstance(target_examples, int):
                examples = random.sample(self.examples, min(target_examples, len(self.examples)))
            elif isinstance(target_examples, list):
                examples = [self.examples[idx] for idx in target_examples]
            else:
                raise ValueError(f"invalid target_examples: {target_examples}")
        else:
            examples = self.examples
        assert self.example_template is not None, "example template must be specified to format examples"
        formatted_examples = []
        for idx, example in enumerate(examples, 1):
            prompt_template = langchain_core.prompts.PromptTemplate.from_template(
                template=self.example_template.template,
                template_format=self.example_template.format,
            )
            assert (
                EXAMPLE_OUTPUT_KEY in prompt_template.input_variables
            ), f"example template must include '{EXAMPLE_OUTPUT_KEY}' variable"
            example_vars = example.input_variables.copy()
            assert EXAMPLE_OUTPUT_KEY not in example_vars, "overlap between input/output variable names"
            example_vars[EXAMPLE_OUTPUT_KEY] = example.output
            # add any variables that might be in the example object directly (but not in its input vars attribute)
            example_vars.update(
                {k: v for k, v in example.dict().items() if k not in [*example_vars, "input_variables"]}
            )
            if EXAMPLE_OPT_INDEX_KEY in prompt_template.input_variables and EXAMPLE_OPT_INDEX_KEY not in example_vars:
                # add the index of the example as an additional variable if needed
                example_vars[EXAMPLE_OPT_INDEX_KEY] = idx  # remember: these are 1-based indices
            if EXAMPLE_OPT_COUNT_KEY in prompt_template.input_variables and EXAMPLE_OPT_COUNT_KEY not in example_vars:
                # add the total number of examples used in this prompt as an additional variable if needed
                example_vars[EXAMPLE_OPT_COUNT_KEY] = len(examples)
            formatted_examples.append(prompt_template.format(**example_vars))
        return self.template_block_separator.join(formatted_examples)

    def create_prompt_template(
        self,
        include_examples: bool = True,
        target_examples: int | list[int] | None = None,
        role_variables: dict[str, typing.Any] | None = None,
        context_variables: dict[str, typing.Any] | None = None,
        examples_block_variables: dict[str, typing.Any] | None = None,
    ) -> langchain_core.prompts.PromptTemplate:
        """Create a LangChain prompt template from role, context, examples, and question templates.

        Args:
            include_examples: Whether to include few-shot examples in the template.
            target_examples: List of examples to target when rendering the prompt. Can pass in
                a list of example indices, or an integer that specifies the number of samples to
                pick randomly. If `None` is provided instead, all examples are included.
            role_variables: Variables to substitute in the role template.
            context_variables: Variables to substitute in the context template.
            examples_block_variables: Variables to substitute in the examples block template. Should
                not include the 'examples_str' variable (will be added directly).

        Returns:
            A `langchain_core.prompts.PromptTemplate` instance ready for use with LangChain.
        """
        template_parts = []
        if self.role is not None:
            role_prompt = langchain_core.prompts.PromptTemplate.from_template(
                template=self.role.template,
                template_format=self.role.format,
            ).format(**(role_variables or {}))
            template_parts.append(role_prompt)
        if self.context is not None:
            context_prompt = langchain_core.prompts.PromptTemplate.from_template(
                template=self.context.template,
                template_format=self.context.format,
            ).format(**(context_variables or {}))
            template_parts.append(context_prompt)
        if include_examples and self.examples:
            assert (
                self.examples_block_template is not None
            ), "examples block template must be specified to render examples"
            examples_block_template = langchain_core.prompts.PromptTemplate.from_template(
                template=self.examples_block_template.template,
                template_format=self.examples_block_template.format,
            )
            assert (
                "examples_str" in examples_block_template.input_variables
            ), "examples block template must include 'examples_str' variable"
            examples_block_variables = examples_block_variables or {}
            assert "examples_str" not in examples_block_variables, "overlap between input/output variable names"
            examples_str = self.get_examples_as_text(target_examples=target_examples)
            examples_block_variables["examples_str"] = examples_str
            examples_block_prompt = examples_block_template.format(**examples_block_variables)
            template_parts.append(examples_block_prompt)
        question_prompt = langchain_core.prompts.PromptTemplate.from_template(
            template=self.question.template,
            template_format=self.question.format,
        )  # for template validation + to get input variables list for combined prompt below
        template_parts.append(question_prompt.template)
        complete_template = self.template_block_separator.join(template_parts)
        combined_prompt = langchain_core.prompts.PromptTemplate(
            template=complete_template,
            input_variables=question_prompt.input_variables,
        )
        return combined_prompt

    def render_prompt(
        self,
        include_examples: bool = True,
        target_examples: int | list[int] | None = None,
        role_variables: dict[str, typing.Any] | None = None,
        context_variables: dict[str, typing.Any] | None = None,
        **question_variables,
    ) -> str:
        """Render a prompt template with the provided variables.

        Args:
            include_examples: Whether to include few-shot examples in the template.
            target_examples: List of examples to target when rendering the prompt. Can pass in
                a list of example indices, or an integer that specifies the number of samples to
                pick randomly. If `None` is provided instead, all examples are included.
            role_variables: Variables to substitute in the role template.
            context_variables: Variables to substitute in the context template.
            **question_variables: Variables to substitute in the question template.

        Returns:
            A string containing the rendered prompt (with no more processing needed).
        """
        template = self.create_prompt_template(
            include_examples=include_examples,
            target_examples=target_examples,
            role_variables=role_variables,
            context_variables=context_variables,
        )
        return template.format(**question_variables)


class VersionedPromptConfig(pydantic.BaseModel):
    """Container for multiple versioned prompt configurations.

    This model supports config files with multiple prompt versions where each version key maps to
    a complete prompt configuration.

    Example YAML structure:
    ```yaml
    v1.0.0:
      metadata:
        name: "My Prompt"
        description: "First version of the prompt"
      template: "This is the v1 template with {variable}."
      input_variables: ["variable"]

    v2.0.0:
      metadata:
        name: "My Prompt"
        description: "Improved version of the prompt"
      template: "This is the improved v2 template with {variable}."
      input_variables: ["variable"]
    ```
    """

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""
    versions: dict[str, PromptConfig]
    """Dictionary mapping version strings to prompt configurations."""
    default_version: str
    """The default version to use if none is specified when asked for a prompt config."""

    @classmethod
    def from_yaml(cls, yaml_file_path: pathlib.Path | str) -> "VersionedPromptConfig":
        """Parse YAML file content into a VersionedPromptConfig object."""
        yaml_file_path = pathlib.Path(yaml_file_path)
        assert yaml_file_path.is_file(), f"YAML file does not exist: {yaml_file_path}"
        yaml_content = yaml_file_path.read_text(encoding="utf-8")
        raw_data = yaml.safe_load(yaml_content)
        versions = {}
        # note: we do not enforce a version pattern since versions might be named after targeted LLMs/APIs
        for version, config_data in raw_data.items():
            if version == DEFAULT_PROMPT_VERSION_KEY:
                continue  # we'll take care of this one below
            assert isinstance(config_data, dict), "each prompt version must be mapped to a config a dictionary"
            # if the version information is not already in the config data, add it here before validation
            if "metadata" not in config_data:
                config_data["metadata"] = {"version": version}
            else:
                assert "version" not in config_data["metadata"], "version cannot be specified in metadata directly"
                config_data["metadata"]["version"] = version
            versions[version] = PromptConfig.model_validate(config_data)
        if not versions:
            raise ValueError("no valid prompt versions found in YAML file")
        if DEFAULT_PROMPT_VERSION_KEY in raw_data:
            default_version = raw_data[DEFAULT_PROMPT_VERSION_KEY]
            assert isinstance(default_version, str) and default_version in versions, "default version does not exist"
        else:
            # set the default version to be the latest one by default if nothing is specified (i.e. last in file)
            default_version = next(reversed(versions))
        return cls(versions=versions, default_version=default_version)

    @property
    def version_count(self) -> int:
        """Number of prompt versions available."""
        return len(self.versions)

    def get_default(self) -> PromptConfig:
        """Get the default prompt configuration."""
        return self.versions[self.default_version]


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
        self._cache: dict[str, PromptConfig] = {}

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
    ) -> PromptConfig:
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
            versioned_config = VersionedPromptConfig.from_yaml(prompt_file)  # noqa
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
    ) -> PromptConfig:
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
        versioned_config = VersionedPromptConfig.from_yaml(prompt_file)  # noqa
        return list(versioned_config.versions.keys())

    def clear_cache(self) -> None:
        """Clear the internal prompt cache."""
        self._cache.clear()
        self.get_prompt_config.cache_clear()
