import logging
import pathlib
import random
import typing

import langchain_core.prompts
import pydantic

import pyine.utils.pydantic_loader

DEFAULT_PROMPT_VERSION_KEY = "__default__"
"""Key used in config files to identify the default prompt version to use if none is specified.

If the config files does NOT contain this key to define the default prompt version to use, the last
prompt version in the file will be used as the default.
"""
INTERNAL_DEFINES_KEY = "__defines__"
"""Key used to store internal defines in YAML files (will be skipped by the YAML parser)."""

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


class PromptTemplate(pydantic.BaseModel):
    """Model for prompt templates."""

    template: str
    """The prompt template string."""
    format: PromptTemplateFormat = "f-string"
    """The format of the prompt template (f-string or jinja2; f-string is preferred for security)."""
    partial_variables: dict[str, typing.Any] | None = None
    """Optional dictionary of partial variables to be used in the template."""

    def get_partially_rendered_prompt(
        self,
        **kwargs,  # extra partial variables (if any are needed)
    ) -> langchain_core.prompts.PromptTemplate:
        """Return a LangChain prompt template with partial variables filled in."""
        return langchain_core.prompts.PromptTemplate.from_template(
            template=self.template,
            template_format=self.format,
            partial_variables=dict(**(self.partial_variables or {}), **kwargs),
        )

    def render_prompt(self, **kwargs) -> str:
        """Render the prompt template with the provided + internal (partial) variables."""
        return self.get_partially_rendered_prompt().format(**kwargs)


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
    output: typing.Any
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
    examples: list[PromptExample] | None = []
    """Examples tied to this prompt that can be used for few-shot/in-context learning."""
    template_block_separator: str = "\n\n"
    """Separator to use between prompt template blocks."""

    @property
    def example_count(self) -> int:
        """Number of examples available for this prompt."""
        if self.examples is not None:
            return len(self.examples)
        return 0

    def get_examples_as_text(
        self,
        target_examples: int | list[int] | None = None,
        extra_variables: dict[str, typing.Any] | None = None,
    ) -> str:
        """Formats examples as text for inclusion in prompts.

        Args:
            target_examples: List of examples to target when rendering the prompt. Can pass in
                a list of example indices, or an integer that specifies the number of samples to
                pick randomly. If `None` is provided instead, all examples are included.
            extra_variables: Extra variables to substitute in the example templates.

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
            prompt_template = self.example_template.get_partially_rendered_prompt(**(extra_variables or {}))
            assert (
                EXAMPLE_OUTPUT_KEY in prompt_template.input_variables
            ), f"example template must include '{EXAMPLE_OUTPUT_KEY}' variable"
            example_vars = example.input_variables.copy()
            assert EXAMPLE_OUTPUT_KEY not in example_vars, "overlap between input/output variable names"
            example_vars[EXAMPLE_OUTPUT_KEY] = example.output
            # add any variables that might be in the example object directly (but not in its input vars attribute)
            example_vars.update(
                {k: v for k, v in example.model_dump().items() if k not in [*example_vars, "input_variables"]}
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

        Note: this implementation will fully render (format) all the role, context, and examples
        prompts, meaning all variables (arguments) for these prompts must have already been
        specified, or they must be specified via the corresponding dictionaries.

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
        rendered_template_parts = []
        if self.role is not None:
            role_prompt = self.role.render_prompt(**(role_variables or {}))
            rendered_template_parts.append(role_prompt)
        if self.context is not None:
            context_prompt = self.context.render_prompt(**(context_variables or {}))
            rendered_template_parts.append(context_prompt)
        if include_examples and self.examples:
            if self.examples_block_template:
                examples_block_template = self.examples_block_template
            else:
                examples_block_template = DefaultExamplesBlockTemplate
            examples_block_template = examples_block_template.get_partially_rendered_prompt()
            assert (
                "examples_str" in examples_block_template.input_variables
            ), "examples block template must include 'examples_str' variable"
            examples_block_variables = examples_block_variables or {}
            assert "examples_str" not in examples_block_variables, "overlap between input/output variable names"
            examples_str = self.get_examples_as_text(
                target_examples=target_examples,
                extra_variables=examples_block_variables,
            )
            examples_block_variables["examples_str"] = examples_str
            examples_block_prompt = examples_block_template.format(**examples_block_variables)
            rendered_template_parts.append(examples_block_prompt)
        # all template parts that might have been created so far are fully rendered ones
        complete_template = langchain_core.prompts.PromptTemplate.from_template(
            template=self.template_block_separator.join([*rendered_template_parts, self.question.template]),
            template_format=self.question.format,
            partial_variables=self.question.partial_variables,
        )
        return complete_template

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
        raw_data = pyine.utils.pydantic_loader.load_yaml_with_pydantic_support(yaml_file_path)
        versions = {}
        # note: we do not enforce a version pattern since versions might be named after targeted LLMs/APIs
        for version, config_data in raw_data.items():
            if version == DEFAULT_PROMPT_VERSION_KEY:
                continue  # we'll take care of this one below
            if version == INTERNAL_DEFINES_KEY:
                continue  # this block contains variable definitions (aliases), skip it
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
