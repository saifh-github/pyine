import logging
import pathlib
import random
import typing

import langchain_core.messages
import langchain_core.prompts
import langchain_core.prompts.chat
import langchain_core.prompts.string
import pydantic

import pyine.utils.pydantic
from pyine.prompts.types import PromptNameType, PromptVersionType

DEFAULT_PROMPT_VERSION_KEY = "__default__"
"""Key used in config files to identify the default prompt version to use if none is specified.

If the config files does NOT contain this key to define the default prompt version to use, the last
prompt version in the file will be used as the default.
"""
INTERNAL_DEFINES_KEY = "__defines__"
"""Key used to store internal defines in YAML files (will be skipped by the YAML parser)."""
EXAMPLE_OUTPUT_KEY = "output"
"""Key used to identify expected outputs in example templates; required to be in all templates."""
EXAMPLE_OPT_INDEX_KEY = "example_idx"
"""Key used to identify example indices in example templates; optional, provided if detected."""
EXAMPLE_OPT_COUNT_KEY = "example_count"
"""Key used to identify the total number of examples in example templates; optional, provided if detected."""

logger = logging.getLogger(__name__)


def _apply_partial[
    TemplateType: langchain_core.prompts.BasePromptTemplate[typing.Any],
](
    template: TemplateType,
    values: dict[str, typing.Any],
) -> TemplateType:
    """Apply partial variables while preserving the concrete prompt template type."""
    if not values:
        return template
    partial_method = typing.cast("typing.Callable[..., TemplateType]", template.partial)  # type: ignore[reportUnknownMemberType]
    return partial_method(**values)


class PromptTemplate(pydantic.BaseModel):
    """Model for prompt templates (simpler version of the LangChain PromptTemplate class)."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    template: str
    """The prompt template string."""
    format: langchain_core.prompts.string.PromptTemplateFormat = "f-string"
    """The format of the prompt template (f-string or jinja2; f-string is preferred for security)."""
    partial_variables: dict[str, typing.Any] | None = None
    """Optional dictionary of partial variables to be used in the template."""
    optional_variables: list[str] | None = None
    """Optional dictionary of optional variables that do not need to be specified when rendering."""

    def get_partially_rendered_prompt(
        self,
        **kwargs: typing.Any,  # extra partial variables (if any are needed)
    ) -> langchain_core.prompts.PromptTemplate:
        """Return a LangChain prompt template with partial variables filled in."""
        merged_partials: dict[str, typing.Any] = {**(self.partial_variables or {}), **kwargs}
        return langchain_core.prompts.PromptTemplate.from_template(
            template=self.template,
            template_format=self.format,
            partial_variables=merged_partials,
        )

    def render_prompt(self, **kwargs: typing.Any) -> str:
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

    name: PromptNameType
    """Name of the prompt."""
    description: str
    """Description of what this prompt does."""
    version: PromptVersionType
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
    examples: list[PromptExample] | None = pydantic.Field(
        default_factory=lambda: typing.cast("list[PromptExample]", []),
    )
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
        examples: list[PromptExample]
        if target_examples is not None:
            if isinstance(target_examples, int):
                examples = random.sample(self.examples, min(target_examples, len(self.examples)))
            else:
                assert isinstance(target_examples, list)
                examples = [self.examples[idx] for idx in target_examples]
        else:
            examples = self.examples
        assert self.example_template is not None, "example template must be specified to format examples"
        formatted_examples: list[str] = []
        for idx, example in enumerate(examples, 1):
            prompt_template = self.example_template.get_partially_rendered_prompt(**(extra_variables or {}))
            assert EXAMPLE_OUTPUT_KEY in prompt_template.input_variables, (
                f"example template must include '{EXAMPLE_OUTPUT_KEY}' variable"
            )
            example_vars: dict[str, typing.Any] = dict(example.input_variables)
            if EXAMPLE_OUTPUT_KEY in example_vars:
                assert example_vars[EXAMPLE_OUTPUT_KEY] == example.output, "unexpected output value in example"
            else:
                # Convert Pydantic models to JSON strings to avoid sandbox issues with method calls in templates
                output_value = example.output
                if isinstance(output_value, pydantic.BaseModel):
                    output_value = output_value.model_dump_json(indent=2)
                example_vars[EXAMPLE_OUTPUT_KEY] = output_value
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

    def get_system_message(
        self,
        return_as_blocks: bool = False,
        include_examples: bool = True,
        target_examples: int | list[int] | None = None,
        role_variables: dict[str, typing.Any] | None = None,
        context_variables: dict[str, typing.Any] | None = None,
        examples_block_variables: dict[str, typing.Any] | None = None,
    ) -> str | list[str]:
        """Get the system message for this prompt.

        Note: this implementation will fully render (format) all the role, context, and examples
        prompts, meaning all variables (arguments) for these prompts must have already been
        specified, or they must be specified via the corresponding dictionaries.

        Args:
            return_as_blocks: Whether to return the system message as a list of message blocks.
            include_examples: Whether to include few-shot examples in the template.
            target_examples: List of examples to target when rendering the prompt. Can pass in
                a list of example indices, or an integer that specifies the number of samples to
                pick randomly. If `None` is provided instead, all examples are included.
            role_variables: Variables to substitute in the role template.
            context_variables: Variables to substitute in the context template.
            examples_block_variables: Variables to substitute in the examples block template. Should
                not include the 'examples_str' variable (will be added directly).

        Returns:
            The fully rendered system message as a string, or as a list of text blocks.
        """
        rendered_template_parts: list[str] = []
        if self.role is not None:
            role_prompt = self.role.render_prompt(**(role_variables or {}))
            rendered_template_parts.append(role_prompt)
        if self.context is not None:
            context_prompt = self.context.render_prompt(**(context_variables or {}))
            rendered_template_parts.append(context_prompt)
        if include_examples and self.examples:
            examples_block_template = self.examples_block_template or DefaultExamplesBlockTemplate
            examples_block_template = examples_block_template.get_partially_rendered_prompt()
            assert "examples_str" in examples_block_template.input_variables, (
                "examples block template must include 'examples_str' variable"
            )
            block_variables = dict(examples_block_variables) if examples_block_variables else {}
            assert "examples_str" not in block_variables, "overlap between input/output variable names"
            examples_str = self.get_examples_as_text(
                target_examples=target_examples,
                extra_variables=block_variables,
            )
            block_variables["examples_str"] = examples_str
            examples_block_prompt = examples_block_template.format(**block_variables)
            rendered_template_parts.append(examples_block_prompt)
        if return_as_blocks:
            return rendered_template_parts
        return self.template_block_separator.join(rendered_template_parts)

    @typing.overload
    def create_prompt_template(
        self,
        use_chat_template: typing.Literal[False] = False,
        include_examples: bool = True,
        target_examples: int | list[int] | None = None,
        partial_vars: dict[str, typing.Any] | None = None,
        role_variables: dict[str, typing.Any] | None = None,
        context_variables: dict[str, typing.Any] | None = None,
        examples_block_variables: dict[str, typing.Any] | None = None,
    ) -> langchain_core.prompts.PromptTemplate: ...

    @typing.overload
    def create_prompt_template(
        self,
        use_chat_template: typing.Literal[True],
        include_examples: bool = True,
        target_examples: int | list[int] | None = None,
        partial_vars: dict[str, typing.Any] | None = None,
        role_variables: dict[str, typing.Any] | None = None,
        context_variables: dict[str, typing.Any] | None = None,
        examples_block_variables: dict[str, typing.Any] | None = None,
    ) -> langchain_core.prompts.chat.ChatPromptTemplate: ...

    def create_prompt_template(
        self,
        use_chat_template: bool = False,
        include_examples: bool = True,
        target_examples: int | list[int] | None = None,
        partial_vars: dict[str, typing.Any] | None = None,
        role_variables: dict[str, typing.Any] | None = None,
        context_variables: dict[str, typing.Any] | None = None,
        examples_block_variables: dict[str, typing.Any] | None = None,
    ) -> langchain_core.prompts.PromptTemplate | langchain_core.prompts.chat.ChatPromptTemplate:
        """Create a LangChain prompt template from role, context, examples, and question templates.

        This function will try to fully render the system message along with the question, so all
        required variables for the system prompt must be provided now.

        Args:
            use_chat_template: Whether to return a chat prompt template or a regular prompt template.
            include_examples: Whether to include few-shot examples in the template.
            target_examples: List of examples to target when rendering the prompt. Can pass in
                a list of example indices, or an integer that specifies the number of samples to
                pick randomly. If `None` is provided instead, all examples are included.
            partial_vars: Partial variables to substitute in the final (output) prompt template.
            role_variables: Variables to substitute in the role template.
            context_variables: Variables to substitute in the context template.
            examples_block_variables: Variables to substitute in the examples block template. Should
                not include the 'examples_str' variable (will be added directly).

        Returns:
            A prompt template instance ready for use with LangChain.
        """
        # for proper escaping of anything in the system message, pre-determine input vars
        question_input_vars = langchain_core.prompts.string.get_template_variables(
            template=self.question.template,
            template_format=self.question.format,
        )
        if self.question.partial_variables:
            question_input_vars = [var for var in question_input_vars if var not in self.question.partial_variables]
        if use_chat_template:
            system_message_text = typing.cast(
                "str",
                self.get_system_message(
                    return_as_blocks=False,
                    include_examples=include_examples,
                    target_examples=target_examples,
                    role_variables=role_variables,
                    context_variables=context_variables,
                    examples_block_variables=examples_block_variables,
                ),
            )
            messages = [
                langchain_core.messages.SystemMessage(content=system_message_text),
                langchain_core.prompts.HumanMessagePromptTemplate.from_template(
                    self.question.template,
                    template_format=self.question.format,
                    partial_variables=self.question.partial_variables or {},
                    optional_variables=self.question.optional_variables or [],
                ),
            ]
            chat_template = langchain_core.prompts.chat.ChatPromptTemplate(
                messages=messages,
                input_variables=question_input_vars,
                partial_variables=self.question.partial_variables or {},
                optional_variables=self.question.optional_variables or [],
            )
            if self.question.optional_variables:
                existing_partial = getattr(chat_template, "partial_variables", {}) or {}
                optional_defaults = {
                    opt_var: "" for opt_var in self.question.optional_variables if opt_var not in existing_partial
                }
                if optional_defaults:
                    chat_template = _apply_partial(chat_template, optional_defaults)
            if partial_vars:
                chat_template = _apply_partial(chat_template, partial_vars)
            return chat_template

        system_message_blocks = typing.cast(
            "list[str]",
            self.get_system_message(
                return_as_blocks=True,
                include_examples=include_examples,
                target_examples=target_examples,
                role_variables=role_variables,
                context_variables=context_variables,
                examples_block_variables=examples_block_variables,
            ),
        )
        prompt_template = langchain_core.prompts.PromptTemplate(
            template=self.template_block_separator.join([*system_message_blocks, self.question.template]),
            template_format=self.question.format,
            input_variables=question_input_vars,
            partial_variables=self.question.partial_variables or {},
            optional_variables=self.question.optional_variables or [],
        )
        if self.question.optional_variables:
            existing_partial = getattr(prompt_template, "partial_variables", {}) or {}
            optional_defaults = {
                opt_var: "" for opt_var in self.question.optional_variables if opt_var not in existing_partial
            }
            if optional_defaults:
                prompt_template = _apply_partial(prompt_template, optional_defaults)
        if partial_vars:
            prompt_template = _apply_partial(prompt_template, partial_vars)
        return prompt_template

    def render_prompt(
        self,
        use_chat_template: bool = False,
        include_examples: bool = True,
        target_examples: int | list[int] | None = None,
        role_variables: dict[str, typing.Any] | None = None,
        context_variables: dict[str, typing.Any] | None = None,
        examples_block_variables: dict[str, typing.Any] | None = None,
        **question_variables: typing.Any,
    ) -> str:
        """Render a prompt template with the provided variables.

        Args:
            use_chat_template: Whether to return a chat prompt template or a regular prompt template.
            include_examples: Whether to include few-shot examples in the template.
            target_examples: List of examples to target when rendering the prompt. Can pass in
                a list of example indices, or an integer that specifies the number of samples to
                pick randomly. If `None` is provided instead, all examples are included.
            role_variables: Variables to substitute in the role template.
            context_variables: Variables to substitute in the context template.
            examples_block_variables: Variables to substitute in the examples block template. Should
                not include the 'examples_str' variable (will be added directly).
            **question_variables: Variables to substitute in the question template.

        Returns:
            A string containing the rendered prompt (with no more processing needed).
        """
        template = self.create_prompt_template(
            use_chat_template=use_chat_template,
            include_examples=include_examples,
            target_examples=target_examples,
            role_variables=role_variables,
            context_variables=context_variables,
            examples_block_variables=examples_block_variables,
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
    versions: dict[PromptVersionType, PromptConfig]
    """Dictionary mapping version identifiers to prompt configurations."""
    default_version: PromptVersionType
    """The default version to use if none is specified when asked for a prompt config."""

    @classmethod
    def from_yaml(cls, yaml_file_path: pathlib.Path | str) -> "VersionedPromptConfig":
        """Parse YAML file content into a VersionedPromptConfig object."""
        raw_data = pyine.utils.pydantic.load_yaml_with_pydantic_support(yaml_file_path)
        if not isinstance(raw_data, dict):
            raise ValueError("prompt configuration YAML must contain a mapping at the top level")
        typed_raw_data = typing.cast("dict[str, typing.Any]", raw_data)
        versions: dict[PromptVersionType, PromptConfig] = {}
        # note: we do not enforce a version pattern since versions might be named after targeted LLMs/APIs
        for version_key, config_data in typed_raw_data.items():
            if version_key == DEFAULT_PROMPT_VERSION_KEY:
                continue  # we'll take care of this one below
            if version_key == INTERNAL_DEFINES_KEY:
                continue  # this block contains variable definitions (aliases), skip it
            if not isinstance(config_data, dict):
                raise ValueError("each prompt version must be mapped to a config dictionary")
            typed_config_data = typing.cast("dict[str, typing.Any]", config_data)
            version: PromptVersionType = version_key
            # if the version information is not already in the config data, add it here before validation
            if "metadata" not in typed_config_data:
                typed_config_data["metadata"] = {"version": version}
            else:
                if "version" in typed_config_data["metadata"]:
                    raise ValueError("version cannot be specified in metadata directly")
                typed_config_data["metadata"]["version"] = version
            versions[version] = PromptConfig.model_validate(typed_config_data)
        if not versions:
            raise ValueError("no valid prompt versions found in YAML file")
        if DEFAULT_PROMPT_VERSION_KEY in typed_raw_data:
            default_version_value = typed_raw_data[DEFAULT_PROMPT_VERSION_KEY]
            if not isinstance(default_version_value, str):
                raise ValueError("default version must be a string")
            default_version: PromptVersionType = default_version_value
            if default_version not in versions:
                raise ValueError("default version does not exist")
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
