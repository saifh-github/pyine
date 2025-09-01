import enum
import typing

import pydantic

if typing.TYPE_CHECKING:
    import langchain_core.output_parsers
    import langchain_core.prompts

    import pyine.prompts.types


class CodeTypeOptions(enum.StrEnum):
    """Code type options for executing algorithms in Python."""

    SIMPLE = "simple"
    """Option that corresponds to simple code to be executed as-is.

    Such 'simple code' captures required inputs and returns expected outputs on its own, and it
    does not contain any function or class declaration.
    """
    SIMPLE_WITH_DECL = "simple-with-declarations"
    """Option that corresponds to code to be executed as-is containing functions/classes."""
    CALLABLE = "callable"
    """Option that corresponds to code that declares a specific callable class or function.

    For execution, the declared callable is meant to be instantiated by a user and used to execute
    the algorithm.
    """


class InputTypeOptions(enum.StrEnum):
    """Input (argument) type options for passing inputs to algorithms in Python."""

    STDIN = "stdin"
    """Code expects to read arguments via standard input, i.e. Python's `input()` function."""
    CLI = "cli"
    """Code expects to read arguments via command-line arguments."""
    FILE = "file"
    """Code tries to read input arguments or data from a file."""
    ENV = "env-vars"
    """Code tries to read arguments from environment variables."""
    CALLABLE = "callable"
    """Code declares a callable object that expects to be provided arguments directly upon use."""
    NO_INPUT = "no-input"
    """Code does not expect any input."""


class OutputTypeOptions(enum.StrEnum):
    """Output type options for returning algorithm results in Python."""

    STDOUT = "stdout"
    """Code prints its final result to standard output."""
    FILE = "file"
    """Code writes its final result to a file."""
    CALLABLE = "callable"
    """Code declares a callable object that will return its final result directly upon use."""
    NO_OUTPUT = "no-output"
    """Code does not return anything."""


class CodeAnalysisResponse(pydantic.BaseModel):
    """Response model for code analysis.

    This model is used to describe the output of a model tasked with analyzing Python code.
    See the corresponding template YAML file for more details.
    """

    model_config = pydantic.ConfigDict(frozen=True, use_enum_values=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    is_deterministic: bool = pydantic.Field(
        description="Specifies whether the code is deterministic",
    )
    imports_nonstandard_packages: bool = pydantic.Field(
        description="Specifies whether the code imports nonstandard packages",
    )
    invalid_syntax: bool = pydantic.Field(
        description="Specifies whether the code contains invalid Python 3 syntax that would prevent execution",
    )
    blocks_execution: bool = pydantic.Field(
        description="Specifies whether the code contains logic that blocks execution, e.g. infinite loops or queries for user input",
    )
    unnecessary_lines: bool = pydantic.Field(
        description="Specifies whether the code contains unnecessary statements that do not contribute to its outputs, e.g. debugging prints or unit tests",
    )
    filesystem_access: bool = pydantic.Field(
        description="Specifies whether the code attempts to interact (read, write, or execute) anything on the filesystem",
    )
    system_commands: bool = pydantic.Field(
        description="Specifies whether the code attempts to run system commands, e.g. `os.system`, `eval`, etc.",
    )
    network_access: bool = pydantic.Field(
        description="Specifies whether the code attempts to open network connections, exchange data, use external APIs, or use networked services in any way",
    )
    code_type: CodeTypeOptions = pydantic.Field(
        description="Specifies the type of implementation used to execute the algorithm",
    )
    input_type: InputTypeOptions = pydantic.Field(
        description="Specifies how the code expects the algorithm to receive its input(s)",
    )
    output_type: OutputTypeOptions = pydantic.Field(
        description="Specifies how the code expects the algorithm to return its output(s)",
    )


def get_output_parser(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> "langchain_core.output_parsers.BaseOutputParser | None":  # noqa
    """Get the output parser for the code analysis prompt (if one should be used)."""
    # note: we currently have a single output structure for all version
    import langchain_core.output_parsers

    return langchain_core.output_parsers.PydanticOutputParser(pydantic_object=CodeAnalysisResponse)


def get_prompt_template(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
) -> "langchain_core.prompts.BasePromptTemplate":
    """Returns the prompt template for the code analysis prompt (manager module override).

    Note: this implementation appropriately fills in all relevant partial variables, if any.
    """
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config("code_analysis", version=version)
    template = prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        role_variables=None,
        context_variables=dict(expected_output_format=get_output_parser(version).get_format_instructions()),
        examples_block_variables=None,
    )
    if partial_vars:
        template = template.partial(**partial_vars)
    return template
