import enum

import langchain_core.output_parsers
import langchain_core.prompts
import pydantic

import pyine.prompts.manager
import pyine.prompts.utils
import pyine.utils.pydantic_loader


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

    model_config = pydantic.ConfigDict(frozen=True)
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


output_parser = langchain_core.output_parsers.PydanticOutputParser(
    pydantic_object=CodeAnalysisResponse,
)
expected_output_format_str = output_parser.get_format_instructions()


def get_prompt_config(
    version: str | None = None,
) -> pyine.prompts.utils.PromptConfig:
    """Get the prompt configuration for the code analysis prompt.

    Args:
        version: The version of the prompt to retrieve. If None, the default version is returned.
    """
    # first, make sure the pydantic loader has already registered this class
    pyine.utils.pydantic_loader.PydanticYAMLLoader.register_models_from_module(__name__)
    # note: this will be cached by the prompt manager
    return pyine.prompts.manager.get_prompt_config(
        prompt_name="code_analysis",
        version=version,
    )


def get_prompt_template(
    version: str | None = None,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
) -> langchain_core.prompts.PromptTemplate:
    """Return the prompt template for the code analysis prompt.

    Note: this implementation appropriately fills in all relevant partial variables, if any.

    Args:
        version: The version of the prompt to retrieve. If None, the default version is returned.
        include_examples: Whether to include few-shot examples in the template.
        target_examples: List of examples to target when rendering the prompt. Can pass in
            a list of example indices, or an integer that specifies the number of samples to
            pick randomly. If `None` is provided instead, all examples are included.
    """
    prompt_config = get_prompt_config(version=version)
    return prompt_config.create_prompt_template(
        include_examples=include_examples,
        target_examples=target_examples,
        role_variables=None,
        context_variables=dict(expected_output_format=expected_output_format_str),
        examples_block_variables=None,
    )
