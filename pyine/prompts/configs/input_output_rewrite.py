"""Utilities for reformatting problem I/O samples into executable test payloads.

This module defines the structured schema and LangChain prompt plumbing used by the
input/output rewrite flow (see `pyine.apps.traces.trace_failure_analyzer`). The
pipeline targets legacy TACO problems whose `input_output` blocks were captured as
loose prose like "L = 0, R = 1" or who never recorded the callable entry point.
Such payloads cannot drive the trace replay tooling, which requires machine-
read-able JSON arguments aligned with the actual solver signature.

Key responsibilities:

- constrain LLM responses with a frozen `InputOutputRewriteResponse` model so
  every candidate patch carries the entrypoint name, JSON-encoded argument lists,
  and deserializable outputs aligned by index;
- expose `get_output_parser()`, handing callers the Pydantic-backed parser whose
  format instructions are embedded in the rewrite prompt; and
- build the rewrite prompt template via `get_prompt_template()`, injecting the
  parser guidance and wiring through optional template knobs (chat mode, example
  selection, role/context overrides).

Together these helpers let higher-level tools iterate on malformed dataset I/O
automatically: an LLM proposes canonicalized samples, the response is validated
here, and downstream consumers can immediately rerun solutions against the
patched tests.
"""

import typing

import pydantic

if typing.TYPE_CHECKING:
    import langchain_core.output_parsers
    import langchain_core.prompts

    import pyine.prompts.types


python_call_pattern = r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"


type PythonCallName = typing.Annotated[
    pydantic.StrictStr,
    pydantic.Field(pattern=python_call_pattern),
]


class InputOutputRewriteResponse(pydantic.BaseModel):
    """Structured response for input/output patch generation."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    inputs: list[pydantic.StrictStr] = pydantic.Field(
        description=(
            "List of JSON-encoded argument payloads preserved as strings (the dataset stores inputs as text).",
        ),
    )
    outputs: list[pydantic.JsonValue] = pydantic.Field(
        description=("List of outputs aligned by index with inputs. Values must be serializable via JSON."),
    )
    fn_name: PythonCallName = pydantic.Field(
        description=("Fully qualified entrypoint name to execute (e.g. 'Solution.solve' or 'countSquares')."),
    )

    def is_valid(
        self,
    ) -> bool:
        """Validate input/output alignment and presence of entrypoint metadata.

        Returns:
            bool: Whether the response contains a callable name and balanced inputs/outputs.
        """
        return bool(self.fn_name) and len(self.inputs) == len(self.outputs)


def get_output_parser(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> "langchain_core.output_parsers.BaseOutputParser | None":  # noqa
    import langchain_core.output_parsers

    return langchain_core.output_parsers.PydanticOutputParser(pydantic_object=InputOutputRewriteResponse)


def get_prompt_template(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
    role_variables: dict[str, typing.Any] | None = None,
    context_variables: dict[str, typing.Any] | None = None,
    examples_block_variables: dict[str, typing.Any] | None = None,
) -> "langchain_core.prompts.BasePromptTemplate":
    """Return the prompt template configured for minimal problem metadata.

    The associated Jinja template now expects only four input variables:
    `question`, `starter_code`, `first_solution`, and `input_output`.
    Consumers should provide pre-trimmed strings for the text fields and a
    JSON-encoded mapping for the `input_output` payload.
    """
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config("input_output_rewrite", version=version)
    base_context_variables = {
        "expected_output_format": get_output_parser(version).get_format_instructions(),
    }
    if context_variables:
        base_context_variables.update(context_variables)

    template = prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        role_variables=role_variables,
        context_variables=base_context_variables,
        examples_block_variables=examples_block_variables,
    )
    if partial_vars:
        template = template.partial(**partial_vars)
    return template
