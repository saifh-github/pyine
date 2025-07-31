import langchain_core.output_parsers
import langchain_core.prompts
import langchain_core.runnables
import pydantic

import pyine.utils.llm_providers


class CallableAnalysisResponse(pydantic.BaseModel):
    """Response model for callable analysis."""

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""
    entrypoint_function_name: str = pydantic.Field(
        ...,  # required
        description=(
            "Specifies the name of the entrypoint function; if the function is inside "
            "a class, this field should ONLY be the name of the function inside that class."
        ),
    )
    entrypoint_function_arg_names: list[str] = pydantic.Field(
        default=...,  # required
        description=(
            "Specifies the names of the arguments that the entrypoint function takes; "
            "if the function takes no arguments, this field should be an empty list."
        ),
    )
    parent_class_name: str = pydantic.Field(
        default="",
        description=(
            "Specifies the name of the parent class of the entrypoint function; this is only "
            "relevant if the entrypoint function is inside a class; if it is not, this field "
            "should be an empty string."
        ),
    )
    parent_class_arg_names: list[str] = pydantic.Field(
        default=[],
        description=(
            "Specifies the names of the arguments that the parent class of the entrypoint "
            "function requires in order to be instantiated; if the class constructor takes "
            "no arguments, this field should be an empty list; this field is only relevant "
            "if the targeted entrypoint function is inside a class."
        ),
    )


callable_analysis_output_parser = langchain_core.output_parsers.PydanticOutputParser(
    pydantic_object=CallableAnalysisResponse,
)

_callable_analysis_expected_output_format_str = callable_analysis_output_parser.get_format_instructions()

_callable_analysis_example_outputs_str = f"""\
Example:
```python
class Solution:

    def __init__(self):
        pass

    def romanToDecimal(self, S):

        # code here
```

Expected output:
{
    CallableAnalysisResponse(
        entrypoint_function_name="romanToDecimal",
        entrypoint_function_arg_names=["S"],
        parent_class_name="Solution",
        parent_class_arg_names=[],
    ).model_dump_json(indent=2)
}

Another example:
```python
# USER CODE TEMPLATE v0.1
def sum_two_numbers(a, b):
    # SOLUTION HERE
```
Expected output:
{
    CallableAnalysisResponse(
        entrypoint_function_name="sum_two_numbers",
        entrypoint_function_arg_names=["a", "b"],
        parent_class_name="",
        parent_class_arg_names=[],
    ).model_dump_json(indent=2)
}
"""

_callable_analysis_template_str = """\
You are an expert at analyzing Python code and determining how to correctly call the most relevant \
function (the "entrypoint") to execute a given algorithm, even when that function is part of a class.

We will give you a code template ("starter code") that we expect will later be fully implemented.

We are currently only interested in how the algorithm would later be executed given that code template. \
Your task is to extract relevant information on the "entrypoint" that will be used for this execution.

If the function is inside a class, we expect to instantiate the class before accessing its function. \
If the function is not inside a class, we expect to use it directly.

{expected_output_format}

{example_outputs}

Here is the "starter code" you must now analyze:

```python
{starter_code}
```
"""


callable_analysis_prompt = langchain_core.prompts.PromptTemplate(
    input_variables=["starter_code", "expected_output_format", "example_outputs"],
    template=_callable_analysis_template_str,
)


def get_chain(**kwargs) -> langchain_core.runnables.Runnable:
    """Get an inference chain based on the above prompt template and parser."""
    llm = pyine.utils.llm_providers.get_llm_from_provider(**kwargs)
    llm_with_structured_output = llm.with_structured_output(CallableAnalysisResponse)
    callable_analysis_chain = langchain_core.runnables.RunnableSequence(
        callable_analysis_prompt.partial(
            expected_output_format=_callable_analysis_expected_output_format_str,
            example_outputs=_callable_analysis_example_outputs_str,
        ),
        llm_with_structured_output,
    )
    return callable_analysis_chain
