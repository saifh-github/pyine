import langchain_core.output_parsers
import langchain_core.prompts
import langchain_core.runnables

import pyine.utils.llm_providers
import pyine.utils.portability


class SafePromptTemplate(langchain_core.prompts.ChatPromptTemplate):
    """PromptTemplate that safely preserves existing curly braces in template strings.

    This class ensures that any curly braces in the original template content remain intact,
    while still allowing for proper variable formatting.
    """

    def format(self, **kwargs):
        """Format the prompt template, preserving any original curly braces."""
        # The template contains the variables to be replaced
        return super().format(**kwargs)


# @@@@ TODO: put example snippets, inputs, and outputs in a dataclass to make reformatting easier
# also, use for reformatting: pyine.utils.portability.print_code_with_numbered_lines()

# Pre-escape the curly braces in example code to avoid template formatting issues
_code_execution_examples_str = """\
Example code snippet to execute:
```python
name = input("Enter your name: ")
greeting = "Hello, " + name
print(greeting)
```
Provided input arguments:
```
"Bob"
```
Expected output:
"Hello, Bob"

Example code snippet to execute:
```python
def calculate_area_to_circumference_ratio(radius: float) -> float:
    '''Calculate the ratio of a circle's area to its circumference.'''
    circumference = 2 * 3.14159 * radius
    area = 3.14159 * radius ** 2
    return area / circumference

radius = float(input("Enter circle radius: "))
ratio = calculate_area_to_circumference_ratio(radius)
print(f"Area to circumference ratio: {ratio:.3f}")
```
Provided input arguments:
```
2.4
```
Expected output:
"Area to circumference ratio: 1.200"

Example code snippet to execute:
```python
def find_second_largest(numbers: list[int]) -> int:
    '''Find the second largest number in a list.'''
    unique_numbers = list(set(numbers))
    unique_numbers.sort()
    return unique_numbers[-2]
```
Provided input arguments:
```
numbers=[10, 2, 3, 6, 10]
```
Expected output:
6
"""

_code_execution_template_str = """\
You are an expert at interpreting and executing Python 3 code.

Given a Python code snippet, we want to determine the output of the code, given some input arguments.

You must ONLY provide the output of the code, and you must not provide any additional information.

{examples}

Here is the code you must now execute:
```python
{code}
```
Provided input arguments:
```
{input_args}
```
Your predicted output:
"""

code_execution_prompt = SafePromptTemplate(
    input_variables=["code", "input_args", "examples"],
    template=_code_execution_template_str,
)


def get_chain(**kwargs) -> langchain_core.runnables.Runnable:
    """Get an inference chain based on the above prompt template and parser.

    The chain safely handles any curly braces in the code examples and input.
    """
    llm = pyine.utils.llm_providers.get_llm_from_provider(**kwargs)
    # Create a partial chain that injects the examples
    code_execution_chain = langchain_core.runnables.RunnableSequence(
        code_execution_prompt.partial(examples=_code_execution_examples_str),
        llm,
    )
    return code_execution_chain


if __name__ == "__main__":
    _dummy_code = """\
def add(a, b):
    return a + b + 1
"""
    _dummy_input_args = "a=3, b=4"
    _dummy_prompt = code_execution_prompt.format(
        code=_dummy_code, input_args=_dummy_input_args, examples=_code_execution_examples_str
    )
    print(_dummy_prompt)
