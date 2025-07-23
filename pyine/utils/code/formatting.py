import black


def format_code(
    code_string: str,
    line_length: int = 120,
    is_pyi: bool = False,
    string_normalization: bool = True,
) -> str:
    """
    Formats a given string of Python code using the black code formatter.

    This function provides a programmatic way to apply black's formatting
    to a string, making it useful for code generation or transformation pipelines.

    Args:
        code_string: The string of Python code to format.
        line_length: The maximum line length to allow.
        is_pyi: If True, formats the code as a .pyi stub file.
        string_normalization: If True, normalizes string quotes and prefixes.

    Returns:
        A string containing the formatted Python code.
    """
    mode = black.Mode(
        line_length=line_length,
        is_pyi=is_pyi,
        string_normalization=string_normalization,
    )
    try:
        return black.format_str(code_string, mode=mode)
    except black.NothingChanged:
        # if the code is already formatted, return the original string
        return code_string


if __name__ == "__main__":
    unformatted_code = """\
def very_long_function_nameeeeeeeeeeeeeeeeeeee(
    argument_one,
    argument_two, argument_three,
    arg4,
    arg5,arg6,arg7_and_another_very_long_name_for_an_argument,
):
    print(
        "This is a poorly formatted line")
    a=1; b=2;
    if a<b:
        print('hello world')
    return a+b

class MyClass:
    def another_method(self, x,y,z):
        return ((x*y)  + z)
"""
    print("Original Code:")
    print("```python")
    print(unformatted_code)
    print("```")
    formatted_code = format_code(unformatted_code)
    print("\n" + "=" * 20 + "\n")
    print("Formatted Code:")
    print("```python")
    print(formatted_code)
    print("```")
