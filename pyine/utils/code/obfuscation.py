import python_minifier

import pyine.utils.code.formatting


def obfuscate_code(
    code_string: str,
    remove_docstrings_and_literals: bool = True,
    rename_local_variables: bool = True,
    preserved_local_names: list[str] | None = None,
    rename_global_variables: bool = True,
    preserve_global_names: list[str] | None = None,
    reformat_output: bool = True,
) -> str:
    """Obfuscates a given string of Python code using the python-minifier library.

    The function provides several options to control the level of obfuscation, focusing on
    transformations that do not significantly alter the code's structure in terms of lines and
    execution flow. The output should be a code snippet whose interpretation and execution (using
    a model) should be similar or more difficult to the original due to the lack of informative
    shortcuts.

    Note: the output code might need reformatting (using e.g. black).

    Args:
        code_string: the string of Python code to obfuscate.
        remove_docstrings_and_literals: whether to remove docstrings and literal statements from
            the code. Literals include all statements that consist entirely of a literal value.
        rename_local_variables: whether to rename local variables. If True, the function will
            rename any non-global names, including variables, internal functions, etc.
        preserved_local_names: a list of local names that should not be renamed (optional).
        rename_global_variables: whether to rename global variables. If True, the function will
            rename any global names, including imports, function names, builtins, etc.
        preserve_global_names: a list of global names that should not be renamed (optional).
        reformat_output: whether to reformat the output code using black.
    """

    output = python_minifier.minify(
        source=code_string,
        remove_annotations=True,
        remove_pass=False,  # to make sure we preserve most execution steps, even if unnecessary
        remove_literal_statements=remove_docstrings_and_literals,
        combine_imports=False,  # to make sure we preserve most execution steps, even if unnecessary
        hoist_literals=False,  # seems to reorganize lines in a much too dense fashion (harder to trace)
        rename_locals=rename_local_variables,
        preserve_locals=preserved_local_names,
        rename_globals=rename_global_variables,
        preserve_globals=preserve_global_names,
        remove_object_base=True,
        convert_posargs_to_args=True,
        preserve_shebang=True,
        remove_asserts=False,  # need to keep behavior intact, and asserts are part of that
        remove_debug=False,  # same as above
        remove_explicit_return_none=False,  # to make sure we preserve most execution steps, even if unnecessary
        remove_builtin_exception_brackets=True,
        constant_folding=False,  # we want similar-or-worse-complexity examples, and this lowers complexity
    )
    if reformat_output:
        output = pyine.utils.code.formatting.format_code(output)
    return output


if __name__ == "__main__":
    _sample_code = """\
import math
import numpy as np

class MyClass:
    '''This is a sample class.'''
    def __init__(self, value):
        self.my_variable = value  # here's a nice comment

    def my_method(self, multiplier):
        '''This method multiplies the variable.'''
        result = self.my_variable * np.abs(multiplier)
        print(f"The result is: {result}")
        return result

def my_function(x, y):
    '''This is a sample function.'''
    return x + y + math.sqrt(np.abs(x - y))

instance = MyClass(10)
instance.my_method(5)
my_function(3, 4)
"""
    print(f"original code:\n```\n{_sample_code}\n```")
    _obfuscated_code = obfuscate_code(_sample_code)
    print("\n------\n")
    print(f"obfuscated code (default settings):\n```\n{_obfuscated_code}\n```")
