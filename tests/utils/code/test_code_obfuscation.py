import io
import sys

import pytest

import pyine.utils.code.obfuscation as code_obf


@pytest.fixture
def sample_code():
    """Provides a sample Python code string for tests."""
    return """\
import math

class MyTestClass:
    '''A sample class for testing.'''
    an_int: int = 1

    def __init__(self, value: int):
        '''Initializes the class.'''
        self.my_variable: int = value
        "A literal string"

    def calculate(self, multiplier: int) -> int:
        '''A sample method.'''
        local_var = self.my_variable * multiplier
        print(f"Result is {local_var + self.an_int}")
        return local_var + self.an_int

def top_level_function(x: int, y: int) -> float:
    '''A sample top-level function.'''
    'Another literal string'
    result = math.sqrt(x**2 + y**2)
    return result

instance = MyTestClass(10)
instance.calculate(5)
top_level_function(3, 4)
"""


def test_default_obfuscation(sample_code):
    """Tests the default obfuscation settings."""
    obfuscated = code_obf.obfuscate_code(sample_code)
    assert "MyTestClass" not in obfuscated
    assert "top_level_function" not in obfuscated
    assert "local_var" not in obfuscated
    assert "A sample class for testing" not in obfuscated
    assert "A literal string" not in obfuscated
    assert ": int" not in obfuscated
    assert "-> float" not in obfuscated


def test_remove_docstrings_disabled(sample_code):
    """Tests that docstrings and literals are preserved when disabled."""
    obfuscated = code_obf.obfuscate_code(sample_code, remove_docstrings_and_literals=False)
    assert "A sample class for testing" in obfuscated
    assert "A literal string" in obfuscated
    assert "Another literal string" in obfuscated


def test_rename_local_variables_disabled(sample_code):
    """Tests that local variable names are preserved when disabled."""
    obfuscated = code_obf.obfuscate_code(sample_code, rename_local_variables=False)
    assert "local_var" in obfuscated
    assert "result" in obfuscated
    # globals are still renamed by default
    assert "MyTestClass" not in obfuscated
    obfuscated = code_obf.obfuscate_code(sample_code, rename_local_variables=True, preserved_local_names=["local_var"])
    assert "local_var" in obfuscated
    assert "result" not in obfuscated


def test_rename_global_variables_disabled(sample_code):
    """Tests that global variable names are preserved when disabled."""
    obfuscated = code_obf.obfuscate_code(sample_code, rename_global_variables=False)
    assert "MyTestClass" in obfuscated
    assert "top_level_function" in obfuscated
    # locals are still renamed by default
    assert "local_var" not in obfuscated
    obfuscated = code_obf.obfuscate_code(
        sample_code, rename_global_variables=True, preserve_global_names=["MyTestClass", "math"]
    )
    assert "MyTestClass" in obfuscated
    assert "math" in obfuscated
    assert "top_level_function" not in obfuscated


def test_obfuscated_code_executes_correctly(sample_code):
    """Tests that the obfuscated code produces the same output as the original."""
    original_stdout = io.StringIO()
    sys.stdout = original_stdout
    exec(sample_code, {})
    sys.stdout = sys.__stdout__
    original_output = original_stdout.getvalue()

    obfuscated_code = code_obf.obfuscate_code(sample_code)
    obfuscated_stdout = io.StringIO()
    sys.stdout = obfuscated_stdout
    exec(obfuscated_code, {})
    sys.stdout = sys.__stdout__
    obfuscated_output = obfuscated_stdout.getvalue()

    assert original_output == obfuscated_output
    assert "Result is 51" in original_output
