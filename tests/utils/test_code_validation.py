import pytest

from src.utils.code_validation import validate_code


def test_invalid_string():
    """Test validation with an invalid string."""
    with pytest.raises(AssertionError):
        validate_code("")
    with pytest.raises(AssertionError):
        validate_code(None)  # noqa
    with pytest.raises(AssertionError):
        validate_code(123)  # noqa


def test_exceeds_max_size():
    """Test validation with code exceeding maximum size."""
    large_code = "x = 1\n" * 50_001  # Creates a string > 100_000 chars
    with pytest.raises(AssertionError, match="code exceeds 100_000 chars"):
        validate_code(large_code, max_size=100_000)

    # Test with custom max_size
    with pytest.raises(AssertionError, match="code exceeds 10 chars"):
        validate_code("x = 1\ny = 2", max_size=10)


def test_contains_markdown_backticks():
    """Test validation with code containing Markdown backticks."""
    with pytest.raises(AssertionError, match="code contains forbidden Markdown elements"):
        validate_code("x = 1\n```python\ny = 2\n```")


def test_contains_response_tag():
    """Test validation with code containing response tags."""
    with pytest.raises(AssertionError, match="code contains forbidden prompt elements"):
        validate_code("x = 1\n</response>")


def test_unbalanced_parentheses():
    """Test validation with unbalanced parentheses."""
    with pytest.raises(AssertionError, match="code possesses unbalanced delimiters"):
        validate_code("def func(x:\n    return x + 1")


def test_unbalanced_brackets():
    """Test validation with unbalanced brackets."""
    with pytest.raises(AssertionError, match="code possesses unbalanced delimiters"):
        validate_code("x = [1, 2, 3\nprint(x)")


def test_unbalanced_braces():
    """Test validation with unbalanced braces."""
    with pytest.raises(AssertionError, match="code possesses unbalanced delimiters"):
        validate_code("data = {'key': 'value'\nprint(data)")


def test_mismatched_delimiters():
    """Test validation with mismatched delimiters."""
    with pytest.raises(AssertionError, match="code possesses unbalanced delimiters"):
        validate_code("x = (1, 2, 3]")


def test_mixed_indentation():
    """Test validation with mixed indentation (spaces and tabs)."""
    mixed_indent_code = "def func():\n    x = 1\n\ty = 2\n    return x + y"
    with pytest.raises(AssertionError, match="code contains mixed indentation"):
        validate_code(mixed_indent_code)


def test_restricted_function_calls():
    """Test validation with restricted function calls."""
    for func in ['exec', 'eval', '__import__', 'compile', 'globals', 'locals']:
        with pytest.raises(AssertionError, match="found potentially problematic function call"):
            validate_code(f"result = {func}('print(\"Hello\")')")


def test_restricted_imports():
    """Test validation with restricted imports."""
    # Direct import
    for module in ['subprocess', 'shutil', 'importlib']:
        with pytest.raises(AssertionError, match="found potentially problematic import"):
            validate_code(f"import {module}")

    # Import from
    for module in ['subprocess', 'shutil', 'importlib']:
        with pytest.raises(AssertionError, match="found potentially problematic import"):
            validate_code(f"from {module} import something")


def test_infinite_while_loop():
    """Test validation with potentially infinite while loop."""
    infinite_loop = """
def func():
    while True:
        print("This will run forever")
    """
    with pytest.raises(AssertionError, match="found potentially infinite while loop without break or return"):
        validate_code(infinite_loop)


def test_infinite_while_loop_with_break():
    """Test that while True with a break is allowed."""
    with_break = """
def func():
    while True:
        print("This will not run forever")
        if condition:
            break
    """
    # Should not raise an exception
    validate_code(with_break)


def test_infinite_while_loop_with_return():
    """Test that while True with a return is allowed."""
    with_return = """
def func():
    while True:
        print("This will not run forever")
        if condition:
            return
    """
    # Should not raise an exception
    validate_code(with_return)


def test_syntax_error():
    """Test validation with code containing syntax errors."""
    with pytest.raises(SyntaxError):
        validate_code("def func()\n    return 'missing colon'")


def test_indentation_error():
    """Test validation with code containing indentation errors."""
    with pytest.raises(IndentationError):
        validate_code("def func():\nreturn 'no indentation'")


def test_nested_problematic_code():
    """Test validation with nested problematic code."""
    nested_problem = """
def outer():
    def inner():
        import subprocess
        return subprocess.run('ls')
    return inner()
"""
    with pytest.raises(AssertionError, match="found potentially problematic import"):
        validate_code(nested_problem)


def test_complex_valid_code():
    """Test that complex but valid code passes validation."""
    valid_code = """
class Calculator:
    def __init__(self, initial=0):
        self.value = initial

    def add(self, x):
        self.value += x
        return self

    def multiply(self, x):
        self.value *= x
        return self

    def get_result(self):
        return self.value

# Create a calculator and perform operations
calc = Calculator(5)
result = calc.add(3).multiply(2).get_result()
print(f"The result is {result}")
"""
    # Should not raise an exception
    validate_code(valid_code)


def test_conditional_risk():
    """Test validation with potentially risky code inside conditionals that won't execute."""
    conditional_risk = """
def safe_function():
    x = 10
    if False:
        exec("print('This will never execute')")
    return x
"""
    # This should still be caught even though it's in an unreachable branch
    with pytest.raises(AssertionError, match="found potentially problematic function call"):
        validate_code(conditional_risk)


def test_disguised_restricted_import():
    """Test validation with attempts to disguise restricted imports."""
    disguised_import = """
mod_name = 'sub' + 'process'
__import__(mod_name)
"""
    # The __import__ should be caught
    with pytest.raises(AssertionError, match="found potentially problematic function call"):
        validate_code(disguised_import)


def test_commented_code():
    """Test that commented problematic code does not trigger validation errors."""
    commented_code = """
# import subprocess
# eval("print('hello')")
x = 10
"""
    # Should not raise an exception
    validate_code(commented_code)
