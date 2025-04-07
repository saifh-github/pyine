import pytest

import pyine.utils.code_validation

validate_code = pyine.utils.code_validation.validate_code


class TestValidateCode:

    def test_invalid_string(self):
        with pytest.raises(AssertionError):
            validate_code("")
        with pytest.raises(AssertionError):
            validate_code(None)  # noqa
        with pytest.raises(AssertionError):
            validate_code(123)  # noqa

    def test_exceeds_max_size(self):
        large_code = "x = 1\n" * 50_001  # Creates a string > 100_000 chars
        with pytest.raises(AssertionError, match="code exceeds 100000 chars"):
            validate_code(large_code, max_size=100_000)
        with pytest.raises(AssertionError, match="code exceeds 10 chars"):
            validate_code("x = 1\ny = 2", max_size=10)

    def test_contains_markdown_backticks(self):
        with pytest.raises(AssertionError, match="code contains forbidden Markdown elements"):
            validate_code("x = 1\n```python\ny = 2\n```")

    def test_contains_response_tag(self):
        with pytest.raises(AssertionError, match="code contains forbidden prompt elements"):
            validate_code("x = 1\n</response>")

    def test_imbalanced_delims(self):
        with pytest.raises(AssertionError, match="code possesses imbalanced delimiters"):
            validate_code("def func(x:\n    return x + 1")
        with pytest.raises(AssertionError, match="code possesses imbalanced delimiters"):
            validate_code("x = [1, 2, 3\nprint(x)")
        with pytest.raises(AssertionError, match="code possesses imbalanced delimiters"):
            validate_code("data = {'key': 'value'\nprint(data)")
        with pytest.raises(AssertionError, match="code possesses imbalanced delimiters"):
            validate_code("x = (1, 2, 3]")

    def test_mixed_indentation(self):
        mixed_indent_code = "def func():\n    x = 1\n\ty = 2\n    return x + y"
        with pytest.raises(AssertionError, match="code contains mixed indentation"):
            validate_code(mixed_indent_code)

    def test_restricted_function_calls(self):
        for func in ["exec", "eval", "__import__", "compile", "globals", "locals"]:
            with pytest.raises(AssertionError, match="found potentially problematic function call"):
                validate_code(f"result = {func}('print(\"Hello\")')")

    def test_restricted_imports(self):
        for module in ["subprocess", "shutil", "importlib"]:
            with pytest.raises(AssertionError, match="found potentially problematic import"):
                validate_code(f"import {module}")
        for module in ["subprocess", "shutil", "importlib"]:
            with pytest.raises(AssertionError, match="found potentially problematic import"):
                validate_code(f"from {module} import something")

    def test_infinite_while_loop(self):
        infinite_loop = \
"""\
def func():
    while True:
        print("This will run forever")
"""
        with pytest.raises(AssertionError, match="found potentially infinite while loop without break or return"):
            validate_code(infinite_loop)
        with_break = \
"""\
def func():
    while True:
        print("This will not run forever")
        if condition:
            break
"""
        validate_code(with_break)  # should not raise an exception
        with_return = \
"""\
def func():
    while True:
        print("This will not run forever")
        if condition:
            return
"""
        validate_code(with_return)  # should not raise an exception

    def test_syntax_errors(self):
        with pytest.raises(SyntaxError):
            validate_code("def func()\n    return 'missing colon'")
        with pytest.raises(IndentationError):
            validate_code("def func():\nreturn 'no indentation'")

    def test_nested_problematic_code(self):
        nested_problem = \
"""\
def outer():
    def inner():
        import subprocess
        return subprocess.run('ls')
    return inner()
"""
        with pytest.raises(AssertionError, match="found potentially problematic import"):
            validate_code(nested_problem)

    def test_complex_valid_code(self):
        valid_code = \
"""\
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
        validate_code(valid_code)

    def test_disguised_restricted_import(self):
        disguised_import = \
"""\
mod_name = 'sub' + 'process'
__import__(mod_name)
"""
        with pytest.raises(AssertionError, match="found potentially problematic function call"):
            validate_code(disguised_import)

    def test_commented_code(self):
        commented_code = \
"""\
# import subprocess
# eval("print('hello')")
x = 10
"""
        validate_code(commented_code)


class TestNearDuplicateCode:

    @pytest.fixture
    def code_snippets(self):
        return [
            """def add_numbers(a, b):
                # This function adds two numbers
                return a + b
            """,
            """def add_numbers(x, y):
                # This function adds two numbers
                return x + y
            """,
            """def add_numbers(a, b):
                # This function adds two numbers
                # Returns the sum
                return a + b  # Return the result
            """,
            """def add_numbers(a, b):

                return a+b
            """,
            """def multiply_numbers(a, b):
                # This function multiplies two numbers
                return a * b
            """,
            """def fetch_data(url):
                import requests
                response = requests.get(url)
                return response.json()
            """
        ]

    def test_find_near_duplicate_code(self, code_snippets):
        duplicates = pyine.utils.code_validation.find_near_duplicate_code(
            code_snippets,
            threshold=5,
            ignore_comments=True,
            ignore_whitespace=True,
        )
        assert len(duplicates) == 6 and list(range(6)) == list(duplicates.keys())
        assert set([match[0] for match in duplicates[0]]) == {1, 2, 3}

    def test_find_near_duplicate_code_clusters(self, code_snippets):
        clusters = pyine.utils.code_validation.find_near_duplicate_code_clusters(
            code_snippets,
            threshold=0.2,
            ignore_comments=True,
            ignore_whitespace=True,
        )
        assert len(clusters) == 3
        assert set(clusters[0]) == {0, 1, 2, 3}
        assert set(clusters[1]) == {4}
        assert set(clusters[2]) == {5}
        clusters = pyine.utils.code_validation.find_near_duplicate_code_clusters(
            code_snippets,
            threshold=0.0,
            ignore_comments=True,
            ignore_whitespace=True,
        )
        assert len(clusters) == 5
        assert set(clusters[0]) == {0, 2}
        assert set(clusters[1]) == {1}
        assert set(clusters[2]) == {3}
        assert set(clusters[3]) == {4}
        assert set(clusters[4]) == {5}
