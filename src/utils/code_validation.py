import ast
import warnings


def validate_code(code_string: str, max_size: int = 100_000):
    """Validate a code snippet for proper formatting and compilation."""

    assert isinstance(code_string, str) and code_string, "code must be a non-empty string"
    assert len(code_string) <= max_size, f"code exceeds {max_size} chars (has {len(code_string)})"
    assert "```" not in code_string, "code contains forbidden Markdown elements (backticks)"
    assert "</response>" not in code_string, "code contains forbidden prompt elements (response end tag)"

    # check for imbalanced braces and delimiters
    brackets = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for char in code_string:
        if char in brackets.keys():
            stack.append(char)
        elif char in brackets.values():
            if not stack or brackets[stack.pop()] != char:
                raise AssertionError("code possesses unbalanced delimiters")
    assert len(stack) == 0, "code possesses imbalanced delimiters"

    # check for indentation issues
    lines = code_string.split("\n")
    has_tabs = any(line.startswith("\t") for line in lines)
    has_spaces = any(line.startswith("    ") for line in lines)
    assert not (has_tabs and has_spaces), "code contains mixed indentation"

    # do simple ast parsing (preliminary to full compilation)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed_tree = ast.parse(code_string)
    for node in ast.walk(parsed_tree):
        restricted_functions = ['exec', 'eval', '__import__', 'compile', 'globals', 'locals']
        restricted_imports = ['subprocess', 'shutil', 'importlib']
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in restricted_functions, "found potentially problematic function call"
        elif isinstance(node, ast.Import):
            for name in node.names:
                assert name.name.split(".")[0] not in restricted_imports, "found potentially problematic import"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert node.module.split(".")[0] not in restricted_imports, "found potentially problematic import"
        elif isinstance(node, ast.While):  # check for infinite while loop
            if isinstance(node.test, ast.Constant) and node.test.value:
                assert any([isinstance(inner, (ast.Break, ast.Return)) for inner in ast.walk(node)]), (
                    "found potentially infinite while loop without break or return"
                )

    # finally, compile to check syntax, but don't execute; will throw an exception if anything goes wrong
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compile(code_string, "<string>", "exec")
