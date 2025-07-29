import pytest

import pyine.utils.code.patching as patching


@pytest.fixture(
    params=[
        {  # tiny diffs (single line)
            "original": "The quick brown fox jumps over the lazy dog.\n",
            "modified": "The quick red fox jumps over the lazy cat.\n",
            "unrelated": "A completely different sentence.\n",
        },
        {  # pretty small diffs (one line diff over a handful)
            "original": """\
def hello():
    print("Hello, world!")
    return
""",
            "modified": """\
def hello():
    print("Hello, everybody!")
    return
""",
            "unrelated": """\
class MyClass:
    pass
""",
        },
        {  # serious diffs (many lines over seemingly normal code)
            "original": """\
import math

class MyTestClass:
    '''A sample class for testing.'''
    an_int: int = 1

    def __init__(self, value: int):
        '''Initializes the class.'''
        self.my_variable: int = value

    def calculate(self, multiplier: int) -> int:
        '''A sample method.'''
        local_var = self.my_variable * multiplier
        print(f"Result is {local_var + self.an_int}")
        return local_var + self.an_int

def top_level_function(x: int, y: int) -> float:
    '''A sample top-level function.'''
    result = math.sqrt(x**2 + y**2)
    return result
""",
            "modified": """\
import math

class MyTestClass:
    '''A sample class for testing.'''
    an_int: int = 1

    def __init__(self, value: int):
        '''Initializes the class.'''
        self.my_variable: int = value * 2  # changed logic

    def calculate(self, multiplier: int) -> int:
        '''A sample method.'''
        local_var = self.my_variable * multiplier
        print(f"The result is {local_var + self.an_int}")  # changed message
        return local_var + self.an_int

    def another_method(self) -> None:
        '''A new method.'''
        pass

def top_level_function(x: int, y: int) -> float:
    '''A sample top-level function.'''
    # removed the implementation
    return 0.0
""",
            "unrelated": """\
import os

class UnrelatedClass:
    def unrelated_method(self):
        return os.getcwd()
""",
        },
    ]
)
def text_samples(request) -> dict[str, str]:
    return request.param


def test_patching_cycle_is_successful(
    text_samples: dict[str, str],
):
    patch_text = patching.compute_patch(
        text_samples["original"],
        text_samples["modified"],
    )
    assert isinstance(patch_text, str)
    assert len(patch_text) > 0
    new_text, succeeded = patching.apply_patch(
        text_samples["original"],
        patch_text,
    )
    assert succeeded
    assert new_text == text_samples["modified"]


def test_patching_cycle_fails_on_unrelated_text(
    text_samples: dict[str, str],
):
    patch_text = patching.compute_patch(
        text_samples["original"],
        text_samples["modified"],
    )
    new_text, succeeded = patching.apply_patch(
        text_samples["unrelated"],
        patch_text,
    )
    assert not succeeded
    assert new_text != text_samples["modified"]


@pytest.mark.parametrize(
    "original_code,modified_code,patch_text",
    [
        (
            """\
def greet(name):
    print(f"Hello, {name}!")
""",
            """\
def greet(name):
    # Greet the user
    print(f"Hi, {name}!")
""",
            """\
--- original
+++ modified
@@ -1,2 +1,3 @@
 def greet(name):
-    print(f"Hello, {name}!")
+    # Greet the user
+    print(f"Hi, {name}!")
""",
        ),
        (
            """\
def factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result
""",
            """\
def factorial(n):
    # Calculate factorial recursively
    if n <= 1:
        return 1
    return n * factorial(n-1)
""",
            """\
--- original
+++ modified
@@ -1,5 +1,5 @@
 def factorial(n):
-    result = 1
-    for i in range(1, n+1):
-        result *= i
-    return result
+    # Calculate factorial recursively
+    if n <= 1:
+        return 1
+    return n * factorial(n-1)
""",
        ),
    ],
)
def test_apply_hardcoded_git_style_patch(original_code, modified_code, patch_text):
    new_text, succeeded = patching.apply_patch(
        original_code,
        patch_text,
    )
    assert succeeded
    assert new_text == modified_code
