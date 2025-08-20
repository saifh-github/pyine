import io
import sys

import pytest

from pyine.utils.code.execution import (
    _unsafe_execute_and_trace_code,  # OK to use here directly w/ hardcoded code strings (will be faster)
)
from pyine.utils.code.execution import (
    EXEC_MODULE_OBJ_NAME,
    EXEC_TRACE_FILE_NAME,
    MockInput,
    MockInputContext,
    TracingCapException,
    execute_and_trace_code,
)


class TestMockInput:

    inputs_str = "line1\nline2\nline3\n"

    @pytest.fixture
    def mock_input(self):
        return MockInput(self.inputs_str)

    def test_readline(self, mock_input):
        assert mock_input.readline() == "line1\n"
        assert mock_input.readline() == "line2\n"
        assert mock_input.readline() == "line3\n"
        assert mock_input.readline() == ""

    def test_read(self, mock_input):
        assert mock_input.read() == self.inputs_str

    def test_readlines(self):
        mock = MockInput("a\nb\nc")
        assert mock.readlines() == ["a\n", "b\n", "c\n"]

    def test_attribute_passthrough(self, monkeypatch):
        mock = MockInput("test")
        stdin_mock = io.StringIO()
        stdin_mock.fileno = lambda: 42
        monkeypatch.setattr(sys, "stdin", stdin_mock)
        assert mock.fileno() == 42

    def test_mock_input_function(self, capsys):
        mock = MockInput("test input")
        result = mock.mock_input()
        assert result == "test input"


class TestMockInputContext:

    def test_context_manager_basics(self):
        original_stdin = sys.stdin
        with MockInputContext("mocked input"):
            assert isinstance(sys.stdin, MockInput)
            assert input("Prompt: ") == "mocked input"
        assert sys.stdin is original_stdin

    def test_exception_handling(self):
        original_stdin = sys.stdin
        with pytest.raises(ValueError):
            with MockInputContext("mock"):
                raise ValueError("Test exception")
        assert sys.stdin is original_stdin  # noqa

    def test_nested_contexts(self):
        original_stdin = sys.stdin
        with MockInputContext("outer"):
            assert input() == "outer"
            with MockInputContext("inner"):
                assert input() == "inner"
        assert sys.stdin is original_stdin

    def test_with_tracing(self):
        """Test mocking works with code tracing (for troubleshooting)"""
        code = """\
value = input("Enter: ")
print(f"Got: {value}")
"""
        result = _unsafe_execute_and_trace_code(code, identifier="dummy", inputs="test input")
        assert result.exception is None
        assert result.identifier == "dummy"
        assert "Got: test input" in result.stdout


def test_basic_input_mocking():
    """Test basic input mocking functionality."""
    code = """\
name = input("What's your name? ")
age = input("What's your age? ")
print(f"Hello, {name}! You are {age} years old.")
"""
    result = _unsafe_execute_and_trace_code(code, inputs="Alice\n30")
    assert result.exception is None
    assert "Hello, Alice! You are 30 years old." in result.stdout


def test_sys_stdin_readline():
    """Test mocking of sys.stdin.readline()."""
    code = """\
import sys
print("Enter your name:")
name = sys.stdin.readline().strip()
print("Enter your country:")
country = sys.stdin.readline().strip()
print(f"{name} is from {country}.")
"""
    result = _unsafe_execute_and_trace_code(code, inputs="Bob\nUSA")
    assert result.exception is None
    assert "Bob is from USA." in result.stdout


def test_sys_stdin_read():
    """Test mocking of sys.stdin.read()."""
    code = """\
import sys
print("Enter multiple lines of text:")
text = sys.stdin.read()
word_count = len(text.split())
print(f"You entered {word_count} words.")
"""
    inputs = "This is a test.\nMultiple lines\nof text."
    result = _unsafe_execute_and_trace_code(code, inputs=inputs)
    assert result.exception is None
    assert "You entered 8 words." in result.stdout


def test_sys_stdin_readlines():
    """Test mocking of sys.stdin.readlines()."""
    code = """\
import sys
print("Enter multiple lines:")
lines = sys.stdin.readlines()
print(f"You entered {len(lines)} lines.")
print(f"First line: {lines[0].strip()}")
"""
    inputs = "First line\nSecond line\nThird line"
    result = _unsafe_execute_and_trace_code(code, inputs=inputs)
    assert result.exception is None
    assert "You entered 3 lines." in result.stdout
    assert "First line: First line" in result.stdout


def test_mixed_input_methods():
    """Test mixing different input methods."""
    code = """\
import sys
name = input("Enter name: ")
print("Enter address:")
address = sys.stdin.readline().strip()
print("Enter additional info:")
info = sys.stdin.read()
print(f"Name: {name}, Address: {address}, Info: {info.strip()}")
"""
    inputs = "Charlie\n123 Main St\nExtra info\nMore details"
    result = _unsafe_execute_and_trace_code(code, inputs=inputs)
    assert result.exception is None
    assert "Name: Charlie, Address: 123 Main St, Info: Extra info\nMore details" in result.stdout


def test_not_enough_inputs():
    """Test behavior when not enough inputs are provided."""
    code = """\
name = input("What's your name? ")
age = input("What's your age? ")
country = input("What's your country? ")
"""
    inputs = "David\n42"
    result = _unsafe_execute_and_trace_code(code, inputs=inputs)
    assert result.exception is not None and result.exception.type == "EOFError"


def test_error_in_executed_code():
    """Test behavior when the executed code contains an error."""
    code = """\
x = 10
y = 0
result = x / y  # Division by zero error
"""
    result = _unsafe_execute_and_trace_code(code, inputs="")
    assert result.exception is not None and result.exception.type == "ZeroDivisionError"


def test_empty_input():
    """Test with empty input strings."""
    code = """\
response = input("Press Enter to continue...")
print("You pressed Enter")
"""
    inputs = ""
    result = _unsafe_execute_and_trace_code(code, inputs=inputs)
    assert result.exception is not None  # should get EOF error with empty inputs

    inputs = "\n"  # now try with a single empty line
    result = _unsafe_execute_and_trace_code(code, inputs=inputs)
    assert result.exception is None and "You pressed Enter" in result.stdout


def test_code_exec_with_sys_exit():
    """Test that code execution with sys.exit() works as expected."""
    code = """\
import sys
print("Hello, world!")
sys.exit(13)
"""
    result = _unsafe_execute_and_trace_code(code)
    assert "Hello, world!" in result.stdout
    assert result.return_value == 13
    assert result.exception is not None
    assert result.exception.type == "SystemExit"
    assert result.exception.message == "13"


def test_code_exec_with_timeout():
    """Test that code execution with a timeout works as expected."""
    code = """\
import time
time.sleep({sleep_time})
"""
    with pytest.raises(TimeoutError):
        execute_and_trace_code(
            code.format(sleep_time=10),
            timeout_seconds=0.1,
            use_safe_execution=True,
            timeout_external_buffer_seconds=0.1,
        )
    with pytest.raises(TimeoutError):
        execute_and_trace_code(
            code.format(sleep_time=10),
            timeout_seconds=0.1,
            use_safe_execution=False,
        )
    results = execute_and_trace_code(
        code.format(sleep_time=0.1),
        timeout_seconds=5.0,
        use_safe_execution=True,
        timeout_external_buffer_seconds=5.0,
    )
    assert results.exception is None


def test_stdout_capture():
    """Test capturing of stdout at individual event level."""
    code = """\
print("Hello, world!")
a = 1 + 2
print("Goodbye, world!")
"""
    result = _unsafe_execute_and_trace_code(code, trace_only_inside_code_string=True)
    assert result.exception is None
    assert "Hello, world!" in result.stdout
    assert "Goodbye, world!" in result.stdout
    valid_trace_steps = [s for s in result.traced_steps if s is not None]
    assert len(valid_trace_steps) == 5  # entrypoint + 3 lines + return
    assert valid_trace_steps[2].stdout == "Hello, world!\n"
    assert valid_trace_steps[3].stdout is None
    assert valid_trace_steps[4].stdout == "Goodbye, world!\n"
    assert result.stdout == "Hello, world!\nGoodbye, world!\n"


def test_tracing_with_blacklist():
    code = """\

import numpy as np

a = 1
b = 2
for i in range(3):
    a += i
    b *= i if i > 0 else 1
c = int(np.sum(np.ones((5, 5)) * 10))        # L8
print(f"Final values: a={a}, b={b}, c={c}")  # L9
"""
    blacklisted_modules = ["numpy", "contextlib", "traceback", "linecache"]
    blacklisted_objects = ["_internal_set_trace", "trace_context", "_get_stack_str"]
    trace_result = _unsafe_execute_and_trace_code(
        code_string=code,
        blacklisted_modules=blacklisted_modules,
        blacklisted_objects=blacklisted_objects,
    )
    assert "Final values: a=4, b=4, c=250" in trace_result.stdout
    assert len(trace_result.traced_steps) > 0
    last_return, last_line = None, None
    for trace_step in trace_result.traced_steps:
        if trace_step is None:
            continue
        assert trace_step.trace_key.object not in blacklisted_objects
        assert not any([trace_step.trace_key.file.startswith(m) for m in blacklisted_modules])
        if trace_step.event_type == "return":
            last_return = trace_step
        if trace_step.event_type == "line":
            last_line = trace_step
    assert last_return.trace_key.line == 10
    assert last_line.trace_key.line == 10


def test_tracing_within_code_only():
    code = """\

import numpy as np

a = 1
b = 2
for i in range(3):
    a += i
    b *= i if i > 0 else 1
c = int(np.sum(np.ones((5, 5)) * 10))        # L8
print(f"Final values: a={a}, b={b}, c={c}")  # L9
"""
    trace_result = _unsafe_execute_and_trace_code(
        code_string=code,
        trace_only_inside_code_string=True,
    )
    assert "Final values: a=4, b=4, c=250" in trace_result.stdout
    assert len(trace_result.traced_steps) > 0
    last_return, last_line = None, None
    for trace_step in trace_result.traced_steps:
        if trace_step is None:
            continue
        assert trace_step.trace_key.file == EXEC_TRACE_FILE_NAME
        if trace_step.event_type == "return":
            last_return = trace_step
        if trace_step.event_type == "line":
            last_line = trace_step
    assert last_return.trace_key.line == 10
    assert last_line.trace_key.line == 10


def test_entrypoint_with_specific_local_vars():
    """Test that the entrypoint can be specified and that specific local variables are reported."""
    code = """\
def some_magic_function(a: str) -> int:
    c = int(a) // 2
    d = 12 ** c
    print(f"{d=}")
    return c
"""
    trace_result = _unsafe_execute_and_trace_code(
        identifier="hello",  # just for logging purposes
        code_string=code,
        inputs="15",
        expected_output="potato",  # not verified internally, just logged
        entrypoint_name="some_magic_function",
        trace_only_inside_code_string=True,
    )
    assert trace_result.identifier == "hello"
    assert len(trace_result.code_blocks) == 1
    assert next(iter(trace_result.code_blocks.values())).name == "some_magic_function"
    assert trace_result.expected_output == "potato"
    assert trace_result.max_var_repr_length is None
    assert trace_result.max_events_per_line is None
    assert f"d={12 ** (15 // 2)}" in trace_result.stdout
    assert trace_result.return_value == 7
    assert trace_result.exception is None
    assert len(trace_result.metadata) > 0
    expected_tags = ["exec:has_entrypoint", "return:has_value", "return:has_stdout"]
    assert len(trace_result.tags) > 0 and all([t in trace_result.tags for t in expected_tags])
    # assume very specific step mapping below, starting from defs, and then the entrypoint call
    valid_steps = [s for s in trace_result.traced_steps if s is not None]  # drop out-of-scope steps
    assert len(valid_steps) == 3 + 6  # should have 3 for defines, 6 for entrypoint call->return
    assert valid_steps[0].event_type == "call"
    assert valid_steps[0].trace_key.object == EXEC_MODULE_OBJ_NAME
    assert valid_steps[0].trace_key.file == EXEC_TRACE_FILE_NAME
    assert valid_steps[0].trace_key.line == 0  # exec init always starts at line zero
    assert not valid_steps[0].arguments  # no arguments for init call
    assert not valid_steps[0].local_variables and not valid_steps[0].global_variables
    assert valid_steps[1].event_type == "line" and valid_steps[1].trace_key.line == 1  # function def
    assert not valid_steps[1].local_variables and not valid_steps[1].global_variables  # def still not done
    assert valid_steps[2].event_type == "return"  # now def should exist
    assert "some_magic_function" in valid_steps[2].global_variables
    # next steps of interest should be the entrypoint call itself
    assert valid_steps[3].event_type == "call"
    assert valid_steps[3].trace_key.object == "some_magic_function"
    assert valid_steps[3].trace_key.file == EXEC_TRACE_FILE_NAME
    assert valid_steps[3].trace_key.line == 1  # should now be going to the function def itself
    assert valid_steps[3].arguments == {"a": "'15'"}  # this is what we passed to the func as input
    assert valid_steps[3].local_variables == {"a": "'15'"}  # immediately also counts as a local variable
    assert "some_magic_function" in valid_steps[3].global_variables  # that global should still exist
    assert valid_steps[4].event_type == "line" and valid_steps[4].trace_key.line == 2  # move up
    assert valid_steps[4].local_variables == {"a": "'15'"}  # there should not be any new locals yet
    assert valid_steps[5].event_type == "line" and valid_steps[5].trace_key.line == 3  # move up again
    assert valid_steps[5].local_variables == {"a": "'15'", "c": "7"}
    # rest should be OK at this point


def test_trace_event_cap():
    """Test that trace caps work correctly."""
    code = """\
i = 0
while i < 1000:
    i += 1
print(f"Final i={i}")
"""
    result_no_caps = _unsafe_execute_and_trace_code(
        code_string=code,
        trace_only_inside_code_string=True,
    )
    assert len([s for s in result_no_caps.traced_steps if s is not None]) > 1000
    assert result_no_caps.stdout == "Final i=1000\n"
    with pytest.raises(TracingCapException):
        _ = _unsafe_execute_and_trace_code(
            code_string=code,
            trace_only_inside_code_string=True,
            max_events_per_line=10,
        )
    result_with_caps = _unsafe_execute_and_trace_code(
        code_string=code,
        trace_only_inside_code_string=True,
        max_events_per_line=10_000,
    )
    assert result_with_caps.stdout == "Final i=1000\n"


def test_trace_max_var_len_cap():
    code = """\
i = 1000
something = ["potato"] * i
print(f"{len(something)=}")
"""
    result_no_caps = _unsafe_execute_and_trace_code(
        code_string=code,
        trace_only_inside_code_string=True,
    )
    assert result_no_caps.stdout == "len(something)=1000\n"
    with pytest.raises(TracingCapException):
        _ = _unsafe_execute_and_trace_code(
            code_string=code,
            trace_only_inside_code_string=True,
            max_var_repr_length=100,
        )
    result_with_caps = _unsafe_execute_and_trace_code(
        code_string=code,
        trace_only_inside_code_string=True,
        max_var_repr_length=10_000,
    )
    assert result_with_caps.stdout == "len(something)=1000\n"


def test_safe_vs_unsafe_tracing():
    """Test that safe tracing provides the same results as unsafe tracing."""
    code = """\
def func(b: str) -> int:
    print("Hello, world!")
    a = 1 + int(b)
    print("Goodbye, world!")
    return a
"""
    safe_trace_result = execute_and_trace_code(
        code_string=code,
        inputs="2",
        entrypoint_name="func",
        trace_only_inside_code_string=True,
    )
    unsafe_trace_result = execute_and_trace_code(
        code_string=code,
        inputs="2",
        entrypoint_name="func",
        trace_only_inside_code_string=True,
    )
    assert safe_trace_result.stdout == unsafe_trace_result.stdout
    valid_safe_steps = [s for s in safe_trace_result.traced_steps if s is not None]
    valid_unsafe_steps = [s for s in unsafe_trace_result.traced_steps if s is not None]
    assert len(valid_safe_steps) == len(valid_unsafe_steps)
    for safe_step, unsafe_step in zip(valid_safe_steps, valid_unsafe_steps):
        assert safe_step.trace_key == unsafe_step.trace_key
        assert safe_step.event_type == unsafe_step.event_type
        assert safe_step.global_variables == unsafe_step.global_variables
        assert safe_step.local_variables == unsafe_step.local_variables
