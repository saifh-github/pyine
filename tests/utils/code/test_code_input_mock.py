import io
import sys

import pytest

from pyine.utils.code.execution import _unsafe_execute_and_trace_code
from pyine.utils.code.input_mock import MockInput, MockInputContext


class TestMockInput:
    inputs_str = "line1\nline2\nline3\n"

    @pytest.fixture
    def mock_input(self) -> MockInput:
        return MockInput(self.inputs_str)

    def test_readline(
        self,
        mock_input: MockInput,
    ) -> None:
        assert mock_input.readline() == "line1\n"
        assert mock_input.readline() == "line2\n"
        assert mock_input.readline() == "line3\n"
        assert mock_input.readline() == ""

    def test_read(
        self,
        mock_input: MockInput,
    ) -> None:
        assert mock_input.read() == self.inputs_str

    def test_readlines(self) -> None:
        mock = MockInput("a\nb\nc")
        assert mock.readlines() == ["a\n", "b\n", "c\n"]

    def test_attribute_passthrough(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = MockInput("test")
        stdin_mock = io.StringIO()
        stdin_mock.fileno = lambda: 42
        monkeypatch.setattr(sys, "stdin", stdin_mock)
        assert mock.fileno() == 42

    def test_mock_input_function(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock = MockInput("test input")
        result = mock.mock_input()
        assert result == "test input"

    def test_buffer_operations(self) -> None:
        mock = MockInput(["alpha", "beta"], encoding="utf-16")
        first_line = mock.buffer.readline()
        assert first_line == "alpha\n".encode("utf-16")
        remaining = mock.buffer.read()
        assert remaining == "beta\n".encode("utf-16")
        mock = MockInput("gamma\ndelta")
        lines = mock.buffer.readlines()
        assert lines == [b"gamma\n", b"delta\n"]
        mock = MockInput("eps\nzet")
        iterated = list(mock.buffer)
        assert iterated == [b"eps\n", b"zet\n"]

    def test_mock_input_raises_eof_when_exhausted(self) -> None:
        mock = MockInput("only one")
        assert mock.mock_input() == "only one"
        with pytest.raises(EOFError):
            mock.mock_input()

    def test_attribute_passthrough_missing_attribute(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = MockInput("data")
        fake_stdin = io.StringIO()
        monkeypatch.setattr(sys, "stdin", fake_stdin)
        with pytest.raises(AttributeError):
            _ = mock.nonexistent_attribute

    def test_attribute_passthrough_calls_original(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class DummyStdIn:
            def __init__(self) -> None:
                self.called = False

            def fileno(self) -> int:
                self.called = True
                return 7

        dummy = DummyStdIn()
        monkeypatch.setattr(sys, "stdin", dummy)
        mock = MockInput("foo")
        assert mock.fileno() == 7
        assert dummy.called


class TestMockInputContext:
    def test_context_manager_basics(self) -> None:
        original_stdin = sys.stdin
        with MockInputContext("mocked input"):
            assert isinstance(sys.stdin, MockInput)
            assert input("Prompt: ") == "mocked input"
        assert sys.stdin is original_stdin

    def test_exception_handling(self) -> None:
        original_stdin = sys.stdin
        with pytest.raises(ValueError), MockInputContext("mock"):
            raise ValueError("Test exception")
        assert sys.stdin is original_stdin  # noqa

    def test_nested_contexts(self) -> None:
        original_stdin = sys.stdin
        with MockInputContext("outer"):
            assert input() == "outer"
            with MockInputContext("inner"):
                assert input() == "inner"
        assert sys.stdin is original_stdin

    def test_with_tracing(self) -> None:
        """Test mocking works with code tracing (for troubleshooting)"""
        code = """\
value = input("Enter: ")
print(f"Got: {value}")
"""
        result = _unsafe_execute_and_trace_code(code, identifier="dummy", inputs="test input")
        assert result.exception is None
        assert result.identifier == "dummy"
        assert "Got: test input" in result.stdout


def test_basic_input_mocking() -> None:
    """Test basic input mocking functionality."""
    code = """\
name = input("What's your name? ")
age = input("What's your age? ")
print(f"Hello, {name}! You are {age} years old.")
"""
    result = _unsafe_execute_and_trace_code(code, inputs="Alice\n30")
    assert result.exception is None
    assert "Hello, Alice! You are 30 years old." in result.stdout


def test_sys_stdin_readline() -> None:
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


def test_sys_stdin_read() -> None:
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


def test_sys_stdin_readlines() -> None:
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


def test_sys_stdin_iteration() -> None:
    """Test that sys.stdin can be iterated with next()."""
    code = """\
import sys
first = next(sys.stdin).rstrip()
second = next(sys.stdin).rstrip()
print(f"{first}|{second}")
"""
    result = _unsafe_execute_and_trace_code(code, inputs="hello\nworld\n")
    assert result.exception is None
    assert "hello|world" in result.stdout


def test_mixed_input_methods() -> None:
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


def test_not_enough_inputs() -> None:
    """Test behavior when not enough inputs are provided."""
    code = """\
name = input("What's your name? ")
age = input("What's your age? ")
country = input("What's your country? ")
"""
    inputs = "David\n42"
    result = _unsafe_execute_and_trace_code(code, inputs=inputs)
    assert result.exception is not None and result.exception.type == "EOFError"


def test_readline_of_closed_file_error_fix() -> None:
    """Test that an error is not raised when reading from a closed file."""
    code = """\
import sys

readline = sys.stdin.buffer.readline
ns = lambda: readline().rstrip()
ni = lambda: int(readline().rstrip())
nm = lambda: map(int, readline().split())
nl = lambda: list(map(int, readline().split()))

def solve():
    s = [x - 97 for x in ns()]
    n = len(s)
    c = 0
    g = {0: 0}
    for x in s:
        c ^= 1 << x
        y = g.get(c, n) + 1
        for i in range(26):
            y = min(y, g.get(c ^ 1 << i, n) + 1)
        g[c] = min(g.get(c, n), y)
    print(y)
    return

solve()
"""
    inputs = "ayxwvutsrqponmlkiihgfedbbz\n"
    result = _unsafe_execute_and_trace_code(code, inputs=inputs)
    assert result.exception is None and result.stdout == "22\n"
