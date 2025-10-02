import contextlib
import functools
import sys
import types
import typing

_orig_stdin = sys.stdin


class MockInput:
    """Mock class for sys.stdin.readline() and input() to read from a provided list of inputs."""

    class _BufferView:
        """A minimal bytes-oriented buffer view over MockInput that mirrors sys.stdin.buffer."""

        def __init__(self, parent: "MockInput", encoding: str = "utf-8") -> None:
            self._parent = parent
            self.encoding = encoding

        def readline(self) -> bytes:
            """Read a single line as bytes, including the trailing newline if available."""
            try:
                next_input = next(self._parent._input_iter)
                return f"{next_input}\n".encode(self.encoding)
            except StopIteration:
                return b""

        def read(self) -> bytes:
            """Read all remaining input as bytes."""
            return b"".join(f"{line}\n".encode(self.encoding) for line in self._parent._input_iter)

        def readlines(self) -> list[bytes]:
            """Read all remaining input lines as a list of bytes."""
            return [f"{line}\n".encode(self.encoding) for line in self._parent._input_iter]

        def __iter__(self) -> "MockInput._BufferView":
            return self

        def __next__(self) -> bytes:
            line = self.readline()
            if line == b"":
                raise StopIteration
            return line

    # noinspection PyUnreachableCode
    def __init__(self, inputs: str = "", encoding: str = "utf-8") -> None:
        """Initialize the MockInput instance with an input string to be read from.

        If the input string contains newlines, each line will be read separately. If it does not
        possess a final newline, one will be added automatically.
        """
        self._orig_inputs = inputs
        self._encoding = encoding
        if not isinstance(inputs, str):
            inputs = "\n".join(inputs) if isinstance(inputs, list) else str(inputs)
        if inputs and not inputs.endswith("\n"):
            inputs += "\n"
        self._input_iter = iter(inputs.splitlines())
        # provide a bytes-oriented buffer view compatible with sys.stdin.buffer
        self.buffer = MockInput._BufferView(self, encoding=self._encoding)

    def readline(self) -> str:
        """Read a line from the iterator of inputs as text."""
        try:
            next_input = next(self._input_iter)
            return f"{next_input}\n"  # add newline as readline would return
        except StopIteration:
            return ""  # return empty string when no more inputs

    def read(self) -> str:
        """Read all remaining inputs into a single string."""
        return "".join(f"{line}\n" for line in self._input_iter)

    def readlines(self) -> list[str]:
        """Read all remaining inputs into a list of strings."""
        return [f"{line}\n" for line in self._input_iter]

    def __getattr__(self, name: str) -> typing.Any:
        """Forward attribute access to sys.stdin for any attributes not found in MockInput.

        Note: we intentionally provide our own 'buffer' attribute; therefore it will not be
        forwarded to the original stdin.
        """
        try:
            stdin_attr = getattr(_orig_stdin, name)
            if callable(stdin_attr):

                def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
                    method = getattr(sys.stdin, name)
                    return method(*args, **kwargs)

                return wrapper
            return stdin_attr
        except AttributeError as e:
            raise AttributeError(f"'{self.__class__.__name__}' nor sys.stdin has attrib '{name}'") from e

    def mock_input(self, _: str = "") -> str:
        """Read the next input from the iterator of inputs."""
        try:
            return next(self._input_iter)
        except StopIteration as e:
            raise EOFError("not enough input lines provided") from e

    def __iter__(self) -> "MockInput":
        """Returns the iterator over all inputs."""
        return self

    def __next__(self) -> str:
        """Returns the next input line."""
        line = self.readline()
        if line == "":
            raise StopIteration
        return line


class MockInputContext(contextlib.AbstractContextManager):
    """Context manager for replacing sys.stdin.readline() and input() with MockInput."""

    def __init__(self, inputs: str = "") -> None:
        """Initialize the MockInput instance with an input string to be read from.

        If the input string contains newlines, each line will be read separately. If it does not
        possess a final newline, one will be added automatically.
        """
        self.mocker = MockInput(inputs)

    def __enter__(self, inputs: str = "") -> None:
        """Replaces sys.stdin.readline() and input() with MockInput."""
        self.original_stdin = sys.stdin
        if isinstance(__builtins__, dict):
            self.original_input = __builtins__["input"]  # type: ignore
            __builtins__["input"] = functools.partial(MockInput.mock_input, self.mocker)  # type: ignore
        else:
            self.original_input = __builtins__.input  # type: ignore
            __builtins__.input = functools.partial(MockInput.mock_input, self.mocker)  # type: ignore
        sys.stdin = self.mocker  # type: ignore

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Restores sys.stdin.readline() and input() to their original values."""
        sys.stdin = self.original_stdin
        if isinstance(__builtins__, dict):
            __builtins__["input"] = self.original_input  # type: ignore
        else:
            __builtins__.input = self.original_input  # type: ignore
