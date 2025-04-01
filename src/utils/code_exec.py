import sys
import io
import contextlib
import typing


def execute_code_with_mocked_input(
    code_string: str,
    inputs: str
) -> typing.Tuple[str, typing.Optional[Exception]]:  # type: ignore
    """Execute Python code with mocked input.

    Replaces calls to input() or sys.stdin.readline() with lines from the
    provided inputs string.

    Args:
        code_string: a string containing arbitrary Python code to execute.
        inputs: a string containing individual lines to be used as input values
            (one line per input call).

    Returns:
        A tuple containing:
            - The captured stdout output as a string
            - Any exception that was raised during execution, or None if execution
              was successful
    """
    # split inputs into lines and create an iterator
    input_lines = inputs.splitlines()
    input_iter = iter(input_lines)

    # mock the input function
    def mock_input(prompt: str = "") -> str:
        try:
            next_input = next(input_iter)
            print(f"providing next mocked input: {next_input}")
            return next_input
        except StopIteration:
            raise EOFError("Not enough input lines provided")

    # create a mock stdin that returns our predefined inputs
    class MockStdin:
        def readline(self) -> str:
            try:
                next_input = next(input_iter)
                print(f"providing next mocked input: {next_input}")
                return f"{next_input}\n"  # add newline as readline would return
            except StopIteration:
                return ""  # return empty string when no more inputs

        def read(self) -> str:
            result = ''.join(f"{line}\n" for line in input_iter)
            print(f"providing mocked read: {result.strip()}")
            return result

        def readlines(self) -> typing.List[str]:
            result = [f"{line}\n" for line in input_iter]
            print(f"providing mocked readlines: {result}")
            return result

        def __getattr__(self, name: str) -> typing.Any:
            # pass through any other attributes to the real stdin
            return getattr(sys.__stdin__, name)

    # store the original stdin, stdout, and input function
    original_stdin = sys.stdin
    original_input = __builtins__["input"]  # type: ignore

    # replace stdin and input with our mocks
    sys.stdin = MockStdin()  # type: ignore
    __builtins__["input"] = mock_input  # type: ignore

    # create StringIO objects to capture stdout and stderr
    stdout_capture = io.StringIO()
    result_exception = None
    try:
        # execute the code with captured stdout and our mocked input
        with contextlib.redirect_stdout(stdout_capture):
            # execute the code
            exec(code_string, {})
    except Exception as e:
        # store any exception that occurred
        result_exception = e
    finally:
        # restore the original stdin and input function
        sys.stdin = original_stdin
        __builtins__["input"] = original_input  # type: ignore

    # return the captured output and any exception
    return stdout_capture.getvalue(), result_exception
