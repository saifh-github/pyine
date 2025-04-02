import sys
import io
import contextlib
import dataclasses
import pprint
import types
import typing

import src.utils.reprod
from src.utils.portable_repr import get_portable_representation as portable_repr



@dataclasses.dataclass(frozen=True)
class TraceKey:
    """Dataclass for storing and exporting execution trace keys."""
    file: str
    """The file name containing the code that was executed."""
    function: str
    """The name of the function that was executed."""
    line: int
    """The number of the code line that was executed."""


@dataclasses.dataclass(frozen=True)
class TraceEvent:
    """Dataclass for storing and exporting execution trace events."""
    event_type: str
    """The type of event that occurred (e.g., "call", "return", "exception")."""
    stack_trace: typing.List[TraceKey]
    """A list of TraceKey instances representing the call stack at the time of the event."""
    variables: typing.Dict[str, typing.Any]
    """A dictionary containing the variables at the time of the event."""
    internal_variables: typing.Dict[str, typing.Any]
    """A dictionary containing internal variables at the time of the event."""
    arguments: typing.Optional[typing.Dict[str, typing.Any]]
    """A dictionary containing the arguments passed to the function at the time of the event."""
    return_value: typing.Optional[typing.Any]
    """The return value of the function at the time of the event."""
    exception: typing.Optional[typing.Dict[str, typing.Any]]
    """A dictionary containing information about the exception that occurred, if any."""
    trace_step: int
    """The trace step number at the time of the event; should be unique for each event."""


@dataclasses.dataclass(frozen=True)
class TraceResult:
    """Dataclass for storing and exporting execution trace results."""
    code_string: str
    """The original code string that was executed."""
    inputs: str
    """The inputs that were available to the code during execution."""
    traced_states: typing.Dict[TraceKey, typing.List[typing.Optional[TraceEvent]]]
    """A dictionary containing the execution trace states at each line of code.
    
    In this dictionary, keys are `TraceKey` instances (combining file, function, and line info)
    and values are lists of `TraceEvent` objects representing the execution state at each
    line of code (prior to execution). If a line has more than `max_events_per_line` events,
    subsequent events objects will be replaced with `None`.
    """
    tracing_steps: int
    """The total number of tracing steps taken during execution."""
    return_value: typing.Optional[typing.Any]
    """The return value of the executed code, if any."""
    exception: typing.Optional[Exception]
    """The exception that occurred during execution, if any."""
    stdout: str
    """The captured stdout output during execution."""
    stderr: str
    """The captured stderr output during execution."""
    env_metadata: typing.Dict[str, typing.Any]
    """A dictionary containing metadata about the execution environment."""


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
            result = "".join(f"{line}\n" for line in input_iter)
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


def execute_and_trace_code(
    code_string: str,
    blacklisted_functions: typing.Optional[typing.List[str]] = None,
    max_events_per_line: typing.Optional[int] = None,
) -> TraceResult:
    """Execute Python code and trace the state of the execution at each line.

    Args:
        code_string: A string containing the Python code to execute and trace.
        blacklisted_functions: A list of function names to exclude from tracing.
        max_events_per_line: The maximum number of events to record per line.

    Returns:
        A `TraceResult` instance containing the execution results.
    """
    traced_states: typing.Dict[TraceKey, typing.List[typing.Optional[TraceEvent]]] = {}
    trace_step = 0  # will be incremented each time the callback function is called

    def _trace_callback(
        frame: types.FrameType,  # noqa
        event: str,
        arg: typing.Any,
    ) -> typing.Optional[typing.Callable]:
        """Callback function for sys.settrace that records execution state at each line."""
        nonlocal trace_step

        trace_key = TraceKey(
            file=frame.f_code.co_filename,
            function=frame.f_code.co_name,
            line=frame.f_lineno
        )
        if event == "call" and blacklisted_functions and trace_key.function in blacklisted_functions:
            return_trace_callback = None  # do not trace that function, skip over it
        else:
            return_trace_callback = _trace_callback  # any non-blacklisted function will be traced
        trace_step += 1  # will reflect the total number of calls to this callback, no matter what
        if trace_key not in traced_states:
            traced_states[trace_key] = []
        if max_events_per_line and len(traced_states[trace_key]) >= max_events_per_line:
            # if we have reached the trace event limit for this line, append `None` instead of event
            trace_event = None
        else:
            # gather the actual event data and create the corresponding object
            stack_trace = []
            current_frame = frame
            while current_frame:
                stack_trace.append(TraceKey(
                    file=current_frame.f_code.co_filename,
                    function=current_frame.f_code.co_name,
                    line=current_frame.f_lineno
                ))
                current_frame = current_frame.f_back
            regular_vars = {
                name: portable_repr(value) for name, value in frame.f_locals.items()
                if not name.startswith("__") and name != "_trace_callback"
            }
            internal_vars = {
                name: portable_repr(value) for name, value in frame.f_locals.items()
                if name.startswith("__") and name != "__builtins__"
            }
            arguments, return_value, exception = None, None, None
            if event == "call":
                arguments = {
                    name: portable_repr(value) for name, value in frame.f_locals.items()
                    if not name.startswith("__")
                }
            elif event == "return":
                return_value = portable_repr(arg)
            elif event == "exception":
                exc_type, exc_value, exc_traceback = arg
                exception = {
                    "type": exc_type.__name__,
                    "message": portable_repr(exc_value),
                }
            trace_event = TraceEvent(
                event_type=event,
                stack_trace=stack_trace,
                variables=regular_vars,
                internal_variables=internal_vars,
                arguments=arguments,
                return_value=return_value,
                exception=exception,
                trace_step=trace_step,
            )

        traced_states[trace_key].append(trace_event)
        return return_trace_callback

    compiled_code = compile(code_string, "<string>", "exec")
    stdout_capture = io.StringIO()  # capture stdout to avoid polluting the output
    namespace = {}  # create a namespace for execution (to keep potential global defs)
    try:
        with contextlib.redirect_stdout(stdout_capture):
            sys.settrace(_trace_callback)
            exec(compiled_code, namespace)
    finally:
        sys.settrace(None)
    trace_result = TraceResult(
        code_string=code_string,
        inputs="", # @@@@@ TODO
        traced_states=traced_states,
        tracing_steps=trace_step,
        return_value=None, # @@@@@ TODO
        exception=None, # @@@@@ TODO
        stdout=stdout_capture.getvalue(),
        stderr=None, # @@@@@ TODO
        env_metadata=src.utils.reprod.get_reprod_metadata(),
    )
    return trace_result


def format_traced_code_execution(
    code_string: str,
    traced_states: typing.Dict[TraceKey, typing.List[TraceEvent]]
) -> str:
    """Format traced code execution results into a readable report.

    Args:
        code_string: The original code string that was executed.
        traced_states: Dictionary of TraceKey instances to lists of TraceEvent objects.

    Returns:
        A formatted string showing the code execution with variable states.
    """
    code_lines = code_string.splitlines()
    result = []
    result.append("Code Execution Trace:")
    result.append("=====================")
    result.append("")

    for line_number, line in enumerate(code_lines, 1):
        result.append(f"Line {line_number}: {line}")
        relevant_events = [
            (key, events)
            for key, events in traced_states.items()
            if key.line == line_number
        ]
        for trace_key, events in relevant_events:
            for event_number, trace_event in enumerate(events, 1):
                if event_number > 1:
                    result.append(f"  Visit #{event_number}:")

                result.append(f"  Event Type: {trace_event.event_type}")
                result.append(f"  Trace Step: {trace_event.trace_step}")

                if trace_event.arguments:
                    result.append("  Arguments:")
                    for arg_name, arg_value in trace_event.arguments.items():
                        try:
                            result.append(f"    {arg_name}: {pprint.pformat(arg_value)}")
                        except Exception:
                            result.append(f"    {arg_name}: <unable to display value>")

                if trace_event.variables:
                    result.append("  Variables:")
                    for var_name, var_value in trace_event.variables.items():
                        try:
                            result.append(f"    {var_name}: {pprint.pformat(var_value)}")
                        except Exception:
                            result.append(f"    {var_name}: <unable to display value>")

                if trace_event.internal_variables:
                    result.append("  Internal Variables:")
                    for intern_name, intern_value in trace_event.internal_variables.items():
                        try:
                            result.append(f"    {intern_name}: {pprint.pformat(intern_value)}")
                        except Exception:
                            result.append(f"    {intern_name}: <unable to display value>")

                if trace_event.return_value is not None:
                    result.append("  Return Value:")
                    try:
                        result.append(f"    {pprint.pformat(trace_event.return_value)}")
                    except Exception:
                        result.append("    <unable to display return value>")

                if trace_event.exception:
                    result.append("  Exception:")
                    for exc_key, exc_value in trace_event.exception.items():
                        result.append(f"    {exc_key}: {exc_value}")

                result.append("")  # add spacing between events

    return "\n".join(result)


if __name__ == "__main__":
    # example usage of tracing + printing (note: gets spammy for large functions!):
    _sample_code = \
"""\

import numpy as np

def potato():
    print("potato")
    
class Something:
    def __init__(self):
        self.potato = "potato"
    def ok():
        todo

some_potato = Something()

something_else = np.array
hllo = np
sdkjnsdkg = np.random.random
sjdgknsd = Something.ok

a = 1
b = 2
for i in range(3):
    a += i
    b *= i if i > 0 else 1
print(f"Final values: a={a}, b={b}")
"""
    _trace_result = execute_and_trace_code(_sample_code)
    _report = format_traced_code_execution(_sample_code, _trace_result.traced_states)
    print(_report)
