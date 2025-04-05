import sys
import io
import contextlib
import dataclasses
import functools
import os
import pprint
import site
import types
import typing

import src.utils.reprod
import src.utils.code_blocks
from src.utils.portable_repr import get_portable_representation as portable_repr

_orig_stdin = sys.stdin


@dataclasses.dataclass(frozen=True)
class TraceKey:
    """Dataclass for storing and exporting execution trace keys."""
    file: str
    """The file name containing the code that was executed."""
    object: str
    """The name of the code object that was executed."""
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
    """A dictionary containing the arguments passed to the code object at the time of the event."""
    return_value: typing.Optional[typing.Any]
    """The return value of the code object at the time of the event."""
    exception: typing.Optional[typing.Dict[str, typing.Any]]
    """A dictionary containing information about the exception that occurred, if any."""
    trace_step_idx: int
    """The trace step index at the time of the event; should be unique for each event."""
    trace_key: TraceKey
    """The trace key associated with this event, for convenience."""


@dataclasses.dataclass(frozen=True)
class TraceResult:
    """Dataclass for storing and exporting execution trace results."""
    code_string: str
    """The original code string that was executed."""
    code_blocks: typing.Dict[int, src.utils.code_blocks.CodeBlock]
    """A dictionary containing the logic blocks of the executed code, indexed by line number."""
    inputs: str
    """The inputs that were available to the code during execution."""
    max_events_per_line: typing.Optional[int]
    """The maximum number of events to record per line (if needed)."""
    traced_steps: typing.List[typing.Optional[TraceEvent]]
    """A list of all traced steps, in order of execution.
    
    The states correspond to variables at each line of code prior to execution. If a line has more
    than `max_events_per_line` events, additional events are substituted by `None` in this list. To
    determine which line an event occurred on, see the `TraceKey` attribute of each event, or the
    `traced_steps_map` dictionary below.
    """
    traced_steps_map: typing.Dict[TraceKey, typing.List[int]]
    """A dictionary containing the indices of each traced step for each line of executed code.
    
    In this dictionary, keys are `TraceKey` instances (combining file, object, and line info)
    and values are lists of indices pointing to `TraceEvent` objects in the above `traced_steps`
    list. If a line has more than `max_events_per_line` events, its corresponding indices list will
    still contain all trace step indices, but some events in `traced_steps` will be substituted with
    `None`.
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
    metadata: typing.Dict[str, typing.Any]
    """A dictionary containing metadata about the execution environment & settings."""


@contextlib.contextmanager
def trace_context(
    trace_callback: typing.Callable,
) -> typing.Iterator[None]:
    """
    Context manager for sys.settrace.

    Args:
        trace_callback: The trace function to set during the context

    Yields:
        None
    """
    old_trace = sys.gettrace()
    sys.settrace(trace_callback)
    try:
        yield
    finally:
        sys.settrace(old_trace)


class MockInput:
    """Mock class for sys.stdin.readline() and input() to read from a provided list of inputs."""

    def __init__(self, inputs: str = ""):
        """Initialize the MockInput instance with an input string to be read from.

        If the input string contains newlines, each line will be read separately. If it does not
        possess a final newline, one will be added automatically.
        """
        self._orig_inputs = inputs
        if inputs and not inputs.endswith("\n"):
            inputs += "\n"
        self._input_iter = iter(inputs.splitlines())

    def readline(self) -> str:
        """Read a line from the iterator of inputs."""
        try:
            next_input = next(self._input_iter)
            return f"{next_input}\n"  # add newline as readline would return
        except StopIteration:
            return ""  # return empty string when no more inputs

    def read(self) -> str:
        """Read all remaining inputs into a single string."""
        result = "".join(f"{line}\n" for line in self._input_iter)
        return result

    def readlines(self) -> typing.List[str]:
        """Read all remaining inputs into a list of strings."""
        result = [f"{line}\n" for line in self._input_iter]
        return result

    def __getattr__(self, name: str) -> typing.Any:
        """Forward attribute access to sys.stdin for any attributes not found in MockInput."""
        try:
            stdin_attr = getattr(_orig_stdin, name)
            if callable(stdin_attr):
                def wrapper(*args, **kwargs):
                    method = getattr(sys.stdin, name)
                    return method(*args, **kwargs)
                return wrapper
            else:
                return stdin_attr
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' nor sys.stdin has attrib '{name}'")

    def mock_input(self, prompt: str = "") -> str:
        """Read the next input from the iterator of inputs."""
        try:
            next_input = next(self._input_iter)
            return next_input
        except StopIteration:
            raise EOFError("Not enough input lines provided")


class MockInputContext(contextlib.AbstractContextManager):
    """Context manager for replacing sys.stdin.readline() and input() with MockInput."""

    def __init__(self, inputs: str = ""):
        """Initialize the MockInput instance with an input string to be read from.

        If the input string contains newlines, each line will be read separately. If it does not
        possess a final newline, one will be added automatically.
        """
        self.mocker = MockInput(inputs)

    def __enter__(self, inputs: str = ""):
        """Replaces sys.stdin.readline() and input() with MockInput."""
        self.original_stdin = sys.stdin
        if isinstance(__builtins__, dict):
            self.original_input = __builtins__["input"]  # type: ignore
            __builtins__["input"] = functools.partial(MockInput.mock_input, self.mocker)  # type: ignore
        else:
            self.original_input = __builtins__.input  # type: ignore
            __builtins__.input = functools.partial(MockInput.mock_input, self.mocker)  # type: ignore
        sys.stdin = self.mocker  # type: ignore


    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restores sys.stdin.readline() and input() to their original values."""
        sys.stdin = self.original_stdin
        if isinstance(__builtins__, dict):
            __builtins__["input"] = self.original_input  # type: ignore
        else:
            __builtins__.input = self.original_input  # type: ignore


def _get_clean_filename(filename: str) -> str:
    if filename == "<string>":
        return filename  # nothing to do
    site_pkgs = site.getsitepackages() + [site.getusersitepackages()]
    for site_dir in site_pkgs:
        if filename.startswith(site_dir):
            return os.path.relpath(filename, site_dir)
    stdlib_dir = os.path.dirname(os.__file__)
    if filename.startswith(stdlib_dir):
        return os.path.relpath(filename, stdlib_dir)
    project_root = os.getcwd()
    if filename.startswith(project_root):
        return os.path.relpath(filename, project_root)
    return filename


def execute_and_trace_code(
    code_string: str,
    inputs: str = "",
    blacklisted_modules: typing.Optional[typing.Iterable[str]] = None,
    blacklisted_objects: typing.Optional[typing.Iterable[str]] = None,
    trace_only_inside_code_string: bool = False,
    max_events_per_line: typing.Optional[int] = None,
    seed: typing.Optional[int] = 42,
) -> TraceResult:
    """Execute Python code and trace the state of the execution at each line.

    Also replaces calls to input() or sys.stdin.readline() with lines from the provided inputs
    string, and captures stdout and stderr during execution.

    Args:
        code_string: a string containing arbitrary Python code to execute and trace.
        inputs: a string containing individual lines to be used as input values
            (one line per input call).
        blacklisted_modules: A list of module names to exclude from tracing.
        blacklisted_objects: A list of object names to exclude from tracing.
        trace_only_inside_code_string: If True, only trace code inside the provided code_string.
        max_events_per_line: The maximum number of events to record per line.
        seed: The seed to use for random number generation. Defaults to 42.

    Returns:
        A `TraceResult` instance containing the execution results.
    """
    try:
        code_blocks = src.utils.code_blocks.identify_code_blocks(code_string)
        compiled_code = compile(code_string, "<string>", "exec")
    except Exception as e:
        raise Exception(f"error while analyzing and compiling code: {e}")
    traced_steps: typing.List[typing.Optional[TraceEvent]] = []
    traced_steps_map: typing.Dict[TraceKey, typing.List[int]] = {}
    last_trace_step_idx = 0  # will be incremented each time the callback is called

    def _trace_callback(
        frame: types.FrameType,  # noqa
        event: str,
        arg: typing.Any,
    ) -> typing.Optional[typing.Callable]:
        """Callback function for sys.settrace that records execution state at each line."""
        nonlocal last_trace_step_idx

        trace_key = TraceKey(
            file=_get_clean_filename(frame.f_code.co_filename),
            object=frame.f_code.co_name,
            line=frame.f_lineno
        )
        return_trace_callback = _trace_callback  # any non-blacklisted object will be traced
        is_blacklisted = (
            (blacklisted_objects and trace_key.object in blacklisted_objects)
            or (blacklisted_modules and trace_key.file.startswith(tuple(blacklisted_modules)))
        )
        is_inside_code_string = trace_key.file == "<string>"
        must_skip = is_blacklisted or (not is_inside_code_string and trace_only_inside_code_string)
        if event == "call" and must_skip:
            return_trace_callback = None  # do not trace that function, skip over it
        if trace_key not in traced_steps_map:
            traced_steps_map[trace_key] = []
        max_event_capped = max_events_per_line and len(traced_steps_map[trace_key]) >= max_events_per_line
        if max_event_capped or must_skip:
            # if we have reached the trace limit or a blacklisted event, append `None` instead of event
            trace_event = None
        else:
            # gather the actual event data and create the corresponding object
            stack_trace = []
            current_frame = frame
            while current_frame:
                stack_trace.append(TraceKey(
                    file=_get_clean_filename(current_frame.f_code.co_filename),
                    object=current_frame.f_code.co_name,
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
                trace_step_idx=last_trace_step_idx,
                trace_key=trace_key,
            )
        traced_steps.append(trace_event)
        traced_steps_map[trace_key].append(last_trace_step_idx)
        last_trace_step_idx += 1  # will reflect the total number of calls to this callback, no matter what
        return return_trace_callback

    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()  # to avoid polluting the output
    caught_exception = None
    global_scope, local_scope = {}, {}
    src.utils.reprod.set_seed(seed)
    try:
        with MockInputContext(inputs):
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                with trace_context(_trace_callback):
                    exec(compiled_code, global_scope, local_scope)
    except Exception as e:
        caught_exception = e  # store any exception that occurred
    reprod_metadata = src.utils.reprod.get_reprod_metadata()
    reprod_metadata["initial_seed"] = seed
    reprod_metadata["max_events_per_line"] = max_events_per_line
    reprod_metadata["blacklisted_modules"] = list(blacklisted_modules or [])
    reprod_metadata["blacklisted_objects"] = list(blacklisted_objects or [])
    trace_result = TraceResult(
        code_string=code_string,
        code_blocks=code_blocks,
        inputs=inputs,
        max_events_per_line=max_events_per_line,
        traced_steps=traced_steps,
        traced_steps_map=traced_steps_map,
        tracing_steps=last_trace_step_idx,
        return_value=None, # @@@@@ TODO
        exception=caught_exception,
        stdout=stdout_capture.getvalue(),
        stderr=stderr_capture.getvalue(),
        metadata=reprod_metadata,
    )
    return trace_result


def format_traced_code_execution(
    trace_result: TraceResult,
) -> str:
    """Format traced code execution results into a readable report.

    Args:
        trace_result: A TraceResult instance containing the execution results.

    Returns:
        A formatted string showing the code execution with variable states.
    """
    code_lines = trace_result.code_string.splitlines()
    result = []
    result.append("Code Execution Trace:")
    result.append("=====================")
    result.append("")
    for line_number, line in enumerate(code_lines, 1):
        result.append(f"Line {line_number}: {line}")
        relevant_events = [
            (key, trace_result.traced_steps_map[key])
            for key in trace_result.traced_steps_map
            if key.line == line_number
        ]
        for trace_key, event_indices in relevant_events:
            result.append(f"  {trace_key}")
            for event_number, trace_step_idx in enumerate(event_indices, 1):
                trace_event = trace_result.traced_steps[trace_step_idx]
                if trace_event is None:
                    result.append("  [Event skipped due to max_events_per_line limit]")
                    continue
                result.append(f"    Visit #{event_number} (step #{trace_event.trace_step_idx}):")
                result.append(f"    Event Type: {trace_event.event_type}")
                result.append(f"    Trace Step: {trace_event.trace_step_idx}")
                if trace_event.arguments:
                    result.append("    Arguments:")
                    for arg_name, arg_value in trace_event.arguments.items():
                        try:
                            result.append(f"      {arg_name}: {pprint.pformat(arg_value)}")
                        except Exception:
                            result.append(f"      {arg_name}: <unable to display value>")
                if trace_event.variables:
                    result.append("    Variables:")
                    for var_name, var_value in trace_event.variables.items():
                        try:
                            result.append(f"      {var_name}: {pprint.pformat(var_value)}")
                        except Exception:
                            result.append(f"      {var_name}: <unable to display value>")
                if trace_event.internal_variables:
                    result.append("    Internal Variables:")
                    for intern_name, intern_value in trace_event.internal_variables.items():
                        try:
                            result.append(f"      {intern_name}: {pprint.pformat(intern_value)}")
                        except Exception:
                            result.append(f"      {intern_name}: <unable to display value>")
                if trace_event.return_value is not None:
                    result.append("    Return Value:")
                    try:
                        result.append(f"      {pprint.pformat(trace_event.return_value)}")
                    except Exception:
                        result.append("      <unable to display return value>")
                if trace_event.exception:
                    result.append("    Exception:")
                    for exc_key, exc_value in trace_event.exception.items():
                        result.append(f"      {exc_key}: {exc_value}")
                result.append("")  # add spacing between events
    return "\n".join(result)


if __name__ == "__main__":
    # example usage of tracing + printing (note: gets spammy for large functions!):
    _sample_code = \
"""\

def potato(a: int) -> int:
    print(f"potato {a}")
    return a + 1
    
class Something:
    def __init__(self):
        self.potato = "potato"
    def ok(self):
        return 10
    class SomethingElse:
        def __init__(self):
            self.potato = "potato2"
        def ok2(self, a: int, b: int):
            todo2

some_potato = Something()

import numpy as np

a = 1
b = 2
for i in range(3):
    a += i
    b *= i if i > 0 else 1
c = int(np.sum(np.ones((5, 5)) * some_potato.ok()))
print(f"Final values: a={a}, b={b}, c={c}")
"""
    _trace_result = execute_and_trace_code(
        code_string=_sample_code,
        inputs="",
        blacklisted_modules=["linecache", "traceback", "pydev", "pydevd_tracing", "contextlib", "numpy"],
        #trace_only_inside_code_string=True,
    )
    print(_trace_result.stdout)
    _report = format_traced_code_execution(
        trace_result=_trace_result,
    )
    print(_report)
    print("Done.")
