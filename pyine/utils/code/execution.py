import asyncio
import collections.abc
import contextlib
import dataclasses
import enum
import logging
import multiprocessing
import os
import pprint
import sys
import time
import traceback
import types
import typing

import pydantic

import pyine.utils.code.args_mapper
import pyine.utils.code.blocks
import pyine.utils.code.input_mock
import pyine.utils.code.output_capture
import pyine.utils.filesystem
import pyine.utils.portability
import pyine.utils.reprod
import pyine.utils.timers

__all__ = [
    "EXEC_TRACE_FILE_NAME",
    "EXEC_BLOCK_OBJ_NAME",
    "EXEC_PARENT_FILE_NAME",
    "TraceKey",
    "TraceEventType",
    "TraceException",
    "TraceEvent",
    "TraceResult",
    "trace_context",
    "execute_and_trace_code",
    "format_traced_code_execution",
]

logger = logging.getLogger(__name__)


EXEC_TRACE_FILE_NAME = "<string>"
"""Name used to identify code lines that were executed and traced in the code string itself."""

EXEC_BLOCK_OBJ_NAME = "<block>"
"""Name used to identify code blocks that can be executed in the code string itself."""

EXEC_MODULE_OBJ_NAME = "<module>"
"""Name used to identify when code modules are 'called' (imported) during execution."""

EXEC_PARENT_FILE_NAME = "<parent>"
"""Name used to identify the parent file executing the code string itself via exec."""

REL_PATH_FROM_ROOT = pyine.utils.filesystem.get_relative_path_to_root(__file__)
"""Relative path from the root of the package to the current file."""

TraceKeyReprType = str
"""Helper type for representing a TraceKey in a string representation (for json dumps)."""

DONT_CATCH_EXCEPTIONS = (
    KeyboardInterrupt,
    GeneratorExit,
    MemoryError,
    asyncio.CancelledError,
)
"""Exceptions that should not be caught when tracing, and that should rise to the top process."""

INTERNAL_EVENT_KEY_PREFIXES = (
    # never report trace events for captured stream operations
    pyine.utils.portability.get_portable_filename(pyine.utils.code.output_capture.__file__),
    # ... add more prefixes here as needed for operations that are related to the tracing scaffolding
)
"""Prefixes for event keys that should not be reported as they relate to the tracing scaffolding."""


class TracingCapException(Exception):
    """Exception signaling that some traced attribute exceeded a predefined cap."""

    pass


class TraceKey(typing.NamedTuple):
    """NamedTuple for storing and exporting execution trace keys."""

    # note: needed instead of frozen dataclass to avoid issues w/ hashing in pydantic models
    file: str
    """The file name containing the code that was executed."""
    object: str
    """The name of the code object that was executed."""
    line: int
    """The number of the code line that was executed."""

    def __repr__(self) -> TraceKeyReprType:
        """Returns a string representation of the trace key."""
        return f"{self.file}:{self.object}:L{self.line:04d}"

    @classmethod
    def from_string(cls, str_repr: str) -> "TraceKey":
        """Creates a TraceKey instance from a string representation."""
        file, object_name, line_str = str_repr.split(":")
        line = int(line_str.lstrip("L"))
        return cls(file, object_name, line)


class TraceEventType(enum.StrEnum):
    """Identifies the type of execution event caught via the sys.settrace callback."""

    # note: do not modify these! they are the actual values used by sys.settrace

    CALL = "call"
    RETURN = "return"
    EXCEPTION = "exception"
    LINE = "line"


class TraceTagType(enum.StrEnum):
    """Identifies the type of trace outcome tags event."""

    HAS_EVENT_BLACKLISTED = "events:has_blacklisted"
    HAS_EVENT_EXCEPTION = "events:has_exception"

    HAS_EXEC_ENTRYPOINT = "exec:has_entrypoint"
    HAS_EXEC_SYSEXIT = "exec:has_sysexit"

    HAS_RETURN_EXCEPTION = "return:has_exception"
    HAS_RETURN_VALUE = "return:has_value"
    HAS_RETURN_STDOUT = "return:has_stdout"
    HAS_RETURN_STDERR = "return:has_stderr"

    HAS_INPUTS_EMPTY = "inputs:empty"

    @classmethod
    def get_step_count_tags(cls, traced_steps: list[typing.Any | None]) -> list[str]:
        """Returns a list of tags for the number of traced steps."""
        return [
            f"total_steps:{cls._get_size_bucket(len(traced_steps))}",
            f"valid_steps:{cls._get_size_bucket(len([s is not None for s in traced_steps]))}",
        ]

    @staticmethod
    def _get_size_bucket(count: int) -> str:
        """Returns a string for the size bucket of the traced steps."""
        if count == 0:
            return "0"
        elif 0 < count <= 10:
            return "1_10"
        elif 10 < count <= 100:
            return "10_100"
        elif 100 < count <= 1000:
            return "100_1k"
        elif 1000 < count <= 10_000:
            return "1k_10k"
        elif 10_000 < count <= 100_000:
            return "10k_100k"
        else:  # > 100_000
            return "100k_plus"


class TraceException(typing.NamedTuple):
    """NamedTuple for storing and exporting execution trace exceptions."""

    type: str
    """The type of exception that occurred."""
    message: str
    """The message of the exception that occurred."""
    origin: TraceKey | None
    """Location where the exception originated (file, object, and line), if available."""
    traceback: str | None
    """Formatted traceback string, if available."""

    def __repr__(self):
        """Returns a string representation of the trace exception."""
        # note: when we expect an exception to be produced in a traced run, this is what gets compared
        return f"{self.type}({self.message})"

    @classmethod
    def from_exception(
        cls,
        exc_type: type[BaseException],  # noqa
        exc_value: BaseException,
        exc_tb: types.TracebackType | None,
    ) -> "TraceException":
        """Build a TraceException from exception triplet, capturing origin and formatted traceback."""
        origin = None
        if exc_tb is not None:
            tb_frames = traceback.extract_tb(exc_tb)
            if tb_frames:
                last = tb_frames[-1]
                origin = TraceKey(
                    file=pyine.utils.portability.get_portable_filename(last.filename),
                    object=last.name,
                    line=last.lineno,
                )
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb)) if exc_tb is not None else None
        return cls(exc_type.__name__, str(exc_value), origin, tb_str)


@dataclasses.dataclass(frozen=True)
class TraceEvent:
    """Dataclass for storing and exporting execution trace events."""

    event_type: TraceEventType
    """The type of event that occurred (e.g., "call", "return", "exception")."""
    stack_trace: list[TraceKey]
    """A list of TraceKey instances representing the call stack at the time of the event."""
    global_variables: dict[str, str]
    """A dictionary containing the global variables at the time of the event."""
    local_variables: dict[str, str]
    """A dictionary containing the local variables at the time of the event."""
    arguments: dict[str, str] | None
    """A dictionary containing the arguments passed to the code object at the time of the event."""
    return_value: str | None
    """The return value of the code object at the time of the event."""
    stdout: str | None
    """The captured stdout output for this event (if any)."""
    stderr: str | None
    """The captured stderr output for this event (if any)."""
    exception: TraceException | None
    """Contains information about the exception that occurred, if any."""
    trace_step_idx: int
    """The trace step index at the time of the event; should be unique for each event."""
    trace_key: TraceKey
    """The trace key associated with this event, for convenience."""

    def __hash__(self):
        """Returns a hash value for the event, considering only relevant unique attributes."""
        return hash((self.trace_step_idx, self.trace_key))

    def __repr__(self):
        """Returns a (partial) string representation of the trace event."""
        return f"step#{self.trace_step_idx:06d}:{self.event_type}@{self.trace_key}"


class TraceResult(pydantic.BaseModel):
    """Dataclass for storing and exporting execution trace results."""

    model_config = pydantic.ConfigDict(frozen=True)
    """Pydantic model configuration (freezes the dataclass)."""
    identifier: str | None
    """An identifier for this trace (used for printing/logging purposes only)."""
    code_string: str
    """The original code string that was executed."""
    code_blocks: dict[TraceKeyReprType, pyine.utils.code.blocks.CodeBlock]
    """A dictionary containing the logic blocks of the executed code, indexed by start line number."""
    inputs: str
    """The inputs that were available to the code during execution."""
    expected_output: str
    """The expected output of the code, if any; should be used for verification/predictions."""
    max_valid_events: int | None
    """The maximum number of valid (in-scope) events that could have been recorded (if such a cap was used)."""
    max_events_per_line: int | None
    """The maximum number of events to record per line (if such a cap was used)."""
    max_var_repr_length: int | None
    """The maximum length of variable representations (if such a cap was used)."""
    traced_steps: list[TraceEvent | None]
    """A list of all traced steps, in order of execution.

    The states correspond to variables at each line of code prior to execution. Steps that are
    `None` should correspond to steps that occurred outside our tracing scope. To determine which
    line a 'valid' (non-None) event occurred on, see the `TraceKey` attribute of that event, or the
    `traced_steps_map` dictionary below.
    """
    traced_steps_map: dict[TraceKeyReprType, list[int]]
    """A dictionary containing the indices of each traced step for each line of executed code.

    In this dictionary, keys are `TraceKey` representations (combining file, object, and line info)
    and values are lists of indices pointing to `TraceEvent` objects in the above `traced_steps`
    list.
    """
    entrypoint_name: str | None
    """The name of the entrypoint function that was executed, if any."""
    entrypoint_step_idx: int | None
    """The trace step index just prior to calling the entrypoint function (if one is called)."""
    return_value: typing.Any | None
    """The return value of the executed code, if any."""
    exception: TraceException | None
    """Contains information about the exception that occurred, if any."""
    stdout: str
    """The captured stdout output during execution (in full)."""
    stderr: str
    """The captured stderr output during execution (in full)."""
    metadata: dict[str, typing.Any]
    """A dictionary containing metadata about the execution environment & settings."""
    tags: list[str]
    """List of tags (labels) associated with this trace, assigned based on tracing outcomes."""

    def __str__(self):
        """Returns a string representation of the trace result based on its identifier."""
        if self.identifier is not None:
            return str(self.identifier)
        else:
            return str(self.model_dump_json())

    @property
    def code_string_with_line_numbers(self) -> str:
        """Returns the code string with line numbers for each line of code."""
        return pyine.utils.portability.get_code_with_numbered_lines(self.code_string)

    @property
    def total_step_count(self) -> int:
        """Returns the total number of recorded tracing steps (including out-of-scope ones).

        Note: this number can vary due to a number of factors that include the trace scaffolding,
        and it is therefore NOT a good idea to use it for any kind of prediction task.
        """
        return len(self.traced_steps)

    @property
    def valid_step_count(self) -> int:
        """Returns the valid number of recorded tracing steps, where valid means in-scope."""
        return len([s for s in self.traced_steps if s is not None])


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


def _execute_in_subprocess(
    *args,  # we will forward all args + kwargs to `_unsafe_execute_and_trace_code`
    result_queue: multiprocessing.Queue,
    identifier: str | None = None,
    timeout_seconds: float = 60,
    **kwargs,
) -> None:
    """Execute 'unsafe' tracing in a separate process (where unsafe means it could crash the main process)."""
    # note: this could not be a local define because it gets pickled for multiprocessing
    result_queue.put(("started", os.getpid()))
    try:
        result = _unsafe_execute_and_trace_code(
            *args,
            identifier=identifier,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        result_queue.put(("returned", result))
    except BaseException as e:  # catch all potential exception types to provide them to the parent
        result_queue.put(("raised", e))


def _safe_execute_and_trace_code(
    *args,  # we will forward all args + kwargs to `_unsafe_execute_and_trace_code`
    identifier: str | None = None,
    timeout_seconds: float = 60,
    timeout_external_buffer_seconds: float = 5,
    sleep_duration_seconds: float = 0.1,
    **kwargs,
) -> "TraceResult":
    """
    Wrapper around `_unsafe_execute_and_trace_code` that handles OS-level crashes.

    'Safe' in this case relates to using process isolation to prevent segfaults, OOM kills, and
    other OS-level crashes from taking down the main process.

    See the `_unsafe_execute_and_trace_code` docstring for more details on forwarded arguments.

    Args:
        identifier: an identifier for this trace (used for printing/logging purposes).
        timeout_seconds: the maximum number of seconds to allow for code execution. Use internallly
            in the forked process, but also here to make sure we kill the process if it freezes for
            some reason. The external check will add a time buffer to whatever number this is.
        timeout_external_buffer_seconds: the buffer to add to the timeout value when checking for
            hung processes in this master process; should be positive float, in seconds. Defaults
            to 5, so we allow 5 extra seconds on top of `timeout_seconds` before raising.
        sleep_duration_seconds: the default sleep duration between checks for process completion.

    Returns:
        A `TraceResult` instance containing the execution results.
    """
    result_queue = multiprocessing.Queue()
    logger.debug(f"launching subprocess for tracing (name={identifier})")
    process = multiprocessing.Process(
        target=_execute_in_subprocess,
        args=args,
        kwargs=dict(
            result_queue=result_queue,
            identifier=identifier,
            timeout_seconds=timeout_seconds,
            **kwargs,
        ),
        name=identifier,
    )
    process.start()
    process_timeout = timeout_seconds + timeout_external_buffer_seconds
    start_time = time.time()
    latest_time_delta = 0
    child_pid, status, returned_val = None, None, None
    # wait until we get the child pid from the queue
    while latest_time_delta < process_timeout and returned_val is None:
        while result_queue.empty():
            latest_time_delta = time.time() - start_time
            if latest_time_delta > process_timeout:
                break
            time.sleep(sleep_duration_seconds)
        if not result_queue.empty():
            if child_pid is None:
                status, child_pid = result_queue.get_nowait()
                assert status == "started"
            else:
                status, returned_val = result_queue.get_nowait()
                assert status in ("returned", "raised")
    if process.is_alive():
        logger.debug(f"killing hanging subprocess for tracing (name={identifier})")
        process.kill()
    elif process.exitcode != 0:
        # note: this might not be an issue, solutions sometimes use sys.exit for outputs
        logger.debug(f"subprocess exited with non-zero exit code (name={identifier}, code={process.exitcode})")
    if status == "returned":
        logger.debug(f"tracing subprocess returned results (name={identifier})")
        assert isinstance(returned_val, TraceResult)
        return returned_val
    elif status == "raised":
        assert isinstance(returned_val, Exception)
        logger.debug(f"tracing subprocess raised exception: {returned_val}")
        raise returned_val
    error_msg = f"tracing subprocess timed out after {latest_time_delta:.3f} seconds (name={identifier})"
    logger.debug(error_msg)
    raise TimeoutError(error_msg)


def _unsafe_execute_and_trace_code(
    code_string: str,
    inputs: typing.Any | None = None,
    expected_output: typing.Any | None = None,
    identifier: str | None = None,
    entrypoint_name: str | None = None,
    blacklisted_modules: typing.Iterable[str] | None = None,
    blacklisted_objects: typing.Iterable[str] | None = None,
    trace_only_inside_code_string: bool = False,
    max_valid_events: int | None = None,
    max_events_per_line: int | None = None,
    max_var_repr_length: int | None = None,
    timeout_seconds: float = 60,
    seed: int | None = 42,
) -> TraceResult:
    """Execute Python code and trace the state of the execution at each line.

    'Unsafe' here means that any segfaults, OOM kills, or other OS-level crashes will take down
    the main process if it called this function directly. Refer to the 'safe' version to avoid that.

    Will replace calls to input() or sys.stdin.readline() with lines from the provided inputs, and
    captures stdout and stderr during execution.

    Args:
        code_string: a string containing arbitrary Python code to execute and trace.
        inputs: input arguments for the execution (either fed via stdin, or provided to entrypoint).
        expected_output: the expected output of the code execution. This output will not be verified
            in this function, and is passed for logging/serialization purposes only.
        identifier: an identifier for this trace (used for printing/logging purposes).
        entrypoint_name: The name of the entrypoint function to execute.
        blacklisted_modules: A list of module names to exclude from tracing.
        blacklisted_objects: A list of object names to exclude from tracing.
        trace_only_inside_code_string: If True, only trace code inside the provided code_string.
        max_valid_events: The maximum number of valid (in-scope) events to allow. If this cap is exceeded,
            a `TracingCapException` will be raised.
        max_events_per_line: The maximum number of events allowed per line. If this cap is exceeded,
            a `TracingCapException` will be raised.
        max_var_repr_length: The maximum length (in chars) allowed for the representation of a
            variable. Above this cap, a `TracingCapException` will be raised.
        timeout_seconds: The maximum number of seconds to allow for code execution.
        seed: The seed to use for random number generation. Defaults to 42.

    Returns:
        A `TraceResult` instance containing the execution results.
    """
    try:
        code_blocks = pyine.utils.code.blocks.identify_code_blocks(code_string)
        code_blocks = {
            TraceKey(
                file=EXEC_TRACE_FILE_NAME,
                object=code_block_data.name if code_block_data.name else EXEC_BLOCK_OBJ_NAME,
                line=code_block_line,
            ): code_block_data
            for code_block_line, code_block_data in code_blocks.items()
        }
        compiled_code = compile(code_string, EXEC_TRACE_FILE_NAME, "exec")
    except Exception as e:
        raise Exception(f"error while analyzing and compiling code: {e}") from e
    traced_steps: list[TraceEvent | None] = []
    traced_steps_map: dict[TraceKeyReprType, list[int]] = {}
    last_trace_step_idx = 0  # will be incremented each time the callback is called
    trace_tags = []  # will be used to store the interesting outcome-related tags for this trace
    stdout_buffer, stderr_buffer = "", ""
    stdout_capture = pyine.utils.code.output_capture.StdStreamCapture(
        stream_name="stdout",
        encoding="utf-8",
        errors="replace",
        fileno_value=1,
    )
    stderr_capture = pyine.utils.code.output_capture.StdStreamCapture(
        stream_name="stderr",
        encoding="utf-8",
        errors="replace",
        fileno_value=2,
    )

    def _capture_buffers() -> tuple[str, str]:  # new_stdout, new_stderr
        nonlocal stdout_buffer, stderr_buffer
        stdout_capture.flush()
        stderr_capture.flush()
        new_stdout = stdout_capture.getvalue()
        new_stderr = stderr_capture.getvalue()
        stdout_capture.seek(0)
        stdout_capture.truncate(0)
        stderr_capture.seek(0)
        stderr_capture.truncate(0)
        stdout_buffer += new_stdout
        stderr_buffer += new_stderr
        return new_stdout, new_stderr

    def _trace_callback(
        frame: types.FrameType,  # noqa
        event: str,
        arg: typing.Any,
    ) -> typing.Callable | None:
        """Callback function for sys.settrace that records execution state at each line."""
        nonlocal last_trace_step_idx

        trace_key = TraceKey(
            file=pyine.utils.portability.get_portable_filename(frame.f_code.co_filename),
            object=frame.f_code.co_name,
            line=frame.f_lineno,
        )
        return_trace_callback = _trace_callback  # any non-blacklisted object will be traced
        # now, determine if we want to keep the event or not
        is_blacklisted = (blacklisted_objects and trace_key.object in blacklisted_objects) or (
            blacklisted_modules and trace_key.file.startswith(tuple(blacklisted_modules))
        )
        if is_blacklisted and TraceTagType.HAS_EVENT_BLACKLISTED not in trace_tags:
            trace_tags.append(TraceTagType.HAS_EVENT_BLACKLISTED)
        is_internal = any([str(trace_key).startswith(prefix) for prefix in INTERNAL_EVENT_KEY_PREFIXES])
        is_inside_code_string = trace_key.file == EXEC_TRACE_FILE_NAME
        must_skip = is_blacklisted or is_internal or (not is_inside_code_string and trace_only_inside_code_string)
        if event == "call" and must_skip:
            return_trace_callback = None  # do not trace that function, skip over it
        trace_key_repr = str(trace_key)
        if trace_key_repr not in traced_steps_map:
            traced_steps_map[trace_key_repr] = []
        max_event_capped = max_events_per_line and len(traced_steps_map[trace_key_repr]) >= max_events_per_line
        if max_event_capped:
            raise TracingCapException(
                f"max events per line ({max_events_per_line}) exceeded for trace at key {trace_key_repr}"
            )
        if must_skip:
            # if we have a blacklisted event, append `None` instead of the event data
            trace_event = None
        else:
            # gather the actual event data and create the corresponding object
            new_stdout, new_stderr = _capture_buffers()
            stack_trace = []
            current_frame = frame
            while current_frame:
                clean_filename = pyine.utils.portability.get_portable_filename(current_frame.f_code.co_filename)
                if clean_filename == REL_PATH_FROM_ROOT:
                    break  # stop tracing the stack once we get to this level
                stack_trace.append(
                    TraceKey(
                        file=clean_filename,
                        object=current_frame.f_code.co_name,
                        line=current_frame.f_lineno,
                    )
                )
                current_frame = current_frame.f_back
            banned_local_var_names = ["_trace_callback", "__builtins__"]
            local_vars = {
                name: pyine.utils.portability.get_portable_representation(value)
                for name, value in frame.f_locals.items()
                if not name.startswith("__") and name not in banned_local_var_names
            }
            if max_var_repr_length is not None and local_vars:
                max_locals_var_repr_len = max([len(v) for v in local_vars.values()])
                if max_locals_var_repr_len > max_var_repr_length:
                    raise TracingCapException(
                        f"max locals repr len exceeded for trace at key {trace_key_repr} "
                        f"(found max len: {max_locals_var_repr_len}, cap: {max_var_repr_length})"
                    )
            global_vars = {
                name: pyine.utils.portability.get_portable_representation(value)
                for name, value in frame.f_globals.items()
                if not name.startswith("__")
            }
            if max_var_repr_length is not None and global_vars:
                max_globals_var_repr_len = max([len(v) for v in global_vars.values()])
                if max_globals_var_repr_len > max_var_repr_length:
                    raise TracingCapException(
                        f"max globals repr len exceeded for trace at key {trace_key_repr} "
                        f"(found max len: {max_globals_var_repr_len}, cap: {max_var_repr_length})"
                    )
            arguments, exec_return_value, exception = None, None, None
            if event == "call":
                arguments = {
                    name: pyine.utils.portability.get_portable_representation(value)
                    for name, value in frame.f_locals.items()
                    if not name.startswith("__")
                }
                if max_var_repr_length is not None and arguments:
                    max_args_var_repr_len = max([len(v) for v in arguments.values()])
                    if max_args_var_repr_len > max_var_repr_length:
                        raise TracingCapException(
                            f"max args repr len exceeded for trace at key {trace_key_repr}"
                            f"(found max len: {max_args_var_repr_len}, cap: {max_var_repr_length})"
                        )
            elif event == "return":
                exec_return_value = pyine.utils.portability.get_portable_representation(arg)
                if max_var_repr_length is not None and len(exec_return_value) > max_var_repr_length:
                    raise TracingCapException(
                        f"max return val repr len exceeded for trace at key {trace_key_repr}"
                        f"(found len: {len(exec_return_value)}, cap: {max_var_repr_length})"
                    )
            elif event == "exception":
                exc_type, exc_value, exc_traceback = arg
                exception = TraceException.from_exception(exc_type, exc_value, exc_traceback)
                if TraceTagType.HAS_EVENT_EXCEPTION not in trace_tags:
                    trace_tags.append(TraceTagType.HAS_EVENT_EXCEPTION)
            trace_event = TraceEvent(
                event_type=TraceEventType(event),
                stack_trace=stack_trace,
                global_variables=global_vars,
                local_variables=local_vars,
                arguments=arguments,
                return_value=exec_return_value,
                stdout=new_stdout if new_stdout else None,
                stderr=new_stderr if new_stderr else None,
                exception=exception,
                trace_step_idx=last_trace_step_idx,
                trace_key=trace_key,
            )
        traced_steps.append(trace_event)
        traced_steps_map[trace_key_repr].append(last_trace_step_idx)
        last_trace_step_idx += 1  # will reflect the total number of calls to this callback, no matter what
        if max_valid_events is not None and last_trace_step_idx >= max_valid_events:
            raise TracingCapException(
                f"max valid events exceeded for trace at key {trace_key_repr} "
                f"(reached the cap of {last_trace_step_idx} in-scope events)"
            )
        return return_trace_callback

    return_value, caught_exception = None, None
    entrypoint_step_idx = None  # only used if we call an entrypoint after exec
    pyine.utils.reprod.set_seed(seed)
    exec_namespace = {}
    if inputs is None or (isinstance(inputs, collections.abc.Sized) and not inputs):
        trace_tags.append(TraceTagType.HAS_INPUTS_EMPTY)
    try:
        with pyine.utils.timers.TimeLimit(timeout_seconds):
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):  # noqa
                if entrypoint_name is not None:
                    with trace_context(_trace_callback):
                        exec(compiled_code, exec_namespace)
                    if entrypoint_name and entrypoint_name in exec_namespace:
                        # note for later: if this is buggy/annoying, could add call inside code string itself
                        entrypoint_step_idx = last_trace_step_idx
                        entrypoint = exec_namespace[entrypoint_name]
                        entrypoint_args, entrypoint_kwargs = pyine.utils.code.args_mapper.map_inputs_to_callable(
                            entrypoint, inputs
                        )
                        with trace_context(_trace_callback):
                            return_value = entrypoint(*entrypoint_args, **entrypoint_kwargs)
                        trace_tags.append(TraceTagType.HAS_EXEC_ENTRYPOINT)
                else:
                    with pyine.utils.code.input_mock.MockInputContext(str(inputs)):
                        with trace_context(_trace_callback):
                            exec(compiled_code, exec_namespace)
                _capture_buffers()
    except (TimeoutError, TracingCapException):
        # we'll let callers handle what happens when code tracing times out or caps are exceeded
        raise
    except DONT_CATCH_EXCEPTIONS:
        # process is probably being interrupted, raise immediately here as well
        raise
    except SystemExit as e:
        # some crazy people return their outputs via sys.exit, so catch those correctly...
        caught_exception = e
        return_value = e.code
        trace_tags.append(TraceTagType.HAS_EXEC_SYSEXIT)
    except Exception as e:
        # otherwise, if it's not a timeout/dontcatch/sysexit, store the exception as part of the results
        caught_exception = e
    reprod_metadata = pyine.utils.reprod.get_reprod_metadata()
    reprod_metadata["entrypoint_name"] = str(entrypoint_name)
    reprod_metadata["blacklisted_modules"] = str(list(blacklisted_modules or []))
    reprod_metadata["blacklisted_objects"] = str(list(blacklisted_objects or []))
    reprod_metadata["trace_only_inside_code_string"] = str(trace_only_inside_code_string)
    reprod_metadata["max_valid_events"] = str(max_valid_events)
    reprod_metadata["max_events_per_line"] = str(max_events_per_line)
    reprod_metadata["max_var_repr_length"] = str(max_var_repr_length)
    reprod_metadata["timeout_seconds"] = str(timeout_seconds)
    reprod_metadata["seed"] = str(seed)
    if return_value is not None:
        trace_tags.append(TraceTagType.HAS_RETURN_VALUE)
    if caught_exception is not None:
        trace_tags.append(TraceTagType.HAS_RETURN_EXCEPTION)
    if stdout_buffer:
        trace_tags.append(TraceTagType.HAS_RETURN_STDOUT)
    if stderr_buffer:
        trace_tags.append(TraceTagType.HAS_RETURN_STDERR)
    trace_tags.extend(TraceTagType.get_step_count_tags(traced_steps))
    try:
        trace_result = TraceResult(
            identifier=identifier,
            code_string=code_string,
            code_blocks={str(block_key): block for block_key, block in code_blocks.items()},
            inputs=str(inputs),
            expected_output=str(expected_output),
            max_valid_events=max_valid_events,
            max_events_per_line=max_events_per_line,
            max_var_repr_length=max_var_repr_length,
            traced_steps=traced_steps,
            traced_steps_map=traced_steps_map,
            entrypoint_name=entrypoint_name,
            entrypoint_step_idx=entrypoint_step_idx,
            return_value=return_value,
            exception=(
                TraceException.from_exception(
                    caught_exception.__class__,
                    caught_exception,
                    caught_exception.__traceback__,
                )
                if caught_exception
                else None
            ),
            stdout=stdout_buffer,
            stderr=stderr_buffer,
            metadata=reprod_metadata,
            tags=trace_tags,
        )
    except pydantic.ValidationError as e:
        print(f"Error while creating TraceResult instance (unrelated to exec): {e}")
        raise e
    return trace_result


def execute_and_trace_code(
    *args,
    use_safe_execution: bool = True,
    **kwargs,
) -> "TraceResult":
    """Convenience function that chooses between safe and unsafe execution and tracing functions.

    By 'safe', we solely mean process-crash-safe, i.e. that the executed code exiting (using e.g.
    `sys.exit`) or crashing due to memory or OS-level issues will not cause the main process to
    also crash.

    See the `_unsafe_execute_and_trace_code` docstring for more details on the tracing arguments.

    Note that if tracing exceeds the specified timeout delay, it will raise `TimeoutError`.

    Args:
        use_safe_execution (bool): If True, use process isolation. If False, use the orig tracing
            function directly. Defaults to True, as there probably isn't much runtime difference
            between the two, especially given

    Returns:
        A `TraceResult` instance containing the execution results.
    """
    if not use_safe_execution:
        return _unsafe_execute_and_trace_code(*args, **kwargs)
    else:
        return _safe_execute_and_trace_code(*args, **kwargs)


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
    result = ["Code Execution Trace:", "=====================", ""]
    for line_number, line in enumerate(code_lines, 1):
        result.append(f"Line {line_number}: {line}")
        relevant_events = []
        for trace_key_repr, step in trace_result.traced_steps_map.items():
            trace_key = TraceKey.from_string(trace_key_repr)
            if trace_key.line == line_number:
                relevant_events.append((trace_key, step))
        for trace_key, event_indices in relevant_events:
            result.append(f"  {trace_key}")
            for event_number, trace_step_idx in enumerate(event_indices, 1):
                trace_event = trace_result.traced_steps[trace_step_idx]
                if trace_event is None:
                    result.append("  [Event skipped due to out-of-tracing-scope]")
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
                if trace_event.global_variables:
                    result.append("    Global Variables:")
                    for var_name, var_value in trace_event.global_variables.items():
                        try:
                            result.append(f"      {var_name}: {pprint.pformat(var_value)}")
                        except Exception:
                            result.append(f"      {var_name}: <unable to display value>")
                if trace_event.local_variables:
                    result.append("    Local Variables:")
                    for intern_name, intern_value in trace_event.local_variables.items():
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
                    result.append(f"      {trace_event.exception}")
                result.append("")  # add spacing between events
    return "\n".join(result)


if __name__ == "__main__":
    # example usage of tracing + printing (note: gets spammy for large functions!):
    _sample_code = """\

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
    _trace_result = _unsafe_execute_and_trace_code(
        code_string=_sample_code,
        inputs="",
        blacklisted_modules=["linecache", "traceback", "pydev", "pydevd_tracing", "contextlib", "numpy"],
        # trace_only_inside_code_string=True,
    )
    print(_trace_result.stdout)
    _report = format_traced_code_execution(
        trace_result=_trace_result,
    )
    print(_report)
    print("Done.")
