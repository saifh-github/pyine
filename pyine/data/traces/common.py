"""Contains common utilities for tracing programs and creating trace datasets."""

from __future__ import annotations

import dataclasses
import functools
import logging
import typing
import warnings

import pydantic

import pyine.data.traces.dataset_utils
import pyine.utils.code.execution
import pyine.utils.code.output_compare
import pyine.utils.code.validation

__all__ = [
    "RemoteTracebackError",
    "TestTuple",
    "TraceExecutionOutcome",
    "TraceRequest",
    "TracingConfig",
    "trace_code_snippet",
]

logger = logging.getLogger(__name__)


class RemoteTracebackError(Exception):
    """Exception wrapper class that wraps another exception and adds a remote traceback."""

    pass


@dataclasses.dataclass(frozen=True)
class TestTuple:
    """Represents a test tuple to use to trace/verify a given solution."""

    test_idx: int
    """Index of the test inputs/outputs pair within the original coding problem."""
    inputs: typing.Any
    """Inputs to use for tracing/verifying a given solution."""
    outputs: typing.Any
    """Expected outputs to check after executing a given solution with the above inputs."""


@dataclasses.dataclass(frozen=True)
class TraceRequest:
    """Represents a code snippet to trace with a specific set of inputs/outputs and identifiers."""

    code_string: str
    """Code snippet to be traced; should already be reformatted/refactored/augmented if needed."""
    trace_id: pyine.data.traces.dataset_utils.TraceIdentifier
    """Pre-determined trace identifier to use for the results of tracing this code string."""
    entrypoint_name: str | None
    """Name of the entrypoint function to use for tracing this code string."""
    test_inputs: typing.Any
    """Inputs to use for tracing this code string."""
    test_outputs: typing.Any
    """Expected (resulting) outputs to check after tracing this code string."""
    metadata: str | None = None
    """Optional metadata to associate with the trace that may be useful for debugging."""

    @property
    def compare_should_fail(self) -> bool:
        """Returns whether this request is associated with buggy code that should NOT succeed output comparisons."""
        return self.trace_id.is_bugged


class TracingConfig(pydantic.BaseModel):
    """Configuration model for code snippet tracing settings."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (freezes the dataclass)."""

    max_trace_events_per_line: pydantic.PositiveInt | None = None
    """Maximum number of trace events per solution code line. If None, no maximum."""
    max_trace_var_repr_length: pydantic.PositiveInt | None = 10_000
    """Maximum length of variable representation strings, in characters."""
    max_trace_valid_events: pydantic.PositiveInt | None = 20_000
    """Maximum number of (valid, in-scope) events allowed per trace. If None, no maximum."""
    execution_timeout_seconds: pydantic.PositiveFloat = 10.0
    """Timeout in seconds for each execution attempt. If exceeded, solution is skipped."""
    execution_seed: int | None = 0  # none = non-deterministic
    """"Seed used to initialize internal RNGs during each execution attempt."""


TraceExecutionOutcome = tuple[
    pyine.utils.code.execution.TraceResult,
    pyine.utils.code.output_compare.CompareResult,
]
"""Alias for a tuple of (trace result, comparison result) for a given trace execution."""


def trace_code_snippet(
    code_snippet: TraceRequest,
    tracing_config: TracingConfig,
    output_compare_config: pyine.utils.code.output_compare.CompareOptions,
) -> TraceExecutionOutcome:
    """Traces a (potentially augmented) solution with a specific input and returns the result.

    Both tracing results and expected vs found output comparison results are returned. The latter
    will be used to determine if the solution will be written to the dataset or discarded.
    """
    test_inputs: typing.Any = code_snippet.test_inputs
    test_outputs: typing.Any = code_snippet.test_outputs
    if code_snippet.entrypoint_name is not None and isinstance(test_inputs, list) and isinstance(test_outputs, list):
        typed_inputs = typing.cast("list[typing.Any]", test_inputs)
        typed_outputs = typing.cast("list[typing.Any]", test_outputs)
        if len(typed_inputs) == len(typed_outputs) == 1:
            test_inputs = typed_inputs[0]
            test_outputs = typed_outputs[0]
        # TODO: confirm that these are fine everywhere and they do not cause some i/o issues?
        # they are meant to be single-input/output wrappers for entrypoint-based execution
    expected_for_compare = typing.cast("typing.Any", test_outputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no need to capture warnings related to traced code
        pyine.utils.code.validation.validate_code(code_snippet.code_string)  # last check before tracing
        trace_result = pyine.utils.code.execution.execute_and_trace_code(
            code_string=code_snippet.code_string,
            inputs=test_inputs,
            expected_output=test_outputs,
            identifier=str(code_snippet.trace_id),
            entrypoint_name=code_snippet.entrypoint_name,
            trace_only_inside_code_string=True,
            max_valid_events=tracing_config.max_trace_valid_events,
            max_events_per_line=tracing_config.max_trace_events_per_line,
            max_var_repr_length=tracing_config.max_trace_var_repr_length,
            timeout_seconds=tracing_config.execution_timeout_seconds,
            seed=tracing_config.execution_seed,
            request_metadata=code_snippet.metadata,
            use_safe_execution=True,  # since this parent is running in a thread, we want to isolate the child
        )
    comp = typing.cast(
        "typing.Callable[[typing.Any, typing.Any], pyine.utils.code.output_compare.CompareResult]",
        functools.partial(
            pyine.utils.code.output_compare.compare,
            options=output_compare_config,
        ),
    )
    default_test_result = None  # will store the most useful test result (across all comparison cases)
    if trace_result.exception is not None:
        # make sure the exception is not one that we are never meant to catch here
        dont_catch_exceptions = {str(t.__name__): t for t in pyine.utils.code.execution.DONT_CATCH_EXCEPTIONS}
        if trace_result.exception.type in dont_catch_exceptions:
            # if it is such a case, it means the tracing process itself failed somehow, so we need to raise
            exc = dont_catch_exceptions[trace_result.exception.type](trace_result.exception.message)
            if trace_result.exception.origin:
                origin_note = f"origin: {trace_result.exception.origin!r}"
            else:
                origin_note = "origin: <unknown>"
            exc.add_note(origin_note)
            if trace_result.exception.traceback:
                exc.add_note("remote traceback:\n" + trace_result.exception.traceback)
                raise exc from RemoteTracebackError(trace_result.exception.traceback)
            raise exc
        # the exec raised a catchable exception; the only way this was a 'success' is if we also expected one
        exception_test_result = comp(str(trace_result.exception), str(expected_for_compare))
        if exception_test_result:
            return (
                trace_result,
                exception_test_result,
            )  # we're done, we can leave already
        exception_test_result.reason = f"execution raised unexpected exception: {trace_result.exception}"
        if trace_result.exception.type == SystemExit.__name__:
            # that was likely called on purpose, i.e. the program finished and produced something
            # ...maybe it's the exit code or exception message itself we need to match?
            exception_msg_test_result = comp(trace_result.exception.message, expected_for_compare)
            if exception_msg_test_result:
                return trace_result, exception_msg_test_result
            exception_exit_code_test_result = comp(trace_result.return_value, expected_for_compare)
            if exception_exit_code_test_result:
                return trace_result, exception_exit_code_test_result
            # if we get here, checks failed, maybe something was printed before exiting?
            # (will jump to stdout checking logic below)
            default_test_result = exception_test_result
        else:
            # other kinds of exception are probably unexpected, and did not match the expected output
            return (
                trace_result,
                exception_test_result,
            )  # return the results immediately, pass or fail
    if code_snippet.entrypoint_name is not None or trace_result.return_value is not None:
        # the executed code returned a value that SHOULD be the expected one
        # (if the code had a specific entrypoint, this is the only possible outcome)
        return_val_test_result = comp(trace_result.return_value, expected_for_compare)
        if return_val_test_result:
            return trace_result, return_val_test_result
        if default_test_result is None:
            return_val_test_result.reason = f"unexpected entrypoint return value: {return_val_test_result.reason}"
            default_test_result = return_val_test_result
    # last chance: if we get here, assume the value we need to check is a printed output (in stdout)
    stdout_test_result = comp(trace_result.stdout, expected_for_compare)
    if stdout_test_result:
        return trace_result, stdout_test_result
    # if the expected outputs are a list of strings, last-last fix attempt: merge them into a string
    if isinstance(test_outputs, list):
        test_outputs_list = typing.cast("list[typing.Any]", test_outputs)
        if all(isinstance(item, str) for item in test_outputs_list):
            test_output_strings = typing.cast("list[str]", test_outputs_list)
            stdout_test_result = comp(trace_result.stdout, "\n".join(test_output_strings))
            if stdout_test_result:
                return trace_result, stdout_test_result
    if default_test_result is None:
        stdout_test_result.reason = f"unexpected stdout output: {stdout_test_result.reason}"
        default_test_result = stdout_test_result
    return trace_result, default_test_result
