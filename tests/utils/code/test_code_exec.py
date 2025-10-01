import sys

import pytest

import pyine.utils.code.blocks
import pyine.utils.code.execution
from pyine.utils.code.execution import (
    EXEC_MODULE_OBJ_NAME,
    EXEC_TRACE_FILE_NAME,
    TraceEvent,
    TraceEventType,
    TraceException,
    TraceKey,
    TraceResult,
    TraceTagType,
    TracingCapException,
    _safe_execute_and_trace_code,
    _unsafe_execute_and_trace_code,
    execute_and_trace_code,
    format_traced_code_execution,
)


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
import os
import sys
print("Hello, world!")
a = 1 + 2
sys.stdout.buffer.write("Goodbye, world!".encode())
print("one last note", file=sys.stderr)
"""
    result = _unsafe_execute_and_trace_code(code, trace_only_inside_code_string=True)
    assert result.exception is None
    assert "Hello, world!" in result.stdout
    assert "Goodbye, world!" in result.stdout
    assert "one last note" in result.stderr
    valid_trace_steps = [s for s in result.traced_steps if s is not None]
    assert result.total_step_count == len(result.traced_steps)
    assert result.valid_step_count == len(valid_trace_steps)
    assert len(valid_trace_steps) == 8  # entrypoint call + 6 lines + final return
    assert valid_trace_steps[4].stdout == "Hello, world!\n"
    assert valid_trace_steps[5].stdout is None
    assert valid_trace_steps[6].stdout == "Goodbye, world!"
    assert valid_trace_steps[7].stderr == "one last note\n"
    assert result.stdout == "Hello, world!\nGoodbye, world!"
    assert "one last note\n" in result.stderr  # might contain extra warnings from pydev/debugger


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
        inputs="'15'",
        expected_output="potato",  # not verified internally, just logged
        entrypoint_name="some_magic_function",
        trace_only_inside_code_string=True,
    )
    assert trace_result.identifier == "hello"
    assert len(trace_result.code_blocks) == 1
    assert next(iter(trace_result.code_blocks.values())).name == "some_magic_function"
    assert trace_result.expected_output == "potato"
    assert trace_result.max_valid_events is None
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


def test_entrypoint_with_multiple_args():
    """Test that the entrypoint can be specified and that multi-args unpacking goes well."""
    code = """\
def some_magic_function(a: str, b: int, c: int = 1) -> int:
    return int(a) + b + c
"""
    inputs_outputs_to_test = [
        (["3", 2.0], 6),
        ({"a": "3", "b": 2}, 6),
        ("'3', 2, 1", 6),
        ("'3', 2.0", 6),
    ]
    for inputs, expected_output in inputs_outputs_to_test:
        trace_result = _unsafe_execute_and_trace_code(
            identifier="hello",  # just for logging purposes
            code_string=code,
            inputs=inputs,
            expected_output=str(expected_output),
            entrypoint_name="some_magic_function",
            trace_only_inside_code_string=True,
        )
        assert trace_result.return_value == expected_output


def test_entrypoint_with_single_arg():
    """Test that the entrypoint can be specified and that single-arg unpacking goes well."""
    code = """\
def single_arg_fn(a):
    return a + a
"""
    inputs_outputs_to_test = [
        (3, 6),
        ("3", 6),
        ("'3'", "33"),
        (2.2, 4.4),
        ([1, 2, 3], [1, 2, 3, 1, 2, 3]),
    ]
    for inputs, expected_output in inputs_outputs_to_test:
        trace_result = _unsafe_execute_and_trace_code(
            identifier="hello",  # just for logging purposes
            code_string=code,
            inputs=inputs,
            expected_output=str(expected_output),
            entrypoint_name="single_arg_fn",
            trace_only_inside_code_string=True,
        )
        assert trace_result.return_value == expected_output


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
    assert result_no_caps.valid_step_count > 1000
    assert result_no_caps.stdout == "Final i=1000\n"
    with pytest.raises(TracingCapException):
        _ = _unsafe_execute_and_trace_code(
            code_string=code,
            trace_only_inside_code_string=True,
            max_events_per_line=10,
        )
    with pytest.raises(TracingCapException):
        _ = _unsafe_execute_and_trace_code(
            code_string=code,
            trace_only_inside_code_string=True,
            max_valid_events=100,
        )
    result_with_caps = _unsafe_execute_and_trace_code(
        code_string=code,
        trace_only_inside_code_string=True,
        max_valid_events=5_000,
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
        use_safe_execution=True,
    )
    unsafe_trace_result = execute_and_trace_code(
        code_string=code,
        inputs="2",
        entrypoint_name="func",
        trace_only_inside_code_string=True,
        use_safe_execution=False,
    )
    assert safe_trace_result.stdout == unsafe_trace_result.stdout
    valid_safe_steps = [s for s in safe_trace_result.traced_steps if s is not None]
    valid_unsafe_steps = [s for s in unsafe_trace_result.traced_steps if s is not None]
    assert len(valid_safe_steps) == len(valid_unsafe_steps)
    for safe_step, unsafe_step in zip(valid_safe_steps, valid_unsafe_steps, strict=False):
        assert safe_step.trace_key == unsafe_step.trace_key
        assert safe_step.event_type == unsafe_step.event_type
        assert safe_step.global_variables == unsafe_step.global_variables
        assert safe_step.local_variables == unsafe_step.local_variables


def _build_dummy_trace_result() -> tuple[TraceResult, TraceEvent, TraceKey]:
    trace_key = TraceKey(file="snippet.py", object="fn", line=1)
    event = TraceEvent(
        event_type=TraceEventType.CALL,
        stack_trace=[trace_key],
        global_variables={"g": "1"},
        local_variables={"l": "2"},
        arguments={"l": "2"},
        return_value=None,
        stdout=None,
        stderr=None,
        exception=None,
        trace_step_idx=0,
        trace_key=trace_key,
    )
    dummy_block = pyine.utils.code.blocks.CodeBlock(
        type=pyine.utils.code.blocks.BlockType.FUNCTION,
        name="fn",
        depth=0,
        parent_line=None,
        start_line=1,
        end_line=2,
    )
    trace_result = TraceResult(
        identifier="identifier",
        code_string="print('hi')\n",
        code_blocks={repr(trace_key): dummy_block},
        inputs="inp",
        expected_output="out",
        max_valid_events=None,
        max_events_per_line=None,
        max_var_repr_length=None,
        traced_steps=[event, None],
        traced_steps_map={repr(trace_key): [0]},
        entrypoint_name=None,
        entrypoint_step_idx=None,
        return_value=123,
        exception=None,
        stdout="ok",
        stderr="",
        metadata={"seed": "42"},
        tags=["tag"],
    )
    return trace_result, event, trace_key


def test_tracekey_roundtrip_and_tags_buckets():
    key = TraceKey(file="foo.py", object="fn", line=12)
    repr_value = repr(key)
    assert repr_value == "foo.py:fn:L0012"
    rebuilt = TraceKey.from_string(repr_value)
    assert rebuilt == key
    tags_zero = TraceTagType.get_step_count_tags([])
    assert tags_zero == ["total_steps:0", "valid_steps:0"]
    tags_small = TraceTagType.get_step_count_tags([None, object(), object()])
    assert tags_small == ["total_steps:1_10", "valid_steps:1_10"]
    tags_large = TraceTagType.get_step_count_tags([object()] * 15)
    assert tags_large[0] == "total_steps:10_100"
    assert tags_large[1] == "valid_steps:10_100"


def test_trace_exception_from_exception_captures_origin():
    def boom():
        raise ValueError("kaboom")

    try:
        boom()
    except ValueError:
        exc_type, exc_value, exc_tb = sys.exc_info()
    assert exc_type is not None and exc_tb is not None and exc_value is not None
    trace_exc = TraceException.from_exception(exc_type, exc_value, exc_tb)
    assert trace_exc.type == "ValueError"
    assert "kaboom" in trace_exc.message
    assert trace_exc.origin is not None
    assert trace_exc.origin.object == "boom"
    assert trace_exc.traceback is not None


def test_trace_event_and_result_helpers():
    trace_result, event, trace_key = _build_dummy_trace_result()
    assert hash(event) == hash((0, trace_key))
    assert repr(event) == "step#000000:call@snippet.py:fn:L0001"
    assert str(trace_result) == "identifier"
    numbered = trace_result.code_string_with_line_numbers
    assert numbered.startswith("L0001:")
    assert trace_result.total_step_count == 2
    assert trace_result.valid_step_count == 1
    formatted = format_traced_code_execution(trace_result)
    assert "Code Execution Trace" in formatted
    assert "Line 1" in formatted


def test_safe_execute_returns_result(monkeypatch: pytest.MonkeyPatch):
    trace_result, _, _ = _build_dummy_trace_result()

    class DummyQueue:
        def __init__(self):
            self._items: list[tuple[str, object]] = []

        def empty(self) -> bool:
            return not self._items

        def get_nowait(self):
            return self._items.pop(0)

        def put(self, value):
            self._items.append(value)

    events = [("started", 42), ("returned", trace_result)]

    class DummyProcess:
        def __init__(self, target=None, args=(), kwargs=None, name=None):
            self._kwargs = kwargs or {}
            self.queue = self._kwargs["result_queue"]
            self.exitcode = 0
            self._alive = False

        def start(self):
            for event in events:
                self.queue.put(event)

        def is_alive(self) -> bool:
            return self._alive

        def kill(self):
            self._alive = False

        def join(self, timeout):
            return None

    monkeypatch.setattr("pyine.utils.code.execution.multiprocessing.Queue", lambda: DummyQueue())
    monkeypatch.setattr("pyine.utils.code.execution.multiprocessing.Process", DummyProcess)

    result = _safe_execute_and_trace_code(identifier="dummy", timeout_seconds=1, code_string="", inputs=None)
    assert result == trace_result


def test_safe_execute_raises_original_exception(monkeypatch: pytest.MonkeyPatch):
    class DummyQueue:
        def __init__(self):
            self._items: list[tuple[str, object]] = []

        def empty(self) -> bool:
            return not self._items

        def get_nowait(self):
            return self._items.pop(0)

        def put(self, value):
            self._items.append(value)

    boom = RuntimeError("boom")
    events = [("started", 84), ("raised", boom)]

    class DummyProcess:
        def __init__(self, target=None, args=(), kwargs=None, name=None):
            self._kwargs = kwargs or {}
            self.queue = self._kwargs["result_queue"]
            self.exitcode = 1
            self._alive = False

        def start(self):
            for event in events:
                self.queue.put(event)

        def is_alive(self) -> bool:
            return self._alive

        def kill(self):
            self._alive = False

        def join(self, timeout):
            return None

    monkeypatch.setattr("pyine.utils.code.execution.multiprocessing.Queue", lambda: DummyQueue())
    monkeypatch.setattr("pyine.utils.code.execution.multiprocessing.Process", DummyProcess)

    with pytest.raises(RuntimeError) as exc_info:
        _safe_execute_and_trace_code(identifier="dummy", timeout_seconds=1, code_string="", inputs=None)
    assert exc_info.value is boom


def test_safe_execute_times_out_and_kills_process(monkeypatch: pytest.MonkeyPatch):
    class DummyQueue:
        def __init__(self):
            self._items: list[tuple[str, object]] = []

        def empty(self) -> bool:
            return not self._items

        def get_nowait(self):
            return self._items.pop(0)

        def put(self, value):
            self._items.append(value)

    events = [("started", 21)]

    class DummyProcess:
        def __init__(self, target=None, args=(), kwargs=None, name=None):
            self._kwargs = kwargs or {}
            self.queue = self._kwargs["result_queue"]
            self.exitcode = -9
            self._alive = True

        def start(self):
            for event in events:
                self.queue.put(event)

        def is_alive(self) -> bool:
            return self._alive

        def kill(self):
            self._alive = False

        def join(self, timeout):
            return None

    monkeypatch.setattr("pyine.utils.code.execution.multiprocessing.Queue", lambda: DummyQueue())
    monkeypatch.setattr("pyine.utils.code.execution.multiprocessing.Process", DummyProcess)

    time_counter = {"value": 0.0}

    def fake_time():
        value = time_counter["value"]
        time_counter["value"] += 0.6
        return value

    monkeypatch.setattr("pyine.utils.code.execution.time.time", fake_time)
    monkeypatch.setattr("pyine.utils.code.execution.time.sleep", lambda _: None)

    with pytest.raises(TimeoutError):
        _safe_execute_and_trace_code(
            identifier="dummy",
            timeout_seconds=1,
            timeout_external_buffer_seconds=0,
            sleep_duration_seconds=0,
            code_string="",
            inputs=None,
        )


def test_trace_tag_type_size_buckets_cover_ranges() -> None:
    buckets = {
        0: "0",
        5: "1_10",
        50: "10_100",
        500: "100_1k",
        5000: "1k_10k",
        50000: "10k_100k",
        150000: "100k_plus",
    }
    for count, expected in buckets.items():
        traced_steps = [object() for _ in range(count)]
        tags = TraceTagType.get_step_count_tags(traced_steps)
        assert tags[0].endswith(expected)
        assert tags[1].endswith(expected)


def test_trace_result_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pyine.utils.code.execution.pyine.utils.portability,
        "get_code_with_numbered_lines",
        lambda code: f"numbered:{code}",
    )
    trace_key = pyine.utils.code.execution.TraceKey("file.py", "func", 10)
    trace_event = pyine.utils.code.execution.TraceEvent(
        event_type=pyine.utils.code.execution.TraceEventType.LINE,
        stack_trace=[trace_key],
        global_variables={"g": "1"},
        local_variables={"x": "2"},
        arguments=None,
        return_value="3",
        stdout="out",
        stderr="err",
        exception=None,
        trace_step_idx=1,
        trace_key=trace_key,
    )
    trace_result = pyine.utils.code.execution.TraceResult(
        identifier="trace-1",
        code_string="print('hi')",
        code_blocks={},
        inputs=None,
        expected_output=None,
        max_valid_events=None,
        max_events_per_line=None,
        max_var_repr_length=None,
        traced_steps=[None, trace_event, None],
        traced_steps_map={},
        entrypoint_name="main",
        entrypoint_step_idx=1,
        return_value="ok",
        exception=None,
        stdout="stdout",
        stderr="",
        metadata={},
        tags=[],
    )
    assert "numbered:" in trace_result.code_string_with_line_numbers
    assert trace_result.total_step_count == 3
    assert trace_result.valid_step_count == 1


def test_trace_context_sets_and_restores_trace() -> None:
    events: list[str] = []

    def tracer(frame, event, arg):  # noqa
        events.append(event)
        return tracer

    original_trace = sys.gettrace()
    with pyine.utils.code.execution.trace_context(tracer):
        _ = sum(range(3))
    assert events
    assert sys.gettrace() is original_trace
