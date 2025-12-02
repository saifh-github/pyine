"""Tests for pyine.data.traces.common module.

Tests the trace_code_snippet function with real code execution, as well as
the supporting data structures (TestTuple, TraceRequest, TracingConfig).
"""

import pydantic
import pytest

import pyine.data.traces.common as common
import pyine.data.traces.dataset_utils as dataset_utils
import pyine.utils.code.output_compare as output_compare

# -----------------------------------------------------------------------------
#                      helper functions and fixtures
# -----------------------------------------------------------------------------


def make_trace_id(
    *,
    augmented: bool = False,
    bugged: bool = False,
) -> dataset_utils.TraceIdentifier:
    """Create a trace identifier for testing."""
    if bugged:
        return dataset_utils.TraceIdentifier("DS", "sub", 0, 0, 0, "issues_todos", 0)
    if augmented:
        return dataset_utils.TraceIdentifier("DS", "sub", 0, 0, 0, "hints_docs", 0)
    return dataset_utils.TraceIdentifier("DS", "sub", 0, 0, 0)


def make_trace_request(
    code: str,
    inputs: object = None,
    outputs: object = None,
    entrypoint: str | None = "solution",
    **kwargs: object,
) -> common.TraceRequest:
    """Create a trace request with common defaults."""
    return common.TraceRequest(
        code_string=code,
        trace_id=kwargs.get("trace_id", make_trace_id()),  # type: ignore
        entrypoint_name=entrypoint,
        test_inputs=inputs,
        test_outputs=outputs,
        metadata=kwargs.get("metadata"),  # type: ignore
    )


@pytest.fixture
def default_tracing_config() -> common.TracingConfig:
    """Create a default tracing configuration for tests."""
    return common.TracingConfig(
        max_trace_valid_events=1000,
        max_trace_events_per_line=50,
        max_trace_var_repr_length=1000,
        execution_timeout_seconds=5.0,
        execution_seed=42,
    )


@pytest.fixture
def default_compare_options() -> output_compare.CompareOptions:
    """Create default compare options for tests."""
    return output_compare.CompareOptions()


# -----------------------------------------------------------------------------
#                            tracing data structures
# -----------------------------------------------------------------------------


class TestTestTuple:
    """Tests for the TestTuple dataclass."""

    def test_creation(self) -> None:
        """Test basic TestTuple creation."""
        tt = common.TestTuple(test_idx=0, inputs=[1, 2], outputs=3)
        assert tt.test_idx == 0
        assert tt.inputs == [1, 2]
        assert tt.outputs == 3

    def test_frozen(self) -> None:
        """Test that TestTuple is immutable."""
        tt = common.TestTuple(test_idx=0, inputs=1, outputs=2)
        with pytest.raises(AttributeError):
            tt.test_idx = 1  # noqa


class TestTraceRequest:
    """Tests for the TraceRequest dataclass."""

    def test_creation(self) -> None:
        """Test basic TraceRequest creation."""
        tid = make_trace_id()
        req = common.TraceRequest(
            code_string="def solution(x): return x",
            trace_id=tid,
            entrypoint_name="solution",
            test_inputs=5,
            test_outputs=5,
        )
        assert req.code_string == "def solution(x): return x"
        assert req.trace_id == tid
        assert req.entrypoint_name == "solution"
        assert req.test_inputs == 5
        assert req.test_outputs == 5

    def test_frozen(self) -> None:
        """Test that TraceRequest is immutable."""
        req = make_trace_request("def solution(x): return x", 1, 1)
        with pytest.raises(AttributeError):
            req.code_string = "other"  # noqa

    def test_compare_should_fail_false_for_normal_trace(self) -> None:
        """Test compare_should_fail is False for non-bugged trace."""
        req = make_trace_request("def solution(x): return x", 1, 1)
        assert req.compare_should_fail is False

    def test_compare_should_fail_true_for_bugged_trace(self) -> None:
        """Test compare_should_fail is True for bugged trace."""
        tid = make_trace_id(bugged=True)
        req = common.TraceRequest(
            code_string="def solution(x): return x + 1",  # intentionally wrong
            trace_id=tid,
            entrypoint_name="solution",
            test_inputs=5,
            test_outputs=5,  # wrong expected
        )
        assert req.compare_should_fail is True


class TestTracingConfig:
    """Tests for the TracingConfig pydantic model."""

    def test_custom_values(self) -> None:
        """Test TracingConfig with custom values."""
        config = common.TracingConfig(
            max_trace_events_per_line=100,
            max_trace_var_repr_length=500,
            max_trace_valid_events=1000,
            execution_timeout_seconds=5.0,
            execution_seed=123,
        )
        assert config.max_trace_events_per_line == 100
        assert config.max_trace_var_repr_length == 500
        assert config.max_trace_valid_events == 1000
        assert config.execution_timeout_seconds == 5.0
        assert config.execution_seed == 123

    def test_frozen(self) -> None:
        """Test that TracingConfig is frozen (immutable)."""
        config = common.TracingConfig()
        with pytest.raises(pydantic.ValidationError):
            config.execution_seed = 999  # noqa

    def test_rejects_extra_fields(self) -> None:
        """Test that TracingConfig rejects unknown fields."""
        with pytest.raises(pydantic.ValidationError):
            common.TracingConfig(unknown_field=123)  # noqa

    def test_rejects_negative_timeout(self) -> None:
        """Test that TracingConfig rejects non-positive timeout."""
        with pytest.raises(pydantic.ValidationError):
            common.TracingConfig(execution_timeout_seconds=-1.0)

    def test_allows_none_seed(self) -> None:
        """Test that TracingConfig allows None for seed (non-deterministic)."""
        config = common.TracingConfig(execution_seed=None)
        assert config.execution_seed is None


class TestRemoteTracebackError:
    """Tests for the RemoteTracebackError exception."""

    def test_can_be_raised(self) -> None:
        """Test that RemoteTracebackError can be raised and caught."""
        with pytest.raises(common.RemoteTracebackError):
            raise common.RemoteTracebackError("remote error")

    def test_inherits_from_exception(self) -> None:
        """Test that RemoteTracebackError is an Exception."""
        err = common.RemoteTracebackError("test")
        assert isinstance(err, Exception)

    def test_message_preserved(self) -> None:
        """Test that error message is preserved."""
        err = common.RemoteTracebackError("specific message")
        assert "specific message" in str(err)


# -----------------------------------------------------------------------------
#                         trace_code_snippet tests
# -----------------------------------------------------------------------------


@pytest.mark.slow
class TestTraceCodeSnippetReturnValue:
    """Tests for trace_code_snippet with return value comparison."""

    def test_successful_return_value_match(
        self,
        default_tracing_config: common.TracingConfig,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test successful execution with matching return value."""
        code = "def solution(x):\n    return x + 1\n"
        request = make_trace_request(code, inputs=5, outputs=6)
        trace_result, compare_result = common.trace_code_snippet(
            request, default_tracing_config, default_compare_options
        )
        assert compare_result  # truthy means success
        assert trace_result.return_value == 6
        assert trace_result.exception is None

    def test_failed_return_value_mismatch(
        self,
        default_tracing_config: common.TracingConfig,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test failed execution with mismatched return value."""
        code = "def solution(x):\n    return x + 1\n"
        request = make_trace_request(code, inputs=5, outputs=99)  # wrong expected
        trace_result, compare_result = common.trace_code_snippet(
            request, default_tracing_config, default_compare_options
        )
        assert not compare_result  # falsy means failure
        assert trace_result.return_value == 6  # actual result

    def test_return_value_with_list_inputs(
        self,
        default_tracing_config: common.TracingConfig,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test with list inputs/outputs that get unwrapped."""
        code = "def solution(x):\n    return sum(x)\n"
        request = make_trace_request(code, inputs=[[1, 2, 3]], outputs=[6])  # single-element lists get unwrapped
        trace_result, compare_result = common.trace_code_snippet(
            request, default_tracing_config, default_compare_options
        )
        assert compare_result
        assert trace_result.return_value == 6


@pytest.mark.slow
class TestTraceCodeSnippetStdout:
    """Tests for trace_code_snippet with stdout comparison."""

    def test_successful_stdout_match(
        self,
        default_tracing_config: common.TracingConfig,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test successful execution with matching stdout output."""
        code = "def solution():\n    print('hello')\n"
        request = make_trace_request(code, inputs=None, outputs="hello")
        trace_result, compare_result = common.trace_code_snippet(
            request, default_tracing_config, default_compare_options
        )
        assert compare_result
        assert "hello" in trace_result.stdout

    def test_stdout_with_no_entrypoint(
        self,
        default_tracing_config: common.TracingConfig,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test script-style code with stdout output."""
        code = "x = 2 + 2\nprint(x)\n"
        request = make_trace_request(code, inputs=None, outputs="4", entrypoint=None)
        trace_result, compare_result = common.trace_code_snippet(
            request, default_tracing_config, default_compare_options
        )
        assert compare_result
        assert "4" in trace_result.stdout

    def test_failed_stdout_mismatch(
        self,
        default_tracing_config: common.TracingConfig,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test failed execution with mismatched stdout output."""
        code = "def solution():\n    print('hello')\n"
        request = make_trace_request(code, inputs=None, outputs="goodbye")
        trace_result, compare_result = common.trace_code_snippet(
            request, default_tracing_config, default_compare_options
        )
        assert not compare_result


@pytest.mark.slow
class TestTraceCodeSnippetExceptions:
    """Tests for trace_code_snippet exception handling."""

    def test_expected_exception_matches(
        self,
        default_tracing_config: common.TracingConfig,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test that expected exception can match as output."""
        code = "def solution(x):\n    raise ValueError('expected error')\n"
        # when the expected output includes the exception string, it can match
        request = make_trace_request(code, inputs=1, outputs="ValueError: expected error")
        trace_result, compare_result = common.trace_code_snippet(
            request, default_tracing_config, default_compare_options
        )
        assert trace_result.exception is not None
        assert "ValueError" in trace_result.exception.type

    def test_unexpected_exception_fails(
        self,
        default_tracing_config: common.TracingConfig,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test that unexpected exception causes failure."""
        code = "def solution(x):\n    raise ValueError('oops')\n"
        request = make_trace_request(code, inputs=1, outputs=42)  # expected number, not exception
        trace_result, compare_result = common.trace_code_snippet(
            request, default_tracing_config, default_compare_options
        )
        assert not compare_result
        assert trace_result.exception is not None

    def test_system_exit_with_code(
        self,
        default_tracing_config: common.TracingConfig,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test that SystemExit exit code can be matched."""
        code = "def solution():\n    import sys\n    sys.exit(42)\n"
        request = make_trace_request(code, inputs=None, outputs=42)
        trace_result, compare_result = common.trace_code_snippet(
            request, default_tracing_config, default_compare_options
        )
        assert trace_result.exception is not None
        assert trace_result.exception.type == "SystemExit"


@pytest.mark.slow
class TestTraceCodeSnippetConfiguration:
    """Tests for trace_code_snippet with different configurations."""

    def test_exceeding_max_valid_events_raises(
        self,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test that exceeding max_valid_events raises TracingCapError."""
        import pyine.utils.code.execution as exec_utils

        code = "def solution():\n    for i in range(1000):\n        x = i\n    return 999\n"
        request = make_trace_request(code, inputs=None, outputs=999)
        config = common.TracingConfig(
            max_trace_valid_events=10,  # very low limit
            execution_timeout_seconds=5.0,
            execution_seed=42,
        )
        # when max_valid_events is exceeded, a TracingCapError is raised
        with pytest.raises(exec_utils.TracingCapError, match="max valid events exceeded"):
            common.trace_code_snippet(request, config, default_compare_options)

    def test_determinism_with_seed(
        self,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test that same seed produces same results."""
        code = "import random\ndef solution():\n    return random.randint(0, 1000)\n"
        request = make_trace_request(code, inputs=None, outputs=None)
        config = common.TracingConfig(
            max_trace_valid_events=100,
            execution_timeout_seconds=5.0,
            execution_seed=42,
        )
        result1, _ = common.trace_code_snippet(request, config, default_compare_options)
        result2, _ = common.trace_code_snippet(request, config, default_compare_options)
        assert result1.return_value == result2.return_value


@pytest.mark.slow
class TestTraceCodeSnippetEdgeCases:
    """Tests for trace_code_snippet edge cases."""

    def test_empty_return_value(
        self,
        default_tracing_config: common.TracingConfig,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test function that returns None."""
        code = "def solution():\n    pass\n"
        request = make_trace_request(code, inputs=None, outputs=None)
        trace_result, compare_result = common.trace_code_snippet(
            request, default_tracing_config, default_compare_options
        )
        assert compare_result
        assert trace_result.return_value is None

    def test_complex_return_value(
        self,
        default_tracing_config: common.TracingConfig,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test function with complex return value."""
        code = "def solution():\n    return {'a': 1, 'b': [1, 2, 3]}\n"
        request = make_trace_request(code, inputs=None, outputs={"a": 1, "b": [1, 2, 3]})
        trace_result, compare_result = common.trace_code_snippet(
            request, default_tracing_config, default_compare_options
        )
        assert compare_result
        assert trace_result.return_value == {"a": 1, "b": [1, 2, 3]}

    def test_trace_has_valid_identifier(
        self,
        default_tracing_config: common.TracingConfig,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test that trace result has correct identifier."""
        code = "def solution(x):\n    return x\n"
        tid = make_trace_id()
        request = common.TraceRequest(
            code_string=code,
            trace_id=tid,
            entrypoint_name="solution",
            test_inputs=42,
            test_outputs=42,
        )
        trace_result, _ = common.trace_code_snippet(request, default_tracing_config, default_compare_options)
        assert trace_result.identifier == str(tid)


# -----------------------------------------------------------------------------
#                      compare_execution_result tests
# -----------------------------------------------------------------------------


class TestCompareExecutionResult:
    """Unit tests for compare_execution_result function."""

    def make_trace_result(
        self,
        return_value: object = None,
        stdout: str = "",
        exception: object = None,
    ) -> object:
        """Create a mock-like TraceResult for testing.

        Uses types.SimpleNamespace to create a lightweight object with only
        the fields needed by compare_execution_result.
        """
        import types

        return types.SimpleNamespace(
            return_value=return_value,
            stdout=stdout,
            exception=exception,
        )

    def test_match_return_value_with_entrypoint(
        self,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test matching return value when entrypoint is specified."""
        result = self.make_trace_result(return_value=42)
        compare_result = common.compare_execution_result(
            trace_result=result,
            expected_output=42,
            compare_options=default_compare_options,
            has_entrypoint=True,
        )
        assert compare_result

    def test_mismatch_return_value_with_entrypoint(
        self,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test mismatched return value when entrypoint is specified."""
        result = self.make_trace_result(return_value=42)
        compare_result = common.compare_execution_result(
            trace_result=result,
            expected_output=99,
            compare_options=default_compare_options,
            has_entrypoint=True,
        )
        assert not compare_result

    def test_match_stdout_without_entrypoint(
        self,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test matching stdout when no entrypoint."""
        result = self.make_trace_result(stdout="hello\n")
        compare_result = common.compare_execution_result(
            trace_result=result,
            expected_output="hello",
            compare_options=default_compare_options,
            has_entrypoint=False,
        )
        assert compare_result

    def test_mismatch_stdout_without_entrypoint(
        self,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test mismatched stdout when no entrypoint."""
        result = self.make_trace_result(stdout="hello\n")
        compare_result = common.compare_execution_result(
            trace_result=result,
            expected_output="goodbye",
            compare_options=default_compare_options,
            has_entrypoint=False,
        )
        assert not compare_result

    def test_match_list_of_strings_stdout(
        self,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test matching merged list of strings against stdout."""
        result = self.make_trace_result(stdout="line1\nline2\n")
        compare_result = common.compare_execution_result(
            trace_result=result,
            expected_output=["line1", "line2"],
            compare_options=default_compare_options,
            has_entrypoint=False,
        )
        assert compare_result

    def test_prefers_return_value_over_stdout(
        self,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test that return value is checked before stdout when entrypoint exists."""
        result = self.make_trace_result(return_value=42, stdout="wrong\n")
        compare_result = common.compare_execution_result(
            trace_result=result,
            expected_output=42,
            compare_options=default_compare_options,
            has_entrypoint=True,
        )
        assert compare_result

    def test_falls_back_to_stdout_if_return_value_none(
        self,
        default_compare_options: output_compare.CompareOptions,
    ) -> None:
        """Test fallback to stdout when return value is None."""
        result = self.make_trace_result(return_value=None, stdout="42\n")
        compare_result = common.compare_execution_result(
            trace_result=result,
            expected_output="42",
            compare_options=default_compare_options,
            has_entrypoint=False,
        )
        assert compare_result
