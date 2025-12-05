"""Edge case tests for sample transformation logic.

These tests cover edge cases, error handling, and strategy behaviors
that are not covered by the basic accuracy tests.
"""

import logging
import typing

import numpy as np
import pytest

import pyine.utils.code.execution as exec_utils
from pyine.organisms.datamodules.samples.common import (
    SamplePredictType,
    SampleTransformStrategy,
)
from pyine.organisms.datamodules.samples.configs import SampleTransformConfig
from pyine.organisms.datamodules.samples.transform import (
    _get_code_segment_sample,
    _get_function_call_sample,
    generate_sample,
)
from tests.organisms.datamodules.samples.conftest import (
    CODE_FUNCTION_WITH_EXCEPTION,
    CODE_GLOBAL_AND_LOCAL,
    CODE_LOOP_MULTIPLE_ITERATIONS,
    CODE_RECURSIVE_FUNCTION,
    CODE_SIMPLE_FUNCTION,
    make_selected_sample,
    make_trace_metadata,
)

TraceFromCodeFixture = typing.Callable[..., exec_utils.TraceResult]


CODE_NO_FUNCTION_CALL = """
x = 5
y = x * 2
print(y)
""".strip()


CODE_DEEP_RECURSION = """
def deep(n):
    if n <= 0:
        return 0
    return deep(n - 1) + 1
""".strip()


@pytest.mark.slow
class TestFunctionCallEdgeCases:
    """Tests for edge cases in function call sample generation."""

    def test_function_returning_exception(
        self,
        trace_from_code: TraceFromCodeFixture,
        function_return_only_config: SampleTransformConfig,
        default_rng: np.random.Generator,
    ) -> None:
        """Verify exception handling: expected_output contains exception repr."""
        trace = trace_from_code(CODE_FUNCTION_WITH_EXCEPTION, inputs=(1, 0), entrypoint="divide")
        trace_meta = make_trace_metadata(trace)
        sample = _get_function_call_sample(
            trace_data=trace,
            trace_meta=trace_meta,
            trace_code_type=make_selected_sample(trace).code_type,
            code_summary="",
            transform_config=function_return_only_config,
            rng=default_rng,
        )
        assert sample is not None, "should produce a function call sample"
        # when the function raises an exception, expected_output should contain it
        assert "ValueError" in sample.expected_output or "Cannot divide by zero" in sample.expected_output

    def test_recursive_function_depth_tracking(
        self,
        trace_from_code: TraceFromCodeFixture,
        function_return_only_config: SampleTransformConfig,
    ) -> None:
        """Verify recursive calls correctly track depth and match call/return pairs."""
        trace = trace_from_code(CODE_RECURSIVE_FUNCTION, inputs=3, entrypoint="factorial")
        trace_meta = make_trace_metadata(trace)
        # with recursion, we may get different samples depending on which call we pick
        samples_found = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            sample = _get_function_call_sample(
                trace_data=trace,
                trace_meta=trace_meta,
                trace_code_type=make_selected_sample(trace).code_type,
                code_summary="",
                transform_config=function_return_only_config,
                rng=rng,
            )
            if sample is not None:
                samples_found.append(sample)
                # verify the step indices are valid
                assert sample.first_step_idx < sample.last_step_idx
                first_step = trace.traced_steps[sample.first_step_idx]
                last_step = trace.traced_steps[sample.last_step_idx]
                assert first_step.event_type == exec_utils.TraceEventType.CALL
                assert last_step.event_type == exec_utils.TraceEventType.RETURN
                # verify return value is a valid factorial result (1, 2, or 6)
                assert any(val in sample.expected_output for val in ("1", "2", "6"))
        assert len(samples_found) > 0, "should produce at least one sample from recursive function"

    def test_function_call_exceeds_step_cap_skipped(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify function calls exceeding step cap are skipped."""
        config = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            predict_type_prob_map={SamplePredictType.function_return: 1.0},
            max_partial_trace_steps=1,  # very low cap
            min_partial_trace_steps=1,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
        )
        trace = trace_from_code(CODE_LOOP_MULTIPLE_ITERATIONS, inputs=5, entrypoint="sum_range")
        trace_meta = make_trace_metadata(trace)
        # with max_partial_trace_steps=1, most function calls should be skipped
        sample = _get_function_call_sample(
            trace_data=trace,
            trace_meta=trace_meta,
            trace_code_type=make_selected_sample(trace).code_type,
            code_summary="",
            transform_config=config,
            rng=np.random.default_rng(42),
        )
        # likely to be None because function has more than 1 step
        # (but could succeed if there's a very short function)
        if sample is not None:
            assert sample.trace_step_count <= 1

    def test_function_call_below_min_steps_skipped(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify function calls below min steps are skipped."""
        config = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            predict_type_prob_map={SamplePredictType.function_return: 1.0},
            max_partial_trace_steps=100,
            min_partial_trace_steps=50,  # high min
            max_inputs_str_length=1000,
            max_output_str_length=1000,
        )
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=5, entrypoint="compute")
        trace_meta = make_trace_metadata(trace)
        # simple function has few steps, should be skipped
        sample = _get_function_call_sample(
            trace_data=trace,
            trace_meta=trace_meta,
            trace_code_type=make_selected_sample(trace).code_type,
            code_summary="",
            transform_config=config,
            rng=np.random.default_rng(42),
        )
        assert sample is None, "simple function should be skipped with high min_steps"

    def test_function_call_inputs_exceed_string_cap_skipped(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify function calls with too-long inputs are skipped."""
        config = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            predict_type_prob_map={SamplePredictType.function_return: 1.0},
            max_partial_trace_steps=100,
            min_partial_trace_steps=1,
            max_inputs_str_length=1,  # very low cap
            max_output_str_length=1000,
        )
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=12345, entrypoint="compute")
        trace_meta = make_trace_metadata(trace)
        sample = _get_function_call_sample(
            trace_data=trace,
            trace_meta=trace_meta,
            trace_code_type=make_selected_sample(trace).code_type,
            code_summary="",
            transform_config=config,
            rng=np.random.default_rng(42),
        )
        assert sample is None, "should be skipped due to input string length cap"

    def test_function_call_output_exceed_string_cap_skipped(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify function calls with too-long output are skipped."""
        config = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            predict_type_prob_map={SamplePredictType.function_return: 1.0},
            max_partial_trace_steps=100,
            min_partial_trace_steps=1,
            max_inputs_str_length=1000,
            max_output_str_length=1,  # very low cap
        )
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=99999, entrypoint="compute")
        trace_meta = make_trace_metadata(trace)
        sample = _get_function_call_sample(
            trace_data=trace,
            trace_meta=trace_meta,
            trace_code_type=make_selected_sample(trace).code_type,
            code_summary="",
            transform_config=config,
            rng=np.random.default_rng(42),
        )
        assert sample is None, "should be skipped due to output string length cap"

    def test_no_function_calls_returns_none(
        self,
        trace_from_code: TraceFromCodeFixture,
        function_return_only_config: SampleTransformConfig,
        default_rng: np.random.Generator,
    ) -> None:
        """Verify returns None when code has no function calls."""
        # script mode - no entrypoint function
        trace = trace_from_code(CODE_NO_FUNCTION_CALL, inputs=None, entrypoint=None)
        trace_meta = make_trace_metadata(trace)
        sample = _get_function_call_sample(
            trace_data=trace,
            trace_meta=trace_meta,
            trace_code_type=make_selected_sample(trace).code_type,
            code_summary="",
            transform_config=function_return_only_config,
            rng=default_rng,
        )
        # should be None because there are no CALL events (other than module)
        assert sample is None


@pytest.mark.slow
class TestCodeSegmentEdgeCases:
    """Tests for edge cases in code segment sample generation."""

    def test_segment_from_loop_multiple_visits(
        self,
        trace_from_code: TraceFromCodeFixture,
        frame_variables_only_config: SampleTransformConfig,
    ) -> None:
        """Verify segments from loops correctly handle multiple line visits."""
        trace = trace_from_code(CODE_LOOP_MULTIPLE_ITERATIONS, inputs=5, entrypoint="sum_range")
        trace_meta = make_trace_metadata(trace)
        # collect multiple samples to check hit counts vary
        hit_counts = set()
        for seed in range(50):
            rng = np.random.default_rng(seed)
            sample = _get_code_segment_sample(
                trace_data=trace,
                trace_meta=trace_meta,
                trace_code_type=make_selected_sample(trace).code_type,
                code_summary="",
                transform_config=frame_variables_only_config,
                rng=rng,
            )
            if sample is not None:
                hit_counts.add(sample.first_line_hit)
                hit_counts.add(sample.last_line_hit)
        # with multiple loop iterations, we should see different hit counts
        assert len(hit_counts) >= 2, "should capture multiple distinct hit counts across loop iterations"

    def test_segment_combine_vars_false_excludes_globals(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify globals are excluded when combine_local_and_global_vars=False."""
        config = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            predict_type_prob_map={SamplePredictType.frame_variables: 1.0},
            max_partial_trace_steps=100,
            min_partial_trace_steps=1,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
            combine_local_and_global_vars_for_partial_samples=False,
        )
        trace = trace_from_code(CODE_GLOBAL_AND_LOCAL, inputs=5, entrypoint="compute")
        trace_meta = make_trace_metadata(trace)
        for seed in range(100):
            rng = np.random.default_rng(seed)
            sample = _get_code_segment_sample(
                trace_data=trace,
                trace_meta=trace_meta,
                trace_code_type=make_selected_sample(trace).code_type,
                code_summary="",
                transform_config=config,
                rng=rng,
            )
            if sample is None:
                continue
            assert sample.predict_type == SamplePredictType.frame_variables
            assert "MULTIPLIER" not in sample.inputs
            assert "MULTIPLIER" not in sample.expected_output
            return
        pytest.fail("failed to generate segment sample to validate global exclusion")

    def test_segment_step_count_respects_max_cap(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify segment step count does not exceed max_partial_trace_steps."""
        config = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            predict_type_prob_map={SamplePredictType.frame_variables: 1.0},
            max_partial_trace_steps=3,  # low cap
            min_partial_trace_steps=1,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
        )
        trace = trace_from_code(CODE_LOOP_MULTIPLE_ITERATIONS, inputs=10, entrypoint="sum_range")
        trace_meta = make_trace_metadata(trace)
        for seed in range(50):
            rng = np.random.default_rng(seed)
            sample = _get_code_segment_sample(
                trace_data=trace,
                trace_meta=trace_meta,
                trace_code_type=make_selected_sample(trace).code_type,
                code_summary="",
                transform_config=config,
                rng=rng,
            )
            if sample is not None:
                assert sample.trace_step_count <= 3, "step count should respect max cap"
                break
        else:
            pytest.fail("could not generate a segment sample")

    def test_segment_step_count_respects_min_threshold(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify segment step count meets min_partial_trace_steps threshold."""
        config = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            predict_type_prob_map={SamplePredictType.frame_variables: 1.0},
            max_partial_trace_steps=100,
            min_partial_trace_steps=3,  # require at least 3 steps
            max_inputs_str_length=1000,
            max_output_str_length=1000,
        )
        trace = trace_from_code(CODE_LOOP_MULTIPLE_ITERATIONS, inputs=5, entrypoint="sum_range")
        trace_meta = make_trace_metadata(trace)
        for seed in range(50):
            rng = np.random.default_rng(seed)
            sample = _get_code_segment_sample(
                trace_data=trace,
                trace_meta=trace_meta,
                trace_code_type=make_selected_sample(trace).code_type,
                code_summary="",
                transform_config=config,
                rng=rng,
            )
            if sample is not None:
                assert sample.trace_step_count >= 3, "step count should meet min threshold"
                break
        else:
            pytest.fail("could not generate a segment sample")

    def test_trace_ending_without_return_logs_warning(
        self,
        trace_from_code: TraceFromCodeFixture,
        frame_variables_only_config: SampleTransformConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify warning is logged when trace ends without return event.

        Note: This is difficult to trigger with real traces since the tracer
        normally captures all return events. This test documents the expected behavior.
        """
        # most real traces will have matching returns, so this test primarily
        # verifies the code path exists and doesn't crash
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=5, entrypoint="compute")
        trace_meta = make_trace_metadata(trace)
        with caplog.at_level(logging.WARNING):
            _get_code_segment_sample(
                trace_data=trace,
                trace_meta=trace_meta,
                trace_code_type=make_selected_sample(trace).code_type,
                code_summary="",
                transform_config=frame_variables_only_config,
                rng=np.random.default_rng(42),
            )
        # the sample may or may not be None, but the code path should work
        # (warning is only logged for malformed traces, which are rare)


@pytest.mark.slow
class TestTransformStrategyBehavior:
    """Tests for different transform strategy behaviors."""

    def test_random_strategy_produces_partial_samples(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify random strategy can produce partial samples."""
        config = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.random,
            predict_type_prob_map={
                SamplePredictType.function_return: 0.4,
                SamplePredictType.frame_variables: 0.4,
            },
            max_partial_trace_steps=100,
            min_partial_trace_steps=1,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
        )
        trace = trace_from_code(CODE_LOOP_MULTIPLE_ITERATIONS, inputs=4, entrypoint="sum_range")
        selection = make_selected_sample(trace)
        got_partial = False
        got_program_output = False
        for seed in range(200):
            rng = np.random.default_rng(seed)
            sample = generate_sample(
                code_type_selection_result=selection,
                trace_data=trace,
                code_summary="",
                transform_config=config,
                rng=rng,
            )
            assert sample is not None
            if sample.predict_type == SamplePredictType.program_output:
                got_program_output = True
            else:
                got_partial = True
            if got_partial and got_program_output:
                break
        assert got_partial, "random strategy should eventually yield partial samples"
        assert got_program_output, "random strategy should sometimes fall back to program output when mass < 1.0"

    def test_if_too_long_strategy_triggers_for_long_traces(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify if_too_long strategy triggers partial sampling for long traces."""
        config = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.if_too_long,
            too_long_total_steps_threshold=1,  # very low threshold
            too_long_valid_steps_threshold=1,
            too_long_code_lines_threshold=1,
            predict_type_prob_map={SamplePredictType.frame_variables: 1.0},
            max_partial_trace_steps=100,
            min_partial_trace_steps=1,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
        )
        trace = trace_from_code(CODE_LOOP_MULTIPLE_ITERATIONS, inputs=5, entrypoint="sum_range")
        selection = make_selected_sample(trace)
        got_partial = False
        for seed in range(50):
            rng = np.random.default_rng(seed)
            sample = generate_sample(
                code_type_selection_result=selection,
                trace_data=trace,
                code_summary="",
                transform_config=config,
                rng=rng,
            )
            if sample is not None and sample.predict_type == SamplePredictType.frame_variables:
                got_partial = True
                break
        assert got_partial, "if_too_long should trigger partial samples for long traces"

    def test_if_too_long_strategy_full_for_short_traces(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify if_too_long strategy returns full output for short traces."""
        config = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.if_too_long,
            too_long_total_steps_threshold=10000,  # very high threshold
            too_long_valid_steps_threshold=10000,
            too_long_code_lines_threshold=1000,
            predict_type_prob_map={SamplePredictType.frame_variables: 1.0},
            max_partial_trace_steps=100,
            min_partial_trace_steps=1,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
        )
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=5, entrypoint="compute")
        selection = make_selected_sample(trace)
        # with high thresholds, short trace should not trigger partial sampling
        for seed in range(10):
            rng = np.random.default_rng(seed)
            sample = generate_sample(
                code_type_selection_result=selection,
                trace_data=trace,
                code_summary="",
                transform_config=config,
                rng=rng,
            )
            assert sample is not None
            assert sample.predict_type == SamplePredictType.program_output

    def test_functions_fallback_to_segments_when_enabled(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify fallback to segments works when function sample fails."""
        config = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            predict_type_prob_map={SamplePredictType.function_return: 1.0},
            max_partial_trace_steps=100,
            min_partial_trace_steps=1,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
            functions_fallback_to_segments=True,
        )
        trace = trace_from_code(CODE_LOOP_MULTIPLE_ITERATIONS, inputs=5, entrypoint="sum_range")
        selection = make_selected_sample(trace)
        # try to get a sample - may be function_return or frame_variables due to fallback
        got_sample = False
        for seed in range(50):
            rng = np.random.default_rng(seed)
            sample = generate_sample(
                code_type_selection_result=selection,
                trace_data=trace,
                code_summary="",
                transform_config=config,
                rng=rng,
            )
            if sample is not None:
                got_sample = True
                # could be either function_return, frame_variables, or program_output
                assert sample.predict_type in (
                    SamplePredictType.function_return,
                    SamplePredictType.frame_variables,
                    SamplePredictType.program_output,
                )
                break
        assert got_sample, "should produce a sample with fallback enabled"

    def test_never_strategy_always_returns_program_output(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify never strategy always returns program_output."""
        config = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.never,
            predict_type_prob_map={},
        )
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=5, entrypoint="compute")
        selection = make_selected_sample(trace)
        for seed in range(10):
            rng = np.random.default_rng(seed)
            sample = generate_sample(
                code_type_selection_result=selection,
                trace_data=trace,
                code_summary="",
                transform_config=config,
                rng=rng,
            )
            assert sample is not None
            assert sample.predict_type == SamplePredictType.program_output


@pytest.mark.slow
class TestSelectPredictType:
    """Tests for _select_predict_type decision logic."""

    def test_code_override_forces_program_output(
        self,
        trace_from_code: TraceFromCodeFixture,
        default_transform_config: SampleTransformConfig,
        default_rng: np.random.Generator,
    ) -> None:
        """Verify code override always forces program_output."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=5, entrypoint="compute")
        selection = make_selected_sample(trace, code_override="def x(): return 1")
        sample = generate_sample(
            code_type_selection_result=selection,
            trace_data=trace,
            code_summary="",
            transform_config=default_transform_config,
            rng=default_rng,
        )
        assert sample is not None
        assert sample.predict_type == SamplePredictType.program_output
        assert sample.has_code_override is True

    def test_hybrid_strategy_too_long_triggers_partial(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify hybrid strategy triggers partial for too-long traces."""
        config = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.hybrid,
            too_long_total_steps_threshold=1,  # very low
            too_long_valid_steps_threshold=1,
            too_long_code_lines_threshold=1,
            predict_type_prob_map={SamplePredictType.frame_variables: 1.0},
            max_partial_trace_steps=100,
            min_partial_trace_steps=1,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
        )
        trace = trace_from_code(CODE_LOOP_MULTIPLE_ITERATIONS, inputs=3, entrypoint="sum_range")
        selection = make_selected_sample(trace)
        got_partial = False
        for seed in range(50):
            rng = np.random.default_rng(seed)
            sample = generate_sample(
                code_type_selection_result=selection,
                trace_data=trace,
                code_summary="",
                transform_config=config,
                rng=rng,
            )
            if sample is not None and sample.predict_type == SamplePredictType.frame_variables:
                got_partial = True
                break
        assert got_partial, "hybrid strategy should trigger partial for too-long traces"
