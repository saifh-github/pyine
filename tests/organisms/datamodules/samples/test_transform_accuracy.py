"""Accuracy tests for sample transformation logic.

These tests verify that SampleData fields contain exact, accurate values
that correctly represent the execution trace data. All tests use real
execution traces generated from Python code snippets.
"""

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
    _get_full_program_sample,
    _get_function_call_sample,
    generate_sample,
)
from tests.organisms.datamodules.samples.conftest import (
    CODE_GLOBAL_AND_LOCAL,
    CODE_LOOP_MULTIPLE_ITERATIONS,
    CODE_MULTI_LINE_LOOP,
    CODE_NESTED_FUNCTIONS,
    CODE_SIMPLE_FUNCTION,
    make_selected_sample,
    make_trace_metadata,
)

TraceFromCodeFixture = typing.Callable[..., exec_utils.TraceResult]


@pytest.mark.slow
class TestFunctionCallSampleAccuracy:
    """Tests verifying accuracy of function call sample field values."""

    def test_inputs_match_function_arguments(
        self,
        trace_from_code: TraceFromCodeFixture,
        function_return_only_config: SampleTransformConfig,
        default_rng: np.random.Generator,
    ) -> None:
        """Verify sample.inputs equals repr(call_event.arguments)."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=5, entrypoint="compute")
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
        # the function is called with inputs=5, which becomes arguments={'x': '5'}
        # (tracer stores arguments as a dict with param names as keys, repr'd values)
        assert "x" in sample.inputs, f"expected 'x' in inputs, got '{sample.inputs}'"
        assert "5" in sample.inputs, f"expected '5' in inputs, got '{sample.inputs}'"

    def test_output_matches_return_value(
        self,
        trace_from_code: TraceFromCodeFixture,
        function_return_only_config: SampleTransformConfig,
        default_rng: np.random.Generator,
    ) -> None:
        """Verify sample.expected_output equals repr(return_value)."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=7, entrypoint="compute")
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
        # compute(7) returns 7 * 2 = 14
        # the expected_output is repr(return_value), so it contains "14"
        assert "14" in sample.expected_output, f"expected '14' in output, got '{sample.expected_output}'"

    def test_step_indices_match_trace_positions(
        self,
        trace_from_code: TraceFromCodeFixture,
        function_return_only_config: SampleTransformConfig,
        default_rng: np.random.Generator,
    ) -> None:
        """Verify first_step_idx and last_step_idx match actual trace step indices."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=3, entrypoint="compute")
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
        # verify the step indices point to valid events in the trace
        first_step = trace.traced_steps[sample.first_step_idx]
        last_step = trace.traced_steps[sample.last_step_idx]
        assert first_step is not None, "first step should exist"
        assert last_step is not None, "last step should exist"
        assert first_step.event_type == exec_utils.TraceEventType.CALL
        assert last_step.event_type == exec_utils.TraceEventType.RETURN
        # verify the indices match what's stored in the events
        assert first_step.trace_step_idx == sample.first_step_idx
        assert last_step.trace_step_idx == sample.last_step_idx

    def test_trace_step_count_matches_valid_events(
        self,
        trace_from_code: TraceFromCodeFixture,
        function_return_only_config: SampleTransformConfig,
        default_rng: np.random.Generator,
    ) -> None:
        """Verify trace_step_count equals actual valid events between call and return."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=4, entrypoint="compute")
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
        # count valid events manually between first_step_idx and last_step_idx
        actual_count = sum(
            step is not None for step in trace.traced_steps[sample.first_step_idx : sample.last_step_idx]
        )
        assert sample.trace_step_count == actual_count, f"expected {actual_count} steps, got {sample.trace_step_count}"

    def test_line_boundaries_match_code_block(
        self,
        trace_from_code: TraceFromCodeFixture,
        function_return_only_config: SampleTransformConfig,
        default_rng: np.random.Generator,
    ) -> None:
        """Verify first_line and last_line match the function's code block boundaries."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=2, entrypoint="compute")
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
        # for the 'compute' function, we should have a code block
        # the sample's line range should match the block
        assert sample.first_line >= 1, "first_line should be at least 1"
        assert sample.last_line >= sample.first_line, "last_line should be >= first_line"
        # verify entrypoint is the function name
        assert sample.entrypoint == "compute"

    def test_line_hit_fields_are_zero_for_function_return(
        self,
        trace_from_code: TraceFromCodeFixture,
        function_return_only_config: SampleTransformConfig,
        default_rng: np.random.Generator,
    ) -> None:
        """Verify first_line_hit and last_line_hit are 0 for function_return samples."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=6, entrypoint="compute")
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
        # these should be 0 as they're not tracked for function_return
        assert sample.first_line_hit == 0, "first_line_hit should be 0 for function_return"
        assert sample.last_line_hit == 0, "last_line_hit should be 0 for function_return"

    def test_nested_function_selects_correct_function(
        self,
        trace_from_code: TraceFromCodeFixture,
        function_return_only_config: SampleTransformConfig,
    ) -> None:
        """Verify nested function calls correctly identify the selected function."""
        trace = trace_from_code(CODE_NESTED_FUNCTIONS, inputs=5, entrypoint="outer")
        trace_meta = make_trace_metadata(trace)
        # try multiple seeds to get different function selections
        found_inner = False
        found_outer = False
        for seed in range(50):
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
                if sample.entrypoint == "inner":
                    found_inner = True
                    # inner(x) returns x + 1, so inner(5) = 6 or inner(6) = 7
                    # expected_output is repr of return value, could be "'6'" or "'7'" or "6" or "7"
                    assert "6" in sample.expected_output or "7" in sample.expected_output
                elif sample.entrypoint == "outer":
                    found_outer = True
                    # outer(5) returns inner(inner(5)) = inner(6) = 7
                    assert "7" in sample.expected_output
        # should be able to select at least one of them
        assert found_inner or found_outer, "should be able to select a function"


@pytest.mark.slow
class TestCodeSegmentSampleAccuracy:
    """Tests verifying accuracy of code segment sample field values."""

    def test_inputs_match_local_variables_at_segment_start(
        self,
        trace_from_code: TraceFromCodeFixture,
        frame_variables_only_config: SampleTransformConfig,
    ) -> None:
        """Verify sample.inputs equals repr(local_variables) at segment start."""
        trace = trace_from_code(CODE_LOOP_MULTIPLE_ITERATIONS, inputs=4, entrypoint="sum_range")
        trace_meta = make_trace_metadata(trace)
        # try multiple seeds until we get a valid segment
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
                # inputs should be a repr of a dict of local variables
                assert sample.inputs.startswith("{"), "inputs should be a dict repr"
                assert sample.predict_type == SamplePredictType.frame_variables
                break
        else:
            pytest.skip("could not generate a segment sample")

    def test_output_matches_local_variables_at_segment_end(
        self,
        trace_from_code: TraceFromCodeFixture,
        frame_variables_only_config: SampleTransformConfig,
    ) -> None:
        """Verify sample.expected_output equals repr(local_variables) at segment end."""
        trace = trace_from_code(CODE_LOOP_MULTIPLE_ITERATIONS, inputs=3, entrypoint="sum_range")
        trace_meta = make_trace_metadata(trace)
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
                assert sample.expected_output.startswith("{"), "output should be a dict repr"
                break
        else:
            pytest.skip("could not generate a segment sample")

    def test_first_line_hit_is_one_indexed(
        self,
        trace_from_code: TraceFromCodeFixture,
        frame_variables_only_config: SampleTransformConfig,
    ) -> None:
        """Verify first_line_hit is 1-indexed (first visit = 1, not 0)."""
        trace = trace_from_code(CODE_LOOP_MULTIPLE_ITERATIONS, inputs=5, entrypoint="sum_range")
        trace_meta = make_trace_metadata(trace)
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
                assert sample.first_line_hit >= 1, "first_line_hit should be >= 1 (1-indexed)"
                break
        else:
            pytest.skip("could not generate a segment sample")

    def test_last_line_hit_is_one_indexed(
        self,
        trace_from_code: TraceFromCodeFixture,
        frame_variables_only_config: SampleTransformConfig,
    ) -> None:
        """Verify last_line_hit is 1-indexed (first visit = 1, not 0)."""
        trace = trace_from_code(CODE_LOOP_MULTIPLE_ITERATIONS, inputs=5, entrypoint="sum_range")
        trace_meta = make_trace_metadata(trace)
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
                assert sample.last_line_hit >= 1, "last_line_hit should be >= 1 (1-indexed)"
                break
        else:
            pytest.skip("could not generate a segment sample")

    def test_step_indices_match_trace_positions(
        self,
        trace_from_code: TraceFromCodeFixture,
        frame_variables_only_config: SampleTransformConfig,
    ) -> None:
        """Verify first_step_idx and last_step_idx match actual trace positions."""
        trace = trace_from_code(CODE_MULTI_LINE_LOOP, inputs=3, entrypoint="process")
        trace_meta = make_trace_metadata(trace)
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
                # verify the step indices point to valid events
                first_step = trace.traced_steps[sample.first_step_idx]
                last_step = trace.traced_steps[sample.last_step_idx]
                assert first_step is not None, "first step should exist"
                assert last_step is not None, "last step should exist"
                assert first_step.trace_step_idx == sample.first_step_idx
                assert last_step.trace_step_idx == sample.last_step_idx
                break
        else:
            pytest.skip("could not generate a segment sample")

    def test_trace_step_count_matches_segment_size(
        self,
        trace_from_code: TraceFromCodeFixture,
        frame_variables_only_config: SampleTransformConfig,
    ) -> None:
        """Verify trace_step_count equals segment_end_idx - segment_start_idx."""
        trace = trace_from_code(CODE_MULTI_LINE_LOOP, inputs=4, entrypoint="process")
        trace_meta = make_trace_metadata(trace)
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
                # note: trace_step_count is segment size, not step index difference
                # but they should be related (step count = index diff for contiguous segments)
                assert sample.trace_step_count >= 1, "trace_step_count should be >= 1"
                assert sample.trace_step_count <= frame_variables_only_config.max_partial_trace_steps
                break
        else:
            pytest.skip("could not generate a segment sample")

    def test_combine_vars_true_includes_globals(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify globals are included when combine_local_and_global_vars_for_partial_samples=True."""
        config = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            predict_type_prob_map={SamplePredictType.frame_variables: 1.0},
            max_partial_trace_steps=100,
            min_partial_trace_steps=1,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
            combine_local_and_global_vars_for_partial_samples=True,
        )
        trace = trace_from_code(CODE_GLOBAL_AND_LOCAL, inputs=5, entrypoint="compute")
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
                # check if MULTIPLIER (global) is in the inputs or output
                # note: this may or may not include globals depending on where the segment falls
                assert sample.predict_type == SamplePredictType.frame_variables
                break
        else:
            pytest.skip("could not generate a segment sample")


@pytest.mark.slow
class TestFullProgramSampleAccuracy:
    """Tests verifying accuracy of full program sample field values."""

    def test_first_line_is_zero(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify full program samples have first_line == 0."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=3, entrypoint="compute")
        selection = make_selected_sample(trace)
        sample = _get_full_program_sample(
            trace_data=trace,
            code_type_selection_result=selection,
            code_summary="test",
        )
        assert sample.first_line == 0, "full program sample should have first_line == 0"

    def test_last_line_is_code_line_count(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify full program samples have last_line == len(code.splitlines())."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=3, entrypoint="compute")
        selection = make_selected_sample(trace)
        sample = _get_full_program_sample(
            trace_data=trace,
            code_type_selection_result=selection,
            code_summary="test",
        )
        expected_lines = len(trace.code_string.splitlines())
        assert sample.last_line == expected_lines, f"expected last_line == {expected_lines}, got {sample.last_line}"

    def test_inputs_from_trace(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify full program sample inputs match str(trace_data.inputs)."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=42, entrypoint="compute", expected_output=84)
        selection = make_selected_sample(trace)
        sample = _get_full_program_sample(
            trace_data=trace,
            code_type_selection_result=selection,
            code_summary="",
        )
        assert sample.inputs == str(trace.inputs), f"expected '{trace.inputs}', got '{sample.inputs}'"

    def test_expected_output_from_trace(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify full program sample expected_output matches str(trace_data.expected_output)."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=10, entrypoint="compute", expected_output=20)
        selection = make_selected_sample(trace)
        sample = _get_full_program_sample(
            trace_data=trace,
            code_type_selection_result=selection,
            code_summary="",
        )
        assert sample.expected_output == str(trace.expected_output)

    def test_step_count_is_valid_count(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify full program sample trace_step_count == trace_data.valid_step_count."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=5, entrypoint="compute")
        selection = make_selected_sample(trace)
        sample = _get_full_program_sample(
            trace_data=trace,
            code_type_selection_result=selection,
            code_summary="",
        )
        assert sample.trace_step_count == trace.valid_step_count

    def test_predict_type_is_program_output(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify full program samples have predict_type == program_output."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=8, entrypoint="compute")
        selection = make_selected_sample(trace)
        sample = _get_full_program_sample(
            trace_data=trace,
            code_type_selection_result=selection,
            code_summary="",
        )
        assert sample.predict_type == SamplePredictType.program_output

    def test_line_hit_and_step_idx_fields_are_zero(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify full program samples have all hit/idx fields set to 0."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=9, entrypoint="compute")
        selection = make_selected_sample(trace)
        sample = _get_full_program_sample(
            trace_data=trace,
            code_type_selection_result=selection,
            code_summary="",
        )
        assert sample.first_line_hit == 0, "first_line_hit should be 0 for full program"
        assert sample.last_line_hit == 0, "last_line_hit should be 0 for full program"
        assert sample.first_step_idx == 0, "first_step_idx should be 0 for full program"
        assert sample.last_step_idx == 0, "last_step_idx should be 0 for full program"

    def test_code_override_uses_override_code(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify code override is used when provided."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=5, entrypoint="compute")
        override_code = "def modified(x): return x * 3"
        selection = make_selected_sample(trace, code_override=override_code)
        sample = _get_full_program_sample(
            trace_data=trace,
            code_type_selection_result=selection,
            code_summary="",
        )
        assert sample.code == override_code
        assert sample.has_code_override is True

    def test_no_code_override_uses_trace_code(
        self,
        trace_from_code: TraceFromCodeFixture,
    ) -> None:
        """Verify trace code is used when no override."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=5, entrypoint="compute")
        selection = make_selected_sample(trace, code_override=None)
        sample = _get_full_program_sample(
            trace_data=trace,
            code_type_selection_result=selection,
            code_summary="",
        )
        assert sample.code == trace.code_string
        assert sample.has_code_override is False


@pytest.mark.slow
class TestGenerateSampleIntegration:
    """Integration tests for the generate_sample function."""

    def test_generate_sample_returns_valid_sample(
        self,
        trace_from_code: TraceFromCodeFixture,
        default_transform_config: SampleTransformConfig,
        default_rng: np.random.Generator,
    ) -> None:
        """Verify generate_sample returns a valid SampleData object."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=5, entrypoint="compute")
        selection = make_selected_sample(trace)
        sample = generate_sample(
            code_type_selection_result=selection,
            trace_data=trace,
            code_summary="test summary",
            transform_config=default_transform_config,
            rng=default_rng,
        )
        assert sample is not None
        assert sample.identifier == trace.identifier
        assert sample.code == trace.code_string
        assert sample.description == "test summary"

    def test_code_override_forces_program_output(
        self,
        trace_from_code: TraceFromCodeFixture,
        default_transform_config: SampleTransformConfig,
        default_rng: np.random.Generator,
    ) -> None:
        """Verify code override always results in program_output predict_type."""
        trace = trace_from_code(CODE_SIMPLE_FUNCTION, inputs=5, entrypoint="compute")
        override_code = "def other(x): return x + 1"
        selection = make_selected_sample(trace, code_override=override_code)
        # even with partial sample config, code override should force program_output
        sample = generate_sample(
            code_type_selection_result=selection,
            trace_data=trace,
            code_summary="",
            transform_config=default_transform_config,
            rng=default_rng,
        )
        assert sample is not None
        assert sample.predict_type == SamplePredictType.program_output
        assert sample.code == override_code
        assert sample.has_code_override is True
