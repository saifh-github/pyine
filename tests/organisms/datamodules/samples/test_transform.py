"""Tests for sample transformation logic in the samples module."""

import numpy as np
import pytest
import pytest_mock

import pyine.utils.code.execution as exec_utils
from pyine.organisms.datamodules.samples.common import (
    SampleCodeType,
    SampleCodeTypeSet,
    SamplePredictType,
    SampleTransformStrategy,
)
from pyine.organisms.datamodules.samples.configs import SampleTransformConfig
from pyine.organisms.datamodules.samples.selection import SelectedSample
from pyine.organisms.datamodules.samples.transform import (
    SelectedPredictTypeResult,
    _get_code_segment_sample,
    _get_function_call_sample,
    generate_sample,
)
from tests.utils.fake_dataset_readers import FakeTraceDataConfig, FakeTraceDatasetReader


@pytest.fixture
def small_fake_reader() -> FakeTraceDatasetReader:
    """Provides a small fake dataset reader for transform tests."""
    cfg = FakeTraceDataConfig(
        dataset_name="FAKE",
        subset_name="transform_test",
        num_problems=1,
        solutions_per_problem=2,
        tests_per_problem=2,
        augmented_per_solution=0,
        entrypoint_name="solution",
        max_events_per_line=50,
        seed=321,
        code_kind="function",
    )
    return FakeTraceDatasetReader(config=cfg)


def make_selected_sample(
    reader: FakeTraceDatasetReader,
    idx: int,
    code_type: SampleCodeTypeSet | None = None,
    code_override: str | None = None,
) -> SelectedSample:
    """Helper to create a SelectedSample from a fake reader."""
    trace_meta = reader.trace_metadata[idx]
    trace_id = trace_meta.trace_id
    parent_id = trace_id.get_augmentless_identifier()
    return SelectedSample(
        parent_id=parent_id,
        trace_id=trace_id,
        trace_meta=trace_meta,
        code_type=code_type or SampleCodeTypeSet.create_default(),
        code_override=code_override,
    )


class TestSelectedPredictTypeResult:
    """Tests for the SelectedPredictTypeResult dataclass."""

    def test_should_skip_when_predict_type_is_none(self) -> None:
        class FakeTrace:
            valid_step_count = 10
            total_step_count = 10
            code_string = "x = 1"

        result = SelectedPredictTypeResult(
            trace_data=FakeTrace(),
            is_too_long=False,
            randomly_picked_partial_sample=False,
            forced_full_output_due_to_code_override=False,
            predict_type=None,
        )
        assert result.should_skip

    def test_is_partial_sample_for_frame_variables(self) -> None:
        class FakeTrace:
            valid_step_count = 10
            total_step_count = 10
            code_string = "x = 1"

        result = SelectedPredictTypeResult(
            trace_data=FakeTrace(),
            is_too_long=False,
            randomly_picked_partial_sample=False,
            forced_full_output_due_to_code_override=False,
            predict_type=SamplePredictType.frame_variables,
        )
        assert result.is_partial_sample

    def test_is_not_partial_sample_for_program_output(self) -> None:
        class FakeTrace:
            valid_step_count = 10
            total_step_count = 10
            code_string = "x = 1"

        result = SelectedPredictTypeResult(
            trace_data=FakeTrace(),
            is_too_long=False,
            randomly_picked_partial_sample=False,
            forced_full_output_due_to_code_override=False,
            predict_type=SamplePredictType.program_output,
        )
        assert not result.is_partial_sample


class TestGenerateSample:
    """Tests for the generate_sample function."""

    def test_full_program_output_with_never_strategy(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        selection = make_selected_sample(small_fake_reader, 0)
        trace_data = small_fake_reader[0]
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.never,
            predict_type_prob_map={},
        )
        rng = np.random.default_rng(0)
        sample = generate_sample(
            code_type_selection_result=selection,
            trace_data=trace_data,
            code_summary="Test description",
            transform_config=cfg,
            rng=rng,
        )
        assert sample is not None
        assert sample.predict_type == SamplePredictType.program_output
        assert sample.identifier == trace_data.identifier
        assert sample.code == trace_data.code_string
        assert sample.description == "Test description"
        assert sample.first_line == 0
        assert sample.last_line == len(trace_data.code_string.splitlines())

    def test_code_override_forces_program_output(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        override_code = "def modified(x): return x + 1"
        selection = make_selected_sample(
            small_fake_reader,
            0,
            code_type=SampleCodeTypeSet(frozenset({SampleCodeType.hinted})),
            code_override=override_code,
        )
        trace_data = small_fake_reader[0]
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            predict_type_prob_map={SamplePredictType.frame_variables: 1.0},
        )
        rng = np.random.default_rng(0)
        sample = generate_sample(
            code_type_selection_result=selection,
            trace_data=trace_data,
            code_summary=None,
            transform_config=cfg,
            rng=rng,
        )
        assert sample is not None
        assert sample.predict_type == SamplePredictType.program_output
        assert sample.code == override_code
        assert sample.has_code_override

    def test_partial_sample_frame_variables(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        selection = make_selected_sample(small_fake_reader, 0)
        trace_data = small_fake_reader[0]
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            max_partial_trace_steps=10,
            min_partial_trace_steps=1,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
            predict_type_prob_map={SamplePredictType.frame_variables: 1.0},
        )
        # try multiple seeds to verify we can produce frame_variables at least sometimes
        got_frame_variables = False
        for seed in range(20):
            rng = np.random.default_rng(seed)
            sample = generate_sample(
                code_type_selection_result=selection,
                trace_data=trace_data,
                code_summary="",
                transform_config=cfg,
                rng=rng,
            )
            assert sample is not None
            assert sample.predict_type in (SamplePredictType.frame_variables, SamplePredictType.program_output)
            if sample.predict_type == SamplePredictType.frame_variables:
                got_frame_variables = True
                assert sample.trace_step_count <= 10
                assert sample.trace_step_count >= 1
        assert got_frame_variables, "should produce at least one frame_variables sample across seeds"

    def test_partial_sample_function_return(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        selection = make_selected_sample(small_fake_reader, 0)
        trace_data = small_fake_reader[0]
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            max_partial_trace_steps=100,
            min_partial_trace_steps=1,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
            predict_type_prob_map={SamplePredictType.function_return: 1.0},
            functions_fallback_to_segments=True,
        )
        # try multiple seeds to verify we can produce function_return or frame_variables
        got_partial = False
        for seed in range(20):
            rng = np.random.default_rng(seed)
            sample = generate_sample(
                code_type_selection_result=selection,
                trace_data=trace_data,
                code_summary="",
                transform_config=cfg,
                rng=rng,
            )
            assert sample is not None
            # with functions_fallback_to_segments=True, may get frame_variables instead
            if sample.predict_type in (SamplePredictType.function_return, SamplePredictType.frame_variables):
                got_partial = True
        assert got_partial, "should produce at least one partial sample across seeds"

    def test_fallback_to_full_when_partial_caps_exceeded(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        selection = make_selected_sample(small_fake_reader, 0)
        trace_data = small_fake_reader[0]
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            max_inputs_str_length=0,
            max_output_str_length=0,
            predict_type_prob_map={SamplePredictType.frame_variables: 1.0},
            fallback_to_orig=True,
        )
        rng = np.random.default_rng(0)
        sample = generate_sample(
            code_type_selection_result=selection,
            trace_data=trace_data,
            code_summary="",
            transform_config=cfg,
            rng=rng,
        )
        assert sample is not None
        assert sample.predict_type == SamplePredictType.program_output

    def test_sample_tags_include_metadata(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        selection = make_selected_sample(small_fake_reader, 0)
        trace_data = small_fake_reader[0]
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.never,
            predict_type_prob_map={},
        )
        rng = np.random.default_rng(0)
        sample = generate_sample(
            code_type_selection_result=selection,
            trace_data=trace_data,
            code_summary="description here",
            transform_config=cfg,
            rng=rng,
        )
        assert sample is not None
        tags = sample.get_tag_list()
        assert any("sample_predict_type:" in t for t in tags)
        assert any("sample_code_type:" in t for t in tags)
        assert any("sample_code_description:" in t for t in tags)


class TestPrivateFunctionCallSample:
    """Tests for the _get_function_call_sample private function."""

    class _FakeKey:
        def __init__(self, obj: str, line: int) -> None:
            self.object = obj
            self.line = line

        def __str__(self) -> str:
            return f"{self.object}:{self.line}"

    class _FakeEvent:
        def __init__(
            self,
            event_type: exec_utils.TraceEventType,
            step_idx: int,
            key: "TestPrivateFunctionCallSample._FakeKey",
            local_vars: dict | None = None,
            global_vars: dict | None = None,
            arguments: object | None = None,
            return_value: object | None = None,
            exception: object | None = None,
        ) -> None:
            self.event_type = event_type
            self.trace_step_idx = step_idx
            self.trace_key = key
            self.local_variables = local_vars or {}
            self.global_variables = global_vars or {}
            self.arguments = arguments
            self.return_value = return_value
            self.exception = exception

    class _FakeTrace:
        def __init__(self, steps: list) -> None:
            self.identifier = "fake/trace/1"
            self.code_string = "x = foo(2) + 1\nprint(x)\n"
            self.code_blocks = {}
            self.inputs = ""
            self.expected_output = ""
            self.traced_steps = steps

        @property
        def valid_step_count(self) -> int:
            return len([s for s in self.traced_steps if s is not None])

    @pytest.fixture
    def nested_call_trace(self) -> _FakeTrace:
        steps = [
            self._FakeEvent(exec_utils.TraceEventType.CALL, 0, self._FakeKey("main", 1)),
            self._FakeEvent(
                exec_utils.TraceEventType.LINE,
                step_idx=1,
                key=self._FakeKey("main", 2),
                local_vars={"x": 1},
            ),
            self._FakeEvent(
                exec_utils.TraceEventType.CALL,
                step_idx=2,
                key=self._FakeKey("foo", 3),
                arguments=(1,),
            ),
            self._FakeEvent(
                exec_utils.TraceEventType.LINE,
                step_idx=3,
                key=self._FakeKey("foo", 101),
                local_vars={"y": 2},
            ),
            self._FakeEvent(
                exec_utils.TraceEventType.RETURN,
                step_idx=4,
                key=self._FakeKey("foo", 103),
                return_value=5,
            ),
            self._FakeEvent(
                exec_utils.TraceEventType.LINE,
                step_idx=5,
                key=self._FakeKey("main", 5),
                local_vars={"x": 6},
            ),
            self._FakeEvent(
                exec_utils.TraceEventType.RETURN,
                step_idx=6,
                key=self._FakeKey("main", 6),
            ),
        ]
        return self._FakeTrace(steps=steps)

    def test_function_call_sample_basic(
        self,
        nested_call_trace: _FakeTrace,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            predict_type_prob_map={SamplePredictType.function_return: 1.0},
            min_partial_trace_steps=1,
        )
        sample = _get_function_call_sample(
            trace_data=nested_call_trace,
            trace_meta=mocker.MagicMock(),
            trace_code_type=SampleCodeTypeSet.create_default(),
            code_summary="potato",
            transform_config=cfg,
            rng=np.random.default_rng(seed=0),
        )
        assert sample is not None
        assert sample.identifier == nested_call_trace.identifier
        assert sample.code == nested_call_trace.code_string
        assert sample.description == "potato"
        assert sample.predict_type == SamplePredictType.function_return
        assert sample.entrypoint == "foo"
        assert sample.inputs == "(1,)"
        assert sample.expected_output == "5"
        assert sample.trace_step_count == 2


class TestPrivateCodeSegmentSample:
    """Tests for the _get_code_segment_sample private function."""

    def test_code_segment_sample_returns_frame_variables(
        self,
        small_fake_reader: FakeTraceDatasetReader,
    ) -> None:
        trace_data = small_fake_reader[0]
        trace_meta = small_fake_reader.trace_metadata[0]
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            predict_type_prob_map={SamplePredictType.frame_variables: 1.0},
            min_partial_trace_steps=1,
            max_partial_trace_steps=5,
        )
        sample = _get_code_segment_sample(
            trace_data=trace_data,
            trace_meta=trace_meta,
            trace_code_type=SampleCodeTypeSet.create_default(),
            code_summary="test summary",
            transform_config=cfg,
            rng=np.random.default_rng(seed=42),
        )
        if sample is not None:
            assert sample.predict_type == SamplePredictType.frame_variables
            assert sample.trace_step_count <= 5
            assert sample.trace_step_count >= 1


class TestTransformStrategies:
    """Tests for different transform strategies."""

    def test_never_strategy_always_returns_full_output(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        selection = make_selected_sample(small_fake_reader, 0)
        trace_data = small_fake_reader[0]
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.never,
            predict_type_prob_map={},
        )
        for seed in range(10):
            rng = np.random.default_rng(seed)
            sample = generate_sample(
                code_type_selection_result=selection,
                trace_data=trace_data,
                code_summary="",
                transform_config=cfg,
                rng=rng,
            )
            assert sample is not None
            assert sample.predict_type == SamplePredictType.program_output

    def test_always_strategy_attempts_partial_sample(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        selection = make_selected_sample(small_fake_reader, 0)
        trace_data = small_fake_reader[0]
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            max_partial_trace_steps=50,
            min_partial_trace_steps=1,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
            predict_type_prob_map={
                SamplePredictType.frame_variables: 0.5,
                SamplePredictType.function_return: 0.5,
            },
            functions_fallback_to_segments=True,
        )
        # verify we can produce at least one partial sample
        got_partial = False
        for seed in range(20):
            rng = np.random.default_rng(seed)
            sample = generate_sample(
                code_type_selection_result=selection,
                trace_data=trace_data,
                code_summary="",
                transform_config=cfg,
                rng=rng,
            )
            assert sample is not None
            if sample.predict_type in (SamplePredictType.frame_variables, SamplePredictType.function_return):
                got_partial = True
        assert got_partial, "always strategy should produce at least one partial sample"

    def test_hybrid_strategy_behavior(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        selection = make_selected_sample(small_fake_reader, 0)
        trace_data = small_fake_reader[0]
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.hybrid,
            too_long_total_steps_threshold=1,
            too_long_valid_steps_threshold=1,
            too_long_code_lines_threshold=1,
            max_partial_trace_steps=50,
            min_partial_trace_steps=1,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
            predict_type_prob_map={SamplePredictType.frame_variables: 1.0},
        )
        # with thresholds set to 1, trace should be "too long" and trigger partial sampling
        got_partial = False
        for seed in range(20):
            rng = np.random.default_rng(seed)
            sample = generate_sample(
                code_type_selection_result=selection,
                trace_data=trace_data,
                code_summary="",
                transform_config=cfg,
                rng=rng,
            )
            assert sample is not None
            if sample.predict_type == SamplePredictType.frame_variables:
                got_partial = True
        assert got_partial, "hybrid strategy should produce partial samples for 'too long' traces"
