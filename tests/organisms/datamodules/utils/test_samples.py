import pytest

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.organisms.datamodules.utils.samples
import pyine.prompts
import pyine.utils.code.execution as exec_utils
import tests.data.utils.env_checks
from pyine.organisms.datamodules.utils.samples import (
    SampleBuilder,
    SampleTransformConfig,
    TraceMetadata,
)
from tests.utils.fake_dataset_readers import FakeTraceDataConfig, FakeTraceDatasetReader


@pytest.fixture()
def small_fake_reader() -> FakeTraceDatasetReader:
    cfg = FakeTraceDataConfig(
        dataset_name="FAKE",
        subset_name="unit",
        num_problems=1,
        solutions_per_problem=2,
        tests_per_problem=2,
        augmented_per_solution=0,
        entrypoint_name="solution",
        max_events_per_line=50,
        seed=321,  # different seed from other tests to avoid coupling
        code_kind="function",
    )
    return FakeTraceDatasetReader(config=cfg)


def make_targets(reader: FakeTraceDatasetReader, indices: list[int]) -> list[TraceMetadata]:
    phash = reader.get_hash()
    targets: list[TraceMetadata] = []
    for i in indices:
        tr: exec_utils.TraceResult = reader[i]
        assert tr.identifier is not None
        targets.append(
            TraceMetadata(
                identifier=str(tr.identifier),
                index=i,
                parent_dataset_hash=phash,
                tags=["unit", "fake"],
            )
        )
    return targets


def test_trace_targeting(small_fake_reader: FakeTraceDatasetReader) -> None:
    # first, check if we can indeed target specific traces
    targets = make_targets(small_fake_reader, [0, 1, 2])
    sb = SampleBuilder(source_data=[small_fake_reader], traces=targets)
    assert len(sb) == len(targets)
    for t in sb.traces:
        assert t.index in small_fake_reader.trace_indices
        assert t.identifier in small_fake_reader.trace_keys
        assert small_fake_reader.trace_keys.index(t.identifier) == t.index
        assert t.parent_dataset_hash == small_fake_reader.get_hash()
        assert t.tags == ["unit", "fake"]
    # also check if we can properly get default trace metadata when targets are not specified
    expected_traces = pyine.organisms.datamodules.utils.samples.get_traces_metadata(
        readers=small_fake_reader,
        base_filter=None,
        verbose=True,
    )
    assert len(expected_traces) == len(small_fake_reader)
    sb2 = SampleBuilder(source_data=small_fake_reader)
    assert len(sb2) == len(expected_traces)
    for t in sb2.traces:
        assert t.index in small_fake_reader.trace_indices
        assert t.identifier in small_fake_reader.trace_keys
        assert small_fake_reader.trace_keys.index(t.identifier) == t.index
        assert t.parent_dataset_hash == small_fake_reader.get_hash()
        assert t.tags != ["unit", "fake"]


class TestSampleBuilderFullSamples:

    def test_full_trace_when_never(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        targets = make_targets(small_fake_reader, [0])
        cfg = SampleTransformConfig(
            partial_sample_decision_strategy="never",
            random_seed=123,
            output_type_prob_map={},
        )
        sb = SampleBuilder(source_data=[small_fake_reader], traces=targets, config=cfg)  # noqa
        assert len(sb) == 1
        sample = sb[0]
        tr = small_fake_reader[0]
        # basic integrity
        assert sample.identifier == tr.identifier
        assert sample.code == tr.code_string
        assert isinstance(sample.description, str)
        assert sample.output_type == "program output"
        assert sample.inputs == tr.inputs
        assert sample.output == tr.expected_output
        # boundaries
        assert sample.first_line == 0
        assert sample.last_line == len(tr.code_string.splitlines())
        # step count should count only non-None (i.e. valid) events
        assert sample.trace_step_count == tr.valid_step_count
        assert sample.trace_step_count == len([s for s in tr.traced_steps if s is not None])


class TestSampleBuilderPartialSamples:

    def test_partial_sample_basic(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        targets = make_targets(small_fake_reader, [0])
        cfg = SampleTransformConfig(
            partial_sample_decision_strategy="always",
            random_seed=42,
            max_partial_trace_steps=3,
            output_type_prob_map={
                "frame variables": 1.0,
            },
        )
        sb = SampleBuilder(source_data=[small_fake_reader], traces=targets, config=cfg)  # noqa
        sample = sb[0]
        tr = small_fake_reader[0]
        # boundaries are sane
        total_lines = len(tr.code_string.splitlines())
        # note: first/last line can be out-of-order if we picked a segment inside a loop
        assert 0 <= sample.first_line <= total_lines
        assert 0 <= sample.last_line <= total_lines
        # sample has content and steps
        assert isinstance(sample.inputs, str)
        assert isinstance(sample.output, str)
        assert isinstance(sample.output_type, str) and sample.output_type == "frame variables"
        assert sample.trace_step_count > 0

    def test_partial_sample_function_call(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        targets = make_targets(small_fake_reader, [0])
        cfg = SampleTransformConfig(
            partial_sample_decision_strategy="always",
            random_seed=13,
            output_type_prob_map={
                "function return": 1.0,
            },
        )
        sb = SampleBuilder(source_data=[small_fake_reader], traces=targets, config=cfg)  # noqa
        sample = sb[0]
        tr = small_fake_reader[0]
        # boundaries are sane
        total_lines = len(tr.code_string.splitlines())
        # note: first/last line should NOT be out-of-order (can't be inside a loop)
        assert 0 <= sample.first_line <= sample.last_line <= total_lines
        # sample has content and steps
        assert isinstance(sample.inputs, str)
        assert isinstance(sample.output, str)
        assert isinstance(sample.output_type, str) and sample.output_type == "function return"
        assert sample.trace_step_count > 0

    def test_caps_enforced_and_fallback(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        targets = make_targets(small_fake_reader, [0])
        cfg = SampleTransformConfig(
            partial_sample_decision_strategy="always",
            random_seed=1,
            max_inputs_str_length=0,  # any non-empty inputs will exceed -> partial skipped
            output_type_prob_map={
                "frame variables": 1.0,
            },
        )
        sb = SampleBuilder(source_data=[small_fake_reader], traces=targets, config=cfg)  # noqa
        sample = sb[0]
        # since caps reject partial sample, we should have fallen back to full program output
        assert sample.output_type == "program output"
        tr = small_fake_reader[0]
        assert sample.output == tr.expected_output


class TestSampleBuilderRealData:

    @pytest.mark.slow
    @pytest.mark.skipif(
        tests.data.utils.env_checks.TACO_TRACES_DATASET_MISSING,
        reason="TACO traces dataset is missing, cannot check sample generation",
    )
    def test_sample_generation_on_taco_traces(self):
        cfg = SampleTransformConfig(
            random_seed=0,
            partial_sample_decision_strategy="hybrid",
            functions_fallback_to_segments=True,
            max_partial_trace_steps=100,
            min_partial_trace_steps=5,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
            output_type_prob_map={
                "program output": 0.5,
                "frame variables": 0.1,
                "function return": 0.4,
            },
        )
        dataset_path = pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO")
        taco_reader = pyine.data.traces.dataset_reader.DatasetReader(dataset_path)
        sb = SampleBuilder(
            source_data=taco_reader,
            traces=None,
            config=cfg,
        )
        assert len(sb) == len(taco_reader)
        for trace_idx in range(len(taco_reader)):
            if trace_idx > 100:
                break  # check up to 100 traces, that should be enough
            trace_data = taco_reader[trace_idx]
            sample = sb[trace_idx]
            assert trace_data.identifier == sample.identifier
            assert trace_data.code_string == sample.code
            assert isinstance(sample.description, str)
            code_lines = trace_data.code_string.splitlines()
            if sample.output_type == "program output":
                assert sample.first_line == 0
                assert sample.last_line == len(code_lines)
                assert sample.inputs == trace_data.inputs
                assert sample.output == trace_data.expected_output
                assert sample.trace_step_count == trace_data.valid_step_count
            else:
                assert isinstance(sample.inputs, str) and len(sample.inputs) <= cfg.max_inputs_str_length
                assert isinstance(sample.output, str) and len(sample.output) <= cfg.max_output_str_length
                assert sample.trace_step_count > 0
                assert cfg.min_partial_trace_steps <= sample.trace_step_count <= cfg.max_partial_trace_steps
                if sample.output_type == "frame variables":
                    assert 0 < sample.first_line < len(code_lines)
                    assert 0 < sample.last_line < len(code_lines)
                else:  # function return
                    assert 0 < sample.first_line <= sample.last_line < len(code_lines)
                    matched_code_blocks = [
                        cb
                        for cb in trace_data.code_blocks.values()
                        if cb.start_line == sample.first_line and cb.end_line == sample.last_line
                    ]
                    if matched_code_blocks:
                        assert any([cb.name in sample.description for cb in matched_code_blocks])


class TestPrivateSampleMethods:
    """Directly test internal sample builders on controlled fake data."""

    class _FakeKey:
        def __init__(
            self,
            obj: str,
            line: int,
        ) -> None:
            self.object = obj
            self.line = line

        def __str__(
            self,
        ) -> str:
            return f"{self.object}:{self.line}"

    class _FakeEvent:
        def __init__(
            self,
            event_type: exec_utils.TraceEventType,
            step_idx: int,
            key: "TestPrivateSampleMethods._FakeKey",
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
        def __init__(
            self,
            steps: list["TestPrivateSampleMethods._FakeEvent"],
        ) -> None:
            self.identifier = "fake/trace/1"
            self.code_string = """\
x = foo(2) + 1
print(x)
"""
            self.code_blocks = {}
            self.inputs = ""
            self.expected_output = ""
            self.traced_steps = steps

    @pytest.fixture()
    def nested_call_trace(
        self,
    ) -> "_FakeTrace":
        # shape:
        # 0 CALL main
        # 1 LINE main
        # 2 CALL foo
        # 3 LINE foo
        # 4 RETURN foo
        # 5 LINE main
        # 6 RETURN main
        steps: list[TestPrivateSampleMethods._FakeEvent] = []
        steps.append(self._FakeEvent(exec_utils.TraceEventType.CALL, 0, self._FakeKey("main", 1)))
        steps.append(self._FakeEvent(exec_utils.TraceEventType.LINE, 1, self._FakeKey("main", 2), local_vars={"x": 1}))
        steps.append(
            self._FakeEvent(
                exec_utils.TraceEventType.CALL,
                2,
                self._FakeKey("foo", 3),
                arguments=(1,),
            )
        )
        steps.append(self._FakeEvent(exec_utils.TraceEventType.LINE, 3, self._FakeKey("foo", 101), local_vars={"y": 2}))
        steps.append(self._FakeEvent(exec_utils.TraceEventType.RETURN, 4, self._FakeKey("foo", 103), return_value=5))
        steps.append(self._FakeEvent(exec_utils.TraceEventType.LINE, 5, self._FakeKey("main", 5), local_vars={"x": 6}))
        steps.append(self._FakeEvent(exec_utils.TraceEventType.RETURN, 6, self._FakeKey("main", 6)))
        return TestPrivateSampleMethods._FakeTrace(steps=steps)

    def test_get_code_segment_sample_skips_inner_calls(
        self, small_fake_reader: FakeTraceDatasetReader, nested_call_trace: "_FakeTrace", mocker
    ) -> None:
        cfg = SampleTransformConfig(
            random_seed=0,
            min_partial_trace_steps=1,
            max_partial_trace_steps=3,
        )
        # traces can be empty since we call private methods directly; reader is only used to pass __init__ checks
        sb = SampleBuilder(source_data=[small_fake_reader], traces=[], config=cfg)  # noqa
        sample = sb._get_code_segment_sample(
            nested_call_trace,
            trace_meta=mocker.MagicMock(),
            target_output_type="frame variables",
        )
        assert sample is not None
        assert sample.identifier == nested_call_trace.identifier
        assert sample.code == nested_call_trace.code_string
        assert sample.output_type == "frame variables"
        assert 2 <= sample.first_line <= 5
        assert 3 <= sample.last_line <= 6
        assert 1 <= sample.trace_step_count <= 3
        assert isinstance(sample.inputs, str)
        assert isinstance(sample.output, str)

    def test_get_function_call_sample_basic(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        nested_call_trace: "_FakeTrace",
        mocker,
    ) -> None:
        cfg = SampleTransformConfig(
            random_seed=0,
            min_partial_trace_steps=1,
        )
        sb = SampleBuilder(source_data=[small_fake_reader], traces=[], config=cfg)  # noqa
        sample = sb._get_function_call_sample(nested_call_trace, trace_meta=mocker.MagicMock())
        assert sample is not None
        assert sample.identifier == nested_call_trace.identifier
        assert sample.code == nested_call_trace.code_string
        assert sample.output_type == "function return"
        assert sample.entrypoint == "foo"
        # since we didn't provide code_blocks mapping, first/last line should equal the call site line
        assert sample.first_line == sample.last_line == 3
        assert sample.inputs == "(1,)"
        assert sample.output == "5"
        assert sample.trace_step_count == 2


def test_code_summary_is_used_from_prompt_db(small_fake_reader: FakeTraceDatasetReader, tmp_path) -> None:
    # create a temporary prompt result DB and insert a single code summary for one solution id
    db_path = tmp_path / "prompt_results.sqlite"
    db = pyine.prompts.PromptResultDB(str(db_path))
    tr0 = small_fake_reader[0]
    assert tr0.identifier is not None
    sol_id_obj = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(tr0.identifier).get_parent_identifier()
    sol_id_str = str(sol_id_obj)
    expected_summary = "This solution computes and prints intermediate values, then returns 2*x."
    db.store(
        identifier=sol_id_str,
        prompt="code summary prompt",
        result=expected_summary,
        prompt_name="code_summary",
    )
    # build SampleBuilder pointing to our temporary DB; force full samples to simplify checks
    cfg = SampleTransformConfig(
        partial_sample_decision_strategy="never",
        random_seed=0,
        output_type_prob_map={},
    )
    sb = SampleBuilder(
        source_data=[small_fake_reader],
        traces=None,
        config=cfg,
        prompt_result_db_path=str(db_path),
    )
    assert len(sb) == len(small_fake_reader)
    # for traces matching the stored solution id, description should match; others should be empty
    for i in range(len(sb)):
        sample = sb[i]
        assert isinstance(sample.description, str)
        tr = small_fake_reader[i]
        assert tr.identifier is not None
        curr_sol_id_str = str(
            pyine.data.traces.dataset_utils.TraceIdentifier.from_string(tr.identifier).get_parent_identifier()
        )
        if curr_sol_id_str == sol_id_str:
            assert sample.description == expected_summary
        else:
            assert sample.description == ""
