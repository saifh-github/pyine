import pathlib

import numpy as np
import pytest
import pytest_mock

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.organisms.datamodules.utils.samples
import pyine.prompts
import pyine.utils.code.execution as exec_utils
import tests.env_checks
from pyine.organisms.datamodules.utils.samples import (
    SampleBuilder,
    SampleFilteringConfig,
    SampleSelectionConfig,
    SampleTransformConfig,
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


def make_targets(
    reader: FakeTraceDatasetReader,
    indices: list[int],
) -> list[pyine.data.traces.dataset_utils.TraceMetadata]:
    targets: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
    for idx in indices:
        tr: exec_utils.TraceResult = reader[idx]
        assert tr.identifier is not None
        targets.append(
            pyine.data.traces.dataset_utils.TraceMetadata(
                identifier=str(tr.identifier),
                parent_dataset_hash=reader.hash,
                index=idx,
                internal_index=idx,
                step_count=tr.valid_step_count,
                code_string=tr.code_string,
                inputs=tr.inputs,
                expected_output=tr.expected_output,
                return_value=tr.return_value,
                exception=tr.exception,
                stdout=tr.stdout,
                stderr=tr.stderr,
                metadata=tr.metadata,
                tags=[*tr.tags, "unit", "fake"],
            )
        )
    return targets


def test_trace_targeting(small_fake_reader: FakeTraceDatasetReader) -> None:
    # first, check if we can indeed target specific traces
    targets = make_targets(small_fake_reader, [0, 1, 2])
    sb = SampleBuilder(source_data=[small_fake_reader], traces=targets)
    assert len(sb) == len(targets)
    for st in sb.selected_traces:
        t = st.trace_meta
        assert t.identifier in small_fake_reader.trace_keys
        assert small_fake_reader.trace_keys.index(t.identifier) == t.index
        assert t.parent_dataset_hash == small_fake_reader.hash
        assert "unit" in t.tags and "fake" in t.tags
    # also check if we can properly get default trace metadata when targets are not specified
    expected_traces = pyine.data.traces.dataset_reader.get_traces_metadata(small_fake_reader)
    assert len(expected_traces) == len(small_fake_reader)
    sb2 = SampleBuilder(source_data=small_fake_reader)
    assert len(sb2) == len(expected_traces)
    for st in sb2.selected_traces:
        t = st.trace_meta
        assert t.identifier in small_fake_reader.trace_keys
        assert small_fake_reader.trace_keys.index(t.identifier) == t.index
        assert t.parent_dataset_hash == small_fake_reader.hash
        assert "unit" not in t.tags and "fake" not in t.tags


class TestSampleBuilderFullSamples:
    def test_full_trace_when_never(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        targets = make_targets(small_fake_reader, [0])
        cfg = SampleTransformConfig(
            transform_strategy="never",
            output_type_prob_map={},
        )
        sb = SampleBuilder(source_data=[small_fake_reader], traces=targets, transform_config=cfg)
        assert len(sb) == 1
        sample = sb[0]
        tr = small_fake_reader[0]
        # basic integrity
        assert sample.identifier == tr.identifier
        assert sample.code == tr.code_string
        assert isinstance(sample.description, str)
        assert sample.output_type == "program_output"
        assert sample.inputs == str(tr.inputs)
        assert sample.expected_output == str(tr.expected_output)
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
            transform_strategy="always",
            max_partial_trace_steps=3,
            output_type_prob_map={
                "frame_variables": 1.0,
            },
        )
        sb = SampleBuilder(source_data=[small_fake_reader], traces=targets, transform_config=cfg)
        sample = sb[0]
        tr = small_fake_reader[0]
        # boundaries are sane
        total_lines = len(tr.code_string.splitlines())
        # note: first/last line can be out-of-order if we picked a segment inside a loop
        assert 0 <= sample.first_line <= total_lines
        assert 0 <= sample.last_line <= total_lines
        # sample has content and steps
        assert isinstance(sample.inputs, str)
        assert isinstance(sample.expected_output, str)
        assert isinstance(sample.output_type, str) and sample.output_type == "frame_variables"
        assert sample.trace_step_count > 0

    def test_partial_sample_function_call(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        targets = make_targets(small_fake_reader, [0])
        cfg = SampleTransformConfig(
            transform_strategy="always",
            output_type_prob_map={
                "function_return": 1.0,
            },
        )
        sb = SampleBuilder(source_data=[small_fake_reader], traces=targets, transform_config=cfg)
        sample = sb[0]
        tr = small_fake_reader[0]
        # boundaries are sane
        total_lines = len(tr.code_string.splitlines())
        # note: first/last line should NOT be out-of-order (can't be inside a loop)
        assert 0 <= sample.first_line <= sample.last_line <= total_lines
        # sample has content and steps
        assert isinstance(sample.inputs, str)
        assert isinstance(sample.expected_output, str)
        assert isinstance(sample.output_type, str) and sample.output_type == "function_return"
        assert sample.trace_step_count > 0

    def test_caps_enforced_and_fallback(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        targets = make_targets(small_fake_reader, [0])
        cfg = SampleTransformConfig(
            transform_strategy="always",
            max_inputs_str_length=0,  # any non-empty inputs will exceed -> partial skipped
            output_type_prob_map={
                "frame_variables": 1.0,
            },
        )
        sb = SampleBuilder(source_data=[small_fake_reader], traces=targets, transform_config=cfg)
        sample = sb[0]
        # since caps reject partial sample, we should have fallen back to full program output
        assert sample.output_type == "program_output"
        tr = small_fake_reader[0]
        assert sample.expected_output == str(tr.expected_output)


class TestSampleBuilderFiltering:
    def test_filtering_by_step_count(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        # first, get all traces without filtering
        sb_no_filter = SampleBuilder(source_data=[small_fake_reader])
        total_traces = len(sb_no_filter)
        assert total_traces > 0
        # now apply a very restrictive step count filter (should filter out some/all traces)
        cfg = SampleFilteringConfig(max_trace_steps=5)
        sb_filtered = SampleBuilder(
            source_data=[small_fake_reader],
            filtering_config=cfg,
        )
        # should have fewer traces (or same if all traces were already below threshold)
        assert len(sb_filtered) <= total_traces
        # verify that all remaining traces satisfy the filter
        for st in sb_filtered.selected_traces:
            tr = small_fake_reader[st.trace_meta.index]
            assert tr.valid_step_count <= 5
        # check that stats are available
        stats = sb_filtered.get_stats()
        assert "filtering/total_traces" in stats
        assert "filtering/kept_traces" in stats
        assert "filtering/filtered_by_step_count" in stats
        assert stats["filtering/total_traces"] == total_traces
        assert stats["filtering/kept_traces"] == len(sb_filtered)

    def test_filtering_by_var_repr_length(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        # get all traces without filtering
        sb_no_filter = SampleBuilder(source_data=[small_fake_reader])
        total_traces = len(sb_no_filter)
        # apply a very restrictive var repr length filter
        cfg = SampleFilteringConfig(max_args_length=10)
        sb_filtered = SampleBuilder(
            source_data=[small_fake_reader],
            filtering_config=cfg,
        )
        # should have fewer traces (or same if all were already below threshold)
        assert len(sb_filtered) <= total_traces
        # verify that all remaining traces satisfy the filter
        for st in sb_filtered.selected_traces:
            tr = small_fake_reader[st.trace_meta.index]
            inputs_str = str(tr.inputs)
            expected_output_str = str(tr.expected_output)
            combined_length = len(inputs_str) + len(expected_output_str)
            assert combined_length <= 10
        # check stats
        stats = sb_filtered.get_stats()
        assert "filtering/filtered_by_var_length" in stats

    def test_filtering_both_criteria(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        # apply both filters at once
        cfg = SampleFilteringConfig(
            max_trace_steps=100,
            max_args_length=500,
        )
        sb_filtered = SampleBuilder(
            source_data=[small_fake_reader],
            filtering_config=cfg,
        )
        # verify all remaining traces satisfy both criteria
        for st in sb_filtered.selected_traces:
            tr = small_fake_reader[st.trace_meta.index]
            assert tr.valid_step_count <= 100
            inputs_str = str(tr.inputs)
            expected_output_str = str(tr.expected_output)
            combined_length = len(inputs_str) + len(expected_output_str)
            assert combined_length <= 500
        # check that stats reflect both filters
        stats = sb_filtered.get_stats()
        assert "filtering/filtered_by_step_count" in stats
        assert "filtering/filtered_by_var_length" in stats

    def test_no_filtering_when_config_is_none(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        # when no filtering config is provided, all traces should be kept
        sb_default = SampleBuilder(source_data=[small_fake_reader])
        sb_empty_config = SampleBuilder(
            source_data=[small_fake_reader],
            filtering_config=SampleFilteringConfig(),
        )
        # both should have the same number of traces
        assert len(sb_default) == len(sb_empty_config)


class TestSampleBuilderRealData:
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.dataset
    @pytest.mark.skipif(
        tests.env_checks.TACO_TRACES_DATASET_MISSING,
        reason="TACO traces dataset is missing, cannot check sample generation",
    )
    def test_sample_generation_on_taco_traces(
        self,
    ) -> None:
        dataset_path = pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO")
        taco_reader = pyine.data.traces.dataset_reader.DatasetReader(dataset_path)
        if len(taco_reader) < 1000:
            pytest.skip("TACO traces dataset is too small, skipping sample generation test")
        cfg = SampleTransformConfig(
            transform_strategy="hybrid",
            functions_fallback_to_segments=True,
            max_partial_trace_steps=100,
            min_partial_trace_steps=5,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
            output_type_prob_map={
                "program_output": 0.5,
                "frame_variables": 0.1,
                "function_return": 0.4,
            },
        )
        sb = SampleBuilder(
            source_data=taco_reader,
            traces=None,
            transform_config=cfg,
        )
        assert len(sb) <= len(taco_reader)
        for sample_idx in range(len(sb)):
            if sample_idx > 100:
                break  # check up to 100 samples, that should be enough
            sample = sb[sample_idx]
            assert sample.identifier in taco_reader.trace_keys
            trace_idx = taco_reader.trace_keys.index(sample.identifier)
            trace_data = taco_reader[trace_idx]
            if sample.has_code_override:
                assert sample.code != trace_data.code_string
            else:
                assert sample.code == trace_data.code_string
            assert isinstance(sample.description, str)
            code_lines = sample.code.splitlines()
            if sample.output_type == "program_output":
                assert sample.first_line == 0
                assert sample.last_line == len(code_lines)
                assert sample.inputs == str(trace_data.inputs)
                assert sample.expected_output == str(trace_data.expected_output)
                assert sample.trace_step_count == trace_data.valid_step_count
            else:
                assert isinstance(sample.inputs, str)
                assert isinstance(sample.expected_output, str)
                assert len(sample.inputs) <= cfg.max_inputs_str_length
                assert len(sample.expected_output) <= cfg.max_output_str_length
                assert 0 < sample.trace_step_count < trace_data.valid_step_count
                assert cfg.min_partial_trace_steps <= sample.trace_step_count <= cfg.max_partial_trace_steps
                if sample.output_type == "frame_variables":
                    assert 0 < sample.first_line < len(code_lines)
                    assert 0 < sample.last_line <= len(code_lines)
                else:  # function_return
                    assert 0 < sample.first_line <= sample.last_line < len(code_lines)


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
        steps.append(
            self._FakeEvent(
                exec_utils.TraceEventType.LINE,
                1,
                self._FakeKey("main", 2),
                local_vars={"x": 1},
            )
        )
        steps.append(
            self._FakeEvent(
                exec_utils.TraceEventType.CALL,
                2,
                self._FakeKey("foo", 3),
                arguments=(1,),
            )
        )
        steps.append(
            self._FakeEvent(
                exec_utils.TraceEventType.LINE,
                3,
                self._FakeKey("foo", 101),
                local_vars={"y": 2},
            )
        )
        steps.append(
            self._FakeEvent(
                exec_utils.TraceEventType.RETURN,
                4,
                self._FakeKey("foo", 103),
                return_value=5,
            )
        )
        steps.append(
            self._FakeEvent(
                exec_utils.TraceEventType.LINE,
                5,
                self._FakeKey("main", 5),
                local_vars={"x": 6},
            )
        )
        steps.append(self._FakeEvent(exec_utils.TraceEventType.RETURN, 6, self._FakeKey("main", 6)))
        return TestPrivateSampleMethods._FakeTrace(steps=steps)

    def test_get_code_segment_sample_skips_inner_calls(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        nested_call_trace: "_FakeTrace",
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        cfg = SampleTransformConfig(
            min_partial_trace_steps=1,
            max_partial_trace_steps=3,
        )
        # traces can be empty since we call private methods directly; reader is only used to pass __init__ checks
        sb = SampleBuilder(source_data=[small_fake_reader], traces=[], transform_config=cfg)
        sample = sb._get_code_segment_sample(
            trace_data=nested_call_trace,
            trace_meta=mocker.MagicMock(),
            trace_code_type="original",
            target_output_type="frame_variables",
            rng=np.random.default_rng(seed=0),
        )
        assert sample is not None
        assert sample.identifier == nested_call_trace.identifier
        assert sample.code == nested_call_trace.code_string
        assert sample.output_type == "frame_variables"
        assert 2 <= sample.first_line <= 5
        assert 3 <= sample.last_line <= 6
        assert 1 <= sample.trace_step_count <= 3
        assert isinstance(sample.inputs, str)
        assert isinstance(sample.expected_output, str)

    def test_get_function_call_sample_basic(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        nested_call_trace: "_FakeTrace",
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        cfg = SampleTransformConfig(
            min_partial_trace_steps=1,
        )
        sb = SampleBuilder(source_data=[small_fake_reader], traces=[], transform_config=cfg)
        sample = sb._get_function_call_sample(
            trace_data=nested_call_trace,
            trace_meta=mocker.MagicMock(),
            trace_code_type="original",
            rng=np.random.default_rng(seed=0),
        )
        assert sample is not None
        assert sample.identifier == nested_call_trace.identifier
        assert sample.code == nested_call_trace.code_string
        assert sample.output_type == "function_return"
        assert sample.entrypoint == "foo"
        # since we didn't provide code_blocks mapping, first/last line should equal the call site line
        assert sample.first_line == sample.last_line == 3
        assert sample.inputs == "(1,)"
        assert sample.expected_output == "5"
        assert sample.trace_step_count == 2


def test_code_summary_is_used_from_prompt_db(
    small_fake_reader: FakeTraceDatasetReader,
    tmp_path: pathlib.Path,
) -> None:
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
        transform_strategy="never",
        output_type_prob_map={},
    )
    sb = SampleBuilder(
        source_data=[small_fake_reader],
        traces=None,
        transform_config=cfg,
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


class TestSelectTraces:
    def test_select_traces_original_and_obfuscated_skip(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        # original: should keep all clusters (no augmented variants exist => one per trace)
        sb_original = SampleBuilder(
            source_data=[small_fake_reader],
            selection_config=SampleSelectionConfig(
                choice_strategy="latest",
                input_type_prob_map={"original": 1.0},
            ),
        )
        selected_traces_meta = [s.trace_meta for s in sb_original.selected_traces]
        code_input_types = [s.code_type for s in sb_original.selected_traces]
        assert len(selected_traces_meta) == len(small_fake_reader)
        assert set(code_input_types) == {"original"}
        code_overrides = [s.code_override for s in sb_original.selected_traces]
        assert all(ovr is None for ovr in code_overrides)
        # obfuscated: none present and no DB fallback for obfuscation => skip everything
        sb_obf = SampleBuilder(
            source_data=[small_fake_reader],
            selection_config=SampleSelectionConfig(
                choice_strategy="latest",
                input_type_prob_map={"obfuscated": 1.0},
            ),
        )
        assert len(sb_obf) == 0
        # hinted without DB allowed: none present => skip everything
        sb_no_db = SampleBuilder(
            source_data=[small_fake_reader],
            selection_config=SampleSelectionConfig(
                allow_db_lookups=False,
                choice_strategy="latest",
                input_type_prob_map={"hinted": 1.0},
            ),
        )
        assert len(sb_no_db) == 0

    def test_select_traces_db_fallback_hinted_latest_and_bugged_random(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        tmp_path: pathlib.Path,
    ) -> None:
        # prepare prompt DB with both trace-level (hints) and solution-level (issues/bugs) overrides
        db_path = tmp_path / "select_traces.sqlite"
        db = pyine.prompts.PromptResultDB(str(db_path))
        # pick one specific trace to attach hints to (trace-level)
        tr0 = small_fake_reader[0]
        assert tr0.identifier is not None
        trace_id_str = str(tr0.identifier)
        # and derive its solution id to attach bugged records (solution-level)
        sol_id_str = str(
            pyine.data.traces.dataset_utils.TraceIdentifier.from_string(trace_id_str).get_parent_identifier()
        )
        # store two hinted variants for the trace (latest should pick the last one)
        db.store(
            identifier=trace_id_str,
            prompt="hint",
            result="hint_v1",
            prompt_name="hints/thingy1",
        )
        db.store(
            identifier=trace_id_str,
            prompt="hint",
            result="hint_v2",
            prompt_name="hints/thingy2",
        )
        # store two bug variants for the solution (random should pick one deterministically by seed)
        db.store(
            identifier=sol_id_str,
            prompt="issues",
            result="bug_A",
            prompt_name="issues/simple",
        )
        db.store(
            identifier=sol_id_str,
            prompt="issues",
            result="bug_B",
            prompt_name="issues/complex",
        )
        # hinted + latest => should include exactly the targeted trace cluster with the latest override
        sb_hinted = SampleBuilder(
            source_data=[small_fake_reader],
            selection_config=SampleSelectionConfig(
                allow_db_lookups=True,
                choice_strategy="latest",
                input_type_prob_map={"hinted": 1.0},
            ),
            prompt_result_db_path=str(db_path),
        )
        # only one cluster should have hints in DB => length 1
        assert len(sb_hinted) == 1
        assert sb_hinted.selected_traces[0].code_type == "hinted"
        assert sb_hinted.selected_traces[0].code_override == "hint_v2"  # latest
        assert (
            sb_hinted.selected_traces[0].trace_meta.identifier == trace_id_str
        )  # same original trace (override applies at use time)
        # bugged + random => for the selected solution, there are as many clusters as test cases
        # only clusters matching the solution id should be selected; others skipped
        sb_bugged = SampleBuilder(
            source_data=[small_fake_reader],
            selection_config=SampleSelectionConfig(
                seed=0,
                allow_db_lookups=True,
                choice_strategy="random",
                input_type_prob_map={"bugged": 1.0},
            ),
            prompt_result_db_path=str(db_path),
        )
        # figure how many tests exist for that solution in the fake reader (two by fixture config)
        # i.e., number of clusters that share the same parent solution id
        expected_bugged_clusters = sum(
            1 for t in sb_bugged.selected_traces if str(t.trace_meta.solution_id) == sol_id_str
        )
        # no other solution has DB bug records, so the builder should contain only those clusters
        assert len(sb_bugged) == expected_bugged_clusters
        assert all(str(t.trace_meta.solution_id) == sol_id_str for t in sb_bugged.selected_traces)
        code_input_types = [s.code_type for s in sb_bugged.selected_traces]
        assert set(code_input_types) == {"bugged"}
        code_overrides = [s.code_override for s in sb_bugged.selected_traces]
        assert all(ovr in {"bug_A", "bug_B"} for ovr in code_overrides)
