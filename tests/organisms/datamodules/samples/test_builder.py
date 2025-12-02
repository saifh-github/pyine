"""Tests for the SampleBuilder class (integration tests)."""

import pathlib

import pytest

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.prompts
import tests.env_checks
from pyine.organisms.datamodules.samples import (
    SampleBuilder,
    SampleCodeType,
    SampleCodeTypeSet,
    SamplePredictType,
    SampleSelectionConfig,
    SampleTransformConfig,
    SampleTransformStrategy,
    TraceFilteringConfig,
)
from tests.utils.fake_dataset_readers import FakeTraceDataConfig, FakeTraceDatasetReader


@pytest.fixture
def small_fake_reader() -> FakeTraceDatasetReader:
    """Provides a small fake dataset reader for builder tests."""
    cfg = FakeTraceDataConfig(
        dataset_name="FAKE",
        subset_name="builder_test",
        num_problems=2,
        solutions_per_problem=2,
        tests_per_problem=2,
        augmented_per_solution=0,
        entrypoint_name="solution",
        max_events_per_line=50,
        seed=654,
        code_kind="function",
    )
    return FakeTraceDatasetReader(config=cfg)


def make_trace_targets(
    reader: FakeTraceDatasetReader,
    indices: list[int],
) -> list[pyine.data.traces.dataset_utils.TraceMetadata]:
    """Helper to create trace metadata targets from reader indices."""
    return [reader.trace_metadata[idx] for idx in indices]


def default_selection_config() -> SampleSelectionConfig:
    """Returns a default selection config with explicit SampleCodeTypeSet keys."""
    return SampleSelectionConfig(
        code_type_prob_map={SampleCodeTypeSet.create_default(): 1.0},
    )


class TestSampleBuilderInitialization:
    """Tests for SampleBuilder initialization."""

    def test_init_with_single_reader(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        sb = SampleBuilder(source_data=small_fake_reader, selection_config=default_selection_config())
        assert len(sb) > 0
        assert len(sb.readers_map) == 1
        assert small_fake_reader.hash in sb.readers_map

    def test_init_with_reader_list(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        sb = SampleBuilder(source_data=[small_fake_reader], selection_config=default_selection_config())
        assert len(sb) > 0

    def test_init_with_specific_traces(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        targets = make_trace_targets(small_fake_reader, [0, 1, 2])
        sb = SampleBuilder(source_data=[small_fake_reader], traces=targets, selection_config=default_selection_config())
        assert len(sb) <= len(targets)

    def test_init_without_traces_targets_all(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        sb = SampleBuilder(source_data=small_fake_reader, selection_config=default_selection_config())
        default_traces = pyine.data.traces.dataset_reader.get_traces_metadata(small_fake_reader)
        assert len(sb.orig_traces) == len(default_traces)


class TestSampleBuilderTraceTargeting:
    """Tests for targeting specific traces in SampleBuilder."""

    def test_targeted_traces_are_indexed_correctly(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        targets = make_trace_targets(small_fake_reader, [0, 1, 2])
        sb = SampleBuilder(source_data=[small_fake_reader], traces=targets, selection_config=default_selection_config())
        for st in sb.selection_results:
            t = st.trace_meta
            assert t.identifier in small_fake_reader.trace_keys
            assert small_fake_reader.trace_keys.index(t.identifier) == t.index
            assert t.parent_dataset_hash == small_fake_reader.hash

    def test_default_trace_metadata_when_targets_not_specified(
        self,
        small_fake_reader: FakeTraceDatasetReader,
    ) -> None:
        sb = SampleBuilder(source_data=small_fake_reader, selection_config=default_selection_config())
        for st in sb.selection_results:
            t = st.trace_meta
            assert t.identifier in small_fake_reader.trace_keys
            assert small_fake_reader.trace_keys.index(t.identifier) == t.index
            assert t.parent_dataset_hash == small_fake_reader.hash


class TestSampleBuilderFullSamples:
    """Tests for generating full program output samples."""

    def test_full_trace_with_never_strategy(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        targets = make_trace_targets(small_fake_reader, [0])
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.never,
            predict_type_prob_map={},
        )
        sb = SampleBuilder(
            source_data=[small_fake_reader],
            traces=targets,
            transform_config=cfg,
            selection_config=default_selection_config(),
        )
        assert len(sb) >= 1
        sample = sb[0]
        tr = small_fake_reader[0]
        assert sample.identifier == tr.identifier
        assert sample.code == tr.code_string
        assert sample.predict_type == SamplePredictType.program_output
        assert sample.inputs == str(tr.inputs)
        assert sample.expected_output == str(tr.expected_output)
        assert sample.first_line == 0
        assert sample.last_line == len(tr.code_string.splitlines())
        assert sample.trace_step_count == tr.valid_step_count


class TestSampleBuilderPartialSamples:
    """Tests for generating partial samples."""

    def test_partial_sample_frame_variables(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        targets = make_trace_targets(small_fake_reader, [0])
        tr = small_fake_reader[0]
        total_lines = len(tr.code_string.splitlines())
        # try multiple seeds to verify we can produce frame_variables at least once
        got_frame_variables = False
        for seed in range(20):
            cfg = SampleTransformConfig(
                seed=seed,
                transform_strategy=SampleTransformStrategy.always,
                max_partial_trace_steps=10,
                min_partial_trace_steps=1,
                max_inputs_str_length=1000,
                max_output_str_length=1000,
                predict_type_prob_map={
                    SamplePredictType.frame_variables: 1.0,
                },
            )
            sb = SampleBuilder(
                source_data=[small_fake_reader],
                traces=targets,
                transform_config=cfg,
                selection_config=default_selection_config(),
            )
            sample = sb[0]
            assert 0 <= sample.first_line <= total_lines
            assert 0 <= sample.last_line <= total_lines
            assert isinstance(sample.inputs, str)
            assert isinstance(sample.expected_output, str)
            assert sample.trace_step_count > 0
            assert sample.predict_type in (SamplePredictType.frame_variables, SamplePredictType.program_output)
            if sample.predict_type == SamplePredictType.frame_variables:
                got_frame_variables = True
                assert sample.trace_step_count <= 10
                assert sample.first_line_hit >= 1, "hit count should be 1-indexed"
                assert sample.last_line_hit >= 1
                assert sample.first_step_idx >= 0
                assert sample.last_step_idx >= sample.first_step_idx
        assert got_frame_variables, "should produce at least one frame_variables sample across seeds"

    def test_partial_sample_function_return(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        targets = make_trace_targets(small_fake_reader, [0])
        tr = small_fake_reader[0]
        total_lines = len(tr.code_string.splitlines())
        # try multiple seeds to verify we can produce a partial sample at least once
        got_partial = False
        for seed in range(20):
            cfg = SampleTransformConfig(
                seed=seed,
                transform_strategy=SampleTransformStrategy.always,
                max_partial_trace_steps=100,
                min_partial_trace_steps=1,
                max_inputs_str_length=1000,
                max_output_str_length=1000,
                predict_type_prob_map={
                    SamplePredictType.function_return: 1.0,
                },
                functions_fallback_to_segments=True,
            )
            sb = SampleBuilder(
                source_data=[small_fake_reader],
                traces=targets,
                transform_config=cfg,
                selection_config=default_selection_config(),
            )
            sample = sb[0]
            assert 0 <= sample.first_line <= sample.last_line <= total_lines
            assert isinstance(sample.inputs, str)
            assert isinstance(sample.expected_output, str)
            assert sample.trace_step_count > 0
            # with functions_fallback_to_segments=True, may get frame_variables instead
            if sample.predict_type in (SamplePredictType.function_return, SamplePredictType.frame_variables):
                got_partial = True
        assert got_partial, "should produce at least one partial sample across seeds"

    def test_fallback_to_full_when_caps_exceeded(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        targets = make_trace_targets(small_fake_reader, [0])
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.always,
            max_inputs_str_length=0,
            predict_type_prob_map={
                SamplePredictType.frame_variables: 1.0,
            },
        )
        sb = SampleBuilder(
            source_data=[small_fake_reader],
            traces=targets,
            transform_config=cfg,
            selection_config=default_selection_config(),
        )
        sample = sb[0]
        assert sample.predict_type == SamplePredictType.program_output
        tr = small_fake_reader[0]
        assert sample.expected_output == str(tr.expected_output)


class TestSampleBuilderFiltering:
    """Tests for trace filtering in SampleBuilder."""

    def test_filtering_by_step_count(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        sb_no_filter = SampleBuilder(source_data=[small_fake_reader], selection_config=default_selection_config())
        total_traces = len(sb_no_filter)
        assert total_traces > 0
        # find traces that exceed the step count threshold
        traces_exceeding_threshold = sum(1 for t in small_fake_reader.trace_metadata if t.step_count > 5)
        if traces_exceeding_threshold == 0:
            pytest.skip("no traces exceed step count threshold, cannot test filtering")
        cfg = TraceFilteringConfig(max_trace_steps=5)
        sb_filtered = SampleBuilder(
            source_data=[small_fake_reader],
            filtering_config=cfg,
            selection_config=default_selection_config(),
        )
        # verify filtering actually removed some traces
        assert len(sb_filtered) < total_traces, "filtering should remove some traces"
        for st in sb_filtered.selection_results:
            tr = small_fake_reader[st.trace_meta.index]
            assert tr.valid_step_count <= 5
        stats = sb_filtered.get_stats()
        assert "orig_trace_count" in stats
        assert "kept_trace_count" in stats
        assert "filtered/by_step_count" in stats
        assert stats["filtered/by_step_count"] > 0

    def test_filtering_by_args_length(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata

        def get_args_length(t: pyine.data.traces.dataset_utils.TraceMetadata) -> int:
            return len(str(t.inputs)) + len(str(t.expected_output))

        args_length_map = {idx: get_args_length(traces[idx]) for idx in range(len(traces))}
        unique_lengths = sorted(set(args_length_map.values()))
        if len(unique_lengths) < 2:
            # all traces have the same args length; set threshold below to filter all
            if unique_lengths[0] <= 1:
                pytest.skip("cannot set max_args_length below 1 to exercise filtering")
            length_threshold = unique_lengths[0] - 1
        else:
            # set threshold to the smallest value so some traces are filtered
            length_threshold = unique_lengths[0]
        cfg = TraceFilteringConfig(max_args_length=length_threshold)
        sb_filtered = SampleBuilder(
            source_data=[small_fake_reader],
            filtering_config=cfg,
            selection_config=default_selection_config(),
        )
        expected_count = sum(1 for al in args_length_map.values() if al <= length_threshold)
        assert len(sb_filtered) == expected_count
        for st in sb_filtered.selection_results:
            tr = small_fake_reader[st.trace_meta.index]
            assert get_args_length(tr) <= length_threshold
        stats = sb_filtered.get_stats()
        assert "filtered/by_var_length" in stats

    def test_no_filtering_when_config_is_none(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        sb_default = SampleBuilder(source_data=[small_fake_reader], selection_config=default_selection_config())
        sb_empty_config = SampleBuilder(
            source_data=[small_fake_reader],
            filtering_config=TraceFilteringConfig(
                max_trace_families=None,
                max_trace_steps=None,
                max_code_line_count=None,
                max_code_line_length=None,
                max_code_length=None,
                max_args_length=None,
            ),
            selection_config=default_selection_config(),
        )
        assert len(sb_default) == len(sb_empty_config)


class TestSampleBuilderSelection:
    """Tests for sample selection in SampleBuilder."""

    def test_select_original_code_types(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        sb = SampleBuilder(
            source_data=[small_fake_reader],
            selection_config=SampleSelectionConfig(
                code_type_prob_map={SampleCodeTypeSet.create_default(): 1.0},
            ),
        )
        code_types = [s.code_type for s in sb.selection_results]
        assert all(ct.is_original for ct in code_types)
        overrides = [s.code_override for s in sb.selection_results]
        assert all(ovr is None for ovr in overrides)

    def test_select_unavailable_augmented_skips(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        sb = SampleBuilder(
            source_data=[small_fake_reader],
            selection_config=SampleSelectionConfig(
                code_type_prob_map={SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated})): 1.0},
                allow_db_lookups=False,
                fallback_to_orig=False,
            ),
        )
        assert len(sb) == 0


class TestSampleBuilderCodeSummaries:
    """Tests for code summary lookup in SampleBuilder."""

    def test_code_summary_from_prompt_db(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        tmp_path: pathlib.Path,
    ) -> None:
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
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.never,
            predict_type_prob_map={},
        )
        sb = SampleBuilder(
            source_data=[small_fake_reader],
            traces=None,
            transform_config=cfg,
            prompt_result_db_path=str(db_path),
            selection_config=default_selection_config(),
        )
        assert len(sb) == len(small_fake_reader)
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


class TestSampleBuilderStats:
    """Tests for SampleBuilder statistics."""

    def test_get_stats_returns_expected_keys(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        sb = SampleBuilder(source_data=[small_fake_reader], selection_config=default_selection_config())
        stats = sb.get_stats()
        expected_keys = [
            "orig_trace_count",
            "kept_trace_count",
            "kept_trace_family_count",
            "filtered/by_step_count",
            "filtered/by_code_length",
            "filtered/by_var_length",
            "filtered/by_trace_family_cap",
            "selected/failed_selections",
            "selected/with_full_trace_support",
            "selected/with_prompt_db_code",
            "selected/with_parent_fallback",
            "summaries_count",
            "sample_count",
        ]
        for key in expected_keys:
            assert key in stats


class TestSampleBuilderIndexAccess:
    """Tests for index access on SampleBuilder."""

    def test_getitem_returns_sample_data(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        sb = SampleBuilder(source_data=[small_fake_reader], selection_config=default_selection_config())
        assert len(sb) > 0
        sample = sb[0]
        assert sample.identifier is not None
        assert sample.code is not None
        assert sample.predict_type is not None

    def test_getitem_out_of_range_raises_error(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        sb = SampleBuilder(source_data=[small_fake_reader], selection_config=default_selection_config())
        with pytest.raises(IndexError):
            _ = sb[len(sb)]
        with pytest.raises(IndexError):
            _ = sb[-len(sb) - 1]

    def test_len_matches_sample_count(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        sb = SampleBuilder(source_data=[small_fake_reader], selection_config=default_selection_config())
        assert len(sb) == len(sb.selection_results.samples)


class TestSampleBuilderRealData:
    """Integration tests with real datasets."""

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.dataset
    @pytest.mark.skipif(
        tests.env_checks.TACO_TRACES_DATASET_MISSING,
        reason="TACO traces dataset is missing, cannot check sample generation",
    )
    def test_sample_generation_on_taco_traces(self) -> None:
        dataset_path = pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO")
        taco_reader = pyine.data.traces.dataset_reader.DatasetReader(dataset_path)
        if len(taco_reader) < 1000:
            pytest.skip("TACO traces dataset is too small, skipping sample generation test")
        cfg = SampleTransformConfig(
            transform_strategy=SampleTransformStrategy.hybrid,
            functions_fallback_to_segments=True,
            max_partial_trace_steps=100,
            min_partial_trace_steps=5,
            max_inputs_str_length=1000,
            max_output_str_length=1000,
            predict_type_prob_map={
                SamplePredictType.program_output: 0.5,
                SamplePredictType.frame_variables: 0.1,
                SamplePredictType.function_return: 0.4,
            },
        )
        sb = SampleBuilder(
            source_data=taco_reader,
            traces=None,
            transform_config=cfg,
            selection_config=default_selection_config(),
        )
        assert len(sb) <= len(taco_reader)
        for sample_idx in range(min(100, len(sb))):
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
            if sample.predict_type == SamplePredictType.program_output:
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
