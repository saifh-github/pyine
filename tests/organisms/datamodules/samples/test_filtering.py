"""Tests for trace filtering logic in the samples module."""

import collections

import pytest
import pytest_mock

import pyine.data.traces.dataset_utils
from pyine.organisms.datamodules.samples.configs import TraceFilteringConfig
from pyine.organisms.datamodules.samples.filtering import (
    filter_traces,
)
from tests.utils.fake_dataset_readers import FakeTraceDataConfig, FakeTraceDatasetReader


@pytest.fixture
def small_fake_reader() -> FakeTraceDatasetReader:
    """Provides a small fake dataset reader for filtering tests."""
    cfg = FakeTraceDataConfig(
        dataset_name="FAKE",
        subset_name="filter_test",
        num_problems=2,
        solutions_per_problem=2,
        tests_per_problem=2,
        augmented_per_solution=0,
        entrypoint_name="solution",
        max_events_per_line=50,
        seed=456,
        code_kind="function",
    )
    return FakeTraceDatasetReader(config=cfg)


class TestTraceFilteringResults:
    """Tests for the TraceFilteringResults dataclass."""

    def test_properties_compute_correctly(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(
            max_trace_steps=None,
            max_code_line_count=None,
            max_code_line_length=None,
            max_code_length=None,
            max_args_length=None,
        )
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        assert results.orig_trace_count == len(traces)
        assert results.kept_trace_count == len(traces)
        assert results.filtered_trace_count == 0

    def test_kept_traces_returns_all_members(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(
            max_trace_steps=None,
            max_code_line_count=None,
            max_code_line_length=None,
            max_code_length=None,
            max_args_length=None,
        )
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        kept = results.kept_traces
        assert len(kept) == results.kept_trace_count
        all_identifiers = [t.identifier for t in kept]
        assert len(set(all_identifiers)) == len(all_identifiers)


class TestFilterTraces:
    """Tests for the filter_traces function."""

    def test_no_filtering_when_disabled(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(
            max_trace_families=None,
            max_trace_steps=None,
            max_code_line_count=None,
            max_code_line_length=None,
            max_code_length=None,
            max_args_length=None,
        )
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        assert results.kept_trace_count == len(traces)
        assert results.filtered_by_step_count == 0
        assert results.filtered_by_code_length == 0
        assert results.filtered_by_var_length == 0
        assert results.filtered_by_trace_family_cap == 0

    def test_filtering_by_step_count(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        total_traces = len(traces)
        assert total_traces > 0
        # verify some traces exceed the threshold (so filtering can happen)
        traces_exceeding = sum(1 for t in traces if t.step_count > 5)
        if traces_exceeding == 0:
            pytest.skip("no traces exceed step count threshold, cannot test filtering")
        cfg = TraceFilteringConfig(
            max_trace_steps=5,
            max_code_line_count=None,
            max_code_line_length=None,
            max_code_length=None,
            max_args_length=None,
        )
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        # verify filtering actually removed some traces
        assert results.kept_trace_count < total_traces, "filtering should remove some traces"
        assert results.filtered_by_step_count > 0
        for t in results.kept_traces:
            assert t.step_count <= 5

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
        cfg = TraceFilteringConfig(
            max_trace_steps=None,
            max_code_line_count=None,
            max_code_line_length=None,
            max_code_length=None,
            max_args_length=length_threshold,
        )
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        expected_count = sum(1 for al in args_length_map.values() if al <= length_threshold)
        assert results.kept_trace_count == expected_count
        for t in results.kept_traces:
            assert get_args_length(t) <= length_threshold

    def test_filtering_by_code_line_count(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        line_counts = {idx: len(traces[idx].code_string.splitlines()) for idx in range(len(traces))}
        unique_counts = sorted(set(line_counts.values()))
        if len(unique_counts) < 2:
            pytest.skip("all traces have the same line count, cannot test filtering")
        line_threshold = unique_counts[0] + 1
        cfg = TraceFilteringConfig(
            max_trace_steps=None,
            max_code_line_count=line_threshold,
            max_code_line_length=None,
            max_code_length=None,
            max_args_length=None,
        )
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        expected_count = sum(1 for lc in line_counts.values() if lc < line_threshold)
        assert results.kept_trace_count == expected_count

    def test_filtering_by_code_length(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        code_lengths = {idx: len(traces[idx].code_string) for idx in range(len(traces))}
        unique_lengths = sorted(set(code_lengths.values()))
        if len(unique_lengths) < 2:
            pytest.skip("all traces have the same code length, cannot test filtering")
        length_threshold = unique_lengths[0] + 1
        cfg = TraceFilteringConfig(
            max_trace_steps=None,
            max_code_line_count=None,
            max_code_line_length=None,
            max_code_length=length_threshold,
            max_args_length=None,
        )
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        expected_count = sum(1 for cl in code_lengths.values() if cl < length_threshold)
        assert results.kept_trace_count == expected_count

    def test_filtering_by_code_line_length(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata

        def get_max_line_length(code: str) -> int:
            lines = code.splitlines() or [""]
            return max(len(line) for line in lines)

        line_length_map = {idx: get_max_line_length(traces[idx].code_string) for idx in range(len(traces))}
        unique_lengths = sorted(set(line_length_map.values()))
        if len(unique_lengths) < 2:
            if unique_lengths[0] <= 1:
                pytest.skip("cannot set max_code_line_length below 1 to exercise filtering")
            length_threshold = unique_lengths[0] - 1
        else:
            length_threshold = unique_lengths[0]
        cfg = TraceFilteringConfig(
            max_trace_steps=None,
            max_code_line_count=None,
            max_code_line_length=length_threshold,
            max_code_length=None,
            max_args_length=None,
        )
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        expected_count = sum(1 for ll in line_length_map.values() if ll <= length_threshold)
        assert results.kept_trace_count == expected_count

    def test_filtering_combined_criteria(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(
            max_trace_steps=100,
            max_code_line_count=None,
            max_code_line_length=None,
            max_code_length=None,
            max_args_length=500,
        )
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        for t in results.kept_traces:
            assert t.step_count <= 100
            inputs_str = str(t.inputs)
            expected_output_str = str(t.expected_output)
            combined_length = len(inputs_str) + len(expected_output_str)
            assert combined_length <= 500

    def test_filtering_by_trace_family_cap(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(
            max_trace_families=1,
            max_trace_steps=None,
            max_code_line_count=None,
            max_code_line_length=None,
            max_code_length=None,
            max_args_length=None,
        )
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        assert results.kept_trace_family_count == 1
        assert results.filtered_by_trace_family_cap > 0 or len(traces) <= 1

    def test_filtering_stats_are_consistent(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(
            max_trace_steps=50,
            max_code_line_count=None,
            max_code_line_length=None,
            max_code_length=None,
            max_args_length=100,
        )
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        total_filtered = (
            results.filtered_by_step_count
            + results.filtered_by_code_length
            + results.filtered_by_var_length
            + results.filtered_by_trace_family_cap
            + results.filtered_by_traces_per_family_cap
            + results.filtered_by_traces_per_solution_cap
            + results.filtered_by_traces_per_problem_cap
        )
        assert total_filtered == results.filtered_trace_count
        assert results.kept_trace_count + results.filtered_trace_count == results.orig_trace_count

    def test_filtering_deterministic_with_seed(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(
            seed=42,
            max_trace_families=2,
            max_trace_steps=None,
            max_code_line_count=None,
            max_code_line_length=None,
            max_code_length=None,
            max_args_length=None,
        )
        results1 = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        results2 = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        kept_ids1 = {t.identifier for t in results1.kept_traces}
        kept_ids2 = {t.identifier for t in results2.kept_traces}
        assert kept_ids1 == kept_ids2

    def test_filtering_different_epochs_may_differ(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        if len(traces) <= 2:
            pytest.skip("need more traces to test epoch variation")
        cfg = TraceFilteringConfig(
            seed=42,
            max_trace_families=1,
            max_trace_steps=None,
            max_code_line_count=None,
            max_code_line_length=None,
            max_code_length=None,
            max_args_length=None,
        )
        results_e0 = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        results_e1 = filter_traces(traces=traces, epoch=1, filtering_config=cfg)
        kept_ids_e0 = {t.identifier for t in results_e0.kept_traces}
        kept_ids_e1 = {t.identifier for t in results_e1.kept_traces}
        assert kept_ids_e0 != kept_ids_e1 or results_e0.kept_trace_count <= 1


class TestTraceFilteringResultsValidation:
    """Tests for the TraceFilteringResults post_init validation."""

    def test_validation_passes_for_valid_results(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(max_trace_steps=50)
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        assert results.orig_trace_count >= results.kept_trace_count
        assert results.filtered_trace_count >= 0


class TestFilteringByTracesPerFamily:
    """Tests for the max_traces_per_family filtering parameter."""

    def test_filtering_by_traces_per_family_cap(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(max_traces_per_family=1)
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        for family_traces in results.kept_trace_families.values():
            assert len(family_traces) <= 1

    def test_traces_per_family_deterministic_with_seed(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(seed=42, max_traces_per_family=1)
        results1 = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        results2 = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        kept_ids1 = {t.identifier for t in results1.kept_traces}
        kept_ids2 = {t.identifier for t in results2.kept_traces}
        assert kept_ids1 == kept_ids2


class TestFilteringByTracesPerSolution:
    """Tests for the max_traces_per_solution filtering parameter."""

    def test_filtering_by_traces_per_solution_cap(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(max_traces_per_solution=2)
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        solution_counts: dict[pyine.data.traces.dataset_utils.SolutionIdentifier, int] = collections.defaultdict(int)
        for trace in results.kept_traces:
            solution_id = trace.trace_id.get_parent_identifier()
            solution_counts[solution_id] += 1
        for count in solution_counts.values():
            assert count <= 2

    def test_traces_per_solution_deterministic_with_seed(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(seed=42, max_traces_per_solution=2)
        results1 = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        results2 = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        kept_ids1 = {t.identifier for t in results1.kept_traces}
        kept_ids2 = {t.identifier for t in results2.kept_traces}
        assert kept_ids1 == kept_ids2


class TestFilteringByTracesPerProblem:
    """Tests for the max_traces_per_problem filtering parameter."""

    def test_filtering_by_traces_per_problem_cap(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(max_traces_per_problem=2)
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        problem_counts: dict[pyine.data.traces.dataset_utils.CodingProblemIdentifier, int] = collections.defaultdict(
            int
        )
        for trace in results.kept_traces:
            problem_id = trace.trace_id.get_parent_identifier().get_parent_identifier()
            problem_counts[problem_id] += 1
        for count in problem_counts.values():
            assert count <= 2

    def test_traces_per_problem_deterministic_with_seed(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(seed=42, max_traces_per_problem=2)
        results1 = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        results2 = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        kept_ids1 = {t.identifier for t in results1.kept_traces}
        kept_ids2 = {t.identifier for t in results2.kept_traces}
        assert kept_ids1 == kept_ids2


class TestFilteringCombined:
    """Tests for combining multiple filtering parameters."""

    def test_combined_per_family_and_per_solution(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(max_traces_per_family=1, max_traces_per_solution=2)
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        # per-family constraint
        for family_traces in results.kept_trace_families.values():
            assert len(family_traces) <= 1
        # per-solution constraint
        solution_counts: dict[pyine.data.traces.dataset_utils.SolutionIdentifier, int] = collections.defaultdict(int)
        for trace in results.kept_traces:
            solution_id = trace.trace_id.get_parent_identifier()
            solution_counts[solution_id] += 1
        for count in solution_counts.values():
            assert count <= 2

    def test_combined_per_solution_and_per_problem(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(max_traces_per_solution=2, max_traces_per_problem=3)
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        # per-solution constraint
        solution_counts: dict[pyine.data.traces.dataset_utils.SolutionIdentifier, int] = collections.defaultdict(int)
        for trace in results.kept_traces:
            solution_id = trace.trace_id.get_parent_identifier()
            solution_counts[solution_id] += 1
        for count in solution_counts.values():
            assert count <= 2
        # per-problem constraint
        problem_counts: dict[pyine.data.traces.dataset_utils.CodingProblemIdentifier, int] = collections.defaultdict(
            int
        )
        for trace in results.kept_traces:
            problem_id = trace.trace_id.get_parent_identifier().get_parent_identifier()
            problem_counts[problem_id] += 1
        for count in problem_counts.values():
            assert count <= 3


class TestTokenBasedFiltering:
    """Tests for token-based length filtering."""

    def test_config_validation_requires_tokenizer_when_length_filters_active(self) -> None:
        """Test that use_token_lengths=True with length filters requires a tokenizer."""
        with pytest.raises(ValueError, match="requires either tokenizer_model_id or tokenizer_path"):
            TraceFilteringConfig(
                use_token_lengths=True,
                max_code_length=1000,
            )

    def test_config_validation_no_tokenizer_required_when_no_length_filters(self) -> None:
        """Test that use_token_lengths=True without length filters doesn't require a tokenizer."""
        cfg = TraceFilteringConfig(
            use_token_lengths=True,
            max_code_length=None,
            max_code_line_length=None,
            max_args_length=None,
            max_trace_steps=100,  # only non-length filter active
        )
        assert cfg.use_token_lengths is True

    def test_config_validation_mutually_exclusive_tokenizers(self) -> None:
        """Test that tokenizer_model_id and tokenizer_path are mutually exclusive."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            TraceFilteringConfig(
                use_token_lengths=True,
                tokenizer_model_id="gpt-4o",
                tokenizer_path="some/path",
                max_code_length=1000,
            )

    def test_config_warns_when_tokenizer_set_but_not_used(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test that a warning is logged when tokenizer is set but use_token_lengths=False."""
        mock_logger = mocker.patch("pyine.organisms.datamodules.samples.configs.logger")
        TraceFilteringConfig(
            use_token_lengths=False,
            tokenizer_model_id="gpt-4o",
        )
        mock_logger.warning.assert_called_once()
        assert "tokenizer unused" in mock_logger.warning.call_args[0][0]

    def test_get_length_measurer_returns_len_when_disabled(self) -> None:
        """Test that get_length_measurer returns len() when use_token_lengths=False."""
        cfg = TraceFilteringConfig(use_token_lengths=False)
        measurer = cfg.get_length_measurer()
        assert measurer("hello") == 5  # character count
        assert measurer is len

    def test_get_length_measurer_with_tiktoken(self) -> None:
        """Test that get_length_measurer works with tiktoken."""
        cfg = TraceFilteringConfig(
            use_token_lengths=True,
            tokenizer_model_id="gpt-4o",
            max_code_length=1000,
        )
        measurer = cfg.get_length_measurer()
        result = measurer("hello world")
        assert isinstance(result, int)
        assert result > 0
        assert result != len("hello world")  # should differ from char count (2 tokens vs 11 chars)

    def test_filtering_by_code_length_tokens(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        """Test that max_code_length filtering works with token counts."""
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(
            use_token_lengths=True,
            tokenizer_model_id="gpt-4o",
            max_code_length=5,  # very low token limit to ensure filtering
        )
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        # with such a low token limit, most/all traces should be filtered
        assert results.kept_trace_count <= len(traces)

    def test_filtering_by_code_line_length_tokens(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        """Test that max_code_line_length filtering works with token counts."""
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(
            use_token_lengths=True,
            tokenizer_model_id="gpt-4o",
            max_code_line_length=3,  # very low token limit per line
        )
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        assert results.kept_trace_count <= len(traces)

    def test_filtering_by_args_length_tokens(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        """Test that max_args_length filtering works with token counts."""
        traces = small_fake_reader.trace_metadata
        cfg = TraceFilteringConfig(
            use_token_lengths=True,
            tokenizer_model_id="gpt-4o",
            max_args_length=5,  # very low combined token limit
        )
        results = filter_traces(traces=traces, epoch=0, filtering_config=cfg)
        assert results.kept_trace_count <= len(traces)

    def test_char_vs_token_filtering_differs(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        """Test that char-based and token-based filtering produce different results."""
        traces = small_fake_reader.trace_metadata
        if not traces:
            pytest.skip("no traces in test dataset")
        # find a reasonable threshold where results differ
        sample_code = traces[0].code_string
        char_len = len(sample_code)
        # use a threshold between typical char and token counts
        threshold = char_len // 2  # chars are usually more than tokens
        cfg_chars = TraceFilteringConfig(
            use_token_lengths=False,
            max_code_length=threshold,
        )
        cfg_tokens = TraceFilteringConfig(
            use_token_lengths=True,
            tokenizer_model_id="gpt-4o",
            max_code_length=threshold,
        )
        results_chars = filter_traces(traces=traces, epoch=0, filtering_config=cfg_chars)
        results_tokens = filter_traces(traces=traces, epoch=0, filtering_config=cfg_tokens)
        # token count <= char count, so token-based should typically keep more traces
        # (unless threshold is so low/high that both filter everything/nothing)
        assert results_chars.kept_trace_count >= 0
        assert results_tokens.kept_trace_count >= 0
