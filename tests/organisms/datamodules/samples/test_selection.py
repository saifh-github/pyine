"""Tests for sample selection logic in the samples module."""

import pathlib

import pytest
from pytest_mock import MockerFixture

import pyine.data.traces.dataset_utils
import pyine.prompts
from pyine.organisms.datamodules.samples.common import (
    SampleCodeType,
    SampleCodeTypeSet,
    TraceDatasetToSampleCodeTypeMappings,
)
from pyine.organisms.datamodules.samples.configs import SampleSelectionConfig
from pyine.organisms.datamodules.samples.selection import (
    SelectedSample,
    select_samples_from_trace_families,
)
from tests.utils.fake_dataset_readers import FakeTraceDataConfig, FakeTraceDatasetReader


@pytest.fixture
def small_fake_reader() -> FakeTraceDatasetReader:
    """Provides a small fake dataset reader for selection tests."""
    cfg = FakeTraceDataConfig(
        dataset_name="FAKE",
        subset_name="select_test",
        num_problems=2,
        solutions_per_problem=2,
        tests_per_problem=2,
        augmented_per_solution=0,  # no fake augmentations, as they use unsupported types
        entrypoint_name="solution",
        max_events_per_line=50,
        seed=789,
        code_kind="function",
    )
    return FakeTraceDatasetReader(config=cfg)


@pytest.fixture
def empty_prompt_db(tmp_path: pathlib.Path) -> pyine.prompts.PromptResultDB:
    """Provides an empty prompt result database."""
    return pyine.prompts.PromptResultDB(str(tmp_path / "empty.sqlite"))


class TestSelectedSample:
    """Tests for the SelectedSample dataclass."""

    def test_basic_creation(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        trace_meta = small_fake_reader.trace_metadata[0]
        trace_id = trace_meta.trace_id
        parent_id = trace_id.get_augmentless_identifier()
        sample = SelectedSample(
            parent_id=parent_id,
            trace_id=trace_id,
            trace_meta=trace_meta,
            code_type=SampleCodeTypeSet.create_default(),
            code_override=None,
        )
        assert sample.parent_id == parent_id
        assert sample.trace_id == trace_id
        assert sample.code_type.is_original
        assert sample.code_override is None

    def test_with_code_override(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        trace_meta = small_fake_reader.trace_metadata[0]
        trace_id = trace_meta.trace_id
        parent_id = trace_id.get_augmentless_identifier()
        override_code = "def modified(): pass"
        sample = SelectedSample(
            parent_id=parent_id,
            trace_id=trace_id,
            trace_meta=trace_meta,
            code_type=SampleCodeTypeSet(frozenset({SampleCodeType.hinted})),
            code_override=override_code,
        )
        assert sample.code_override == override_code
        assert sample.code_type.is_hinted


def _default_selection_config() -> SampleSelectionConfig:
    """Returns a default selection config with explicit SampleCodeTypeSet keys."""
    return SampleSelectionConfig(
        code_type_prob_map={SampleCodeTypeSet.create_default(): 1.0},
    )


class TestSampleSelectionResults:
    """Tests for the SampleSelectionResults dataclass."""

    def test_len_returns_sample_count(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        empty_prompt_db: pyine.prompts.PromptResultDB,
    ) -> None:
        traces = small_fake_reader.trace_metadata
        trace_data = TraceDatasetToSampleCodeTypeMappings.create_from_traces(traces, empty_prompt_db)
        results = select_samples_from_trace_families(
            trace_data=trace_data,
            epoch=0,
            selection_config=_default_selection_config(),
            prompt_result_db=empty_prompt_db,
        )
        assert len(results) == len(results.samples)

    def test_getitem_returns_selected_sample(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        empty_prompt_db: pyine.prompts.PromptResultDB,
    ) -> None:
        traces = small_fake_reader.trace_metadata
        trace_data = TraceDatasetToSampleCodeTypeMappings.create_from_traces(traces, empty_prompt_db)
        results = select_samples_from_trace_families(
            trace_data=trace_data,
            epoch=0,
            selection_config=_default_selection_config(),
            prompt_result_db=empty_prompt_db,
        )
        assert len(results) > 0, "selection should return at least one sample with default config"
        sample = results[0]
        assert isinstance(sample, SelectedSample)
        assert sample.trace_id is not None
        assert sample.code_type is not None

    def test_get_code_type_counts(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        empty_prompt_db: pyine.prompts.PromptResultDB,
    ) -> None:
        traces = small_fake_reader.trace_metadata
        trace_data = TraceDatasetToSampleCodeTypeMappings.create_from_traces(traces, empty_prompt_db)
        results = select_samples_from_trace_families(
            trace_data=trace_data,
            epoch=0,
            selection_config=_default_selection_config(),
            prompt_result_db=empty_prompt_db,
        )
        counts = results.get_code_type_counts()
        total_count = sum(counts.values())
        assert total_count == len(results)


class TestSelectSamplesFromTraceFamilies:
    """Tests for the select_samples_from_trace_families function."""

    def test_select_original_only(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        empty_prompt_db: pyine.prompts.PromptResultDB,
    ) -> None:
        traces = small_fake_reader.trace_metadata
        trace_data = TraceDatasetToSampleCodeTypeMappings.create_from_traces(traces, empty_prompt_db)
        selection_config = SampleSelectionConfig(
            code_type_prob_map={SampleCodeTypeSet.create_default(): 1.0},
        )
        results = select_samples_from_trace_families(
            trace_data=trace_data,
            epoch=0,
            selection_config=selection_config,
            prompt_result_db=empty_prompt_db,
        )
        for sample in results.samples:
            assert sample.code_type.is_original
            assert sample.code_override is None

    def test_select_augmented_skips_when_unavailable(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        empty_prompt_db: pyine.prompts.PromptResultDB,
    ) -> None:
        traces = [t for t in small_fake_reader.trace_metadata if not t.is_augmented]
        trace_data = TraceDatasetToSampleCodeTypeMappings.create_from_traces(traces, empty_prompt_db)
        selection_config = SampleSelectionConfig(
            allow_db_lookups=False,
            code_type_prob_map={SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated})): 1.0},
            fallback_to_orig=False,
        )
        results = select_samples_from_trace_families(
            trace_data=trace_data,
            epoch=0,
            selection_config=selection_config,
            prompt_result_db=empty_prompt_db,
        )
        assert len(results) == 0
        assert results.failed_selections > 0

    def test_fallback_to_original_when_enabled(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        empty_prompt_db: pyine.prompts.PromptResultDB,
    ) -> None:
        traces = [t for t in small_fake_reader.trace_metadata if not t.is_augmented]
        trace_data = TraceDatasetToSampleCodeTypeMappings.create_from_traces(traces, empty_prompt_db)
        selection_config = SampleSelectionConfig(
            allow_db_lookups=False,
            code_type_prob_map={SampleCodeTypeSet(frozenset({SampleCodeType.obfuscated})): 1.0},
            fallback_to_orig=True,
        )
        results = select_samples_from_trace_families(
            trace_data=trace_data,
            epoch=0,
            selection_config=selection_config,
            prompt_result_db=empty_prompt_db,
        )
        assert len(results) > 0
        assert results.samples_with_parent_fallback > 0
        for sample in results.samples:
            assert sample.code_type.is_original

    def test_samples_per_family(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        empty_prompt_db: pyine.prompts.PromptResultDB,
    ) -> None:
        traces = small_fake_reader.trace_metadata
        trace_data = TraceDatasetToSampleCodeTypeMappings.create_from_traces(traces, empty_prompt_db)
        selection_config = SampleSelectionConfig(
            code_type_prob_map={SampleCodeTypeSet.create_default(): 1.0},
            samples_per_family=2,
        )
        results = select_samples_from_trace_families(
            trace_data=trace_data,
            epoch=0,
            selection_config=selection_config,
            prompt_result_db=empty_prompt_db,
        )
        num_families = len(trace_data.trace_families)
        expected_samples = num_families * 2
        assert len(results) == expected_samples

    def test_selection_is_deterministic_with_seed(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        empty_prompt_db: pyine.prompts.PromptResultDB,
    ) -> None:
        traces = small_fake_reader.trace_metadata
        trace_data = TraceDatasetToSampleCodeTypeMappings.create_from_traces(traces, empty_prompt_db)
        selection_config = SampleSelectionConfig(
            seed=42,
            code_type_prob_map={SampleCodeTypeSet.create_default(): 1.0},
        )
        results1 = select_samples_from_trace_families(
            trace_data=trace_data,
            epoch=0,
            selection_config=selection_config,
            prompt_result_db=empty_prompt_db,
        )
        results2 = select_samples_from_trace_families(
            trace_data=trace_data,
            epoch=0,
            selection_config=selection_config,
            prompt_result_db=empty_prompt_db,
        )
        ids1 = [s.trace_id for s in results1.samples]
        ids2 = [s.trace_id for s in results2.samples]
        assert ids1 == ids2

    def test_stats_are_consistent(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        empty_prompt_db: pyine.prompts.PromptResultDB,
    ) -> None:
        traces = small_fake_reader.trace_metadata
        trace_data = TraceDatasetToSampleCodeTypeMappings.create_from_traces(traces, empty_prompt_db)
        results = select_samples_from_trace_families(
            trace_data=trace_data,
            epoch=0,
            selection_config=_default_selection_config(),
            prompt_result_db=empty_prompt_db,
        )
        total_from_stats = (
            results.samples_with_full_trace_support
            + results.samples_with_prompt_db_code
            + results.samples_with_parent_fallback
        )
        assert total_from_stats == len(results)


class TestSelectionWithPromptDb:
    """Tests for selection using prompt result database lookups."""

    @pytest.fixture
    def prompt_db_with_hints(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        tmp_path: pathlib.Path,
    ) -> tuple[pyine.prompts.PromptResultDB, str]:
        """Creates a prompt DB with hint records for one trace."""
        db_path = tmp_path / "hints.sqlite"
        db = pyine.prompts.PromptResultDB(str(db_path))
        trace_meta = small_fake_reader.trace_metadata[0]
        trace_id_str = str(trace_meta.trace_id)
        hint_code = "# hint: this is hinted code\ndef solution(x): return x * 2"
        db.store(
            identifier=trace_id_str,
            prompt="hints/docs",
            result=hint_code,
            prompt_name="hints/docs",
            tags=["augment:hinted"],
        )
        return db, trace_id_str

    def test_db_lookup_finds_hinted_code(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        prompt_db_with_hints: tuple[pyine.prompts.PromptResultDB, str],
        mocker: MockerFixture,
    ) -> None:
        db, trace_id_str = prompt_db_with_hints
        traces = small_fake_reader.trace_metadata
        # patch list_prompts to include our test prompt name
        mocker.patch("pyine.prompts.manager.list_prompts", return_value=["hints/docs"])
        trace_data = TraceDatasetToSampleCodeTypeMappings.create_from_traces(traces, db)
        selection_config = SampleSelectionConfig(
            allow_db_lookups=True,
            code_type_prob_map={SampleCodeTypeSet(frozenset({SampleCodeType.hinted})): 1.0},
            fallback_to_orig=False,
        )
        results = select_samples_from_trace_families(
            trace_data=trace_data,
            epoch=0,
            selection_config=selection_config,
            prompt_result_db=db,
        )
        hinted_samples = [s for s in results.samples if s.code_type.is_hinted]
        assert len(hinted_samples) >= 1
        assert any(s.code_override is not None for s in hinted_samples)

    def test_db_lookup_disabled_skips_db_augmentations(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        prompt_db_with_hints: tuple[pyine.prompts.PromptResultDB, str],
    ) -> None:
        db, _ = prompt_db_with_hints
        traces = [t for t in small_fake_reader.trace_metadata if not t.is_augmented]
        trace_data = TraceDatasetToSampleCodeTypeMappings.create_from_traces(traces, db)
        selection_config = SampleSelectionConfig(
            allow_db_lookups=False,
            code_type_prob_map={SampleCodeTypeSet(frozenset({SampleCodeType.hinted})): 1.0},
            fallback_to_orig=False,
        )
        results = select_samples_from_trace_families(
            trace_data=trace_data,
            epoch=0,
            selection_config=selection_config,
            prompt_result_db=db,
        )
        assert len(results) == 0


class TestTraceDatasetToSampleCodeTypeMappings:
    """Tests for the TraceDatasetToSampleCodeTypeMappings class."""

    def test_create_from_traces_builds_families(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        empty_prompt_db: pyine.prompts.PromptResultDB,
    ) -> None:
        traces = small_fake_reader.trace_metadata
        mappings = TraceDatasetToSampleCodeTypeMappings.create_from_traces(traces, empty_prompt_db)
        assert len(mappings.trace_families) > 0
        total_members = sum(len(family) for family in mappings.trace_families.values())
        assert total_members == len(traces)

    def test_trace_metadata_lut_contains_all_traces(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        empty_prompt_db: pyine.prompts.PromptResultDB,
    ) -> None:
        traces = small_fake_reader.trace_metadata
        mappings = TraceDatasetToSampleCodeTypeMappings.create_from_traces(traces, empty_prompt_db)
        for trace in traces:
            assert trace.trace_id in mappings.trace_metadata_lut
            assert mappings.trace_metadata_lut[trace.trace_id] == trace

    def test_solutions_lut_maps_correctly(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        empty_prompt_db: pyine.prompts.PromptResultDB,
    ) -> None:
        traces = small_fake_reader.trace_metadata
        mappings = TraceDatasetToSampleCodeTypeMappings.create_from_traces(traces, empty_prompt_db)
        for solution_id, parent_ids in mappings.solutions_to_trace_family_parent_lut.items():
            for parent_id in parent_ids:
                assert parent_id in mappings.trace_families
                family = mappings.trace_families[parent_id]
                assert all(m.target_trace_meta.solution_id == solution_id for m in family.values())

    def test_problems_lut_maps_correctly(
        self,
        small_fake_reader: FakeTraceDatasetReader,
        empty_prompt_db: pyine.prompts.PromptResultDB,
    ) -> None:
        traces = small_fake_reader.trace_metadata
        mappings = TraceDatasetToSampleCodeTypeMappings.create_from_traces(traces, empty_prompt_db)
        for problem_id, parent_ids in mappings.problems_to_trace_family_parent_lut.items():
            for parent_id in parent_ids:
                assert parent_id in mappings.trace_families
                family = mappings.trace_families[parent_id]
                assert all(m.target_trace_meta.problem_id == problem_id for m in family.values())
