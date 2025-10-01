import pytest

import pyine.data.deltas.dataset_utils as deltas_utils
import pyine.data.traces.dataset_utils as traces_utils
import pyine.utils.code.execution as exec_utils
from tests.utils.fake_dataset_readers import (
    FakeDeltaDatasetReader,
    FakeTraceDataConfig,
    FakeTraceDatasetReader,
)


@pytest.fixture()
def small_config() -> FakeTraceDataConfig:
    return FakeTraceDataConfig(
        dataset_name="FAKE",
        subset_name="unit",
        num_problems=1,
        solutions_per_problem=2,
        tests_per_problem=2,
        augmented_per_solution=1,
        entrypoint_name="solution",
        max_events_per_line=50,
        seed=123,
        code_kind="function",
    )


@pytest.fixture()
def trace_reader(
    small_config: FakeTraceDataConfig,
) -> FakeTraceDatasetReader:
    return FakeTraceDatasetReader(config=small_config)


@pytest.fixture()
def delta_reader(
    small_config: FakeTraceDataConfig,
) -> FakeDeltaDatasetReader:
    return FakeDeltaDatasetReader(config=small_config)


class TestFakeTraceDatasetReader:
    def test_lengths_and_keys_are_consistent(
        self,
        trace_reader: FakeTraceDatasetReader,
        small_config: FakeTraceDataConfig,
    ) -> None:
        expected_traces = (
            small_config.num_problems
            * small_config.solutions_per_problem
            * small_config.tests_per_problem
            * (1 + small_config.augmented_per_solution)
        )
        assert len(trace_reader) == expected_traces
        assert len(trace_reader._trace_indices) == expected_traces
        assert len(trace_reader.trace_keys) == expected_traces
        assert len(trace_reader._problem_indices) == small_config.num_problems
        assert len(trace_reader.problem_keys) == small_config.num_problems

        assert "parent_dataset" in trace_reader.metadata
        assert "dataset_name" in trace_reader.metadata["parent_dataset"]

    def test_getitem_by_index_and_key(
        self,
        trace_reader: FakeTraceDatasetReader,
    ) -> None:
        idx = 0
        trace_by_idx = trace_reader[idx]
        assert isinstance(trace_by_idx, exec_utils.TraceResult)

        key = trace_reader.trace_keys[idx]
        trace_by_key = trace_reader[key]
        assert isinstance(trace_by_key, exec_utils.TraceResult)
        assert trace_by_key.identifier == trace_by_idx.identifier

    def test_get_problem_data_and_mapping(
        self,
        trace_reader: FakeTraceDatasetReader,
    ) -> None:
        idx = 0
        problem = trace_reader.get_problem_data(idx)
        assert isinstance(problem, traces_utils.CodingProblem)

        key = trace_reader.trace_keys[idx]
        parent_problem_key = trace_reader.trace_key_to_problem_key[key]
        assert parent_problem_key in trace_reader.problem_keys

    def test_augment_parent_mappings(
        self,
        trace_reader: FakeTraceDatasetReader,
        small_config: FakeTraceDataConfig,
    ) -> None:
        expected_aug_count = (
            small_config.num_problems
            * small_config.solutions_per_problem
            * small_config.tests_per_problem
            * small_config.augmented_per_solution
        )
        assert len(trace_reader.augment_key_to_parent_trace_key) == expected_aug_count
        for aug_key, parent_key in trace_reader.augment_key_to_parent_trace_key.items():
            assert parent_key in trace_reader.trace_keys
            assert aug_key in trace_reader.trace_keys

    def test_index_and_key_errors(
        self,
        trace_reader: FakeTraceDatasetReader,
    ) -> None:
        with pytest.raises(IndexError):
            _ = trace_reader[len(trace_reader)]
        with pytest.raises(KeyError):
            _ = trace_reader["__non_existing_key__"]
        with pytest.raises(IndexError):
            _ = trace_reader.get_problem_data(len(trace_reader))
        with pytest.raises(KeyError):
            _ = trace_reader.get_problem_data("__non_existing_key__")


class TestFakeDeltaDatasetReader:
    def test_lengths_and_keys_are_consistent(
        self,
        delta_reader: FakeDeltaDatasetReader,
    ) -> None:
        assert len(delta_reader._deltas_indices) == len(delta_reader)
        assert len(delta_reader.deltas_keys) == len(delta_reader)
        assert len(delta_reader._trace_indices) == len(delta_reader.trace_keys) == len(delta_reader)
        # deltas keys match the corresponding trace keys + suffix
        for tkey, dkey in zip(delta_reader.trace_keys, delta_reader.deltas_keys, strict=False):
            assert dkey.startswith(tkey)
            assert dkey.endswith(traces_utils.DELTAS_SUFFIX)

    def test_getitem_and_get_trace_data(
        self,
        delta_reader: FakeDeltaDatasetReader,
    ) -> None:
        idx = 0
        deltas_by_idx = delta_reader[idx]
        assert isinstance(deltas_by_idx, deltas_utils.TraceDeltaList)
        tkey = delta_reader.trace_keys[idx]
        deltas_by_tkey = delta_reader[tkey]
        assert isinstance(deltas_by_tkey, deltas_utils.TraceDeltaList)
        dkey = delta_reader.deltas_keys[idx]
        deltas_by_dkey = delta_reader[dkey]
        assert isinstance(deltas_by_dkey, deltas_utils.TraceDeltaList)
        # get the corresponding trace and verify id consistency
        trace_res = delta_reader.get_trace_data(idx)
        assert isinstance(trace_res, exec_utils.TraceResult)
        assert deltas_by_idx.trace_id == str(trace_res.identifier)

    def test_index_and_key_errors(
        self,
        delta_reader: FakeDeltaDatasetReader,
    ) -> None:
        with pytest.raises(IndexError):
            _ = delta_reader[len(delta_reader)]
        with pytest.raises(AssertionError):
            _ = delta_reader["__non_existing_key__"]
