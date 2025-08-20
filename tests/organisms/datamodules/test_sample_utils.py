import pytest

import pyine.utils.code.execution as exec_utils
from pyine.organisms.datamodules.sample_utils import (
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


def test_len_matches_targets(small_fake_reader: FakeTraceDatasetReader) -> None:
    targets = make_targets(small_fake_reader, [0, 1, 2])
    cfg = SampleTransformConfig(partial_sample_decision_strategy="never")  # noqa
    sb = SampleBuilder(readers=[small_fake_reader], traces=targets, config=cfg)  # noqa
    assert len(sb) == len(targets)


class TestSampleBuilderFullSamples:

    def test_full_trace_when_never(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        targets = make_targets(small_fake_reader, [0])
        cfg = SampleTransformConfig(
            partial_sample_decision_strategy="never",
            random_seed=123,
        )
        sb = SampleBuilder(readers=[small_fake_reader], traces=targets, config=cfg)  # noqa
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
        # step count should count only non-None events
        assert sample.trace_step_count == len([s for s in tr.traced_steps if s is not None])


class TestSampleBuilderPartialSamples:

    def test_partial_sample_basic(self, small_fake_reader: FakeTraceDatasetReader) -> None:
        targets = make_targets(small_fake_reader, [0])
        cfg = SampleTransformConfig(
            partial_sample_decision_strategy="always",
            random_seed=42,
            max_partial_trace_steps=3,
            output_type_prob_map={  # noqa
                "frame variables": 1.0,
            },
        )
        sb = SampleBuilder(readers=[small_fake_reader], traces=targets, config=cfg)  # noqa
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
            output_type_prob_map={  # noqa
                "function return": 1.0,
            },
        )
        sb = SampleBuilder(readers=[small_fake_reader], traces=targets, config=cfg)  # noqa
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
            output_type_prob_map={  # noqa
                "frame variables": 1.0,
            },
        )
        sb = SampleBuilder(readers=[small_fake_reader], traces=targets, config=cfg)  # noqa
        sample = sb[0]
        # since caps reject partial sample, we should have fallen back to full program output
        assert sample.output_type == "program output"
        tr = small_fake_reader[0]
        assert sample.output == tr.expected_output
