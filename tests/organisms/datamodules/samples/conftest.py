"""Shared fixtures for sample transform tests."""

import typing

import numpy as np
import pytest

import pyine.data.traces.dataset_utils
import pyine.utils.code.execution as exec_utils
from pyine.organisms.datamodules.samples.common import (
    SampleCodeTypeSet,
    SamplePredictType,
    SampleTransformStrategy,
)
from pyine.organisms.datamodules.samples.configs import SampleTransformConfig
from pyine.organisms.datamodules.samples.selection import SampleSelectionSource, SelectedSample

# test code snippets for various scenarios

CODE_SIMPLE_FUNCTION = """
def compute(x):
    result = x * 2
    return result
""".strip()

CODE_FUNCTION_WITH_EXCEPTION = """
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
""".strip()

CODE_RECURSIVE_FUNCTION = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""".strip()

CODE_NESTED_FUNCTIONS = """
def inner(x):
    return x + 1

def outer(x):
    y = inner(x)
    z = inner(y)
    return z
""".strip()

CODE_LOOP_MULTIPLE_ITERATIONS = """
def sum_range(n):
    total = 0
    for i in range(n):
        total += i
    return total
""".strip()

CODE_GLOBAL_AND_LOCAL = """
MULTIPLIER = 10

def compute(x):
    local_val = x * MULTIPLIER
    return local_val
""".strip()

CODE_MULTI_LINE_LOOP = """
def process(n):
    result = 0
    for i in range(n):
        temp = i * 2
        result += temp
    return result
""".strip()


@pytest.fixture
def trace_from_code() -> typing.Callable[..., exec_utils.TraceResult]:
    """Factory fixture to generate real traces from code snippets.

    Tests using this fixture should be marked with @pytest.mark.slow
    as real execution tracing takes more time than mocked tests.
    """

    def _trace(
        code: str,
        inputs: typing.Any,
        entrypoint: str | None = None,
        expected_output: typing.Any | None = None,
        identifier: str = "TEST/unit/p000000/s0000/t0000",
    ) -> exec_utils.TraceResult:
        return exec_utils.execute_and_trace_code(
            code_string=code,
            inputs=inputs,
            expected_output=expected_output,
            identifier=identifier,
            entrypoint_name=entrypoint,
            trace_only_inside_code_string=True,
            use_safe_execution=False,
            seed=42,
        )

    return _trace


@pytest.fixture
def default_transform_config() -> SampleTransformConfig:
    """Returns a SampleTransformConfig suitable for testing partial samples."""
    return SampleTransformConfig(
        transform_strategy=SampleTransformStrategy.always,
        predict_type_prob_map={
            SamplePredictType.function_return: 0.5,
            SamplePredictType.frame_variables: 0.5,
        },
        max_partial_trace_steps=100,
        min_partial_trace_steps=1,
        max_inputs_str_length=1000,
        max_output_str_length=1000,
    )


@pytest.fixture
def function_return_only_config() -> SampleTransformConfig:
    """Config that only produces function_return samples."""
    return SampleTransformConfig(
        transform_strategy=SampleTransformStrategy.always,
        predict_type_prob_map={SamplePredictType.function_return: 1.0},
        max_partial_trace_steps=100,
        min_partial_trace_steps=1,
        max_inputs_str_length=1000,
        max_output_str_length=1000,
    )


@pytest.fixture
def frame_variables_only_config() -> SampleTransformConfig:
    """Config that only produces frame_variables samples."""
    return SampleTransformConfig(
        transform_strategy=SampleTransformStrategy.always,
        predict_type_prob_map={SamplePredictType.frame_variables: 1.0},
        max_partial_trace_steps=100,
        min_partial_trace_steps=1,
        max_inputs_str_length=1000,
        max_output_str_length=1000,
    )


@pytest.fixture
def default_rng() -> np.random.Generator:
    """Returns a deterministic RNG for tests."""
    return np.random.default_rng(seed=42)


def make_trace_metadata(
    trace_result: exec_utils.TraceResult,
    dataset: str = "TEST",
    subset: str = "unit",
    problem_idx: int = 0,
    solution_idx: int = 0,
    test_idx: int = 0,
) -> pyine.data.traces.dataset_utils.TraceMetadata:
    """Creates a TraceMetadata object from a TraceResult for testing.

    If the trace_result has an identifier, this function uses a matching TraceIdentifier
    to ensure consistency.
    """
    # use the trace_result's identifier if available, otherwise construct one
    if trace_result.identifier is not None:
        identifier_str = trace_result.identifier
    else:
        trace_id = pyine.data.traces.dataset_utils.TraceIdentifier(
            dataset=dataset,
            subset=subset,
            problem_idx=problem_idx,
            solution_idx=solution_idx,
            test_idx=test_idx,
        )
        identifier_str = str(trace_id)
    return pyine.data.traces.dataset_utils.TraceMetadata(
        identifier=identifier_str,
        parent_dataset_hash="test_hash",
        index=0,
        internal_index=0,
        step_count=trace_result.valid_step_count,
        code_string=trace_result.code_string,
        inputs=trace_result.inputs,
        expected_output=trace_result.expected_output,
        return_value=trace_result.return_value,
        exception=trace_result.exception,
        stdout=trace_result.stdout,
        stderr=trace_result.stderr,
        metadata=trace_result.metadata,
        tags=[],
    )


def make_selected_sample(
    trace_result: exec_utils.TraceResult,
    trace_meta: pyine.data.traces.dataset_utils.TraceMetadata | None = None,
    code_type: SampleCodeTypeSet | None = None,
    code_override: str | None = None,
) -> SelectedSample:
    """Creates a SelectedSample from a TraceResult for testing."""
    if trace_meta is None:
        trace_meta = make_trace_metadata(trace_result)
    trace_id = trace_meta.trace_id
    parent_id = trace_id.get_augmentless_identifier()
    return SelectedSample(
        parent_id=parent_id,
        trace_id=trace_id,
        trace_meta=trace_meta,
        code_type=code_type or SampleCodeTypeSet.create_default(),
        selection_source=SampleSelectionSource.full_trace,
        code_override=code_override,
    )
