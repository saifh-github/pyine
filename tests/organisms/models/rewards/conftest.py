"""Shared fixtures and helpers for reward package tests."""

import pytest

import pyine.organisms.datamodules.samples.common
import pyine.organisms.models.rewards.core.types


def make_sample_data(
    identifier: str,
    *,
    description: str = "",
    code: str = "print('hi')",
) -> pyine.organisms.datamodules.samples.common.SampleData:
    """Build a minimal `SampleData` instance for reward tests."""
    return pyine.organisms.datamodules.samples.common.SampleData(
        identifier=identifier,
        code=code,
        description=description,
        entrypoint="",
        first_line=0,
        last_line=1,
        inputs="",
        expected_output="hi\n",
        predict_type=pyine.organisms.datamodules.samples.common.SamplePredictType.program_output,
        code_type="original",
        trace_step_count=1,
        comma_separated_tags="",
        has_code_override=False,
        complexity_metrics={},
    )


@pytest.fixture
def sample_data() -> pyine.organisms.datamodules.samples.common.SampleData:
    """Default SampleData instance for tests."""
    return make_sample_data("test_sample")


def make_sample_context(
    *,
    prompt: str = "test prompt",
    model_output: str = "test output",
    identifier: str = "test_sample",
    parsed: pyine.organisms.models.rewards.core.types.ParsedOutput | None = None,
) -> pyine.organisms.models.rewards.core.types.SampleContext:
    """Build a SampleContext for testing."""
    return pyine.organisms.models.rewards.core.types.SampleContext(
        prompt=prompt,
        model_output=model_output,
        sample_data=make_sample_data(identifier),
        parsed=parsed,
    )


@pytest.fixture
def sample_context() -> pyine.organisms.models.rewards.core.types.SampleContext:
    """Default SampleContext instance for tests."""
    return make_sample_context()
