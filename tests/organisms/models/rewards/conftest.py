"""Shared fixtures and helpers for reward package tests."""

import pyine.organisms.datamodules.samples.common
import pyine.organisms.models.rewards.core.types
import pyine.utils.parsing


def make_sample_data(
    identifier: str,
    *,
    description: str = "",
    code: str = "print('hi')",
    code_type: str = "original",
    comma_separated_tags: str = "",
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
        code_type=code_type,
        trace_step_count=1,
        comma_separated_tags=comma_separated_tags,
        has_code_override=False,
        complexity_metrics={},
    )


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


def make_parsed_output(
    raw: str,
    *,
    final_answer: str | None = None,
    reasoning: str | None = None,
    final_tag: str = "final",
    include_diagnostics: bool = True,
) -> pyine.organisms.models.rewards.core.types.ParsedOutput:
    """Build a ParsedOutput with optional tag diagnostics for testing.

    When `include_diagnostics=True`, this populates the `fields` dict with tag statistics
    that mimic what `TagsOutputParser` would produce, enabling bonus reward tests.
    """
    fields: dict[str, str] = {}
    if include_diagnostics:
        open_re = pyine.utils.parsing.compile_open_tag_regex(final_tag)
        close_re = pyine.utils.parsing.compile_close_tag_regex(final_tag)
        open_count = len(open_re.findall(raw))
        close_matches = list(close_re.finditer(raw))
        close_count = len(close_matches)
        base = f"tags/{final_tag}/"
        fields[f"{base}open_count"] = str(open_count)
        fields[f"{base}close_count"] = str(close_count)
        fields[f"{base}block_count"] = str(min(open_count, close_count))
        fields[f"{base}has_nested_open"] = "false"
        fields[f"{base}has_stray_close"] = "false"
        fields[f"{base}has_unclosed_open"] = "false"
        if close_matches:
            last_close_end = close_matches[-1].end()
            fields[f"{base}last_close_end"] = str(last_close_end)
            trailing = raw[last_close_end:]
            fields[f"{base}stops_after_last_close"] = str(trailing.strip() == "").lower()
    return pyine.organisms.models.rewards.core.types.ParsedOutput(
        raw=raw,
        final_answer=final_answer,
        reasoning=reasoning,
        fields=fields,
    )
