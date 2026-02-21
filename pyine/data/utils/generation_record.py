"""Shared record schema and builder for generation-based LMDB exports.

Both ``DiskRewardLogger`` (reward pipeline) and ``DiskEvalLogger`` (eval pipeline) produce
LMDB records that share a common set of columns. This module defines the shared schema as a
TypedDict and provides a builder function to construct it.
"""

import collections.abc
import typing


class SharedGenerationRecordFields(typing.TypedDict):
    """Common columns shared between DiskRewardLogger and DiskEvalLogger records.

    All keys are always present (total=True). Unset fields use None explicitly
    for schema stability across records.
    """

    sample_id: str
    """Unique sample identifier."""
    model_output: str | None
    """Raw model output string."""
    prompt: str | None
    """Prompt text used as model input."""
    expected_output: str | None
    """Ground-truth expected output."""
    reasoning: str | None
    """Extracted reasoning field from parsed output, if available."""
    final_answer: str | None
    """Extracted final answer field from parsed output, if available."""
    predict_type: str | None
    """Prediction type for this sample (e.g., program_output, frame_variables)."""
    code_type: str | None
    """Code type for this sample (e.g., original, obfuscated)."""
    has_code_override: bool | None
    """Whether the sample has a code override applied."""
    pregenerated_output: str | None
    """Pre-generated model output, if this sample used one."""
    tags: list[str] | None
    """Arbitrary tags for grouping/filtering."""
    categories: list[str] | None
    """Category labels derived from sample metadata."""
    key_prefix: str
    """Key prefix used in the LMDB key for this record."""


def build_shared_record_fields(
    sample_id: str,
    *,
    model_output: str | None = None,
    prompt: str | None = None,
    expected_output: str | None = None,
    reasoning: str | None = None,
    final_answer: str | None = None,
    predict_type: str | None = None,
    code_type: str | None = None,
    has_code_override: bool | None = None,
    pregenerated_output: str | None = None,
    tags: collections.abc.Sequence[str] | None = None,
    categories: collections.abc.Sequence[str] | None = None,
    key_prefix: str = "",
) -> SharedGenerationRecordFields:
    """Build the shared portion of a generation record dict.

    Coerces sequences to lists for JSON serialization. Does NOT normalize ``key_prefix``; callers
    are responsible for normalizing once before calling this function. All keys are always present
    in the returned dict.

    Args:
        sample_id: Unique identifier for the sample.
        model_output: Raw model output string.
        prompt: Prompt text used as model input.
        expected_output: Ground-truth expected output.
        reasoning: Extracted reasoning field from parsed output.
        final_answer: Extracted final answer field from parsed output.
        predict_type: Prediction type for this sample.
        code_type: Code type for this sample.
        has_code_override: Whether sample has a code override.
        pregenerated_output: Pre-generated model output, if used.
        tags: Arbitrary tags for grouping/filtering.
        categories: Category labels derived from sample metadata.
        key_prefix: Key prefix used in LMDB keys (stored as-is).

    Returns:
        A dict with all shared columns populated.
    """
    return SharedGenerationRecordFields(
        sample_id=sample_id,
        model_output=model_output,
        prompt=prompt,
        expected_output=expected_output,
        reasoning=reasoning,
        final_answer=final_answer,
        predict_type=predict_type,
        code_type=code_type,
        has_code_override=has_code_override,
        pregenerated_output=pregenerated_output,
        tags=list(tags) if tags is not None else None,
        categories=list(categories) if categories is not None else None,
        key_prefix=key_prefix,
    )
