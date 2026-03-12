"""Shared record schema and builder for generation-based LMDB exports.

Both ``DiskRewardLogger`` (reward pipeline) and ``DiskEvalLogger`` (eval pipeline) produce
LMDB records that share a common set of columns. This module defines the shared schema as a
TypedDict and provides a builder function to construct it. Additionally, it provides a utility
to extract structured message lists from records for training pipelines.
"""

import collections.abc
import logging
import typing

logger = logging.getLogger(__name__)

_warned_prompt_fallback = False


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
    sample_data: dict[str, typing.Any] | None
    """Serialized SampleData NamedTuple from the datamodule pipeline."""


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
    sample_data: typing.Any = None,  # pyine.organisms.datamodules.samples.common.SampleData | None
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
        sample_data: Full SampleData NamedTuple to serialize into the record.

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
        sample_data=sample_data._asdict() if sample_data is not None else None,
    )


def restore_sample_data_from_record(
    record: dict[str, typing.Any],
) -> typing.Any:
    """Reconstruct a SampleData from a serialized LMDB record.

    Handles enum coercion (predict_type string -> SamplePredictType).

    Args:
        record: A single LMDB record dict containing a ``sample_data`` nested dict.

    Returns:
        Reconstructed ``pyine.organisms.datamodules.samples.common.SampleData`` NamedTuple.

    Raises:
        ValueError: If ``record["sample_data"]`` is None, missing, or malformed.
    """
    import pyine.organisms.datamodules.samples.common  # runtime import to avoid reverse dependency

    raw = record.get("sample_data")
    if raw is None:
        raise ValueError(
            f"record {record.get('sample_id', '<unknown>')!r} has no sample_data "
            "(None or missing); cannot restore SampleData"
        )
    if not isinstance(raw, dict):
        raise ValueError(
            f"record {record.get('sample_id', '<unknown>')!r} has sample_data of type "
            f"{type(raw).__name__}; expected dict"
        )
    fixed = dict(typing.cast("dict[str, typing.Any]", raw))
    if "predict_type" in fixed:
        fixed["predict_type"] = pyine.organisms.datamodules.samples.common.SamplePredictType(fixed["predict_type"])
    # filter to known fields and drop any extras (forward-compat with schema additions/removals)
    known_fields = set(pyine.organisms.datamodules.samples.common.SampleData._fields)
    extra_keys = set(fixed.keys()) - known_fields
    if extra_keys:
        logger.debug(f"dropping unknown sample_data fields: {sorted(extra_keys)}")
        fixed = {k: v for k, v in fixed.items() if k in known_fields}
    return pyine.organisms.datamodules.samples.common.SampleData(**fixed)


def build_messages_from_record(
    record: dict[str, typing.Any],
    model_output: str,
) -> list[dict[str, str]]:
    """Build a structured message list from an LMDB record and model output.

    Attempts to reconstruct the conversation as a list of role-attributed messages suitable for
    ``tokenizer.apply_chat_template()``.

    Resolution order:
    1. If ``record["prompt_messages"]`` exists, use it as-is and append the assistant reply.
    2. Else, if ``record["prompt"]`` exists, construct a two-message conversation (user + assistant)
       with a warning about manual reassembly.
    3. Otherwise, raise ``ValueError``, as the record is malformed.

    Args:
        record: full LMDB record dict (produced by DiskRewardLogger or DiskEvalLogger).
        model_output: Model output string to use as the assistant message content.

    Returns:
        List of messages, i.e. ``{"role": ..., "content": ...}`` dicts.

    Raises:
        ValueError: If neither ``prompt_messages`` nor ``prompt`` is available in the record.
    """
    sample_id = record.get("sample_id", "<unknown>")
    prompt_messages = record.get("prompt_messages")
    if prompt_messages is not None:
        if not isinstance(prompt_messages, list) or not prompt_messages:
            raise ValueError(f"record {sample_id!r} has invalid prompt_messages; expected non-empty list")
        validated_messages = typing.cast("list[dict[str, str]]", prompt_messages)
        for msg in validated_messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ValueError(
                    f"record {sample_id!r} has invalid prompt_messages; each message must have role and content"
                )
        if validated_messages[-1].get("role") == "assistant":
            raise ValueError(
                f"record {sample_id!r} has prompt_messages ending with an assistant turn; "
                "this would cause a duplicate assistant message, fix the data export "
                "or strip the trailing assistant message from prompt_messages"
            )
        messages: list[dict[str, str]] = [dict(msg) for msg in validated_messages]
        messages.append({"role": "assistant", "content": model_output})
        return messages
    prompt = record.get("prompt")
    if prompt is not None:
        global _warned_prompt_fallback  # noqa: PLW0603
        if not _warned_prompt_fallback:
            logger.warning(
                f"record {sample_id!r} (and possibly others) has 'prompt' but no 'prompt_messages'; "
                "manually reassembling inputs; multi-turn context/system prompts may be lost"
            )
            _warned_prompt_fallback = True
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": model_output},
        ]
    raise ValueError(f"record for {sample_id!r} has neither 'prompt_messages' nor 'prompt'; cannot build messages")
