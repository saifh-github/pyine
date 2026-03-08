"""Shared helpers for formatting correctness records into model-ready inputs."""

from __future__ import annotations

import typing

import pyine.data.utils.generation_record
import pyine.evals.correctness.types as correctness_types
import pyine.utils.transformers.data

if typing.TYPE_CHECKING:
    import transformers


def get_text_value_from_record(
    record: correctness_types.EvalRecord,
    text_field: str,
) -> str:
    """Return the configured text field from an evaluation record.

    Args:
        record: Correctness evaluation record.
        text_field: Name of the ``EvalRecord`` attribute to use as assistant output.

    Returns:
        The configured text value.

    Raises:
        TypeError: If the selected field is not a string.
    """
    text_value = getattr(record, text_field, None)
    if not isinstance(text_value, str):
        raise TypeError(
            f"record {record.sample_id!r} has non-string value for text_field={text_field!r}: "
            f"{type(text_value).__name__}"
        )
    return text_value


def build_messages_from_eval_record(
    record: correctness_types.EvalRecord,
    text_field: str,
) -> list[dict[str, str]]:
    """Build the conversation messages for a correctness evaluation record.

    Args:
        record: Correctness evaluation record.
        text_field: Name of the ``EvalRecord`` attribute to use as assistant output.

    Returns:
        Structured conversation messages for the selected record/output field.
    """
    text_value = get_text_value_from_record(record, text_field)
    return pyine.data.utils.generation_record.build_messages_from_record(
        record=record.record,
        model_output=text_value,
    )


def format_record_for_tokenizer(
    record: correctness_types.EvalRecord,
    tokenizer: transformers.PreTrainedTokenizerBase,
    text_field: str,
) -> str:
    """Format one correctness record into the exact text fed to a tokenizer.

    Args:
        record: Correctness evaluation record.
        tokenizer: Tokenizer whose chat template/fallback formatting should be used.
        text_field: Name of the ``EvalRecord`` attribute to use as assistant output.

    Returns:
        Model-ready text string.
    """
    messages = build_messages_from_eval_record(record, text_field)
    return pyine.utils.transformers.data.format_messages_to_text(messages, tokenizer)


def format_records_for_tokenizer(
    records: list[correctness_types.EvalRecord],
    tokenizer: transformers.PreTrainedTokenizerBase,
    text_field: str,
) -> tuple[list[str], bool]:
    """Format correctness records into model-ready text strings.

    Args:
        records: Correctness evaluation records.
        tokenizer: Tokenizer whose chat template/fallback formatting should be used.
        text_field: Name of the ``EvalRecord`` attribute to use as assistant output.

    Returns:
        A pair of ``(texts, add_special_tokens)`` where ``add_special_tokens`` matches the
        training path semantics: special tokens are added only when the tokenizer does not already
        embed them via a chat template.
    """
    texts = [format_record_for_tokenizer(record, tokenizer, text_field) for record in records]
    add_special_tokens = not pyine.utils.transformers.data.tokenizer_has_chat_template(tokenizer)
    return texts, add_special_tokens


def get_input_formatting_metadata(
    tokenizer: transformers.PreTrainedTokenizerBase,
    text_field: str,
) -> dict[str, typing.Any]:
    """Return structured metadata describing correctness input construction.

    Args:
        tokenizer: Tokenizer whose formatting behavior is used.
        text_field: Name of the ``EvalRecord`` attribute used as assistant output.

    Returns:
        JSON-serializable metadata describing the effective formatting mode.
    """
    has_chat_template = pyine.utils.transformers.data.tokenizer_has_chat_template(tokenizer)
    return {
        "text_field": text_field,
        "input_construction": "eval_record_messages",
        "input_formatting_mode": "chat_template" if has_chat_template else "role_tagged_text",
        "tokenizer_has_chat_template": has_chat_template,
        "add_special_tokens": not has_chat_template,
        "truncation_side": getattr(tokenizer, "truncation_side", None),
    }
