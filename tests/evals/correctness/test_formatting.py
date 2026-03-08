"""Tests for shared correctness input-formatting helpers."""

from __future__ import annotations

import dataclasses
import typing

import pytest

import pyine.evals.correctness.formatting as correctness_formatting
import pyine.evals.correctness.types as correctness_types


def _make_record(
    *,
    model_output: str = "model output",
    final_answer: str = "final answer",
) -> correctness_types.EvalRecord:
    return correctness_types.EvalRecord(
        sample_id="sample-1",
        problem_id="problem-1",
        attempt_index=0,
        model_output=model_output,
        final_answer=final_answer,
        expected_output="expected",
        label=True,
        code_type="original",
        tags=[],
        record={
            "sample_id": "sample-1",
            "prompt": "solve the problem",
        },
        difficulty_score=None,
    )


class _ChatTemplateTokenizer:
    chat_template = "dummy-template"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = False,
    ) -> str:
        assert not tokenize
        return " <sep> ".join(f"{message['role']}={message['content']}" for message in conversation)


class TestBuildMessagesFromEvalRecord:
    def test_uses_requested_text_field(self) -> None:
        record = _make_record(model_output="reasoning trace", final_answer="42")

        messages = correctness_formatting.build_messages_from_eval_record(record, text_field="final_answer")

        assert messages[-1] == {"role": "assistant", "content": "42"}

    def test_raises_for_non_string_text_field(self) -> None:
        record = dataclasses.replace(_make_record(), final_answer=typing.cast("typing.Any", None))

        with pytest.raises(TypeError, match="non-string value"):
            correctness_formatting.build_messages_from_eval_record(record, text_field="final_answer")


class TestFormatRecordsForTokenizer:
    def test_returns_role_tagged_text_without_chat_template(self) -> None:
        record = _make_record(model_output="reasoning trace")
        tokenizer = typing.cast("typing.Any", object())

        texts, add_special_tokens = correctness_formatting.format_records_for_tokenizer(
            [record],
            tokenizer,
            text_field="model_output",
        )

        assert texts == ["user: solve the problem\n\nassistant: reasoning trace"]
        assert add_special_tokens is True

    def test_disables_special_tokens_when_chat_template_is_used(self) -> None:
        record = _make_record(model_output="reasoning trace")
        tokenizer = _ChatTemplateTokenizer()

        texts, add_special_tokens = correctness_formatting.format_records_for_tokenizer(
            [record],
            tokenizer,  # type: ignore[arg-type]
            text_field="model_output",
        )

        assert texts == ["user=solve the problem <sep> assistant=reasoning trace"]
        assert add_special_tokens is False

    def test_get_input_formatting_metadata_without_chat_template(self) -> None:
        tokenizer = typing.cast("typing.Any", object())

        metadata = correctness_formatting.get_input_formatting_metadata(tokenizer, text_field="model_output")

        assert metadata == {
            "text_field": "model_output",
            "input_construction": "eval_record_messages",
            "input_formatting_mode": "role_tagged_text",
            "tokenizer_has_chat_template": False,
            "add_special_tokens": True,
        }

    def test_get_input_formatting_metadata_with_chat_template(self) -> None:
        metadata = correctness_formatting.get_input_formatting_metadata(
            _ChatTemplateTokenizer(),
            text_field="final_answer",
        )

        assert metadata == {
            "text_field": "final_answer",
            "input_construction": "eval_record_messages",
            "input_formatting_mode": "chat_template",
            "tokenizer_has_chat_template": True,
            "add_special_tokens": False,
        }
