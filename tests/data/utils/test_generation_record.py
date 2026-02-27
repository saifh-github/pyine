"""Tests for the shared generation record schema and builder."""

import pytest

import pyine.data.utils.generation_record


class TestBuildSharedRecordFields:
    def test_all_fields_populated(self) -> None:
        record = pyine.data.utils.generation_record.build_shared_record_fields(
            sample_id="test_id",
            model_output="output",
            prompt="prompt",
            expected_output="expected",
            reasoning="reasoning",
            final_answer="answer",
            predict_type="program_output",
            code_type="original",
            has_code_override=False,
            pregenerated_output="pregen",
            tags=["tag1", "tag2"],
            categories=["cat1"],
            key_prefix="valid/",
        )
        assert record["sample_id"] == "test_id"
        assert record["model_output"] == "output"
        assert record["prompt"] == "prompt"
        assert record["expected_output"] == "expected"
        assert record["reasoning"] == "reasoning"
        assert record["final_answer"] == "answer"
        assert record["predict_type"] == "program_output"
        assert record["code_type"] == "original"
        assert record["has_code_override"] is False
        assert record["pregenerated_output"] == "pregen"
        assert record["tags"] == ["tag1", "tag2"]
        assert record["categories"] == ["cat1"]
        assert record["key_prefix"] == "valid/"

    def test_defaults_produce_explicit_nones(self) -> None:
        record = pyine.data.utils.generation_record.build_shared_record_fields("id_1")
        assert record["model_output"] is None
        assert record["prompt"] is None
        assert record["expected_output"] is None
        assert record["reasoning"] is None
        assert record["final_answer"] is None
        assert record["predict_type"] is None
        assert record["code_type"] is None
        assert record["has_code_override"] is None
        assert record["pregenerated_output"] is None
        assert record["tags"] is None
        assert record["categories"] is None
        assert record["key_prefix"] == ""

    def test_sequences_coerced_to_lists(self) -> None:
        record = pyine.data.utils.generation_record.build_shared_record_fields(
            "id_1",
            tags=("t1", "t2"),
            categories=frozenset(["c1"]),
        )
        assert isinstance(record["tags"], list)
        assert set(record["tags"]) == {"t1", "t2"}
        assert isinstance(record["categories"], list)
        assert record["categories"] == ["c1"]

    def test_key_prefix_passthrough(self) -> None:
        record = pyine.data.utils.generation_record.build_shared_record_fields(
            "id_1",
            key_prefix="no_trailing_slash",
        )
        assert record["key_prefix"] == "no_trailing_slash"

    def test_all_typed_dict_keys_present(self) -> None:
        record = pyine.data.utils.generation_record.build_shared_record_fields("minimal")
        expected_keys = set(pyine.data.utils.generation_record.SharedGenerationRecordFields.__annotations__.keys())
        assert set(record.keys()) == expected_keys


class TestBuildMessagesFromRecord:
    def test_with_prompt_messages(self) -> None:
        record = {
            "prompt_messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        messages = pyine.data.utils.generation_record.build_messages_from_record(record, "Hi there!")
        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "Hello"}
        assert messages[2] == {"role": "assistant", "content": "Hi there!"}

    def test_with_prompt_fallback(self) -> None:
        record = {"prompt": "What is 2+2?"}
        messages = pyine.data.utils.generation_record.build_messages_from_record(record, "4")
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "What is 2+2?"}
        assert messages[1] == {"role": "assistant", "content": "4"}

    def test_prefers_prompt_messages_over_prompt(self) -> None:
        record = {
            "prompt_messages": [{"role": "user", "content": "from messages"}],
            "prompt": "from prompt",
        }
        messages = pyine.data.utils.generation_record.build_messages_from_record(record, "reply")
        assert messages[0]["content"] == "from messages"

    def test_missing_both_raises(self) -> None:
        with pytest.raises(ValueError, match="neither"):
            pyine.data.utils.generation_record.build_messages_from_record({}, "output")

    def test_does_not_mutate_original_messages(self) -> None:
        original = [{"role": "user", "content": "hi"}]
        record = {"prompt_messages": original}
        messages = pyine.data.utils.generation_record.build_messages_from_record(record, "hello")
        assert len(original) == 1  # original not modified
        assert len(messages) == 2

    def test_trailing_assistant_in_prompt_messages_raises(self) -> None:
        record = {
            "prompt_messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "already here"},
            ],
        }
        with pytest.raises(ValueError, match="duplicate assistant message"):
            pyine.data.utils.generation_record.build_messages_from_record(record, "new reply")

    def test_empty_prompt_messages_raises(self) -> None:
        record = {"prompt_messages": []}
        with pytest.raises(ValueError, match="prompt_messages"):
            pyine.data.utils.generation_record.build_messages_from_record(record, "hello")

    def test_invalid_prompt_messages_raises(self) -> None:
        record = {"prompt_messages": [{"role": "user"}]}
        with pytest.raises(ValueError, match="prompt_messages"):
            pyine.data.utils.generation_record.build_messages_from_record(record, "hello")
