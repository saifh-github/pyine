"""Tests for the shared generation record schema and builder."""

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
