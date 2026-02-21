"""Tests for the text parsing utilities in pyine.utils.parsing."""

import pydantic
import pytest

import pyine.utils.parsing


class TestCompileOpenTagRegex:
    def test_matches_simple_tag(self) -> None:
        regex = pyine.utils.parsing.compile_open_tag_regex("final")
        assert regex.search("<final>") is not None

    def test_matches_tag_with_attributes(self) -> None:
        regex = pyine.utils.parsing.compile_open_tag_regex("final")
        assert regex.search('<final attr="value">') is not None

    def test_case_insensitive(self) -> None:
        regex = pyine.utils.parsing.compile_open_tag_regex("final")
        assert regex.search("<FINAL>") is not None
        assert regex.search("<Final>") is not None

    def test_empty_tag_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            pyine.utils.parsing.compile_open_tag_regex("")

    def test_whitespace_tag_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            pyine.utils.parsing.compile_open_tag_regex("   ")

    def test_escapes_special_chars(self) -> None:
        regex = pyine.utils.parsing.compile_open_tag_regex("tag.name")
        assert regex.search("<tag.name>") is not None
        assert regex.search("<tagXname>") is None


class TestCompileCloseTagRegex:
    def test_matches_close_tag(self) -> None:
        regex = pyine.utils.parsing.compile_close_tag_regex("final")
        assert regex.search("</final>") is not None

    def test_case_insensitive(self) -> None:
        regex = pyine.utils.parsing.compile_close_tag_regex("final")
        assert regex.search("</FINAL>") is not None

    def test_empty_tag_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            pyine.utils.parsing.compile_close_tag_regex("")

    def test_escapes_special_chars(self) -> None:
        regex = pyine.utils.parsing.compile_close_tag_regex("tag.name")
        assert regex.search("</tag.name>") is not None
        assert regex.search("</tagXname>") is None


class TestModuleExports:
    def test_all_exports(self) -> None:
        assert "compile_open_tag_regex" in pyine.utils.parsing.__all__
        assert "compile_close_tag_regex" in pyine.utils.parsing.__all__
        assert "normalize_path_prefix" in pyine.utils.parsing.__all__
        assert "extract_tag_blocks" in pyine.utils.parsing.__all__
        assert "select_tag_block" in pyine.utils.parsing.__all__
        assert "parse_json_or_python_literal" in pyine.utils.parsing.__all__


class TestExtractTagBlocks:
    def test_extracts_single_block(self) -> None:
        text = "prefix <final>answer</final> suffix"
        result = pyine.utils.parsing.extract_tag_blocks(text, "final")
        assert result.blocks == ("answer",)
        assert result.block_count == 1
        assert result.open_count == 1
        assert result.close_count == 1
        assert not result.has_malformed_structure

    def test_extracts_multiple_blocks(self) -> None:
        text = "<final>first</final> middle <final>second</final>"
        result = pyine.utils.parsing.extract_tag_blocks(text, "final")
        assert result.blocks == ("first", "second")
        assert result.block_count == 2

    def test_detects_nested_open(self) -> None:
        text = "<final>outer <final>inner</final></final>"
        result = pyine.utils.parsing.extract_tag_blocks(text, "final")
        assert result.has_nested_open
        assert result.has_malformed_structure

    def test_detects_stray_close(self) -> None:
        text = "content</final>more"
        result = pyine.utils.parsing.extract_tag_blocks(text, "final")
        assert result.has_stray_close
        assert result.has_malformed_structure

    def test_detects_unclosed_open(self) -> None:
        text = "<final>unclosed content"
        result = pyine.utils.parsing.extract_tag_blocks(text, "final")
        assert result.has_unclosed_open
        assert result.has_malformed_structure

    def test_case_insensitive(self) -> None:
        text = "<FINAL>content</Final>"
        result = pyine.utils.parsing.extract_tag_blocks(text, "final")
        assert result.blocks == ("content",)

    def test_tracks_block_start_offsets(self) -> None:
        text = "abc<final>x</final>def<final>y</final>"
        result = pyine.utils.parsing.extract_tag_blocks(text, "final")
        assert result.block_start_offsets == (3, 22)

    def test_tracks_block_end_offsets(self) -> None:
        text = "abc<final>x</final>def<final>y</final>"
        result = pyine.utils.parsing.extract_tag_blocks(text, "final")
        # first block: <final>x</final> ends at index 19 (after </final>)
        # second block: <final>y</final> ends at index 38 (after </final>)
        assert result.block_end_offsets == (19, 38)
        # verify offsets are correct by checking extracted spans
        assert text[result.block_start_offsets[0] : result.block_end_offsets[0]] == "<final>x</final>"
        assert text[result.block_start_offsets[1] : result.block_end_offsets[1]] == "<final>y</final>"


class TestSelectTagBlock:
    def test_first_policy(self) -> None:
        text = "<tag>first</tag><tag>second</tag>"
        result = pyine.utils.parsing.extract_tag_blocks(text, "tag")
        selection = pyine.utils.parsing.select_tag_block(result, policy="first")
        assert selection is not None
        content, idx = selection
        assert content == "first"
        assert idx == 0

    def test_last_policy(self) -> None:
        text = "<tag>first</tag><tag>second</tag>"
        result = pyine.utils.parsing.extract_tag_blocks(text, "tag")
        selection = pyine.utils.parsing.select_tag_block(result, policy="last")
        assert selection is not None
        content, idx = selection
        assert content == "second"
        assert idx == 1

    def test_error_policy_raises_on_multiple(self) -> None:
        text = "<tag>first</tag><tag>second</tag>"
        result = pyine.utils.parsing.extract_tag_blocks(text, "tag")
        with pytest.raises(ValueError, match="expected exactly one"):
            pyine.utils.parsing.select_tag_block(result, policy="error")

    def test_error_policy_ok_with_single(self) -> None:
        text = "<tag>only</tag>"
        result = pyine.utils.parsing.extract_tag_blocks(text, "tag")
        selection = pyine.utils.parsing.select_tag_block(result, policy="error")
        assert selection is not None
        content, idx = selection
        assert content == "only"
        assert idx == 0

    def test_returns_none_when_no_blocks(self) -> None:
        text = "no tags here"
        result = pyine.utils.parsing.extract_tag_blocks(text, "tag")
        selection = pyine.utils.parsing.select_tag_block(result)
        assert selection is None


class TestParseJsonOrPythonLiteral:
    def test_parses_json_dict(self) -> None:
        result = pyine.utils.parsing.parse_json_or_python_literal('{"key": "value"}')
        assert result.success
        assert result.value == {"key": "value"}

    def test_parses_python_dict(self) -> None:
        result = pyine.utils.parsing.parse_json_or_python_literal("{'key': 'value'}")
        assert result.success
        assert result.value == {"key": "value"}
        assert result.method == "literal_eval"

    def test_parses_python_tuple(self) -> None:
        result = pyine.utils.parsing.parse_json_or_python_literal("(1, 2, 3)")
        assert result.success
        assert result.value == (1, 2, 3)
        assert result.method == "literal_eval"

    def test_parses_json_list(self) -> None:
        result = pyine.utils.parsing.parse_json_or_python_literal("[1, 2, 3]")
        assert result.success
        assert result.value == [1, 2, 3]

    def test_strips_markdown_fences(self) -> None:
        result = pyine.utils.parsing.parse_json_or_python_literal('```json\n{"key": "value"}\n```')
        assert result.success
        assert result.value == {"key": "value"}

    def test_strips_python_fences(self) -> None:
        result = pyine.utils.parsing.parse_json_or_python_literal("```python\n{'key': 'value'}\n```")
        assert result.success
        assert result.value == {"key": "value"}

    def test_fails_on_invalid_syntax(self) -> None:
        result = pyine.utils.parsing.parse_json_or_python_literal("not valid {syntax")
        assert not result.success
        assert result.error is not None

    def test_fails_on_empty_string(self) -> None:
        result = pyine.utils.parsing.parse_json_or_python_literal("")
        assert not result.success

    def test_preserves_none_value(self) -> None:
        result = pyine.utils.parsing.parse_json_or_python_literal("None")
        assert result.success
        assert result.value is None
        assert result.method == "literal_eval"


# -------------------------------- shared output parser tests --------------------------------


class TestParsedOutput:
    def test_defaults(self) -> None:
        parsed = pyine.utils.parsing.ParsedOutput(raw="hello")
        assert parsed.raw == "hello"
        assert parsed.final_answer is None
        assert parsed.reasoning is None
        assert parsed.fields == {}

    def test_all_fields(self) -> None:
        parsed = pyine.utils.parsing.ParsedOutput(
            raw="raw", final_answer="answer", reasoning="reason", fields={"key": "val"}
        )
        assert parsed.raw == "raw"
        assert parsed.final_answer == "answer"
        assert parsed.reasoning == "reason"
        assert parsed.fields == {"key": "val"}

    def test_frozen(self) -> None:
        parsed = pyine.utils.parsing.ParsedOutput(raw="hello")
        with pytest.raises(AttributeError):
            parsed.raw = "world"  # noqa


class TestParsingConfig:
    def test_default_creation(self) -> None:
        config = pyine.utils.parsing.ParsingConfig()
        assert config.mode == "tags"
        assert config.final_tag == "final"
        assert config.reasoning_tag == "reasoning"
        assert config.fallback_policy == "none"

    def test_rejects_empty_tag(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            pyine.utils.parsing.ParsingConfig(final_tag="")

    def test_rejects_same_final_and_reasoning_tag(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            pyine.utils.parsing.ParsingConfig(final_tag="same", reasoning_tag="same")

    def test_frozen(self) -> None:
        config = pyine.utils.parsing.ParsingConfig()
        with pytest.raises(pydantic.ValidationError):
            config.mode = "other"  # type: ignore[reportAttributeAccessIssue]


class TestTagsOutputParser:
    def test_extracts_final_and_reasoning(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            mode="tags",
            final_tag="final",
            reasoning_tag="reasoning",
            fallback_policy="none",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<reasoning>r</reasoning>\n<final>a</final>")
        assert parsed.final_answer == "a"
        assert parsed.reasoning == "r"
        assert parsed.fields["tags/final/open_count"] == "1"
        assert parsed.fields["tags/final/close_count"] == "1"
        assert parsed.fields["tags/final/block_count"] == "1"

    def test_fallback_last_line(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            mode="tags",
            final_tag="final",
            reasoning_tag="reasoning",
            fallback_policy="last_line",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "hello\nworld\n")
        assert parsed.final_answer == "world"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  hello world  ", "hello world"),
            ("   \n\n  ", None),
        ],
    )
    def test_fallback_entire_output(
        self,
        raw: str,
        expected: str | None,
    ) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            mode="tags",
            final_tag="final",
            reasoning_tag="reasoning",
            fallback_policy="entire_output",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", raw)
        assert parsed.final_answer == expected

    def test_multi_tag_policy_first(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            mode="tags",
            final_tag="final",
            reasoning_tag="reasoning",
            fallback_policy="none",
            multi_tag_policy="first",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<final>a</final><final>b</final>")
        assert parsed.final_answer == "a"

    def test_multi_tag_policy_error(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            mode="tags",
            final_tag="final",
            reasoning_tag="reasoning",
            fallback_policy="none",
            multi_tag_policy="error",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        with pytest.raises(ValueError, match="expected exactly one"):
            parser.parse("prompt", "<final>a</final><final>b</final>")

    def test_strict_mode_rejects_stray_close(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            mode="tags",
            final_tag="final",
            reasoning_tag="reasoning",
            fallback_policy="none",
            strict=True,
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        with pytest.raises(ValueError, match="malformed tag structure"):
            parser.parse("prompt", "</final>")

    def test_enabled_fields_final_only_skips_reasoning(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            enabled_fields="final_only",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<reasoning>r</reasoning><final>a</final>")
        assert parsed.final_answer == "a"
        assert parsed.reasoning is None

    def test_enabled_fields_reasoning_only_skips_final(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            enabled_fields="reasoning_only",
            fallback_policy="none",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<reasoning>r</reasoning><final>a</final>")
        assert parsed.final_answer is None
        assert parsed.reasoning == "r"

    def test_reasoning_from_outside_final_overrides_reasoning_tag(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            enabled_fields="both",
            reasoning_from_outside_final=True,
            fallback_policy="none",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<reasoning>r</reasoning> pre <final>a</final>")
        assert parsed.final_answer == "a"
        assert parsed.reasoning == "<reasoning>r</reasoning> pre"
        assert "tags/reasoning/open_count" not in parsed.fields

    def test_reasoning_from_outside_final_works_in_reasoning_only_mode(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            enabled_fields="reasoning_only",
            reasoning_from_outside_final=True,
            fallback_policy="none",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "prefix\n<final>a</final>")
        assert parsed.final_answer is None
        assert parsed.reasoning == "prefix"

    def test_reasoning_from_outside_final_uses_selected_final_block(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            enabled_fields="both",
            reasoning_from_outside_final=True,
            multi_tag_policy="last",
            fallback_policy="none",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "r1<final>a</final>mid<final>b</final>")
        assert parsed.final_answer == "b"
        assert parsed.reasoning == "r1<final>a</final>mid"

    def test_reasoning_from_outside_final_falls_back_to_reasoning_tag_when_no_final(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            enabled_fields="both",
            reasoning_from_outside_final=True,
            fallback_policy="none",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<reasoning>r</reasoning>")
        assert parsed.final_answer is None
        assert parsed.reasoning == "r"

    def test_reasoning_from_outside_final_captures_suffix_only(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            enabled_fields="both",
            reasoning_from_outside_final=True,
            fallback_policy="none",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<final>a</final> suffix text")
        assert parsed.final_answer == "a"
        assert parsed.reasoning == "suffix text"

    def test_reasoning_from_outside_final_captures_prefix_and_suffix(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            enabled_fields="both",
            reasoning_from_outside_final=True,
            fallback_policy="none",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "prefix text <final>a</final> suffix text")
        assert parsed.final_answer == "a"
        assert parsed.reasoning == "prefix text\nsuffix text"

    def test_reasoning_from_outside_final_empty_when_no_surrounding_text(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            enabled_fields="both",
            reasoning_from_outside_final=True,
            fallback_policy="none",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<final>a</final>")
        assert parsed.final_answer == "a"
        assert parsed.reasoning == ""

    def test_reasoning_from_entire_output_when_no_final_answer_uses_full_output(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            enabled_fields="both",
            reasoning_from_entire_output_when_no_final_answer=True,
            fallback_policy="none",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "Just some output with no tags")
        assert parsed.final_answer is None
        assert parsed.reasoning == "Just some output with no tags"

    def test_reasoning_from_entire_output_when_no_final_answer_does_not_override_reasoning_tag(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            enabled_fields="both",
            reasoning_from_entire_output_when_no_final_answer=True,
            fallback_policy="none",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<reasoning>r</reasoning>")
        assert parsed.final_answer is None
        assert parsed.reasoning == "r"

    def test_reasoning_from_entire_output_when_no_final_answer_skips_when_answer_present(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            enabled_fields="both",
            reasoning_from_entire_output_when_no_final_answer=True,
            fallback_policy="none",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<final>a</final>")
        assert parsed.final_answer == "a"
        assert parsed.reasoning == ""

    def test_reasoning_from_entire_output_when_no_final_answer_respects_reasoning_only_mode(self) -> None:
        config = pyine.utils.parsing.ParsingConfig(
            enabled_fields="reasoning_only",
            reasoning_from_entire_output_when_no_final_answer=True,
            fallback_policy="none",
        )
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<final>a</final>")
        assert parsed.final_answer is None
        assert parsed.reasoning == ""

    def test_raw_preserved(self) -> None:
        config = pyine.utils.parsing.ParsingConfig()
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<final>answer</final>")
        assert parsed.raw == "<final>answer</final>"

    def test_no_tags_returns_none(self) -> None:
        config = pyine.utils.parsing.ParsingConfig()
        parser = pyine.utils.parsing.TagsOutputParser(config)
        parsed = parser.parse("prompt", "no tags here")
        assert parsed.final_answer is None

    def test_rejects_unsupported_mode(self) -> None:
        config = pyine.utils.parsing.ParsingConfig()
        object.__setattr__(config, "mode", "unsupported")
        with pytest.raises(ValueError, match="unsupported"):
            pyine.utils.parsing.TagsOutputParser(config)


class TestReasoningFromOutsideFinalEdgeCases:
    """Test edge cases to ensure reasoning is never None when reasoning_from_outside_final=True."""

    @pytest.fixture
    def config(self) -> pyine.utils.parsing.ParsingConfig:
        return pyine.utils.parsing.ParsingConfig(
            enabled_fields="both",
            reasoning_from_outside_final=True,
            fallback_policy="none",
        )

    @pytest.fixture
    def parser(
        self,
        config: pyine.utils.parsing.ParsingConfig,
    ) -> pyine.utils.parsing.TagsOutputParser:
        return pyine.utils.parsing.TagsOutputParser(config)

    @pytest.fixture
    def reasoning_only_parser(self) -> pyine.utils.parsing.TagsOutputParser:
        config = pyine.utils.parsing.ParsingConfig(
            enabled_fields="reasoning_only",
            reasoning_from_outside_final=True,
            fallback_policy="none",
        )
        return pyine.utils.parsing.TagsOutputParser(config)

    def test_only_final_tags_no_surrounding_text(
        self,
        parser: pyine.utils.parsing.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "<final>42</final>")
        assert parsed.final_answer == "42"
        assert parsed.reasoning == ""
        assert parsed.reasoning is not None

    def test_text_before_final_tags(
        self,
        parser: pyine.utils.parsing.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "Let me think...\n<final>42</final>")
        assert parsed.final_answer == "42"
        assert parsed.reasoning == "Let me think..."
        assert parsed.reasoning is not None

    def test_text_after_final_tags(
        self,
        parser: pyine.utils.parsing.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "<final>42</final>\nThat's my answer!")
        assert parsed.final_answer == "42"
        assert parsed.reasoning == "That's my answer!"
        assert parsed.reasoning is not None

    def test_text_before_and_after_final_tags(
        self,
        parser: pyine.utils.parsing.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "Reasoning here\n<final>42</final>\nMore thoughts")
        assert parsed.final_answer == "42"
        assert parsed.reasoning == "Reasoning here\nMore thoughts"
        assert parsed.reasoning is not None

    def test_no_final_tags_no_reasoning_tags_returns_empty(
        self,
        parser: pyine.utils.parsing.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "Just some output with no tags")
        assert parsed.final_answer is None  # no final tags found
        assert parsed.reasoning == ""  # no reasoning tags either, so empty string
        assert parsed.reasoning is not None

    def test_no_final_tags_falls_back_to_reasoning_tags(
        self,
        parser: pyine.utils.parsing.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "<reasoning>This is reasoning</reasoning>")
        assert parsed.final_answer is None  # no final tags found
        assert parsed.reasoning == "This is reasoning"  # falls back to reasoning tags
        assert parsed.reasoning is not None

    def test_empty_output(
        self,
        parser: pyine.utils.parsing.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "")
        assert parsed.final_answer is None
        assert parsed.reasoning == ""
        assert parsed.reasoning is not None

    def test_only_whitespace_output(
        self,
        parser: pyine.utils.parsing.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "   \n\t  ")
        assert parsed.final_answer is None
        assert parsed.reasoning == ""
        assert parsed.reasoning is not None

    def test_unclosed_final_tag(
        self,
        parser: pyine.utils.parsing.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "Some text <final>unclosed")
        assert parsed.final_answer is None  # invalid tag, not extracted
        assert parsed.reasoning == ""  # no valid tags, returns empty string
        assert parsed.reasoning is not None

    def test_empty_final_tags(
        self,
        parser: pyine.utils.parsing.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "prefix<final></final>suffix")
        assert parsed.final_answer is None  # empty final becomes None
        assert parsed.reasoning == "prefix\nsuffix"
        assert parsed.reasoning is not None

    def test_reasoning_only_mode_no_final_tags(
        self,
        reasoning_only_parser: pyine.utils.parsing.TagsOutputParser,
    ) -> None:
        parsed = reasoning_only_parser.parse("prompt", "Just reasoning text")
        assert parsed.final_answer is None  # not requested
        assert parsed.reasoning == ""  # no tags found, returns empty string
        assert parsed.reasoning is not None
