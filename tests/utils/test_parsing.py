"""Tests for the text parsing utilities in pyine.utils.parsing."""

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
