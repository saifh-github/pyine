"""Tests for the tag parsing utilities in pyine.utils.strings."""

import pytest

import pyine.utils.strings


class TestCompileOpenTagRegex:
    def test_matches_simple_tag(self) -> None:
        regex = pyine.utils.strings.compile_open_tag_regex("final")
        assert regex.search("<final>") is not None

    def test_matches_tag_with_attributes(self) -> None:
        regex = pyine.utils.strings.compile_open_tag_regex("final")
        assert regex.search('<final attr="value">') is not None

    def test_case_insensitive(self) -> None:
        regex = pyine.utils.strings.compile_open_tag_regex("final")
        assert regex.search("<FINAL>") is not None
        assert regex.search("<Final>") is not None

    def test_empty_tag_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            pyine.utils.strings.compile_open_tag_regex("")

    def test_whitespace_tag_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            pyine.utils.strings.compile_open_tag_regex("   ")

    def test_escapes_special_chars(self) -> None:
        regex = pyine.utils.strings.compile_open_tag_regex("tag.name")
        assert regex.search("<tag.name>") is not None
        assert regex.search("<tagXname>") is None


class TestCompileCloseTagRegex:
    def test_matches_close_tag(self) -> None:
        regex = pyine.utils.strings.compile_close_tag_regex("final")
        assert regex.search("</final>") is not None

    def test_case_insensitive(self) -> None:
        regex = pyine.utils.strings.compile_close_tag_regex("final")
        assert regex.search("</FINAL>") is not None

    def test_empty_tag_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            pyine.utils.strings.compile_close_tag_regex("")

    def test_escapes_special_chars(self) -> None:
        regex = pyine.utils.strings.compile_close_tag_regex("tag.name")
        assert regex.search("</tag.name>") is not None
        assert regex.search("</tagXname>") is None


class TestModuleExports:
    def test_all_exports(self) -> None:
        assert "compile_open_tag_regex" in pyine.utils.strings.__all__
        assert "compile_close_tag_regex" in pyine.utils.strings.__all__
        assert "normalize_path_prefix" in pyine.utils.strings.__all__
