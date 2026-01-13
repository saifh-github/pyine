"""Test all edge cases for reasoning extraction with reasoning_from_outside_final."""

import pytest

import pyine.organisms.models.rewards.core.configs
import pyine.organisms.models.rewards.core.parser


class TestReasoningFromOutsideFinalEdgeCases:
    """Test edge cases to ensure reasoning is never None when reasoning_from_outside_final=True."""

    @pytest.fixture
    def config(self) -> pyine.organisms.models.rewards.core.configs.ParsingConfig:
        return pyine.organisms.models.rewards.core.configs.ParsingConfig(
            enabled_fields="both",
            reasoning_from_outside_final=True,
            fallback_policy="none",
        )

    @pytest.fixture
    def parser(
        self,
        config: pyine.organisms.models.rewards.core.configs.ParsingConfig,
    ) -> pyine.organisms.models.rewards.core.parser.TagsOutputParser:
        return pyine.organisms.models.rewards.core.parser.TagsOutputParser(config)

    @pytest.fixture
    def reasoning_only_parser(self) -> pyine.organisms.models.rewards.core.parser.TagsOutputParser:
        config = pyine.organisms.models.rewards.core.configs.ParsingConfig(
            enabled_fields="reasoning_only",
            reasoning_from_outside_final=True,
            fallback_policy="none",
        )
        return pyine.organisms.models.rewards.core.parser.TagsOutputParser(config)

    def test_only_final_tags_no_surrounding_text(
        self,
        parser: pyine.organisms.models.rewards.core.parser.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "<final>42</final>")
        assert parsed.final_answer == "42"
        assert parsed.reasoning == ""
        assert parsed.reasoning is not None

    def test_text_before_final_tags(
        self,
        parser: pyine.organisms.models.rewards.core.parser.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "Let me think...\n<final>42</final>")
        assert parsed.final_answer == "42"
        assert parsed.reasoning == "Let me think..."
        assert parsed.reasoning is not None

    def test_text_after_final_tags(
        self,
        parser: pyine.organisms.models.rewards.core.parser.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "<final>42</final>\nThat's my answer!")
        assert parsed.final_answer == "42"
        assert parsed.reasoning == "That's my answer!"
        assert parsed.reasoning is not None

    def test_text_before_and_after_final_tags(
        self,
        parser: pyine.organisms.models.rewards.core.parser.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "Reasoning here\n<final>42</final>\nMore thoughts")
        assert parsed.final_answer == "42"
        assert parsed.reasoning == "Reasoning here\nMore thoughts"
        assert parsed.reasoning is not None

    def test_no_final_tags_no_reasoning_tags_returns_empty(
        self,
        parser: pyine.organisms.models.rewards.core.parser.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "Just some output with no tags")
        assert parsed.final_answer is None  # no final tags found
        assert parsed.reasoning == ""  # no reasoning tags either, so empty string
        assert parsed.reasoning is not None

    def test_no_final_tags_falls_back_to_reasoning_tags(
        self,
        parser: pyine.organisms.models.rewards.core.parser.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "<reasoning>This is reasoning</reasoning>")
        assert parsed.final_answer is None  # no final tags found
        assert parsed.reasoning == "This is reasoning"  # falls back to reasoning tags
        assert parsed.reasoning is not None

    def test_empty_output(
        self,
        parser: pyine.organisms.models.rewards.core.parser.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "")
        assert parsed.final_answer is None
        assert parsed.reasoning == ""
        assert parsed.reasoning is not None

    def test_only_whitespace_output(
        self,
        parser: pyine.organisms.models.rewards.core.parser.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "   \n\t  ")
        assert parsed.final_answer is None
        assert parsed.reasoning == ""
        assert parsed.reasoning is not None

    def test_unclosed_final_tag(
        self,
        parser: pyine.organisms.models.rewards.core.parser.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "Some text <final>unclosed")
        assert parsed.final_answer is None  # invalid tag, not extracted
        assert parsed.reasoning == ""  # no valid tags, returns empty string
        assert parsed.reasoning is not None

    def test_empty_final_tags(
        self,
        parser: pyine.organisms.models.rewards.core.parser.TagsOutputParser,
    ) -> None:
        parsed = parser.parse("prompt", "prefix<final></final>suffix")
        assert parsed.final_answer is None  # empty final becomes None
        assert parsed.reasoning == "prefix\nsuffix"
        assert parsed.reasoning is not None

    def test_reasoning_only_mode_no_final_tags(
        self,
        reasoning_only_parser: pyine.organisms.models.rewards.core.parser.TagsOutputParser,
    ) -> None:
        parsed = reasoning_only_parser.parse("prompt", "Just reasoning text")
        assert parsed.final_answer is None  # not requested
        assert parsed.reasoning == ""  # no tags found, returns empty string
        assert parsed.reasoning is not None
