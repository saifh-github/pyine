import pytest

import pyine.organisms.models.rewards.core.configs
import pyine.organisms.models.rewards.core.parser


class TestTagsOutputParser:
    def test_extracts_final_and_reasoning(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.ParsingConfig(
            mode="tags",
            final_tag="final",
            reasoning_tag="reasoning",
            fallback_policy="none",
        )
        parser = pyine.organisms.models.rewards.core.parser.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<reasoning>r</reasoning>\n<final>a</final>")
        assert parsed.final_answer == "a"
        assert parsed.reasoning == "r"
        assert parsed.fields["tags/final/open_count"] == "1"
        assert parsed.fields["tags/final/close_count"] == "1"
        assert parsed.fields["tags/final/block_count"] == "1"

    def test_fallback_last_line(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.ParsingConfig(
            mode="tags",
            final_tag="final",
            reasoning_tag="reasoning",
            fallback_policy="last_line",
        )
        parser = pyine.organisms.models.rewards.core.parser.TagsOutputParser(config)
        parsed = parser.parse("prompt", "hello\nworld\n")
        assert parsed.final_answer == "world"

    def test_multi_tag_policy_first(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.ParsingConfig(
            mode="tags",
            final_tag="final",
            reasoning_tag="reasoning",
            fallback_policy="none",
            multi_tag_policy="first",
        )
        parser = pyine.organisms.models.rewards.core.parser.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<final>a</final><final>b</final>")
        assert parsed.final_answer == "a"

    def test_multi_tag_policy_error(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.ParsingConfig(
            mode="tags",
            final_tag="final",
            reasoning_tag="reasoning",
            fallback_policy="none",
            multi_tag_policy="error",
        )
        parser = pyine.organisms.models.rewards.core.parser.TagsOutputParser(config)
        with pytest.raises(ValueError, match="expected exactly one"):
            parser.parse("prompt", "<final>a</final><final>b</final>")

    def test_strict_mode_rejects_stray_close(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.ParsingConfig(
            mode="tags",
            final_tag="final",
            reasoning_tag="reasoning",
            fallback_policy="none",
            strict=True,
        )
        parser = pyine.organisms.models.rewards.core.parser.TagsOutputParser(config)
        with pytest.raises(ValueError, match="malformed tag structure"):
            parser.parse("prompt", "</final>")

    def test_enabled_fields_final_only_skips_reasoning(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.ParsingConfig(
            enabled_fields="final_only",
        )
        parser = pyine.organisms.models.rewards.core.parser.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<reasoning>r</reasoning><final>a</final>")
        assert parsed.final_answer == "a"
        assert parsed.reasoning is None

    def test_enabled_fields_reasoning_only_skips_final(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.ParsingConfig(
            enabled_fields="reasoning_only",
            fallback_policy="none",
        )
        parser = pyine.organisms.models.rewards.core.parser.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<reasoning>r</reasoning><final>a</final>")
        assert parsed.final_answer is None
        assert parsed.reasoning == "r"

    def test_reasoning_from_final_prefix_overrides_reasoning_tag(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.ParsingConfig(
            enabled_fields="both",
            reasoning_from_final_prefix=True,
            fallback_policy="none",
        )
        parser = pyine.organisms.models.rewards.core.parser.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<reasoning>r</reasoning> pre <final>a</final>")
        assert parsed.final_answer == "a"
        assert parsed.reasoning == "<reasoning>r</reasoning> pre"
        assert "tags/reasoning/open_count" not in parsed.fields

    def test_reasoning_from_final_prefix_works_in_reasoning_only_mode(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.ParsingConfig(
            enabled_fields="reasoning_only",
            reasoning_from_final_prefix=True,
            fallback_policy="none",
        )
        parser = pyine.organisms.models.rewards.core.parser.TagsOutputParser(config)
        parsed = parser.parse("prompt", "prefix\n<final>a</final>")
        assert parsed.final_answer is None
        assert parsed.reasoning == "prefix"

    def test_reasoning_from_final_prefix_uses_selected_final_block(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.ParsingConfig(
            enabled_fields="both",
            reasoning_from_final_prefix=True,
            multi_tag_policy="last",
            fallback_policy="none",
        )
        parser = pyine.organisms.models.rewards.core.parser.TagsOutputParser(config)
        parsed = parser.parse("prompt", "r1<final>a</final>mid<final>b</final>")
        assert parsed.final_answer == "b"
        assert parsed.reasoning == "r1<final>a</final>mid"

    def test_reasoning_from_final_prefix_falls_back_to_reasoning_tag_when_no_final(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.ParsingConfig(
            enabled_fields="both",
            reasoning_from_final_prefix=True,
            fallback_policy="none",
        )
        parser = pyine.organisms.models.rewards.core.parser.TagsOutputParser(config)
        parsed = parser.parse("prompt", "<reasoning>r</reasoning>")
        assert parsed.final_answer is None
        assert parsed.reasoning == "r"
