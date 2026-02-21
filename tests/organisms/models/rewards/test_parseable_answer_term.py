"""Tests for the parseable_answer reward term."""

import pytest

import pyine.organisms.models.rewards.core.types as reward_types
import pyine.organisms.models.rewards.terms.format.parseable_answer as parseable_answer_term
import pyine.utils.parsing
import tests.organisms.models.rewards.conftest as rewards_conftest


class TestParseableAnswerTermConfig:
    def test_default_config_values(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig()
        assert config.final_tag == "final"
        assert config.reward_if_present == 1.0
        assert config.reward_if_missing == 0.0
        assert config.bonus_if_stops_after_final_tag == 0.0
        assert config.bonus_if_single_final_block == 0.0

    def test_custom_config_values(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig(
            final_tag="answer",
            reward_if_present=2.0,
            reward_if_missing=0.5,
            bonus_if_stops_after_final_tag=0.25,
            bonus_if_single_final_block=0.1,
        )
        assert config.final_tag == "answer"
        assert config.reward_if_present == 2.0
        assert config.reward_if_missing == 0.5

    def test_empty_final_tag_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            parseable_answer_term.ParseableAnswerTermConfig(final_tag="")

    def test_whitespace_final_tag_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            parseable_answer_term.ParseableAnswerTermConfig(final_tag="   ")

    def test_final_tag_is_stripped(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig(final_tag="  answer  ")
        assert config.final_tag == "answer"

    def test_negative_reward_rejected(self) -> None:
        with pytest.raises(ValueError):
            parseable_answer_term.ParseableAnswerTermConfig(reward_if_present=-1.0)

    def test_negative_bonus_rejected(self) -> None:
        with pytest.raises(ValueError):
            parseable_answer_term.ParseableAnswerTermConfig(bonus_if_stops_after_final_tag=-0.5)


class TestParseableAnswerTerm:
    def test_returns_reward_if_present_when_final_answer_exists(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig(
            reward_if_present=1.0,
            reward_if_missing=0.0,
        )
        term = parseable_answer_term.ParseableAnswerTerm(config)
        parsed = pyine.utils.parsing.ParsedOutput(raw="<final>answer</final>", final_answer="answer")
        ctx = rewards_conftest.make_sample_context(parsed=parsed)
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["has_final_answer"] == 1

    def test_returns_reward_if_missing_when_no_final_answer(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig(
            reward_if_present=1.0,
            reward_if_missing=0.5,
        )
        term = parseable_answer_term.ParseableAnswerTerm(config)
        parsed = pyine.utils.parsing.ParsedOutput(raw="no tags here", final_answer=None)
        ctx = rewards_conftest.make_sample_context(parsed=parsed)
        result = term(ctx)
        assert result.value == 0.5
        assert result.metrics["has_final_answer"] == 0

    def test_returns_reward_if_missing_when_parsed_is_none(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig(
            reward_if_present=1.0,
            reward_if_missing=0.25,
        )
        term = parseable_answer_term.ParseableAnswerTerm(config)
        ctx = rewards_conftest.make_sample_context(parsed=None)
        result = term(ctx)
        assert result.value == 0.25
        assert result.metrics["has_final_answer"] == 0

    def test_empty_final_answer_treated_as_missing(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig(
            reward_if_present=1.0,
            reward_if_missing=0.0,
        )
        term = parseable_answer_term.ParseableAnswerTerm(config)
        parsed = pyine.utils.parsing.ParsedOutput(raw="<final></final>", final_answer="")
        ctx = rewards_conftest.make_sample_context(parsed=parsed)
        result = term(ctx)
        assert result.value == 0.0
        assert result.metrics["has_final_answer"] == 0

    def test_whitespace_only_final_answer_treated_as_missing(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig(
            reward_if_present=1.0,
            reward_if_missing=0.0,
        )
        term = parseable_answer_term.ParseableAnswerTerm(config)
        parsed = pyine.utils.parsing.ParsedOutput(raw="<final>   </final>", final_answer="   ")
        ctx = rewards_conftest.make_sample_context(parsed=parsed)
        result = term(ctx)
        assert result.value == 0.0
        assert result.metrics["has_final_answer"] == 0

    def test_bonus_for_single_final_block(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig(
            reward_if_present=1.0,
            bonus_if_single_final_block=0.5,
        )
        term = parseable_answer_term.ParseableAnswerTerm(config)
        parsed = pyine.utils.parsing.ParsedOutput(
            raw="<final>answer</final>",
            final_answer="answer",
            fields={
                "tags/final/open_count": "1",
                "tags/final/close_count": "1",
            },
        )
        ctx = rewards_conftest.make_sample_context(parsed=parsed)
        result = term(ctx)
        assert result.value == 1.5
        assert result.metrics["has_single_final_block"] == 1

    def test_no_bonus_for_multiple_final_blocks(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig(
            reward_if_present=1.0,
            bonus_if_single_final_block=0.5,
        )
        term = parseable_answer_term.ParseableAnswerTerm(config)
        parsed = pyine.utils.parsing.ParsedOutput(
            raw="<final>a</final><final>b</final>",
            final_answer="b",
            fields={
                "tags/final/open_count": "2",
                "tags/final/close_count": "2",
            },
        )
        ctx = rewards_conftest.make_sample_context(parsed=parsed)
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["has_single_final_block"] == 0

    def test_bonus_for_stops_after_final_tag(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig(
            reward_if_present=1.0,
            bonus_if_stops_after_final_tag=0.25,
        )
        term = parseable_answer_term.ParseableAnswerTerm(config)
        parsed = pyine.utils.parsing.ParsedOutput(
            raw="<final>answer</final>",
            final_answer="answer",
            fields={
                "tags/final/open_count": "1",
                "tags/final/close_count": "1",
                "tags/final/stops_after_last_close": "true",
            },
        )
        ctx = rewards_conftest.make_sample_context(parsed=parsed)
        result = term(ctx)
        assert result.value == 1.25
        assert result.metrics["stops_after_final_tag"] == 1

    def test_no_bonus_when_text_follows_final_tag(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig(
            reward_if_present=1.0,
            bonus_if_stops_after_final_tag=0.25,
        )
        term = parseable_answer_term.ParseableAnswerTerm(config)
        parsed = pyine.utils.parsing.ParsedOutput(
            raw="<final>answer</final> extra text",
            final_answer="answer",
            fields={
                "tags/final/open_count": "1",
                "tags/final/close_count": "1",
                "tags/final/stops_after_last_close": "false",
            },
        )
        ctx = rewards_conftest.make_sample_context(parsed=parsed)
        result = term(ctx)
        assert result.value == 1.0
        assert result.metrics["stops_after_final_tag"] == 0

    def test_combined_bonuses(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig(
            reward_if_present=1.0,
            bonus_if_single_final_block=0.5,
            bonus_if_stops_after_final_tag=0.25,
        )
        term = parseable_answer_term.ParseableAnswerTerm(config)
        parsed = pyine.utils.parsing.ParsedOutput(
            raw="<final>answer</final>",
            final_answer="answer",
            fields={
                "tags/final/open_count": "1",
                "tags/final/close_count": "1",
                "tags/final/stops_after_last_close": "true",
            },
        )
        ctx = rewards_conftest.make_sample_context(parsed=parsed)
        result = term(ctx)
        assert result.value == 1.75
        assert result.metrics["has_single_final_block"] == 1
        assert result.metrics["stops_after_final_tag"] == 1

    def test_fallback_to_regex_scan_without_diagnostics(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig(
            reward_if_present=1.0,
            bonus_if_single_final_block=0.5,
            bonus_if_stops_after_final_tag=0.25,
        )
        term = parseable_answer_term.ParseableAnswerTerm(config)
        parsed = pyine.utils.parsing.ParsedOutput(
            raw="<final>answer</final>",
            final_answer="answer",
            fields={},  # no diagnostics from parser
        )
        ctx = rewards_conftest.make_sample_context(parsed=parsed)
        result = term(ctx)
        assert result.value == 1.75  # bonuses should still apply via regex fallback
        assert result.metrics["final_tag_open_count"] == 1
        assert result.metrics["final_tag_close_count"] == 1

    def test_custom_final_tag(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig(
            final_tag="answer",
            reward_if_present=1.0,
            bonus_if_single_final_block=0.5,
        )
        term = parseable_answer_term.ParseableAnswerTerm(config)
        parsed = pyine.utils.parsing.ParsedOutput(
            raw="<answer>42</answer>",
            final_answer="42",
            fields={},  # will use regex fallback
        )
        ctx = rewards_conftest.make_sample_context(parsed=parsed)
        result = term(ctx)
        assert result.value == 1.5
        assert result.metrics["has_final_answer"] == 1
        assert result.metrics["has_single_final_block"] == 1

    def test_case_insensitive_tag_matching(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig(
            final_tag="final",
            reward_if_present=1.0,
            bonus_if_single_final_block=0.5,
        )
        term = parseable_answer_term.ParseableAnswerTerm(config)
        parsed = pyine.utils.parsing.ParsedOutput(
            raw="<FINAL>answer</FINAL>",
            final_answer="answer",
            fields={},
        )
        ctx = rewards_conftest.make_sample_context(parsed=parsed)
        result = term(ctx)
        assert result.value == 1.5
        assert result.metrics["final_tag_open_count"] == 1

    def test_reset_is_noop(self) -> None:
        config = parseable_answer_term.ParseableAnswerTermConfig()
        term = parseable_answer_term.ParseableAnswerTerm(config)
        run_ctx = reward_types.RunInitContext()
        term.reset(run_ctx)  # should not raise


class TestParseableAnswerTermFactory:
    def test_factory_creates_term_from_spec(self) -> None:
        import pyine.organisms.models.rewards.core.configs as reward_configs

        spec = reward_configs.RewardTermSpec(
            name="test",
            type="parseable_answer",
            params={
                "final_tag": "result",
                "reward_if_present": 2.0,
                "reward_if_missing": 0.1,
            },
        )
        term = parseable_answer_term._factory(spec, parser=None)
        assert isinstance(term, parseable_answer_term.ParseableAnswerTerm)
        parsed = pyine.utils.parsing.ParsedOutput(raw="<result>x</result>", final_answer="x")
        ctx = rewards_conftest.make_sample_context(parsed=parsed)
        result = term(ctx)
        assert result.value == 2.0

    def test_factory_validates_params(self) -> None:
        import pyine.organisms.models.rewards.core.configs as reward_configs

        spec = reward_configs.RewardTermSpec(
            name="test",
            type="parseable_answer",
            params={"final_tag": ""},
        )
        with pytest.raises(ValueError, match="cannot be empty"):
            parseable_answer_term._factory(spec, parser=None)


class TestParseableAnswerTermRegistration:
    def test_term_is_registered(self) -> None:
        import pyine.organisms.models.rewards.core.registry as reward_registry
        import pyine.organisms.models.rewards.terms

        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        factory = reward_registry.get_term_factory("parseable_answer")
        assert factory is not None

    def test_alias_is_registered(self) -> None:
        import pyine.organisms.models.rewards.core.registry as reward_registry
        import pyine.organisms.models.rewards.terms

        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        canonical = reward_registry.get_term_factory("parseable_answer")
        alias = reward_registry.get_term_factory("format/parseable_answer")
        assert alias is canonical
        assert reward_registry.get_global_registry().resolve_term_type("format/parseable_answer") == "parseable_answer"
