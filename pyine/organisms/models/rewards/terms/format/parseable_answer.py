"""Built-in term that rewards parseable final answers."""

import pydantic

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.registry as reward_registry
import pyine.organisms.models.rewards.core.term as reward_term
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.utils.parsing


class ParseableAnswerTermConfig(reward_types.BaseConfig):
    """Configuration for `ParseableAnswerTerm`.

    Attributes:
        final_tag: Tag name used to locate the final block in raw output.
        reward_if_present: Reward value when a final answer is available.
        reward_if_missing: Reward value when no final answer is available.
        bonus_if_stops_after_final_tag: Bonus when the model stops after `</final_tag>`.
        bonus_if_single_final_block: Bonus when exactly one `<final_tag>...</final_tag>` block exists.
    """

    final_tag: str = "final"
    """Tag name used to locate the final block in raw output."""

    reward_if_present: pydantic.NonNegativeFloat = 1.0
    """Reward value when a final answer is available."""
    reward_if_missing: pydantic.NonNegativeFloat = 0.0
    """Reward value when no final answer is available."""

    bonus_if_stops_after_final_tag: pydantic.NonNegativeFloat = 0.0
    """Bonus when no non-whitespace follows the final block."""
    bonus_if_single_final_block: pydantic.NonNegativeFloat = 0.0
    """Bonus when the output contains exactly one final block."""

    @pydantic.field_validator("final_tag")
    @classmethod
    def _validate_final_tag(
        cls,
        value: str,
    ) -> str:
        """Validate and normalize `final_tag`."""
        tag = value.strip()
        if not tag:
            raise ValueError("final_tag cannot be empty")
        return tag


class ParseableAnswerTerm(reward_term.BaseRewardTerm):
    """Rewards samples whose parsed output contains a final answer.

    This term checks `SampleContext.parsed.final_answer` (not raw output tags). It requires parsing
    to be enabled in `RewardManager` (via `ParsingConfig`) to produce non-zero rewards. If parsing
    is disabled or `parsed.final_answer` is empty, the sample receives `reward_if_missing`.

    Pairs well with tag-based prompts and `TagsOutputParser`.
    """

    def __init__(
        self,
        config: ParseableAnswerTermConfig,
    ) -> None:
        """Create the term from validated configuration."""
        self._config = config
        self._final_open_re = pyine.utils.parsing.compile_open_tag_regex(config.final_tag)
        self._final_close_re = pyine.utils.parsing.compile_close_tag_regex(config.final_tag)

    def _get_final_tag_stats(
        self,
        parsed: pyine.utils.parsing.ParsedOutput,
    ) -> tuple[int, int, bool | None]:
        """Return `(open_count, close_count, stops_after_final)` for the configured final tag.

        This prefers diagnostics emitted by `TagsOutputParser` when available, and falls back to a
        local regex scan otherwise.

        `stops_after_final` is None if the final closing tag cannot be located.
        """
        open_key = f"tags/{self._config.final_tag}/open_count"
        close_key = f"tags/{self._config.final_tag}/close_count"
        stops_key = f"tags/{self._config.final_tag}/stops_after_last_close"
        if open_key in parsed.fields and close_key in parsed.fields:
            open_count = int(parsed.fields[open_key])
            close_count = int(parsed.fields[close_key])
            stops_after_final = None
            if stops_key in parsed.fields:
                stops_after_final = parsed.fields[stops_key] == "true"
            return open_count, close_count, stops_after_final

        raw = parsed.raw
        open_count = len(self._final_open_re.findall(raw))
        close_matches = list(self._final_close_re.finditer(raw))
        close_count = len(close_matches)
        if not close_matches:
            return open_count, close_count, None
        trailing = raw[close_matches[-1].end() :]
        stops_after_final = trailing.strip() == ""
        return open_count, close_count, stops_after_final

    def __call__(
        self,
        sample_ctx: reward_types.SampleContext,
    ) -> reward_types.TermResult:
        """Compute the reward contribution for a sample.

        Args:
            sample_ctx: Sample context (expects `parsed` to be present for best behavior).

        Returns:
            A `TermResult` whose `value` is based on whether a final answer was extracted, plus
            optional bonuses for having exactly one final block (`bonus_if_single_final_block`) and
            for stopping immediately after the final tag (`bonus_if_stops_after_final_tag`).
        """
        parsed = sample_ctx.parsed
        has_final = parsed is not None and parsed.final_answer is not None and bool(parsed.final_answer.strip())
        value = float(self._config.reward_if_present if has_final else self._config.reward_if_missing)
        metrics: dict[str, reward_types.MetricValue] = {"has_final_answer": int(has_final)}

        if has_final and parsed is not None:
            open_count, close_count, stops_after_final = self._get_final_tag_stats(parsed)
            metrics["final_tag_open_count"] = open_count
            metrics["final_tag_close_count"] = close_count

            if open_count == 1 and close_count == 1:
                value += float(self._config.bonus_if_single_final_block)
                metrics["has_single_final_block"] = 1
            else:
                metrics["has_single_final_block"] = 0

            if stops_after_final is True:
                value += float(self._config.bonus_if_stops_after_final_tag)
                metrics["stops_after_final_tag"] = 1
            else:
                metrics["stops_after_final_tag"] = 0
        return reward_types.TermResult(
            value=float(value),
            metrics=metrics,
        )


def _factory(
    spec: reward_configs.RewardTermSpec,
    *,
    parser: pyine.utils.parsing.OutputParser | None,
) -> reward_types.RewardTerm:
    """Build a `ParseableAnswerTerm` from a term spec.

    If a `TagsOutputParser` is provided and `final_tag` is not explicitly set in params, the
    term's `final_tag` defaults to the parser's configured tag to avoid accidental mismatches.
    """
    params = dict(spec.params)
    if "final_tag" not in params and parser is not None:
        if isinstance(parser, pyine.utils.parsing.TagsOutputParser):
            params["final_tag"] = parser.final_tag
    config = ParseableAnswerTermConfig.model_validate(params)
    return ParseableAnswerTerm(config)


reward_registry.register_term("parseable_answer", _factory)
reward_registry.register_term_aliases("parseable_answer", ["format/parseable_answer"])
