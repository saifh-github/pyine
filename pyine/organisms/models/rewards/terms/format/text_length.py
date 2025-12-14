"""Built-in term that maps text lengths to non-negative rewards.

This term is a flexible "budget shaping" primitive for experiments. It can measure length over
multiple sources (prompt, output, parsed fields, or sample metadata fields) and map length to a
reward via piecewise-linear rules.

Important note:
    Some sources (like `prompt` or `sample_*`) are not controlled by the model; rewards based on
    those act as per-sample constants (i.e., sample weighting / curriculum shaping), not as a
    direct incentive on the model's output.
"""

import enum
import math
import typing

import pydantic

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.registry as reward_registry
import pyine.organisms.models.rewards.core.term as reward_term
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.utils.openai


class LengthSource(enum.StrEnum):
    """Available sources for length computation."""

    PROMPT = "prompt"
    """Measure the prompt length."""
    MODEL_OUTPUT = "model_output"
    """Measure the full model output length."""
    PARSED_REASONING = "parsed_reasoning"
    """Measure the parsed reasoning field length."""
    PARSED_FINAL_ANSWER = "parsed_final_answer"
    """Measure the parsed final answer field length."""
    SAMPLE_CODE = "sample_code"
    """Measure the sample code field length."""


class LengthUnit(enum.StrEnum):
    """Units supported for length computation."""

    CHARS = "chars"
    """Count characters."""
    LINES = "lines"
    """Count lines (splitlines)."""
    OPENAI_TOKENS = "openai_tokens"
    """Estimate token count using OpenAI's tokenizer."""


class MissingTextPolicy(enum.StrEnum):
    """Policy used when the selected source text is unavailable."""

    SKIP = "skip"
    """Skip this component (contributes 0 reward)."""
    EMPTY = "empty"
    """Treat missing text as empty string."""
    ERROR = "error"
    """Raise an error if text is missing."""


# type aliases for Pydantic Literal validation
type LengthSourceType = typing.Literal[
    "prompt",
    "model_output",
    "parsed_reasoning",
    "parsed_final_answer",
    "sample_code",
]
"""Available sources for length computation (type alias for Literal validation)."""

type LengthUnitType = typing.Literal["chars", "lines", "openai_tokens"]
"""Units supported for length computation (type alias for Literal validation)."""

type MissingTextPolicyType = typing.Literal["skip", "empty", "error"]
"""Policy used when the selected source text is unavailable (type alias for Literal validation)."""


class LengthRewardComponentConfig(reward_types.BaseConfig):
    """One length-based reward component.

    A component computes a numeric length from a selected source string, then maps it to a reward
    using a simple piecewise-linear function:
    - for lengths <= `effective_start_length`, reward is `reward_at_start`;
    - for lengths >= `end_length`, reward is `reward_at_end`;
    - otherwise, reward interpolates linearly between the two.

    If `flat_until_length` is set, it becomes the `effective_start_length` (and `start_length` is
    ignored), enabling a common "fixed until threshold, then decay" behavior.
    """

    name: str = "length"
    """Stable component name used to namespace emitted metrics."""
    enabled: bool = True
    """Whether this component is active."""
    source: LengthSourceType = "model_output"
    """Which text source to measure (prompt, output, parsed fields, or sample data fields)."""
    unit: LengthUnitType = "chars"
    """Length unit (`chars`, `lines`, or `openai_tokens`)."""
    openai_model_id: str | None = None
    """Model id used when `unit=\"openai_tokens\"` (passed to `pyine.utils.openai.estimate_token_count`)."""
    strip_whitespace: bool = False
    """Whether to strip leading/trailing whitespace from the source text before measuring length."""
    length_offset: int = 0
    """Offset added to the measured length before reward mapping (can shift thresholds)."""
    start_length: pydantic.NonNegativeInt = 0
    """Left clamp boundary for reward mapping when `flat_until_length` is unset."""
    flat_until_length: pydantic.NonNegativeInt | None = None
    """If set, reward is flat (reward_at_start) until this length, then ramps until `end_length`."""
    end_length: pydantic.NonNegativeInt = 0
    """Right clamp boundary for reward mapping (where reward reaches `reward_at_end`)."""
    reward_at_start: pydantic.NonNegativeFloat = 0.0
    """Reward value used at or below the effective start length."""
    reward_at_end: pydantic.NonNegativeFloat = 0.0
    """Reward value used at or above `end_length`."""
    weight: pydantic.NonNegativeFloat = 1.0
    """Scalar multiplier applied to this component's reward."""
    missing_text_policy: MissingTextPolicyType = "skip"
    """Behavior if the source text is unavailable (e.g., parsed fields not present)."""
    emit_metrics: bool = True
    """Whether to emit per-component diagnostic metrics (length, reward, missing flags)."""

    @pydantic.field_validator("name")
    @classmethod
    def _validate_name(
        cls,
        value: str,
    ) -> str:
        """Validate and normalize `name`."""
        name = value.strip()
        if not name:
            raise ValueError("component name cannot be empty")
        return name

    @pydantic.model_validator(mode="after")
    def _validate_lengths_and_units(self) -> "LengthRewardComponentConfig":
        """Validate length mapping parameters and unit-specific requirements."""
        if self.flat_until_length is None:
            if int(self.end_length) < int(self.start_length):
                raise ValueError("end_length must be >= start_length when flat_until_length is unset")
        else:
            if int(self.flat_until_length) < int(self.start_length):
                raise ValueError("flat_until_length must be >= start_length")
            if int(self.end_length) < int(self.flat_until_length):
                raise ValueError("end_length must be >= flat_until_length")
        if self.unit == "openai_tokens":
            model_id = "" if self.openai_model_id is None else self.openai_model_id.strip()
            if not model_id:
                raise ValueError("openai_model_id is required when unit='openai_tokens'")
        return self


class TextLengthTermConfig(reward_types.BaseConfig):
    """Configuration for `TextLengthTerm`.

    The term applies multiple `LengthRewardComponentConfig` rules and sums their rewards.

    Tip:
        If you include components that depend only on fixed per-sample fields (e.g. `prompt`,
        `sample_code`), those components behave like sample weighting rather
        than a direct incentive on the model output.
    """

    components: list[LengthRewardComponentConfig] = pydantic.Field(min_length=1)
    """Ordered list of length-based reward components to sum."""

    clip_max: pydantic.NonNegativeFloat | None = None
    """Optional maximum clip for the total term reward (after summing components)."""


class TextLengthTerm(reward_term.BaseRewardTerm):
    """Reward term that maps text lengths to non-negative rewards.

    This term is intentionally flexible: you can model common "budget shaping" behaviors by
    configuring one or more components, each of which can be flat, increasing, or decreasing over
    a length interval. It supports multiple text sources including prompt, model output, and
    parsed fields.
    """

    def __init__(
        self,
        config: TextLengthTermConfig,
    ) -> None:
        """Create the term from validated configuration."""
        self._config = config

    @staticmethod
    def _get_source_text(
        sample_ctx: reward_types.SampleContext,
        *,
        source: LengthSourceType,
    ) -> str | None:
        """Get the source text for length measurement, or None if not available."""
        if source == "prompt":
            return sample_ctx.prompt
        if source == "model_output":
            return sample_ctx.model_output
        if source == "parsed_reasoning":
            return None if sample_ctx.parsed is None else sample_ctx.parsed.reasoning
        if source == "parsed_final_answer":
            return None if sample_ctx.parsed is None else sample_ctx.parsed.final_answer
        if source == "sample_code":
            return sample_ctx.sample_data.code
        raise ValueError(f"unknown length source: {source}")

    @staticmethod
    def _compute_length(
        text: str,
        *,
        unit: LengthUnitType,
        openai_model_id: str | None,
    ) -> int:
        """Compute a length for the given text in the requested unit."""
        if unit == "chars":
            return len(text)
        if unit == "lines":
            return len(text.splitlines()) if text else 0
        if unit == "openai_tokens":
            assert openai_model_id is not None
            return int(pyine.utils.openai.estimate_token_count(text=text, model_id=openai_model_id))
        raise ValueError(f"unknown length unit: {unit}")

    @staticmethod
    def _map_length_to_reward(
        length: int,
        *,
        start_length: int,
        flat_until_length: int | None,
        end_length: int,
        reward_at_start: float,
        reward_at_end: float,
    ) -> float:
        """Map a length to a reward using a flat + linear interpolation rule."""
        effective_start = start_length if flat_until_length is None else int(flat_until_length)
        if length <= effective_start:
            return float(reward_at_start)
        if length >= end_length:
            return float(reward_at_end)
        denom = float(max(1, end_length - effective_start))
        t = float(length - effective_start) / denom
        reward = float(reward_at_start) + (float(reward_at_end) - float(reward_at_start)) * t
        return float(max(0.0, reward))

    def __call__(
        self,
        sample_ctx: reward_types.SampleContext,
    ) -> reward_types.TermResult:
        """Compute the term reward and optional per-component metrics."""
        total = 0.0
        metrics: dict[str, bool | int | float] = {}
        for component in self._config.components:
            if not component.enabled or float(component.weight) == 0.0:
                continue
            raw_text = self._get_source_text(sample_ctx, source=component.source)
            is_missing = raw_text is None
            if is_missing:
                if component.missing_text_policy == "error":
                    source_hints = {
                        "parsed_reasoning": "ensure ParsingConfig.enabled_fields includes reasoning",
                        "parsed_final_answer": "ensure ParsingConfig.enabled_fields includes final",
                        "sample_code": "ensure sample_data.code is populated",
                    }
                    hint = source_hints.get(component.source, "check that the source is available")
                    raise ValueError(
                        f"TextLengthTerm component '{component.name}' requires source='{component.source}' "
                        f"but it is not available. Suggestions: {hint}; alternatively, set "
                        f"missing_text_policy='skip' or 'empty' to handle missing text gracefully, "
                        f"or set RewardTermSpec.require_parsed=True to fail early if parsing is missing"
                    )
                if component.missing_text_policy == "skip":
                    if component.emit_metrics:
                        metrics[f"{component.name}/is_missing"] = True
                    continue
                if component.missing_text_policy == "empty":
                    raw_text = ""
                else:
                    raise ValueError(f"unknown missing_text_policy: {component.missing_text_policy}")
            assert raw_text is not None
            text = raw_text.strip() if component.strip_whitespace else raw_text
            length = self._compute_length(
                text,
                unit=component.unit,
                openai_model_id=component.openai_model_id,
            )
            length_with_offset = int(length) + int(component.length_offset)
            reward = self._map_length_to_reward(
                length_with_offset,
                start_length=int(component.start_length),
                flat_until_length=None if component.flat_until_length is None else int(component.flat_until_length),
                end_length=int(component.end_length),
                reward_at_start=float(component.reward_at_start),
                reward_at_end=float(component.reward_at_end),
            )
            reward = float(component.weight) * float(reward)
            if component.emit_metrics:
                metrics[f"{component.name}/is_missing"] = False
                metrics[f"{component.name}/length"] = int(length)
                metrics[f"{component.name}/length_with_offset"] = int(length_with_offset)
                metrics[f"{component.name}/reward"] = float(reward)
            total += float(reward)
        if self._config.clip_max is not None:
            total = min(float(total), float(self._config.clip_max))
        if not math.isfinite(float(total)):
            raise ValueError(f"computed non-finite reward: {total}")
        return reward_types.TermResult(value=float(total), metrics=metrics)


def _factory(
    spec: reward_configs.RewardTermSpec,
    *,
    parser: reward_types.OutputParser | None,
) -> reward_types.RewardTerm:
    """Build a `TextLengthTerm` from a term spec."""
    del parser  # unused; term reads `SampleContext.parsed` directly when configured
    config = TextLengthTermConfig.model_validate(spec.params)
    return TextLengthTerm(config)


reward_registry.register_term("text_length", _factory)
reward_registry.register_term_aliases("text_length", ["format/text_length"])
