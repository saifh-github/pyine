"""Configuration models for the reward manager system.

These are Pydantic v2 models intended to be embedded in higher-level experiment configs. The
configs describe which terms are enabled, how aggregation is performed, and how (and how often)
reward breakdowns/metrics are logged.
"""

import typing

import pydantic

import pyine.evals.utils
import pyine.organisms.models.rewards.core.types as reward_types

type RewardTermParamsType = dict[str, pydantic.JsonValue]
"""JSON-serializable parameter payload passed to reward term factories."""


class AggregationConfig(reward_types.BaseConfig):
    """Configuration for reward aggregation.

    Currently supports weighted sum with optional clipping.
    """

    strategy: typing.Literal["weighted_sum"] = "weighted_sum"
    """Aggregation strategy (currently only weighted sum)."""
    clip_total_min: float | None = None
    """Optional minimum clip value applied to the total reward."""
    clip_total_max: float | None = None
    """Optional maximum clip value applied to the total reward."""
    clip_term_min: float | None = None
    """Optional minimum clip value applied to each unweighted term value."""
    clip_term_max: float | None = None
    """Optional maximum clip value applied to each unweighted term value."""
    return_raw_breakdown: bool = False
    """Whether to include raw (pre-clipping, pre-weighting) values in RewardOutput.raw_terms."""

    @pydantic.model_validator(mode="after")
    def _validate_clip_ranges(self) -> "AggregationConfig":
        """Validate that clip_min <= clip_max when both are set."""
        if self.clip_total_min is not None and self.clip_total_max is not None:
            if self.clip_total_min > self.clip_total_max:
                raise ValueError(
                    f"clip_total_min ({self.clip_total_min}) cannot be greater than "
                    f"clip_total_max ({self.clip_total_max})"
                )
        if self.clip_term_min is not None and self.clip_term_max is not None:
            if self.clip_term_min > self.clip_term_max:
                raise ValueError(
                    f"clip_term_min ({self.clip_term_min}) cannot be greater than clip_term_max ({self.clip_term_max})"
                )
        return self


class VerbosityScalingConfig(reward_types.BaseConfig):
    """Configuration for verbosity-based reward scaling.

    When enabled, applies a multiplicative factor to the aggregated reward based on output length.
    Supports two modes: absolute (threshold-based decay) and relative (group-normalized sigmoid).

    Verbosity scaling requires a tokenizer source for token counting. Either provide a HF tokenizer
    argument to RewardManager, or set parsing.openai_tokenizer_model to use tiktoken. Token counting
    is automatically enabled when verbosity_scaling.enabled=True.

    Important:
        For relative mode, "group" means samples sharing the same prompt (identified by trace_id
        in GRPO training). TRL adapter groups completions before computing relative scaling.

    Relative mode and max_factor:
        In relative mode, samples at or below the group mean get factor ~max_factor, not necessarily 1.0.
        If max_factor < 1.0, even "non-verbose" samples will receive some penalty. Set max_factor=1.0
        if you want no penalty for samples at or below the mean.

    Skip behavior in relative mode:
        When relative mode cannot compute meaningful scaling (group size < 2, or std = 0), the factor
        is set to 1.0 regardless of max_factor. This is a safe fallback that avoids penalizing samples
        when relative comparison is impossible. A warning is emitted once per scaler instance.

    Ordering with clip_total:
        Verbosity scaling is applied AFTER aggregation and any clip_total_min/clip_total_max clipping.
        This means the final scaled reward may fall outside the clip bounds. If you need strict bounds,
        set min_factor appropriately or apply additional post-processing.

    Note on negative rewards:
        Multiplication by factor < 1 moves negative rewards toward 0 (i.e., increases them).
        If this is undesired, ensure rewards are non-negative before scaling.
    """

    enabled: bool = True
    """Whether verbosity scaling is active."""
    length_source: reward_types.LengthSource = reward_types.LengthSource.model_output
    """Which text source to measure for length-based scaling.

    Only model_output, parsed_reasoning, and parsed_final_answer are currently supported
    (others will raise at config validation time).
    """
    mode: typing.Literal["absolute", "relative"] = "absolute"
    """Scaling mode: 'absolute' uses fixed thresholds, 'relative' uses group normalization."""
    decay_type: typing.Literal["linear", "exponential"] = "linear"
    """Decay function for absolute mode."""
    threshold_tokens: pydantic.NonNegativeInt = 0
    """Token count below which no penalty is applied (absolute mode)."""
    end_tokens: pydantic.PositiveInt = 1000
    """Token count at which factor reaches min_factor (absolute mode, linear decay)."""
    decay_rate: pydantic.PositiveFloat = 0.001
    """Decay rate for absolute-exponential mode: factor = max_factor * exp(-decay_rate * excess)."""
    temperature: pydantic.PositiveFloat = 1.0
    """Sigmoid temperature for relative mode (higher = gentler slope)."""
    min_factor: pydantic.NonNegativeFloat = 0.0
    """Minimum value for the verbosity factor (floor). Must be <= max_factor."""
    max_factor: pydantic.PositiveFloat = 1.0
    """Maximum value for the verbosity factor (ceiling). Factor starts here and decays toward min_factor."""
    emit_metrics: bool = True
    """Whether to emit verbosity-related metrics."""
    skip_negative_rewards: bool = True
    """If True, skip scaling for samples with negative rewards (factor=1.0).

    Since multiplication by factor < 1 moves negative rewards toward 0 (increases them),
    this option prevents that behavior. When enabled, only positive rewards are scaled.
    """

    _SUPPORTED_LENGTH_SOURCES: typing.ClassVar[frozenset[reward_types.LengthSource]] = frozenset(
        {
            reward_types.LengthSource.model_output,
            reward_types.LengthSource.parsed_reasoning,
            reward_types.LengthSource.parsed_final_answer,
        }
    )

    @pydantic.model_validator(mode="after")
    def _validate_config(self) -> "VerbosityScalingConfig":
        """Validate config constraints."""
        if self.length_source not in self._SUPPORTED_LENGTH_SOURCES:
            supported = ", ".join(sorted(s.value for s in self._SUPPORTED_LENGTH_SOURCES))
            raise ValueError(
                f"verbosity_scaling length_source={self.length_source.value!r} is not supported; "
                f"supported sources are: {supported}"
            )
        if float(self.min_factor) > float(self.max_factor):
            raise ValueError(f"min_factor ({self.min_factor}) must be <= max_factor ({self.max_factor})")
        if float(self.max_factor) > 1.0:
            raise ValueError(f"max_factor ({self.max_factor}) must be <= 1.0 for penalty behavior")
        if self.mode == "absolute" and self.decay_type == "linear":
            if int(self.end_tokens) <= int(self.threshold_tokens):
                raise ValueError(
                    f"end_tokens ({self.end_tokens}) must be > threshold_tokens ({self.threshold_tokens}) "
                    "when mode='absolute' and decay_type='linear'"
                )
        return self


class ParsingConfig(reward_types.BaseConfig):
    """Configuration for output parsing.

    Parsing is optional. When enabled, the manager parses each sample output once and exposes the
    result via `SampleContext.parsed` to all reward terms.
    """

    mode: typing.Literal["tags"] = "tags"
    """Parsing mode for model outputs."""
    enabled_fields: typing.Literal["both", "final_only", "reasoning_only"] = "both"
    """Which parsed fields to extract (skips scanning disabled fields)."""
    final_tag: str = "final"
    """Tag name used to extract the final answer when using `mode=\"tags\"`."""
    reasoning_from_outside_final: bool = False
    """If True, set reasoning to all text outside the selected `<final_tag>` block (before and/or after)."""
    reasoning_from_entire_output_when_no_final_answer: bool = False
    """If True and no final answer is present, treat the entire model output as reasoning.

    This is useful when the model fails to provide an answer field at all (e.g., no `<final_tag>`
    block and no `fallback_policy` match). When enabled, and the parser would otherwise produce an
    empty reasoning string, it will use the full output instead.
    """
    reasoning_tag: str = "reasoning"
    """Tag name used to extract reasoning when using `mode=\"tags\"`."""
    fallback_policy: typing.Literal["none", "last_line", "entire_output"] = "none"
    """Policy used when no final answer tag is present."""
    multi_tag_policy: typing.Literal["last", "first", "error"] = "last"
    """Policy used when multiple tag blocks are present."""
    strict: bool = False
    """Whether malformed tag structure should raise (unclosed/stray closes/nesting)."""
    capture_diagnostics: bool = True
    """Whether to include tag diagnostics in `ParsedOutput.fields`."""
    track_token_lengths: bool = False
    """Whether to track token-based lengths in addition to character lengths.

    When enabled, requires either a tokenizer argument to RewardManager or openai_tokenizer_model.

    Note: Token counting is also enabled when verbosity_scaling.enabled=True, even if this is False.
    """
    openai_tokenizer_model: str | None = None
    """OpenAI model ID for tiktoken tokenizer (e.g., 'gpt-4').

    Used for token counting when track_token_lengths=True or verbosity_scaling.enabled=True,
    and no HF tokenizer is provided to RewardManager.
    """

    @pydantic.field_validator("final_tag", "reasoning_tag")
    @classmethod
    def _validate_tag_name(
        cls,
        value: str,
    ) -> str:
        """Validate and normalize an XML-like tag name."""
        name = value.strip()
        if not name:
            raise ValueError("tag name cannot be empty")
        return name

    @pydantic.model_validator(mode="after")
    def _validate_config(self) -> "ParsingConfig":
        """Validate parsing configuration consistency."""
        if self.enabled_fields == "reasoning_only" and self.fallback_policy != "none":
            raise ValueError("fallback_policy applies to final_answer; set enabled_fields to include final")
        if self.enabled_fields == "both" and self.final_tag == self.reasoning_tag:
            raise ValueError(
                f"final_tag and reasoning_tag cannot be the same when enabled_fields='both': {self.final_tag!r}"
            )
        return self


class LoggingConfig(reward_types.BaseConfig):
    """Configuration for reward logging.

    Logging is intentionally term-agnostic: terms emit scalar metrics; the logger decides what
    gets logged and how often based on frequency settings. The manager handles key scoping.

    Note: When enabled=True, a logger must be passed to RewardManager() or a ValueError is raised.
    """

    enabled: bool = True
    """Whether reward logging is enabled."""

    log_total: bool = True
    """Whether to include the total reward in logs."""
    log_terms: bool = True
    """Whether to include per-term weighted values in logs."""
    log_metrics: bool = True
    """Whether to include term-emitted and parsing metrics in logs."""
    log_tables: bool = True
    """Whether to log per-sample reward breakdowns to a W&B table (if supported by logger)."""

    # frequency settings (controlled by logger, not manager)
    scalar_log_every_n_generations: pydantic.PositiveInt = 30
    """Emit scalar metrics to wandb every N generations (1-indexed). Logger gates emission internally."""
    table_row_every_n_generations: pydantic.PositiveInt = 60
    """Add a row to the table buffer every N generations (1-indexed). Logger gates rows internally."""
    table_flush_every_n_generations: pydantic.PositiveInt = 1000
    """Flush the table buffer every N generations (fallback if table_max_rows not reached)."""
    table_max_rows: pydantic.PositiveInt = 1000
    """Maximum number of rows kept in the in-memory table buffer before forcing a flush."""

    # metric names
    step_metric_key: str = "train/global_step"
    """Key used for the step metric in WandB logging.

    Defaults to "train/global_step" to align with HuggingFace Trainer's WandbCallback, which calls
    `wandb.define_metric("*", step_metric="train/global_step")`. Change this if using a different
    trainer or custom step tracking.
    """

    # distributed logging
    main_process_only: bool = True
    """Whether reward logging should only happen on the main (rank 0) process."""
    gather_distributed_summaries: bool = True
    """Whether to gather run summaries across ranks and log them on rank 0."""
    barrier_before_finalize: bool = True
    """Whether to barrier all ranks before emitting run-level summaries."""

    # category extraction
    category_extraction_config: pyine.evals.utils.SampleCategoryExtractionConfig | None = None
    """Configuration for extracting categories from sample data for category-wise reward logging.

    When set, the RewardManager will accumulate rewards by category (e.g., code_type, predict_type)
    and log category-wise metrics during evaluation. If None, category-wise logging is disabled.
    """


class RewardTermSpec(reward_types.BaseConfig):
    """Specification for a single reward term instance.

    `type` is a registry key (see `core.registry`). `params` is validated by the term implementation.
    """

    name: str
    """Stable term name used as breakdown/logging key."""
    type: str
    """Registry key for the term implementation (see `core.registry`)."""
    weight: float = 1.0
    """Scalar weight applied to the term value during aggregation."""
    enabled: bool = True
    """Whether the term is active."""
    require_parsed: bool = False
    """Whether this term requires `SampleContext.parsed` to be populated."""
    params: RewardTermParamsType = pydantic.Field(default_factory=dict)
    """Term-specific configuration payload (validated by the term implementation)."""

    @pydantic.field_validator("name")
    @classmethod
    def _validate_name(
        cls,
        value: str,
    ) -> str:
        """Validate and normalize the configured term name."""
        name = value.strip()
        if not name:
            raise ValueError("term name cannot be empty")
        return name

    @pydantic.field_validator("type")
    @classmethod
    def _validate_type_key(
        cls,
        value: str,
    ) -> str:
        """Validate and normalize the configured registry type key."""
        key = value.strip()
        if not key:
            raise ValueError("term type cannot be empty")
        return key

    @pydantic.field_validator("weight")
    @classmethod
    def _validate_weight(
        cls,
        value: typing.Any,
    ) -> float:
        """Validate that the configured term weight is finite."""
        import math

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("weight must be a number")
        float_value = float(value)
        if not math.isfinite(float_value):
            raise ValueError("weight must be finite")
        return float_value


class RewardManagerConfig(reward_types.BaseConfig):
    """Configuration for a `RewardManager` instance.

    This config is validated at manager construction time:
    - duplicate term names are rejected
    - unknown term types are rejected (via the registry)
    """

    terms: list[RewardTermSpec] = pydantic.Field(min_length=1)
    """Ordered list of reward term specs to evaluate."""
    aggregation: AggregationConfig = pydantic.Field(default_factory=AggregationConfig)
    """Aggregation strategy and clipping configuration."""
    logging: LoggingConfig = pydantic.Field(default_factory=LoggingConfig)
    """Logging configuration (frequency, scoping, toggles)."""
    parsing: ParsingConfig | None = None
    """Optional parsing configuration used to populate `SampleContext.parsed`."""
    verbosity_scaling: VerbosityScalingConfig | None = None
    """Optional verbosity-based reward scaling configuration."""

    @pydantic.model_validator(mode="after")
    def _validate_terms_unique(self) -> "RewardManagerConfig":
        """Ensure term names are unique to keep breakdown keys stable."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for term in self.terms:
            if term.name in seen:
                duplicates.add(term.name)
            seen.add(term.name)
        if duplicates:
            raise ValueError(f"duplicate term names: {sorted(duplicates)}")
        return self
