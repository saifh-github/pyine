"""Configuration models for the reward manager system.

These are Pydantic v2 models intended to be embedded in higher-level experiment configs. The
configs describe which terms are enabled, how aggregation is performed, and how (and how often)
reward breakdowns/metrics are logged.
"""

import typing

import pydantic

import pyine.organisms.models.rewards.core.types as reward_types
import pyine.utils.pydantic as pyine_pydantic_utils  # noqa: F401  # pyright: ignore[reportUnusedImport]

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


class OutputConfig(reward_types.BaseConfig):
    """Configuration for `RewardManager.compute()` return values."""

    return_breakdown_default: bool = False
    """Whether `compute()` returns a breakdown by default."""
    return_unweighted_breakdown: bool = False
    """Whether to also include unweighted values in breakdown output."""


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
    def _validate_enabled_fields(self) -> "ParsingConfig":
        """Validate that enabled fields are consistent with fallback behavior."""
        if self.enabled_fields == "reasoning_only" and self.fallback_policy != "none":
            raise ValueError("fallback_policy applies to final_answer; set enabled_fields to include final")
        return self


class LoggingConfig(reward_types.BaseConfig):
    """Configuration for reward logging.

    Logging is intentionally term-agnostic: terms emit scalar metrics; the manager decides what
    gets logged, how often, and with what key scoping.
    """

    enabled: bool = False
    """Whether reward logging is enabled."""
    wandb_key_prefix: str = ""
    """Optional extra key prefix applied by `WandBRewardLogger` (applies to scalars and table key)."""
    log_total: bool = True
    """Whether to include the total reward in logs."""
    log_terms: bool = True
    """Whether to include per-term weighted values in logs."""
    log_metrics: bool = True
    """Whether to include term-emitted metrics in logs."""
    log_every_n_examples: pydantic.PositiveInt = 1
    """Log every N examples (frequency gate)."""
    scope_prefix: str = "reward/"
    """Prefix for all emitted logging keys (e.g., `reward/`)."""
    main_process_only: bool = True
    """Whether reward logging should only happen on the main (rank 0) process."""
    gather_distributed_summaries: bool = False
    """Whether to gather run summaries across ranks and log them on rank 0."""
    barrier_before_finalize: bool = False
    """Whether to barrier all ranks before emitting run-level summaries."""

    log_tables: bool = False
    """Whether to log per-sample reward breakdowns to a W&B table (if supported by logger)."""
    table_key: str = "reward/rewards_table"
    """W&B key under which the per-sample rewards table is logged."""
    table_flush_every_n_logs: pydantic.PositiveInt = 100
    """Flush the rewards table every N logger calls (after frequency gating)."""
    table_max_rows: pydantic.PositiveInt = 1000
    """Maximum number of rows kept in the in-memory table buffer before forcing a flush."""


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
    output: OutputConfig = pydantic.Field(default_factory=OutputConfig)
    """Return-shape configuration for `RewardManager.compute()`."""
    parsing: ParsingConfig | None = None
    """Optional parsing configuration used to populate `SampleContext.parsed`."""

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
