"""Configuration models for the reward manager system.

These are Pydantic v2 models intended to be embedded in higher-level experiment configs. The
configs describe which terms are enabled, how aggregation is performed, and how (and how often)
reward breakdowns/metrics are logged.
"""

import typing

import pydantic

import pyine.evals.utils
import pyine.organisms.models.rewards.core.difficulty as difficulty_module
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
        in GRPO training). Grouping is handled by `RewardManager.compute_batch()` based on
        `SampleData.identifier` (trace id) so that multiple generations for the same prompt are
        normalized together.

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


class DifficultyConfig(reward_types.BaseConfig):
    """Configuration for task difficulty estimation and logging.

    When enabled, computes difficulty metrics for each sample based on code complexity
    metrics and/or trace execution characteristics. Metrics are logged alongside rewards
    to analyze how model performance scales with task difficulty.

    Difficulty Axes:
        Task difficulty for code execution reasoning can be decomposed into two axes:

        1. **Reasoning depth**: The number of sequential reasoning steps required to solve the
           problem, reflecting how far logical dependencies span across the trace. This is
           well-captured by `trace_step_count` (the default primary_source).

        2. **Computational burden**: The amount and brittleness of exact symbolic/numeric
           manipulation within and across steps --- how much state (variables, values) must be
           faithfully carried and updated. We currently lack good metrics for this axis; the
           `halstead_effort` (included in secondary_sources by default) serves as a rough
           code-level proxy, but it measures static code complexity rather than dynamic
           execution state. Better metrics would require trace-level variable statistics or
           evaluations using reference reasoning models directly.

    Binning Strategy:
        Bins are always defined in difficulty-score-space (after normalization). The bin edges
        depend on the normalization_mode:
        - fixed_range: uniform edges in [0, 1] (no overflow bin needed);
        - log: uniform edges in [0, bin_max_score], plus an overflow bin (edge=inf) if
          scores can exceed the last edge (i.e., no score_clip_max, or score_clip_max > bin_max_score).

    Important:
        When `SampleData.has_code_override=True`, the trace_step_count and complexity_metrics are
        still from the original traced code. This means that they might not be reliable indicators
        of sample difficulty (depending on how the code was modified). The `code_override_mode`
        setting controls handling in those cases.
    """

    enabled: bool = True
    """Whether difficulty estimation is active."""

    primary_source: str = "trace_step_count"
    """Primary metric key for difficulty computation.

    Supported sources:

    Execution-relation (affected by has_code_override, use code_override_mode to control):
    - 'trace_step_count': number of trace steps (default);
    - 'segment_span': last_line - first_line;
    - any key from complexity_metrics (e.g., 'halstead_effort', 'cyclomatic_complexity').

    Context-related (valid even with has_code_override):
    - 'code_length': character length of sample code;
    - 'inputs_length': character length of inputs;
    - 'output_length': character length of expected output (from sample data, not model prediction).

    Token-based sources (require tokenizer):
    - 'code_tokens': token count of sample code;
    - 'inputs_tokens': token count of inputs;
    - 'output_tokens': token count of expected output (from sample data, not model prediction).

    To analyze total context size, configure all three token sources as secondary_sources
    and sum them in post-processing. The breakdown is more useful than pre-aggregating.
    """

    secondary_sources: frozenset[str] = frozenset({"halstead_effort"})
    """Additional sources to log raw values for (e.g., to analyze computational burden axis).

    Default includes 'halstead_effort' as a rough proxy for computational burden.

    Logging secondary sources may be useful for downstream/offline difficulty analyses.
    """

    normalization_mode: typing.Literal["none", "log", "fixed_range"] = "log"
    """Specifies how to normalize raw difficulty values.

    Supported modes:
    - none: Use raw values directly (requires explicit bin_edges)
    - log: Apply log1p transform; score is unbounded but well-behaved (typically 0-7)
    - fixed_range: For known-range metrics; score in [0, 1]. Uses source_ranges config.
    """

    source_ranges: dict[str, tuple[float, float, bool]] | None = None
    """Custom ranges for fixed_range normalization.

    Keys are targeted source names, values are `(min, max, invert)` tuples. Score is computed as
    `(clamped - min) / (max - min)`, then if `invert=True`, `score = 1 - score`. Use invert=True
    when higher raw values indicate easier tasks (lower difficulty). Defaults merged with
    DEFAULT_SOURCE_RANGES (which includes maintainability_index).

    Example: `{'custom_metric': (0.0, 1000.0, False)}` for a metric whose values typically range
    from 0 to 1000, and where 0.0 indicates an easier problem.
    """

    score_clip_max: float | None = None
    """Optional ceiling applied to normalized score. Prevents rare outliers from dominating bins.

    When set and <= bin_max_score (or the auto-derived bin max), no overflow bin is added since all
    scores are bounded within the bin range. When score_clip_max > bin_max_score, an overflow bin
    is still added for scores between bin_max_score and score_clip_max.
    """

    num_difficulty_bins: int = 10
    """Number of bins for difficulty-stratified reward logging."""

    bin_max_score: float | None = None
    """Maximum score for bin edges (log mode only).

    If None, defaults to score_clip_max if set, otherwise log1p(10_000) = 9.21...; scores above this
    go into an overflow bin.
    """

    bin_edges: list[float] | None = None
    """Optional manual bin edges. If provided, overrides automatic edge computation.

    Should be a sorted list of floats. The number of bins = len(bin_edges) - 1.

    Example: `[0.0, 1.0, 2.0, 3.0, float('inf')]` for 4 bins with overflow.
    """

    code_override_mode: difficulty_module.CodeOverrideMode = difficulty_module.CodeOverrideMode.skip
    """How to handle samples with `has_code_override=True`, i.e. without reliable trace coverage.

    Default is 'skip' because trace/complexity metrics are from the original code, not the
    overridden code the model actually saw. For context difficulty (token sources), use
    secondary_sources which still work with overrides.

    Options:
    - 'skip': skip execution difficulty metrics, context metrics (token sources) still logged;
    - 'use_original': use original traced metrics (may not reflect actual prediction task difficulty);
    - 'recompute_step_count': recompute trace_step_count from segment indices if available,
      otherwise fall back to original trace_step_count.
    """

    track_percentiles: typing.Literal["disabled", "eval_only", "sampled", "always"] = "disabled"
    """When to track difficulty score distribution for percentile computation.

    Stores all difficulty scores in memory for accurate median/p90/p99.

    Options:
    - 'disabled': do not track percentiles;
    - 'eval_only': only track during eval phase;
    - 'sampled': track every `sample_every_n_generations` generations;
    - 'always': track every generation (use with caution for long runs).
    """

    track_per_term_rewards: typing.Literal["disabled", "eval_only", "sampled", "always"] = "disabled"
    """When to track per-term reward stats within difficulty bins.

    Logs per-bin statistics for each individual reward term (in addition to total reward).
    This multiplies the number of logged metrics by the number of terms.

    Options:
    - 'disabled': do not track per-term rewards;
    - 'eval_only': only track during eval phase;
    - 'sampled': track every `sample_every_n_generations` generations;
    - 'always': track every generation (use with caution for long runs).
    """

    track_bin_quantiles: typing.Literal["disabled", "eval_only", "sampled", "always"] = "disabled"
    """When to track per-difficulty-bin reward quantiles (p10/p50/p90).

    Stores all reward values per difficulty bin in memory for quantile computation.

    Options:
    - 'disabled': do not track bin quantiles;
    - 'eval_only': only track during eval phase;
    - 'sampled': track every `sample_every_n_generations` generations;
    - 'always': track every generation (use with caution for long runs).
    """

    table_mode: typing.Literal["disabled", "eval_only", "sampled", "always"] = "disabled"
    """When to log the lightweight per-sample difficulty table.

    Logs a minimal wandb table (difficulty_samples) with columns: step, generation_count, sample_id,
    primary_source, raw_primary, difficulty_score, difficulty_bin, reward_total, predict_type,
    code_type, has_code_override. Secondary raw values (e.g., halstead_effort) are added as dynamic
    columns when present. Useful for downstream "difficulty vs reward" analysis.

    Options:
    - 'disabled': do not log the table;
    - 'eval_only': only log during eval phase (prefix contains 'eval');
    - 'sampled': log every `sample_every_n_generations` generations;
    - 'always': log every sample in all phases (use with caution).
    """

    sample_every_n_generations: pydantic.PositiveInt = 100
    """Sampling interval for 'sampled' mode tracking options.

    When any tracking option (track_percentiles, track_per_term_rewards, track_bin_quantiles,
    table_mode) is set to 'sampled', values are tracked/logged every N generations. This
    provides a representative sample without storing everything.
    """

    @pydantic.model_validator(mode="after")
    def _validate_config(self) -> "DifficultyConfig":
        """Validate configuration consistency."""
        # validate num_difficulty_bins
        if self.num_difficulty_bins < 1:
            raise ValueError("num_difficulty_bins must be >= 1")
        # validate bin_edges if provided
        if self.bin_edges is not None:
            if len(self.bin_edges) < 2:
                raise ValueError("bin_edges must have at least 2 values")
            if self.bin_edges != sorted(self.bin_edges):
                raise ValueError("bin_edges must be sorted in ascending order")
        # validate bin_max_score
        if self.bin_max_score is not None and self.bin_max_score <= 0:
            raise ValueError("bin_max_score must be positive if set")
        # validate score_clip_max
        if self.score_clip_max is not None and self.score_clip_max <= 0:
            raise ValueError("score_clip_max must be positive if set")
        # CRITICAL: normalization_mode="none" is incoherent with auto-binning
        if self.normalization_mode == "none" and self.bin_edges is None:
            raise ValueError(
                "normalization_mode='none' requires explicit bin_edges; "
                "raw scores have unbounded ranges, making auto-binning meaningless. "
                "Either provide bin_edges or use normalization_mode='log'"
            )
        # CRITICAL: fixed_range requires source with known range for auto-binning
        if self.normalization_mode == "fixed_range" and self.bin_edges is None:
            all_ranges = {**difficulty_module.DEFAULT_SOURCE_RANGES, **(self.source_ranges or {})}
            if self.primary_source not in all_ranges:
                known_sources = ", ".join(sorted(all_ranges.keys()))
                raise ValueError(
                    f"normalization_mode='fixed_range' with auto-binning requires primary_source "
                    f"to have a known range. Known sources: {known_sources}. "
                    f"Got primary_source={self.primary_source!r}. "
                    "Either use normalization_mode='log', provide explicit bin_edges, "
                    "add an entry to source_ranges, or choose a known-range source."
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

    Metric Indexing:
        Different metric types are indexed to different x-axes in WandB:

        - **Per-generation metrics** (reward/total, reward/terms/*, parsing/*) are indexed to
          `{prefix}/generation_count`. This ensures each logged generation has a unique x-coordinate,
          avoiding aggregation issues when multiple generations are logged within the same trainer
          step (e.g., with gradient accumulation or multiple generations per prompt in GRPO).

        - **Batch-level metrics** (reward/batch/*) are indexed to `{prefix}/batch_count`. This
          ensures each logged batch has a unique x-coordinate, avoiding aggregation issues when
          multiple batches are processed within the same trainer step (e.g., gradient accumulation).

        - **Run-level summaries** (reward/run/*) are indexed to `step_metric_key` (default:
          "train/global_step"), which should be an optimizer-step-related index.

        The logger automatically configures these step metrics via `wandb.define_metric()`.

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
    log_batch_stats: bool = True
    """Whether to log per-batch total reward statistics (mean/std and rolling aggregates).

    The overhead is minimal since batch stats are gathered alongside per-sample global generation
    counts in a single distributed collective operation.
    """

    # frequency settings (controlled by logger, not manager)
    log_every_n_generations: pydantic.PositiveInt = 30
    """Log per-generation metrics (scalars and table rows) every N generations (1-indexed).

    This controls both scalar logging and table row additions. When this generation count is
    reached, scalars are emitted and (if log_tables=True) a row is added to the table buffer.
    These metrics are indexed to `{prefix}/generation_count` in WandB, not `step_metric_key`.
    """
    table_max_rows: pydantic.PositiveInt = 100
    """Maximum rows in the table buffer before flushing to W&B.

    When the buffer reaches this size, it's flushed and cleared. This controls both memory
    usage and how frequently table data appears in W&B. Lower values mean more frequent
    uploads but more W&B API calls; higher values batch more data per upload.

    Note: Tables are flushed in two scenarios: (1) when the buffer reaches this size, and
    (2) at phase transitions when flush_stats() or finalize_run() is called. There is no
    separate periodic flush interval—set this value to control flush frequency during phases.
    """

    # metric names
    step_metric_key: str = "train/global_step"
    """Key used as the x-axis for run-level summaries (reward/run/*) in WandB.

    Defaults to "train/global_step" to align with HuggingFace Trainer's WandbCallback; should
    generally correspond to an index linked with optimizer steps.

    Note: Per-generation metrics use `{prefix}/generation_count` and batch-level metrics
    use `{prefix}/batch_count` as their x-axes. These are configured automatically via
    `wandb.define_metric()` to ensure unique x-coordinates.
    """

    # distributed logging
    main_process_only: bool = True
    """Whether reward logging should only happen on the main (rank 0) process.

    When True (default), logging frequency gating (log_every_n_generations) uses LOCAL
    generation counts, so rank 0's N-th sample triggers logging. This ensures
    predictable logging frequency regardless of how samples are distributed across ranks.
    The x-axis value in logged metrics is still the GLOBAL generation count for cross-rank
    alignment.

    When False, all ranks log and frequency gating uses GLOBAL generation counts, so the
    N-th global sample triggers logging regardless of which rank processes it. This requires
    all ranks to have a logger configured.
    """
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
    difficulty: DifficultyConfig | None = None
    """Optional task difficulty estimation configuration."""

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

    # term names that would collide with metric namespaces or cause confusion
    _RESERVED_TERM_NAMES: typing.ClassVar[frozenset[str]] = frozenset(
        {
            # top-level metric namespaces
            "difficulty",  # difficulty estimation metrics (difficulty/*)
            "parsing",  # parsing metrics (parsing/*)
            "verbosity",  # verbosity scaling metrics (verbosity/*)
            "failures",  # failure tracking metrics (failures/*)
            "categories",  # category-wise metrics (categories/*)
            # reward/* sibling keys (would create reward/terms/{name} alongside reward/{name})
            "total",  # reward/total
            "batch",  # reward/batch/*
            "run",  # reward/run/*
            "metrics",  # reward/metrics/*
            "raw_terms",  # reward/raw_terms/*
            "terms",  # reward/terms/* (recursive)
        }
    )

    @pydantic.model_validator(mode="after")
    def _validate_term_names_not_reserved(self) -> "RewardManagerConfig":
        """Ensure term names don't collide with reserved metric namespaces.

        Forbids:
        - Slashes in term names (would create confusing metric key hierarchies)
        - Reserved names that collide with existing metric namespaces
        """
        for term in self.terms:
            if "/" in term.name:
                raise ValueError(
                    f"term name '{term.name}' contains '/'; slashes are forbidden in term names "
                    "to avoid confusing metric key hierarchies"
                )
            if term.name in self._RESERVED_TERM_NAMES:
                raise ValueError(
                    f"term name '{term.name}' is reserved; it would collide with existing metric "
                    f"namespaces. Reserved names: {sorted(self._RESERVED_TERM_NAMES)}"
                )
        return self
