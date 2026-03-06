"""Configuration models for the reward manager system.

These are Pydantic v2 models intended to be embedded in higher-level experiment configs. The
configs describe which terms are enabled, how aggregation is performed, and how (and how often)
reward breakdowns/metrics are logged.
"""

import pathlib
import typing

import pydantic

import pyine.evals.common
import pyine.evals.utils
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.utils.code.difficulty as difficulty_utils
import pyine.utils.parsing

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


class CorrectnessClassifierScalingConfig(reward_types.BaseConfig):
    """Configuration for correctness-classifier-based reward scaling.

    When enabled, applies a multiplicative factor to the aggregated reward based on a pretrained
    classifier's predicted probability that the model output is "correct". The classifier is
    loaded lazily on first use (or eagerly via ``reset()``).

    The scaling factor is ``clamp(classifier_prob, min_factor, max_factor)``, so the scaled
    reward is bounded to ``[aggregated * min_factor, aggregated * max_factor]`` for non-negative
    aggregated rewards.

    Ordering with other scalers:
        Correctness scaling is applied BEFORE verbosity scaling. This means
        ``verbosity/pre_scaling_reward`` contains the post-classifier-scaled value, while
        ``correctness_classifier/pre_scaling_reward`` always contains the raw aggregated reward
        from terms.

    Note on negative rewards:
        Multiplying a negative reward by factor < 1 moves it toward 0 (softens the penalty).
        When ``skip_negative_rewards=True`` (default), negative rewards pass through unscaled
        (factor=1.0).
    """

    enabled: bool = True
    """Master toggle for classifier correctness scaling."""
    checkpoint_path: str
    """Path to pretrained classifier checkpoint directory (model + tokenizer)."""
    max_seq_length: pydantic.PositiveInt = pyine.evals.common.PASS_AT_K_DEFAULTS.max_new_tokens
    """Tokenization length limit; defaults to PASS_AT_K_DEFAULTS.max_new_tokens (10k).

    Validated at model load time against the effective limit (minimum of tokenizer.model_max_length
    and model.config.max_position_embeddings); a ValueError is raised if this value exceeds it.
    """
    temperature: pydantic.PositiveFloat = 1.0
    """Temperature for classifier logits before softmax (>1 softer, <1 sharper)."""
    min_factor: pydantic.NonNegativeFloat = 0.0
    """Minimum scaling factor (clamp from below)."""
    max_factor: pydantic.PositiveFloat = 1.0
    """Maximum scaling factor (clamp from above)."""
    only_for_keyword_samples: bool = False
    """If True, skip non-keyword samples (factor=neutral_factor)."""
    neutral_factor: typing.Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 1.0
    """Factor when skipped (e.g. non-keyword sample). 1.0 means no scaling. Constrained to [0, 1]."""
    positive_label: str = "correct"
    """Label name for the positive class, resolved from model.config.label2id."""
    allow_positive_label_fallback: bool = False
    """If True, fall back to class index 1 when label2id is missing or positive_label absent."""
    device: str | None = None
    """Device override; if None, auto-detects cuda -> mps -> cpu."""
    strict_single_turn: bool = True
    """If True, raise ValueError on serialized multi-turn JSON prompts; if False, warn once.

    Warning: in distributed training, a per-sample raise can deadlock if only one rank hits a bad
    sample before a collective sync. Set to False for distributed runs unless the dataset has been
    prevalidated to contain only single-turn prompts.
    """
    skip_negative_rewards: bool = True
    """If True, use factor=1.0 for negative aggregated rewards (avoids making penalties less negative)."""
    emit_metrics: bool = True
    """If True, emit per-sample diagnostic metrics (correctness_classifier/*)."""

    @pydantic.model_validator(mode="after")
    def _validate_factor_range(self) -> "CorrectnessClassifierScalingConfig":
        """Validate factor constraints."""
        if float(self.min_factor) > float(self.max_factor):
            raise ValueError(f"min_factor ({self.min_factor}) must be <= max_factor ({self.max_factor})")
        if float(self.max_factor) > 1.0:
            raise ValueError(f"max_factor ({self.max_factor}) must be <= 1.0 for scaling behavior")
        return self


class ParsingConfig(pyine.utils.parsing.ParsingConfig):
    """Reward-specific parsing config with token counting support.

    Extends the shared ``ParsingConfig`` with fields used only by the reward pipeline
    for token-based length tracking.
    """

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


class LoggingConfig(reward_types.BaseConfig):
    """Configuration for reward logging.

    Logging is intentionally term-agnostic: terms emit scalar metrics; the logger decides what
    gets logged and how often based on frequency settings. The manager handles key scoping.

    Metric Indexing:
        Different metric types are indexed to different x-axes in WandB:

        - **Per-generation metrics** (reward/total, reward/terms/*, reward/metrics/*) are indexed to
          `{prefix}/generation_count`. This ensures each logged generation has a unique x-coordinate,
          avoiding aggregation issues when multiple generations are logged within the same trainer
          step (e.g., with gradient accumulation or multiple generations per prompt in GRPO).

        - **Batch-level metrics** (reward/batch/*) are indexed to `{prefix}/batch_count`. This
          ensures each logged batch has a unique x-coordinate, avoiding aggregation issues when
          multiple batches are processed within the same trainer step (e.g., gradient accumulation).

        - **Run-level summaries** (reward/run/*, parsing/*) are indexed to `step_metric_key` (default:
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
    """Whether to include term-emitted metrics in per-sample logs."""
    log_tables: bool = True
    """Whether to log per-sample reward breakdowns to a W&B table (if supported by logger)."""
    log_batch_stats: bool = True
    """Whether to log per-batch total reward statistics (mean/std).

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
    separate periodic flush interval-set this value to control flush frequency during phases.
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

    Note: When ``expect_all_rank_logging=True``, the manager relaxes this check for
    per-sample logging so that non-main ranks also call ``log_sample()``. Phase
    summaries and batch stats are unaffected and remain main-process-only.
    """
    expect_all_rank_logging: bool = False
    """Whether the manager should expect logging on all ranks.

    When True, the manager (1) raises ValueError if constructed without a logger on any
    rank (including non-main ranks), and (2) relaxes the ``main_process_only`` early
    return in ``_maybe_log_sample`` so non-main ranks also log samples.

    Set automatically by the training harness when
    ``GenerationExportConfig.export_all_ranks=True`` to fail fast on misconfiguration.
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

    # histogram settings
    histogram_num_bins: int = 50
    """Number of bins for histogram visualizations. Must be > 0."""
    histogram_max_samples: int = 100_000
    """Maximum samples to retain for total reward histogram building (uses reservoir sampling).

    Bounds memory usage when processing many samples per phase. Set to 0 to disable
    limit (not recommended for long runs).

    Note: This setting only applies to the total reward histogram. Difficulty score and per-bin
    histograms are controlled by DifficultyConfig tracking flags (track_percentiles,
    track_bin_quantiles). When enabled, those histograms store all values for the phase without
    bounds, which may cause memory growth in very long runs.
    """

    @pydantic.model_validator(mode="after")
    def _validate_expect_all_rank_logging(self) -> typing.Self:
        if self.expect_all_rank_logging and not self.enabled:
            raise ValueError(
                "expect_all_rank_logging=True requires enabled=True; "
                "cannot require all-rank loggers when logging is disabled"
            )
        return self

    @pydantic.field_validator("histogram_num_bins")
    @classmethod
    def _validate_histogram_num_bins(
        cls,
        value: int,
    ) -> int:
        """Validate histogram_num_bins is positive."""
        if value <= 0:
            raise ValueError("histogram_num_bins must be > 0")
        return value

    @pydantic.field_validator("histogram_max_samples")
    @classmethod
    def _validate_histogram_max_samples(
        cls,
        value: int,
    ) -> int:
        """Validate histogram_max_samples is non-negative."""
        if value < 0:
            raise ValueError("histogram_max_samples must be >= 0")
        return value


class GenerationExportConfig(pydantic.BaseModel):
    """Configuration for exporting RL-generated completions to disk for later SFT re-import.

    The exported LMDB contains per-sample reward records keyed by phase prefix, sample identifier,
    and generation count. For richest metadata, set ``logging.log_tables=True`` in the reward
    manager config (ensures reasoning/final_answer fields are populated).
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    output_path: pathlib.Path
    """Directory path for the output LMDB dataset.

    Must point to a fresh (non-existent or empty) directory. Each exported record is stored
    in a LMDB database as a JSON-ZSTD entry keyed by ``{key_prefix}{sample_id}/{generation_count}``.
    """
    log_every_n_generations: pydantic.PositiveInt = 1
    """How often to write a record, in number of generations.

    Set to 1 (default) to export every generation. Higher values subsample the export,
    which can reduce disk usage for long runs where not every completion is needed.
    """
    export_all_ranks: bool = False
    """Whether to export generations on all distributed ranks.

    When True, each rank should write to its own path (e.g. `{output_path}/rank_{global_rank}/`)
    so that all completions are captured across distributed training. When False (default), only the
    logging rank (determined by LoggingConfig.main_process_only) exports to ``output_path`` directly.

    Callers outside the standard training harness (``create_model_organism_reward_components``)
    must also set ``LoggingConfig.expect_all_rank_logging=True`` and ensure all ranks
    receive a logger to avoid silently dropping data.

    Note: When combined with ``main_process_only=False``, frequency gating uses global generation
    counts. If ``log_every_n_generations > 1``, different ranks may export different subsets.
    Use ``log_every_n_generations=1`` (the default) to guarantee all completions are captured.
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
    correctness_classifier_scaling: CorrectnessClassifierScalingConfig | None = None
    """Optional correctness-classifier-based reward scaling configuration."""
    verbosity_scaling: VerbosityScalingConfig | None = None
    """Optional verbosity-based reward scaling configuration."""
    difficulty: difficulty_utils.DifficultyConfig | None = None
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
            "correctness_classifier",  # correctness scaling metrics (correctness_classifier/*)
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
