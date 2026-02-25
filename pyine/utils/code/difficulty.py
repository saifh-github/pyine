"""Difficulty estimation utilities shared across rewards and evals pipelines.

Provides shared configuration, normalization, and scoring logic for per-sample difficulty
estimation based on trace metadata, code complexity metrics, and length/token features.
Reward-specific aggregation and logging are intentionally excluded from this module.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import typing

import pydantic

if typing.TYPE_CHECKING:
    import pyine.organisms.datamodules.samples as samples


class CodeOverrideMode(enum.StrEnum):
    """Specify how to treat samples with ``has_code_override=True``.

    Overrides indicate that the code text may not match the traced execution metadata. These
    modes control whether execution-derived metrics remain valid and how missing scores should
    be handled.
    """

    skip = enum.auto()
    """Skip execution difficulty metrics; context metrics (token sources) still logged.

    Use when overrides change program logic (e.g., injected bugs, code replacements). In these
    cases, execution-derived difficulty from the original trace is not reliable.
    """
    use_original = enum.auto()
    """Use original traced metrics (may not reflect actual code).

    Useful when overrides only affect docstrings/comments (hints) and do not change execution
    behavior, or when you prefer comparability with the original trace.
    """
    recompute_step_count = enum.auto()
    """Recompute trace_step_count from segment indices; use original complexity metrics.

    Uses ``last_step_idx - first_step_idx + 1`` if both indices are > 0. Falls back to the
    original ``trace_step_count`` if indices are unavailable.
    """


DEFAULT_SOURCE_RANGES: dict[str, tuple[float, float, bool]] = {
    # (min, max, invert): score = (clamped - min) / (max - min), then 1 - score if invert
    "maintainability_index": (0.0, 100.0, True),  # higher MI = easier, so invert
}
"""Default source ranges for fixed-range normalization.

Currently includes only ``maintainability_index`` because it has a standardized [0, 100] range.
Other complexity metrics are unbounded and should typically use ``normalization_mode=\"log\"`` or
explicit ``source_ranges``.
"""

TOKEN_SOURCES: frozenset[str] = frozenset(
    {
        "code_tokens",
        "inputs_tokens",
        "output_tokens",
    }
)
"""Difficulty sources that require a tokenizer (sample-intrinsic context measures).

``output_tokens`` refers to the expected output from sample data, not the model prediction.
"""

EXECUTION_SOURCES: frozenset[str] = frozenset(
    {
        "trace_step_count",
        "segment_span",
        # note: all complexity_metrics keys are also execution sources
    }
)
"""Execution-related difficulty sources (affected by has_code_override)."""

CONTEXT_SOURCES: frozenset[str] = frozenset(
    {
        "code_tokens",
        "inputs_tokens",
        "output_tokens",
        "code_length",
        "inputs_length",
        "output_length",
    }
)
"""Context-related difficulty sources (could be valid even with has_code_override)."""

_DEFAULT_LOG_BIN_MAX_SCORE = math.log1p(10_000)
"""Default bin_max_score for log normalization.

``log1p(10_000)`` is approximately 9.21.
"""


class DifficultyConfig(pydantic.BaseModel):
    """Configuration for task difficulty estimation.

    Captures which metadata source to use, how to normalize it, and how to handle code overrides.
    Used by both the reward pipeline (via ``DifficultyEstimator``) and the code execution eval
    pipeline for per-sample difficulty scoring.

    Difficulty Axes:
        Task difficulty for code execution reasoning can be decomposed into two axes:

        1. **Reasoning depth**: The number of sequential reasoning steps required to solve the
           problem, reflecting how far logical dependencies span across the trace. This is
           well-captured by ``trace_step_count`` (the default primary_source).

        2. **Computational burden**: The amount and brittleness of exact symbolic/numeric
           manipulation within and across steps --- how much state (variables, values) must be
           faithfully carried and updated. We currently lack good metrics for this axis; the
           ``halstead_effort`` (included in secondary_sources by default) serves as a rough
           code-level proxy, but it measures static code complexity rather than dynamic
           execution state. Better metrics would require trace-level variable statistics or
           evaluations using reference reasoning models directly.

    Binning Strategy:
        Bins are always defined in difficulty-score-space (after normalization). The bin edges
        depend on the normalization_mode:

        - ``fixed_range``: uniform edges in [0, 1] (no overflow bin needed);
        - ``log``: uniform edges in [0, bin_max_score], plus an overflow bin (edge=inf) if
          scores can exceed the last edge (i.e., no score_clip_max, or
          score_clip_max > bin_max_score).

    Important:
        When ``SampleData.has_code_override=True``, the trace_step_count and complexity_metrics
        are still from the original traced code. This means that they might not be reliable
        indicators of sample difficulty (depending on how the code was modified). The
        ``code_override_mode`` setting controls handling in those cases.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic model configuration (immutable)."""

    enabled: bool = True
    """Whether difficulty estimation is active."""

    primary_source: str = "trace_step_count"
    """Primary metric key for difficulty computation.

    Supported sources:

    Execution-related (affected by has_code_override, use code_override_mode to control):
      - ``trace_step_count``: number of trace steps (default);
      - ``segment_span``: last_line - first_line;
      - any key from complexity_metrics (e.g., ``halstead_effort``, ``cyclomatic_complexity``).

    Context-related (valid even with has_code_override):
      - ``code_length``: character length of sample code;
      - ``inputs_length``: character length of inputs;
      - ``output_length``: character length of expected output (from sample data, not model
        prediction).

    Token-based sources (require tokenizer):
      - ``code_tokens``: token count of sample code;
      - ``inputs_tokens``: token count of inputs;
      - ``output_tokens``: token count of expected output (from sample data, not model prediction).

    To analyze total context size, configure all three token sources as secondary_sources
    and sum them in post-processing. The breakdown is more useful than pre-aggregating.
    """

    secondary_sources: frozenset[str] = frozenset({"halstead_effort"})
    """Additional sources to log raw values for (e.g., to analyze computational burden axis).

    Default includes ``halstead_effort`` as a rough proxy for computational burden. Secondary
    source values are logged raw (without normalization) for downstream analysis.
    """

    normalization_mode: typing.Literal["none", "log", "fixed_range"] = "log"
    """Specifies how to normalize raw difficulty values.

    Supported modes:
      - ``none``: use raw values directly (requires explicit bin_edges);
      - ``log``: apply log1p transform; score is unbounded but well-behaved (typically 0--7);
      - ``fixed_range``: for known-range metrics; score in [0, 1]. Uses source_ranges config.
    """

    source_ranges: dict[str, tuple[float, float, bool]] | None = None
    """Custom ranges for fixed_range normalization.

    Keys are source names, values are ``(min, max, invert)`` tuples. Score is computed as
    ``(clamped - min) / (max - min)``; if ``invert=True``, ``score = 1 - score``. Use
    ``invert=True`` when higher raw values indicate easier tasks (lower difficulty). Merged with
    ``DEFAULT_SOURCE_RANGES`` (which includes ``maintainability_index``).

    Example: ``{'custom_metric': (0.0, 1000.0, False)}`` for a metric ranging 0--1000 where
    higher values mean harder problems.
    """

    score_clip_max: float | None = None
    """Optional ceiling applied to normalized score. Prevents rare outliers from dominating bins.

    When set and <= bin_max_score (or the auto-derived bin max), no overflow bin is added since
    all scores are bounded within the bin range. When score_clip_max > bin_max_score, an overflow
    bin is still added for scores between bin_max_score and score_clip_max.
    """

    num_difficulty_bins: int = 10
    """Number of bins for difficulty-stratified analysis."""

    bin_max_score: float | None = None
    """Maximum score for bin edges (log mode only).

    If None, defaults to score_clip_max if set, otherwise ``log1p(10_000)`` (~9.21); scores
    above this go into an overflow bin.
    """

    bin_edges: list[float] | None = None
    """Optional manual bin edges. If provided, overrides automatic edge computation.

    Should be a sorted list of floats. The number of bins = ``len(bin_edges) - 1``.

    Example: ``[0.0, 1.0, 2.0, 3.0, float('inf')]`` for 4 bins with overflow.
    """

    code_override_mode: CodeOverrideMode = CodeOverrideMode.use_original
    """How to handle samples with ``has_code_override=True``.

    Default is ``use_original`` to keep difficulty scores available in evals where overrides
    are limited to docstrings/comments and do not alter execution behavior.

    Options:
      - ``skip``: skip execution difficulty metrics; context metrics (token sources) still logged;
      - ``use_original``: use original traced metrics (may not reflect actual prediction task
        difficulty);
      - ``recompute_step_count``: recompute trace_step_count from segment indices if available,
        otherwise fall back to original trace_step_count.
    """

    track_percentiles: typing.Literal["disabled", "eval_only", "always"] = "disabled"
    """When to track difficulty score distribution for percentile computation.

    Stores all difficulty scores in memory for accurate median/p90/p99. Use with caution for
    long training runs when set to ``always``.
    """

    track_per_term_rewards: typing.Literal["disabled", "eval_only", "always"] = "disabled"
    """When to track per-term reward stats within difficulty bins.

    Logs per-bin statistics for each individual reward term (in addition to total reward).
    This multiplies the number of logged metrics by the number of terms.
    """

    track_bin_quantiles: typing.Literal["disabled", "eval_only", "always"] = "disabled"
    """When to track per-difficulty-bin reward quantiles (p10/p50/p90).

    Stores all reward values per difficulty bin in memory for quantile computation. Use with
    caution for long training runs when set to ``always``.
    """

    @pydantic.model_validator(mode="after")
    def _validate_config(self) -> DifficultyConfig:
        """Validate configuration consistency."""
        if self.num_difficulty_bins < 1:
            raise ValueError("num_difficulty_bins must be >= 1")
        if self.bin_edges is not None:
            if len(self.bin_edges) < 2:
                raise ValueError("bin_edges must have at least 2 values")
            if self.bin_edges != sorted(self.bin_edges):
                raise ValueError("bin_edges must be sorted in ascending order")
        if self.bin_max_score is not None and self.bin_max_score <= 0:
            raise ValueError("bin_max_score must be positive if set")
        if self.score_clip_max is not None and self.score_clip_max <= 0:
            raise ValueError("score_clip_max must be positive if set")
        if self.normalization_mode == "none" and self.bin_edges is None:
            raise ValueError(
                "normalization_mode='none' requires explicit bin_edges; "
                "raw scores have unbounded ranges, making auto-binning meaningless. "
                "Either provide bin_edges or use normalization_mode='log'"
            )
        if self.normalization_mode == "fixed_range" and self.bin_edges is None:
            all_ranges = {**DEFAULT_SOURCE_RANGES, **(self.source_ranges or {})}
            if self.primary_source not in all_ranges:
                known_sources = ", ".join(sorted(all_ranges.keys()))
                raise ValueError(
                    "normalization_mode='fixed_range' with auto-binning requires primary_source "
                    f"to have a known range. Known sources: {known_sources}. "
                    f"Got primary_source={self.primary_source!r}. "
                    "Either use normalization_mode='log', provide explicit bin_edges, "
                    "add an entry to source_ranges, or choose a known-range source."
                )
        return self


@dataclasses.dataclass(frozen=True)
class DifficultyResult:
    """Per-sample difficulty computation result.

    Captures the normalized score (when available), the raw primary value, and secondary raw
    values for optional analysis.
    """

    source: str
    """Primary difficulty source name."""
    score: float | None
    """Normalized difficulty score (None if unavailable)."""
    raw_primary: float | None
    """Raw primary source value before normalization."""
    bin_index: int | None
    """Bin index for the normalized score (None if unavailable)."""
    secondary_raw: dict[str, float]
    """Raw values for secondary sources that were available."""
    primary_missing: bool
    """Whether the primary source was missing/unavailable."""
    execution_skipped: bool
    """Whether execution-related difficulty was skipped due to code overrides."""
    has_code_override: bool
    """Whether the sample had a code override."""


def is_execution_source(
    source: str,
) -> bool:
    """Return True if a source is execution-related (affected by code overrides).

    Args:
        source: Source key name.

    Returns:
        True if the source depends on execution metadata, otherwise False.
    """
    if source in EXECUTION_SOURCES:
        return True
    return source not in CONTEXT_SOURCES and source not in TOKEN_SOURCES


def _compute_bin_edges(
    config: DifficultyConfig,
) -> list[float]:
    """Compute bin edges based on normalization mode and config.

    Args:
        config: Difficulty configuration.

    Returns:
        List of bin edges (length = num_bins + 1, plus overflow when needed).
    """
    if config.bin_edges is not None:
        return list(config.bin_edges)
    num_bins = config.num_difficulty_bins
    if config.normalization_mode == "fixed_range":
        return [idx / num_bins for idx in range(num_bins + 1)]
    bin_max = config.bin_max_score
    if bin_max is None:
        bin_max = config.score_clip_max if config.score_clip_max is not None else _DEFAULT_LOG_BIN_MAX_SCORE
    edges = [idx * bin_max / num_bins for idx in range(num_bins + 1)]
    needs_overflow = config.score_clip_max is None or config.score_clip_max > bin_max
    if needs_overflow:
        edges.append(float("inf"))
    return edges


def _get_bin_index(
    score: float,
    bin_edges: list[float],
) -> int:
    """Return bin index for a score based on bin edges.

    Args:
        score: Normalized difficulty score.
        bin_edges: Ordered list of bin edges.

    Returns:
        Index of the bin into which the score falls.
    """
    for bin_idx in range(len(bin_edges) - 1):
        if score < bin_edges[bin_idx + 1]:
            return bin_idx
    return len(bin_edges) - 2


def _get_effective_trace_step_count(
    sample_data: samples.SampleData,
    config: DifficultyConfig,
) -> float | None:
    """Resolve the effective trace step count with override handling.

    Args:
        sample_data: Sample metadata.
        config: Difficulty configuration.

    Returns:
        Effective trace step count, or None when skipped.
    """
    if not sample_data.has_code_override:
        return float(sample_data.trace_step_count)
    mode = config.code_override_mode
    if mode == CodeOverrideMode.skip:
        return None
    if mode == CodeOverrideMode.use_original:
        return float(sample_data.trace_step_count)
    if mode == CodeOverrideMode.recompute_step_count:
        if sample_data.first_step_idx > 0 and sample_data.last_step_idx > 0:
            return float(sample_data.last_step_idx - sample_data.first_step_idx + 1)
        return float(sample_data.trace_step_count)
    return None


def get_raw_value(
    sample_data: samples.SampleData,
    source: str,
    *,
    config: DifficultyConfig,
    token_counter: typing.Callable[[str], int] | None = None,
) -> float | None:
    """Extract raw difficulty value from sample data.

    Args:
        sample_data: Sample metadata containing trace and complexity fields.
        source: Source key (e.g., ``trace_step_count``, ``code_length``, or a complexity metric).
        config: Difficulty configuration.
        token_counter: Optional token counting function for token-based sources.

    Returns:
        Raw numeric value, or None when unavailable.

    Raises:
        ValueError: When a token-based source is requested without a token_counter.
    """
    if sample_data.has_code_override and config.code_override_mode == CodeOverrideMode.skip:
        if is_execution_source(source):
            return None
    if source == "trace_step_count":
        return _get_effective_trace_step_count(sample_data, config)
    if source == "code_length":
        return float(len(sample_data.code))
    if source == "inputs_length":
        return float(len(sample_data.inputs))
    if source == "output_length":
        return float(len(sample_data.expected_output))
    if source == "segment_span":
        span = sample_data.last_line - sample_data.first_line
        return float(span) if span > 0 else None
    if source == "code_tokens":
        if token_counter is None:
            raise ValueError("token_counter required for code_tokens source")
        return float(token_counter(sample_data.code))
    if source == "inputs_tokens":
        if token_counter is None:
            raise ValueError("token_counter required for inputs_tokens source")
        return float(token_counter(sample_data.inputs))
    if source == "output_tokens":
        if token_counter is None:
            raise ValueError("token_counter required for output_tokens source")
        return float(token_counter(sample_data.expected_output))
    complexity = sample_data.complexity_metrics
    if not complexity or source not in complexity:
        return None
    return float(complexity[source])


def normalize_value(
    raw_value: float,
    source: str,
    config: DifficultyConfig,
) -> float:
    """Normalize a raw difficulty value based on config.

    Args:
        raw_value: Raw value for the source.
        source: Source key name.
        config: Difficulty configuration.

    Returns:
        Normalized score.
    """
    if config.normalization_mode == "none":
        score = raw_value
    elif config.normalization_mode == "log":
        score = math.log1p(max(0.0, raw_value))
    elif config.normalization_mode == "fixed_range":
        source_ranges = {**DEFAULT_SOURCE_RANGES, **(config.source_ranges or {})}
        if source not in source_ranges:
            raise ValueError(
                f"fixed_range normalization requires source '{source}' to have a known range; "
                "this should have been caught by config validation"
            )
        low, high, invert = source_ranges[source]
        clamped = max(low, min(high, raw_value))
        score = (clamped - low) / (high - low) if high > low else 0.5
        if invert:
            score = 1.0 - score
    else:
        score = raw_value
    if config.score_clip_max is not None:
        score = min(score, config.score_clip_max)
    return score


class DifficultyScorer:
    """Compute per-sample difficulty metrics using a shared configuration."""

    def __init__(
        self,
        config: DifficultyConfig,
        token_counter: typing.Callable[[str], int] | None = None,
    ) -> None:
        """Initialize the difficulty scorer.

        Args:
            config: Difficulty configuration.
            token_counter: Optional token counting function for token-based sources.

        Raises:
            ValueError: When token-based sources are configured without a token_counter.
        """
        self._config = config
        self._token_counter = token_counter
        all_sources = {config.primary_source} | set(config.secondary_sources)
        if all_sources & TOKEN_SOURCES and token_counter is None:
            raise ValueError(
                f"difficulty sources {all_sources & TOKEN_SOURCES} require a token_counter; "
                "provide a tokenizer or avoid token-based sources"
            )
        self._bin_edges = _compute_bin_edges(config)

    @property
    def config(self) -> DifficultyConfig:
        """Return the difficulty configuration."""
        return self._config

    @property
    def bin_edges(self) -> list[float]:
        """Return the computed bin edges."""
        return list(self._bin_edges)

    def compute_score(
        self,
        sample_data: samples.SampleData,
    ) -> float | None:
        """Return only the normalized difficulty score, if available.

        Args:
            sample_data: Sample metadata.

        Returns:
            Normalized score or None when unavailable.
        """
        return self.compute_result(sample_data).score

    def compute_result(
        self,
        sample_data: samples.SampleData,
    ) -> DifficultyResult:
        """Compute the full difficulty result for a sample.

        Args:
            sample_data: Sample metadata.

        Returns:
            DifficultyResult with score, raw values, and availability flags.
        """
        primary_source = self._config.primary_source
        primary_is_execution = is_execution_source(primary_source)
        skip_execution = sample_data.has_code_override and self._config.code_override_mode == CodeOverrideMode.skip
        secondary_raw: dict[str, float] = {}
        if skip_execution and primary_is_execution:
            for secondary_source in self._config.secondary_sources:
                if secondary_source in CONTEXT_SOURCES or secondary_source in TOKEN_SOURCES:
                    raw_secondary = get_raw_value(
                        sample_data,
                        secondary_source,
                        config=self._config,
                        token_counter=self._token_counter,
                    )
                    if raw_secondary is not None:
                        secondary_raw[secondary_source] = raw_secondary
            return DifficultyResult(
                source=primary_source,
                score=None,
                raw_primary=None,
                bin_index=None,
                secondary_raw=secondary_raw,
                primary_missing=True,
                execution_skipped=True,
                has_code_override=sample_data.has_code_override,
            )
        raw_primary = get_raw_value(
            sample_data,
            primary_source,
            config=self._config,
            token_counter=self._token_counter,
        )
        if raw_primary is None:
            return DifficultyResult(
                source=primary_source,
                score=None,
                raw_primary=None,
                bin_index=None,
                secondary_raw={},
                primary_missing=True,
                execution_skipped=False,
                has_code_override=sample_data.has_code_override,
            )
        score = normalize_value(raw_primary, primary_source, self._config)
        bin_index = _get_bin_index(score, self._bin_edges)
        for secondary_source in self._config.secondary_sources:
            raw_secondary = get_raw_value(
                sample_data,
                secondary_source,
                config=self._config,
                token_counter=self._token_counter,
            )
            if raw_secondary is not None:
                secondary_raw[secondary_source] = raw_secondary
        return DifficultyResult(
            source=primary_source,
            score=score,
            raw_primary=raw_primary,
            bin_index=bin_index,
            secondary_raw=secondary_raw,
            primary_missing=False,
            execution_skipped=False,
            has_code_override=sample_data.has_code_override,
        )
