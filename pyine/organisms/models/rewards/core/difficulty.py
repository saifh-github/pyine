"""Task difficulty estimation for reward analysis.

This module provides a DifficultyEstimator class that computes difficulty metrics from SampleData
fields (trace_step_count, complexity_metrics, string lengths, etc.). Difficulty metrics are purely
diagnostic, i.e. they do not modify rewards.

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

Important:
    When SampleData.has_code_override=True, the trace_step_count and complexity_metrics
    are from the original traced code, not the overridden code. The estimator handles
    this via the configured `code_override_mode`.
"""

import enum
import math
import typing

import pyine.organisms.datamodules.samples
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.utils.stats as stats_utils

if typing.TYPE_CHECKING:
    import pyine.organisms.models.rewards.core.configs as reward_configs


class CodeOverrideMode(enum.StrEnum):
    """How to handle samples with has_code_override=True."""

    skip = enum.auto()
    """Skip execution difficulty metrics; context metrics (token sources) still logged."""
    use_original = enum.auto()
    """Use original traced metrics (may not reflect actual code)."""
    recompute_step_count = enum.auto()
    """Recompute trace_step_count from segment indices; use original complexity metrics.

    Uses (last_step_idx - first_step_idx + 1) if both indices are > 0 (0 means "not applicable").
    Falls back to original trace_step_count if indices are unavailable.
    """


DEFAULT_SOURCE_RANGES: dict[str, tuple[float, float, bool]] = {
    # (min, max, invert): score = (clamped - min) / (max - min), then 1 - score if invert
    "maintainability_index": (0.0, 100.0, True),  # higher MI = easier, so invert
}
"""Default source ranges for fixed-range normalization (can be overridden in config).

Only `maintainability_index` is pre-defined because it's the only common complexity metric with a
standardized range (0-100). Other metrics like cyclomatic_complexity, halstead_effort, etc. are
unbounded and should use `normalization_mode="log"` or be calibrated empirically per dataset via
the `source_ranges` config option.
"""

TOKEN_SOURCES: frozenset[str] = frozenset(
    {
        "code_tokens",
        "inputs_tokens",
        "output_tokens",
    }
)
"""Difficulty sources that require a tokenizer (sample-intrinsic context measures).

Note: `output_tokens` refers to the expected output from sample data, not the model's prediction.
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
"""Context-related difficulty sources (valid even with has_code_override).

Note: `output_*` sources refer to the expected output from sample data, not the model's prediction.
"""

_DEFAULT_LOG_BIN_MAX_SCORE = math.log1p(10_000)
"""Default bin_max_score for log normalization: log1p(10_000) is approximately 9.21."""


def _percentile_index(n: int, percentile: float) -> int:
    """Get 0-indexed position for percentile using nearest-rank method.

    Args:
        n: Number of sorted values.
        percentile: Percentile value (0-100).

    Returns:
        Index into sorted array for the given percentile.
    """
    if n <= 1:
        return 0
    # nearest-rank method: rank = ceil(p/100 * n), index = rank - 1
    rank = math.ceil(percentile / 100.0 * n)
    return max(0, min(rank - 1, n - 1))


class DifficultyEstimator:
    """Computes difficulty metrics from sample metadata.

    Difficulty metrics are purely diagnostic; they do not modify rewards but are logged alongside
    them to analyze performance vs task difficulty.

    See the module docstring for the conceptual framing of difficulty axes (reasoning depth vs
    computational burden).

    Binning Strategy:
        Bins are always defined in difficulty score-space (after normalization), not raw-space. For
        fixed_range normalization: we use uniform edges in [0, 1]. For log normalization: we use
        uniform edges in [0, bin_max_score] + overflow bin. This ensures bins are interpretable
        and stable across a run.
    """

    def __init__(
        self,
        config: "reward_configs.DifficultyConfig",
        token_counter: typing.Callable[[str], int] | None = None,
    ) -> None:
        """Initialize the difficulty estimator.

        Args:
            config: Difficulty configuration.
            token_counter: Optional function to count tokens in a string.

        Raises:
            ValueError: If token sources are used but no token_counter is provided.
        """
        self._config = config
        self._token_counter = token_counter
        # validate token sources have tokenizer
        all_sources = {config.primary_source} | set(config.secondary_sources)
        if all_sources & TOKEN_SOURCES and token_counter is None:
            raise ValueError(
                f"difficulty sources {all_sources & TOKEN_SOURCES} require a token_counter; "
                "ensure RewardManager has a tokenizer configured"
            )
        # phase tracking for phase-aware config options
        self._is_eval_phase = False
        # compute bin edges based on normalization mode
        self._bin_edges = self._compute_bin_edges()
        # tracking stats for run-level aggregation
        self._score_stats = stats_utils.RunningStats()
        # track values for percentile computation (populated when track_percentiles is active)
        self._score_values: list[float] = []
        # counters for missing/invalid samples
        self._missing_count = 0
        self._override_skip_count = 0
        self._total_count = 0
        # per-bin reward stats
        self._bin_reward_stats: list[stats_utils.RunningStats] = [
            stats_utils.RunningStats() for _ in range(len(self._bin_edges) - 1)
        ]
        # per-term bin stats (lazy-initialized when first term is seen)
        self._bin_term_reward_stats: dict[str, list[stats_utils.RunningStats]] = {}
        # for correlation/slope computation (running sums, not stored pairs; more memory efficient)
        self._corr_stats = stats_utils.RunningCorrStats()
        # per-bin reward values for quantile computation (when track_bin_quantiles is active)
        self._bin_reward_values: list[list[float]] = [[] for _ in range(len(self._bin_edges) - 1)]
        # secondary source stats for downstream two-axis analysis
        self._secondary_stats: dict[str, stats_utils.RunningStats] = {}
        self._secondary_corr_stats: dict[str, stats_utils.RunningCorrStats] = {}

    def set_key_prefix(self, key_prefix: str) -> None:
        """Set the key prefix (used to determine phase for tracking options).

        This aligns with the logger's key prefix convention. The estimator determines
        whether it's in an "eval" phase by checking if "eval" appears in the prefix.

        Args:
            key_prefix: Key prefix (e.g., "train/", "eval/"). Tracking options configured
                with "eval_only" mode will only be active when "eval" is in the prefix.
        """
        self._is_eval_phase = "eval" in key_prefix.lower()

    def _is_mode_active(
        self,
        mode: str,
        generation_count: int | None = None,
    ) -> bool:
        """Check if a phase-aware mode setting is currently active.

        Args:
            mode: The mode setting ("disabled", "eval_only", "sampled", "always").
            generation_count: Current generation count (required for "sampled" mode).

        Returns:
            True if the mode is active for the current phase/generation.
        """
        if mode == "disabled":
            return False
        if mode == "always":
            return True
        if mode == "eval_only":
            return self._is_eval_phase
        if mode == "sampled":
            if generation_count is None:
                return False  # can't determine without generation count
            return generation_count % self._config.sample_every_n_generations == 0
        return False

    def _should_track_percentiles(self, generation_count: int | None = None) -> bool:
        """Check if percentile tracking is active for the current phase/generation."""
        return self._is_mode_active(self._config.track_percentiles, generation_count)

    def _should_track_per_term_rewards(self, generation_count: int | None = None) -> bool:
        """Check if per-term reward tracking is active for the current phase/generation."""
        return self._is_mode_active(self._config.track_per_term_rewards, generation_count)

    def _should_track_bin_values(self, generation_count: int | None = None) -> bool:
        """Check if bin value tracking is active (for quantiles)."""
        return self._is_mode_active(self._config.track_bin_quantiles, generation_count)

    def _compute_bin_edges(self) -> list[float]:
        """Compute bin edges based on normalization mode and config.

        Note:
            This method assumes config validation has already run (via DifficultyConfig's
            pydantic validator). It does not re-validate constraints like bin_edges sorting,
            normalization_mode compatibility, or source_ranges availability.

        Returns:
            List of bin edges (length = num_bins + 1 for standard bins, or num_bins + 2 if
            overflow bin is used).
        """
        config = self._config
        num_bins = config.num_difficulty_bins
        # if manual edges provided, use them
        if config.bin_edges is not None:
            return list(config.bin_edges)
        if config.normalization_mode == "fixed_range":
            # uniform edges in [0, 1] (no overflow needed since score is clamped)
            return [i / num_bins for i in range(num_bins + 1)]
        # log normalization: uniform edges in [0, bin_max_score]
        bin_max = config.bin_max_score
        if bin_max is None:
            # use score_clip_max if set, otherwise default
            bin_max = config.score_clip_max if config.score_clip_max is not None else _DEFAULT_LOG_BIN_MAX_SCORE
        edges = [i * bin_max / num_bins for i in range(num_bins + 1)]
        # add overflow bin only if scores can exceed the last bin edge
        # (i.e., no clipping, OR clipping is above the last edge)
        needs_overflow = config.score_clip_max is None or config.score_clip_max > bin_max
        if needs_overflow:
            edges.append(float("inf"))
        return edges

    def _get_bin_index(
        self,
        score: float,
    ) -> int:
        """Get bin index for a difficulty score."""
        for bin_idx in range(len(self._bin_edges) - 1):
            if score < self._bin_edges[bin_idx + 1]:
                return bin_idx
        return len(self._bin_edges) - 2  # last bin (overflow if present)

    def _track_secondary_value(
        self,
        source: str,
        raw_value: float,
        reward_total: float,
    ) -> None:
        """Track secondary source value for correlation analysis."""
        if source not in self._secondary_stats:
            self._secondary_stats[source] = stats_utils.RunningStats()
            self._secondary_corr_stats[source] = stats_utils.RunningCorrStats()
        self._secondary_stats[source].update(raw_value)
        self._secondary_corr_stats[source].update(raw_value, reward_total)

    def reset(self) -> None:
        """Reset estimator state for a new run."""
        self._score_stats = stats_utils.RunningStats()
        self._score_values = []
        self._missing_count = 0
        self._override_skip_count = 0
        self._total_count = 0
        self._bin_reward_stats = [stats_utils.RunningStats() for _ in range(len(self._bin_edges) - 1)]
        self._bin_term_reward_stats = {}
        self._corr_stats = stats_utils.RunningCorrStats()
        self._bin_reward_values = [[] for _ in range(len(self._bin_edges) - 1)]
        self._secondary_stats = {}
        self._secondary_corr_stats = {}

    def _get_effective_trace_step_count(
        self,
        sample_data: pyine.organisms.datamodules.samples.SampleData,
    ) -> float | None:
        """Get effective trace step count, handling code overrides."""
        if not sample_data.has_code_override:
            return float(sample_data.trace_step_count)
        mode = self._config.code_override_mode
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
        self,
        sample_data: pyine.organisms.datamodules.samples.SampleData,
        source: str,
    ) -> float | None:
        """Extract raw difficulty value from sample data.

        Args:
            sample_data: Sample data containing trace/complexity info.
            source: Source key - special keys (trace_step_count, code_length, inputs_length,
                    output_length, segment_span, code_tokens, inputs_tokens, output_tokens)
                    or any key in complexity_metrics.

        Returns:
            Raw numeric difficulty value, or None if not available.
        """
        # execution/structural sources
        if source == "trace_step_count":
            return self._get_effective_trace_step_count(sample_data)
        if source == "code_length":
            return float(len(sample_data.code))
        if source == "inputs_length":
            return float(len(sample_data.inputs))
        if source == "output_length":  # expected output from sample data, not model prediction
            return float(len(sample_data.expected_output))
        if source == "segment_span":
            span = sample_data.last_line - sample_data.first_line
            return float(span) if span > 0 else None
        # token-based context difficulty sources (sample-intrinsic measures)
        if source == "code_tokens":
            assert self._token_counter is not None, "token_counter required for code_tokens source"
            return float(self._token_counter(sample_data.code))
        if source == "inputs_tokens":
            assert self._token_counter is not None, "token_counter required for inputs_tokens source"
            return float(self._token_counter(sample_data.inputs))
        if source == "output_tokens":  # expected output from sample data, not model prediction
            assert self._token_counter is not None, "token_counter required for output_tokens source"
            return float(self._token_counter(sample_data.expected_output))
        # complexity metrics
        complexity = sample_data.complexity_metrics
        if not complexity or source not in complexity:
            return None
        if sample_data.has_code_override and self._config.code_override_mode == CodeOverrideMode.skip:
            return None
        return float(complexity[source])

    def normalize(
        self,
        raw_value: float,
        source: str,
    ) -> float:
        """Apply normalization to a raw difficulty value.

        Returns:
            Normalized score. Range depends on normalization_mode:
            - "none": unbounded (raw value);
            - "log": unbounded (log1p), typically 0-7 for common ranges;
            - "fixed_range": [0, 1] after clamp/invert.
        """
        if self._config.normalization_mode == "none":
            score = raw_value
        elif self._config.normalization_mode == "log":
            score = math.log1p(max(0.0, raw_value))
        elif self._config.normalization_mode == "fixed_range":
            source_ranges = {**DEFAULT_SOURCE_RANGES, **(self._config.source_ranges or {})}
            assert source in source_ranges, (
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
        if self._config.score_clip_max is not None:
            score = min(score, self._config.score_clip_max)
        return score

    def compute(
        self,
        sample_data: pyine.organisms.datamodules.samples.SampleData,
        reward_total: float,
        weighted_terms: dict[str, float] | None = None,
        generation_count: int | None = None,
    ) -> dict[str, reward_types.MetricValue]:
        """Compute difficulty metrics for a sample.

        Args:
            sample_data: Sample data containing trace/complexity info.
            reward_total: Total reward value for this sample.
            weighted_terms: Optional per-term reward values (for per-term binning when enabled).
            generation_count: Optional monotonic generation count (required for "sampled" mode tracking).

        Returns:
            Dictionary of difficulty metrics to merge into RewardOutput.metrics.
        """
        self._total_count += 1
        metrics: dict[str, reward_types.MetricValue] = {
            "difficulty/source": self._config.primary_source,
            "difficulty/execution_skipped": 0,
            "difficulty/primary_missing": 0,
            "difficulty/has_code_override": int(sample_data.has_code_override),
        }
        # determine if primary source is an execution source (affected by code override)
        # token sources (sample_code_tokens, inputs_tokens, etc.) are context, not execution
        primary_is_execution = (
            self._config.primary_source in EXECUTION_SOURCES
            or (
                self._config.primary_source not in CONTEXT_SOURCES and self._config.primary_source not in TOKEN_SOURCES
            )  # complexity_metrics are execution
        )
        # check code override handling; skip execution metrics but still compute context metrics
        skip_execution = sample_data.has_code_override and self._config.code_override_mode == CodeOverrideMode.skip
        if skip_execution:
            self._override_skip_count += 1
            metrics["difficulty/execution_skipped"] = 1
            if primary_is_execution:
                metrics["difficulty/primary_missing"] = 1
                # still log secondary context sources even when primary is skipped
                for secondary_source in self._config.secondary_sources:
                    if secondary_source in CONTEXT_SOURCES or secondary_source in TOKEN_SOURCES:
                        raw_secondary = self.get_raw_value(sample_data, secondary_source)
                        if raw_secondary is not None:
                            metrics[f"difficulty/raw/{secondary_source}"] = raw_secondary
                return metrics
        raw_primary = self.get_raw_value(sample_data, self._config.primary_source)
        if raw_primary is None:
            self._missing_count += 1
            metrics["difficulty/primary_missing"] = 1
            return metrics
        difficulty_score = self.normalize(raw_primary, self._config.primary_source)
        metrics["difficulty/raw_primary"] = raw_primary
        metrics["difficulty/score"] = difficulty_score
        self._score_stats.update(difficulty_score)
        if self._should_track_percentiles(generation_count):
            self._score_values.append(difficulty_score)
        bin_idx = self._get_bin_index(difficulty_score)
        self._bin_reward_stats[bin_idx].update(reward_total)
        metrics["difficulty/bin_index"] = bin_idx
        self._corr_stats.update(difficulty_score, reward_total)
        if self._should_track_bin_values(generation_count):
            self._bin_reward_values[bin_idx].append(reward_total)
        if self._should_track_per_term_rewards(generation_count) and weighted_terms:
            for term_name, term_value in weighted_terms.items():
                if term_name not in self._bin_term_reward_stats:
                    self._bin_term_reward_stats[term_name] = [
                        stats_utils.RunningStats() for _ in range(len(self._bin_edges) - 1)
                    ]
                self._bin_term_reward_stats[term_name][bin_idx].update(term_value)
        for secondary_source in self._config.secondary_sources:
            raw_secondary = self.get_raw_value(sample_data, secondary_source)
            if raw_secondary is not None:
                metrics[f"difficulty/raw/{secondary_source}"] = raw_secondary
                self._track_secondary_value(secondary_source, raw_secondary, reward_total)
        return metrics

    def get_run_summaries(self) -> dict[str, reward_types.MetricValue]:
        """Get aggregated difficulty metrics for run-level logging.

        Returns dict with difficulty score distribution stats, quality flags, bin edges, and per-bin
        reward stats.
        """
        if self._total_count == 0:
            return {}
        metrics: dict[str, reward_types.MetricValue] = {}
        metrics["missing_ratio"] = self._missing_count / self._total_count
        metrics["override_skip_ratio"] = self._override_skip_count / self._total_count
        if self._score_stats.count > 0:
            assert self._score_stats.min is not None and self._score_stats.max is not None
            metrics["mean"] = self._score_stats.mean()
            metrics["std"] = self._score_stats.std()
            metrics["min"] = float(self._score_stats.min)
            metrics["max"] = float(self._score_stats.max)
            metrics["count"] = self._score_stats.count
        else:
            metrics["count"] = 0
        # percentiles (only if values were tracked during any phase)
        if self._score_values:
            sorted_values = sorted(self._score_values)
            n = len(sorted_values)
            metrics["median"] = sorted_values[_percentile_index(n, 50)]
            metrics["p90"] = sorted_values[_percentile_index(n, 90)]
            metrics["p99"] = sorted_values[_percentile_index(n, 99)]
            # recommended percentile-based bin edges for calibrating future runs;
            # these edges divide the observed distribution into equal-mass bins
            for edge_idx in range(self._config.num_difficulty_bins + 1):
                pct = int(edge_idx * 100 / self._config.num_difficulty_bins)
                metrics[f"recommended_edge_{edge_idx}"] = sorted_values[_percentile_index(n, pct)]
        if self._score_stats.count > 0:
            # correlation and slope between difficulty and reward (catches trends bins can hide)
            metrics.update(self._corr_stats.to_metrics(prefix="reward"))
            # secondary source stats and correlations
            for source, corr_stats in self._secondary_corr_stats.items():
                metrics.update(corr_stats.to_metrics(prefix=f"secondary/{source}/reward"))
                if source in self._secondary_stats:
                    metrics.update(self._secondary_stats[source].to_metrics(prefix=f"secondary/{source}"))
        # bin edges (critical for interpretability)
        # convention: inf edges are logged as -1.0 (sentinel value)
        # the has_overflow_bin flag indicates whether the last bin is an overflow bin
        has_overflow = self._bin_edges[-1] == float("inf")
        metrics["num_bins"] = len(self._bin_edges) - 1
        metrics["has_overflow_bin"] = int(has_overflow)
        for edge_idx, edge in enumerate(self._bin_edges):
            if edge == float("inf"):
                metrics[f"bin_edge_{edge_idx}"] = -1.0  # sentinel: -1.0 means infinity
            else:
                metrics[f"bin_edge_{edge_idx}"] = edge
        # per-bin reward stats
        for bin_idx, bin_stats in enumerate(self._bin_reward_stats):
            if bin_stats.count > 0:
                assert bin_stats.min is not None and bin_stats.max is not None
                prefix = f"bin_{bin_idx}"
                metrics[f"{prefix}/count"] = bin_stats.count
                metrics[f"{prefix}/reward_mean"] = bin_stats.mean()
                metrics[f"{prefix}/reward_std"] = bin_stats.std()
                metrics[f"{prefix}/reward_min"] = float(bin_stats.min)
                metrics[f"{prefix}/reward_max"] = float(bin_stats.max)
                # per-bin quantiles (if values were tracked during any phase)
                if self._bin_reward_values:
                    bin_vals = sorted(self._bin_reward_values[bin_idx])
                    if bin_vals:
                        bin_n = len(bin_vals)
                        metrics[f"{prefix}/reward_p10"] = bin_vals[_percentile_index(bin_n, 10)]
                        metrics[f"{prefix}/reward_p50"] = bin_vals[_percentile_index(bin_n, 50)]
                        metrics[f"{prefix}/reward_p90"] = bin_vals[_percentile_index(bin_n, 90)]
        # per-term per-bin reward stats (if enabled)
        for term_name, term_bin_stats in self._bin_term_reward_stats.items():
            for bin_idx, bin_stats in enumerate(term_bin_stats):
                if bin_stats.count > 0:
                    assert bin_stats.min is not None and bin_stats.max is not None
                    prefix = f"bin_{bin_idx}/term_{term_name}"
                    metrics[f"{prefix}/reward_mean"] = bin_stats.mean()
                    metrics[f"{prefix}/reward_std"] = bin_stats.std()
        return metrics

    def merge(
        self,
        other: "DifficultyEstimator",
    ) -> None:
        """Merge another estimator's state into this one (for distributed gathering).

        Raises:
            ValueError: If bin counts don't match (config mismatch).
        """
        # validate bin count match before merging
        expected_bins = len(self._bin_edges) - 1
        if len(other._bin_reward_stats) != expected_bins:
            raise ValueError(
                f"cannot merge difficulty estimators with different bin counts: "
                f"expected {expected_bins}, got {len(other._bin_reward_stats)}"
            )
        self._score_stats.merge(other._score_stats)
        self._score_values.extend(other._score_values)
        self._missing_count += other._missing_count
        self._override_skip_count += other._override_skip_count
        self._total_count += other._total_count
        for bin_idx in range(len(self._bin_reward_stats)):
            self._bin_reward_stats[bin_idx].merge(other._bin_reward_stats[bin_idx])
        for term_name, other_stats in other._bin_term_reward_stats.items():
            if term_name not in self._bin_term_reward_stats:
                self._bin_term_reward_stats[term_name] = [
                    stats_utils.RunningStats() for _ in range(len(self._bin_edges) - 1)
                ]
            for bin_idx, other_bin_stats in enumerate(other_stats):
                self._bin_term_reward_stats[term_name][bin_idx].merge(other_bin_stats)
        self._corr_stats.merge(other._corr_stats)
        assert len(self._bin_reward_values) == len(other._bin_reward_values) == expected_bins
        for bin_idx in range(expected_bins):
            self._bin_reward_values[bin_idx].extend(other._bin_reward_values[bin_idx])
        for source, other_stats in other._secondary_stats.items():
            if source not in self._secondary_stats:
                self._secondary_stats[source] = stats_utils.RunningStats()
            self._secondary_stats[source].merge(other_stats)
        for source, other_corr in other._secondary_corr_stats.items():
            if source not in self._secondary_corr_stats:
                self._secondary_corr_stats[source] = stats_utils.RunningCorrStats()
            self._secondary_corr_stats[source].merge(other_corr)

    def as_state(self) -> dict[str, typing.Any]:
        """Serialize estimator state for checkpointing."""
        return {
            "score_stats": self._score_stats.as_state(),
            "score_values": self._score_values,
            "missing_count": self._missing_count,
            "override_skip_count": self._override_skip_count,
            "total_count": self._total_count,
            "bin_reward_stats": [s.as_state() for s in self._bin_reward_stats],
            "bin_term_reward_stats": {
                term: [s.as_state() for s in stats] for term, stats in self._bin_term_reward_stats.items()
            },
            "corr_stats": self._corr_stats.as_state(),
            "bin_reward_values": self._bin_reward_values,
            "secondary_stats": {k: v.as_state() for k, v in self._secondary_stats.items()},
            "secondary_corr_stats": {k: v.as_state() for k, v in self._secondary_corr_stats.items()},
        }

    @classmethod
    def from_state(
        cls,
        config: "reward_configs.DifficultyConfig",
        state: dict[str, typing.Any],
        token_counter: typing.Callable[[str], int] | None = None,
    ) -> "DifficultyEstimator":
        """Deserialize estimator from checkpoint state.

        Raises:
            ValueError: If state has mismatched bin counts (config changed since checkpoint).
        """
        estimator = cls(config, token_counter=token_counter)
        expected_bins = len(estimator._bin_edges) - 1
        if "bin_reward_stats" in state and len(state["bin_reward_stats"]) != expected_bins:
            raise ValueError(
                f"cannot restore difficulty estimator: state has {len(state['bin_reward_stats'])} bins, "
                f"but config defines {expected_bins} bins (check num_difficulty_bins or bin_edges)"
            )
        if "bin_reward_values" in state and state["bin_reward_values"]:
            if len(state["bin_reward_values"]) != expected_bins:
                raise ValueError(
                    f"cannot restore difficulty estimator: state has {len(state['bin_reward_values'])} "
                    f"bin_reward_values, but config defines {expected_bins} bins"
                )
        if "score_stats" in state:
            estimator._score_stats = stats_utils.RunningStats.from_state(state["score_stats"])
        # only restore score_values if config enables percentile tracking (drop otherwise to save memory)
        if "score_values" in state and config.track_percentiles != "disabled":
            estimator._score_values = state["score_values"]
        estimator._missing_count = state.get("missing_count", 0)
        estimator._override_skip_count = state.get("override_skip_count", 0)
        estimator._total_count = state.get("total_count", 0)
        if "bin_reward_stats" in state:
            estimator._bin_reward_stats = [stats_utils.RunningStats.from_state(s) for s in state["bin_reward_stats"]]
        # only restore bin_term_reward_stats if config enables per-term tracking (drop otherwise to save memory)
        if "bin_term_reward_stats" in state and config.track_per_term_rewards != "disabled":
            for term, stats in state["bin_term_reward_stats"].items():
                if len(stats) != expected_bins:
                    raise ValueError(
                        f"cannot restore difficulty estimator: term '{term}' has {len(stats)} bins, "
                        f"but config defines {expected_bins} bins"
                    )
            estimator._bin_term_reward_stats = {
                term: [stats_utils.RunningStats.from_state(s) for s in stats]
                for term, stats in state["bin_term_reward_stats"].items()
            }
        if "corr_stats" in state:
            estimator._corr_stats = stats_utils.RunningCorrStats.from_state(state["corr_stats"])
        # only restore bin_reward_values if config enables bin quantile tracking (drop otherwise to save memory)
        if "bin_reward_values" in state and config.track_bin_quantiles != "disabled":
            assert len(state["bin_reward_values"]) == expected_bins
            estimator._bin_reward_values = state["bin_reward_values"]
        if "secondary_stats" in state:
            estimator._secondary_stats = {
                k: stats_utils.RunningStats.from_state(v) for k, v in state["secondary_stats"].items()
            }
        if "secondary_corr_stats" in state:
            estimator._secondary_corr_stats = {
                k: stats_utils.RunningCorrStats.from_state(v) for k, v in state["secondary_corr_stats"].items()
            }
        return estimator
