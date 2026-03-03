"""Task difficulty estimation for reward analysis.

This module provides a DifficultyEstimator class that computes difficulty metrics from SampleData
fields (trace_step_count, complexity_metrics, string lengths, etc.). Difficulty metrics are purely
diagnostic, i.e. they do not modify rewards.

For more information on strategies to measure 'task difficulty' across computational depth and
burden axes, see the `pyine.utils.code.difficulty` module.

Important:
    When SampleData.has_code_override=True, the trace_step_count and complexity_metrics
    are from the original traced code, not the overridden code. The estimator handles
    this via the configured `code_override_mode`.
"""

import typing

import pyine.organisms.datamodules.samples
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.utils.code.difficulty as difficulty_utils
import pyine.utils.stats as stats_utils


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
        config: difficulty_utils.DifficultyConfig,
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
        self._scorer = difficulty_utils.DifficultyScorer(
            config,
            token_counter=token_counter,
        )
        # phase tracking for phase-aware config options
        self._is_eval_phase = False
        # compute bin edges based on normalization mode
        self._bin_edges = self._scorer.bin_edges
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
        mode: typing.Literal["disabled", "eval_only", "always"],
    ) -> bool:
        """Check if a phase-aware mode setting is currently active.

        Args:
            mode: The mode setting ("disabled", "eval_only", "always").

        Returns:
            True if the mode is active for the current phase.
        """
        if mode == "disabled":
            return False
        if mode == "always":
            return True
        if mode == "eval_only":
            return self._is_eval_phase
        return False

    def _should_track_percentiles(self) -> bool:
        """Check if percentile tracking is active for the current phase."""
        return self._is_mode_active(self._config.track_percentiles)

    def _should_track_per_term_rewards(self) -> bool:
        """Check if per-term reward tracking is active for the current phase."""
        return self._is_mode_active(self._config.track_per_term_rewards)

    def _should_track_bin_values(self) -> bool:
        """Check if bin value tracking is active (for quantiles)."""
        return self._is_mode_active(self._config.track_bin_quantiles)

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

    def get_raw_value(
        self,
        sample_data: pyine.organisms.datamodules.samples.SampleData,
        source: str,
    ) -> float | None:
        """Return raw difficulty value for a given source."""
        return difficulty_utils.get_raw_value(
            sample_data,
            source,
            config=self._config,
            token_counter=self._token_counter,
        )

    def normalize(
        self,
        raw_value: float,
        source: str,
    ) -> float:
        """Normalize a raw difficulty value based on the estimator config."""
        return difficulty_utils.normalize_value(raw_value, source, self._config)

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

    def compute(
        self,
        sample_data: pyine.organisms.datamodules.samples.SampleData,
        reward_total: float,
        weighted_terms: dict[str, float] | None = None,
    ) -> dict[str, reward_types.MetricValue]:
        """Compute difficulty metrics for a sample.

        Args:
            sample_data: Sample data containing trace/complexity info.
            reward_total: Total reward value for this sample.
            weighted_terms: Optional per-term reward values (for per-term binning when enabled).

        Returns:
            Dictionary of difficulty metrics to merge into RewardOutput.metrics.
        """
        self._total_count += 1
        result = self._scorer.compute_result(sample_data)
        skip_execution = (
            result.has_code_override and self._config.code_override_mode == difficulty_utils.CodeOverrideMode.skip
        )
        if skip_execution:
            self._override_skip_count += 1
        metrics: dict[str, reward_types.MetricValue] = {
            "difficulty/source": self._config.primary_source,
            "difficulty/execution_skipped": int(skip_execution),
            "difficulty/primary_missing": int(result.primary_missing),
            "difficulty/has_code_override": int(result.has_code_override),
        }
        if result.primary_missing:
            if not result.execution_skipped:
                self._missing_count += 1
            for secondary_source, raw_secondary in result.secondary_raw.items():
                metrics[f"difficulty/raw/{secondary_source}"] = raw_secondary
            return metrics
        if result.score is None or result.bin_index is None:
            raise ValueError("difficulty result missing score or bin_index after primary was present")
        if result.raw_primary is None:
            raise ValueError("difficulty result missing raw_primary after primary was present")
        metrics["difficulty/raw_primary"] = result.raw_primary
        metrics["difficulty/score"] = result.score
        self._score_stats.update(result.score)
        if self._should_track_percentiles():
            self._score_values.append(result.score)
        bin_idx = result.bin_index
        self._bin_reward_stats[bin_idx].update(reward_total)
        metrics["difficulty/bin_index"] = bin_idx
        self._corr_stats.update(result.score, reward_total)
        if self._should_track_bin_values():
            self._bin_reward_values[bin_idx].append(reward_total)
        if self._should_track_per_term_rewards() and weighted_terms:
            for term_name, term_value in weighted_terms.items():
                if term_name not in self._bin_term_reward_stats:
                    self._bin_term_reward_stats[term_name] = [
                        stats_utils.RunningStats() for _ in range(len(self._bin_edges) - 1)
                    ]
                self._bin_term_reward_stats[term_name][bin_idx].update(term_value)
        for secondary_source, raw_secondary in result.secondary_raw.items():
            metrics[f"difficulty/raw/{secondary_source}"] = raw_secondary
            self._track_secondary_value(secondary_source, raw_secondary, reward_total)
        return metrics

    def get_bin_stats(self) -> list[stats_utils.RunningStats]:
        """Return per-bin reward stats (list indexed by bin_idx)."""
        return self._bin_reward_stats

    def get_bin_edges(self) -> list[float]:
        """Return bin edges."""
        return self._bin_edges

    def get_bin_term_stats(self) -> dict[str, list[stats_utils.RunningStats]]:
        """Return per-term per-bin stats."""
        return self._bin_term_reward_stats

    def get_histogram_data(self) -> dict[str, list[float] | list[list[float]]]:
        """Return raw values for histogram logging."""
        return {
            "score_values": self._score_values,
            "bin_reward_values": self._bin_reward_values,
        }

    def get_secondary_stats(self) -> dict[str, stats_utils.RunningStats]:
        """Return raw secondary source statistics for table logging."""
        return dict(self._secondary_stats)

    def get_secondary_corr_stats(self) -> dict[str, stats_utils.RunningCorrStats]:
        """Return secondary source correlation statistics for table logging."""
        return dict(self._secondary_corr_stats)

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
            metrics["score/mean"] = self._score_stats.mean()
            metrics["score/std"] = self._score_stats.std()
            metrics["score/min"] = float(self._score_stats.min)
            metrics["score/max"] = float(self._score_stats.max)
            metrics["score/count"] = self._score_stats.count
        else:
            metrics["score/count"] = 0
        # percentiles (only if values were tracked during any phase)
        if self._score_values:
            sorted_values = sorted(self._score_values)
            n = len(sorted_values)
            metrics["score/median"] = sorted_values[stats_utils.percentile_index(n, 50)]
            metrics["score/p90"] = sorted_values[stats_utils.percentile_index(n, 90)]
            metrics["score/p99"] = sorted_values[stats_utils.percentile_index(n, 99)]
            # recommended percentile-based bin edges for calibrating future runs;
            # these edges divide the observed distribution into equal-mass bins
            for edge_idx in range(self._config.num_difficulty_bins + 1):
                pct = int(edge_idx * 100 / self._config.num_difficulty_bins)
                metrics[f"score/recommended_edge_{edge_idx}"] = sorted_values[stats_utils.percentile_index(n, pct)]
        if self._score_stats.count > 0:
            # correlation and slope between difficulty and reward (catches trends bins can hide)
            metrics.update(self._corr_stats.to_metrics(prefix="reward"))
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
                        metrics[f"{prefix}/reward_p10"] = bin_vals[stats_utils.percentile_index(bin_n, 10)]
                        metrics[f"{prefix}/reward_p50"] = bin_vals[stats_utils.percentile_index(bin_n, 50)]
                        metrics[f"{prefix}/reward_p90"] = bin_vals[stats_utils.percentile_index(bin_n, 90)]
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
        config: difficulty_utils.DifficultyConfig,
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
