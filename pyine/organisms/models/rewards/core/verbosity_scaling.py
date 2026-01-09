"""Verbosity-based reward scaling implementation.

This module provides a VerbosityScaler class that applies multiplicative scaling to aggregated
rewards based on output length. The scaler supports two modes:

- Absolute mode: Factor decays from max_factor toward min_factor based on token thresholds
- Relative mode: Factor computed via rescaled sigmoid where mean -> ~max_factor, above-mean -> <max_factor

The verbosity factor is normally in [min_factor, max_factor] and is applied as:
    final_reward = aggregated_reward * verbosity_factor

Important:
    In relative mode, samples at or below the group mean get factor ~max_factor (not necessarily 1.0).
    If max_factor < 1.0, even "non-verbose" samples will receive some penalty. Set max_factor=1.0
    if you want no penalty for samples at or below the mean.

Skip behavior in relative mode:
    When relative mode cannot compute meaningful scaling (group size < 2, or std = 0), the factor
    is set to 1.0 regardless of max_factor. This is a safe fallback that avoids penalizing samples
    when relative comparison is impossible. A warning is emitted once per scaler instance.

Note on negative rewards:
    Multiplication by factor < 1 moves negative rewards toward 0 (i.e., increases them).
    If this is undesired, ensure rewards are non-negative before scaling.
"""

import collections.abc
import math
import statistics
import warnings

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.types as reward_types


def _stable_sigmoid(x: float) -> float:
    """Numerically stable sigmoid using tanh.

    This formulation avoids overflow for large positive x values.
    """
    return 0.5 * (1.0 + math.tanh(x / 2.0))


class VerbosityScaler:
    """Applies multiplicative verbosity scaling to aggregated rewards.

    Supports two modes:
    - absolute: Factor decays from max_factor toward min_factor based on token thresholds
    - relative: Factor computed via rescaled sigmoid where mean -> ~max_factor, above-mean -> <max_factor

    Important:
        In relative mode, samples at or below the group mean get factor ~max_factor (not necessarily 1.0).
        If max_factor < 1.0, even "non-verbose" samples will receive some penalty. Set max_factor=1.0
        if you want no penalty for samples at or below the mean.

    Skip behavior in relative mode:
        When relative mode cannot compute meaningful scaling (group size < 2, or std = 0), the factor
        is set to 1.0 regardless of max_factor. This is a safe fallback that avoids penalizing samples
        when relative comparison is impossible. A warning is emitted once per scaler instance.

    Note on negative rewards:
        Multiplication by factor < 1 moves negative rewards toward 0 (i.e., increases them).
        If this is undesired, ensure rewards are non-negative before scaling.
    """

    def __init__(
        self,
        config: reward_configs.VerbosityScalingConfig,
    ) -> None:
        """Create scaler from config.

        Args:
            config: Verbosity scaling configuration.

        Note:
            VerbosityScaler relies on token counts. When verbosity scaling is enabled,
            the RewardManager automatically enables token counting (regardless of
            parsing.track_token_lengths).
        """
        self._config = config
        self._warned_std_zero = False

    @property
    def is_relative_mode(self) -> bool:
        """Whether this scaler uses relative (group-aware) mode."""
        return self._config.mode == "relative"

    def get_token_count(
        self,
        cache: reward_types.TokenCountCache,
    ) -> int | None:
        """Get token count for configured source from cache.

        Args:
            cache: Token count cache populated by RewardManager.

        Returns:
            Token count for the configured source, or None if the source is not in cache.

        Note:
            Token counts for model_output are always available. Token counts for
            parsed_reasoning and parsed_final_answer require parsing to be enabled
            and the respective fields to be non-None in the parsed output.
        """
        return cache.get(self._config.length_source)

    def compute_absolute_factor(
        self,
        token_count: int,
    ) -> float:
        """Compute factor using absolute thresholds.

        Returns max_factor below threshold, decays toward min_factor above.

        Args:
            token_count: Number of tokens in the output.

        Returns:
            Verbosity factor in [min_factor, max_factor].
        """
        max_factor = float(self._config.max_factor)
        min_factor = float(self._config.min_factor)
        threshold = int(self._config.threshold_tokens)
        if token_count <= threshold:
            return max_factor
        excess = token_count - threshold
        if self._config.decay_type == "linear":
            end_tokens = int(self._config.end_tokens)
            span = max(1, end_tokens - threshold)
            ratio = min(1.0, float(excess) / float(span))
            raw_factor = max_factor - (max_factor - min_factor) * ratio
        else:  # exponential
            raw_factor = max_factor * math.exp(-float(self._config.decay_rate) * float(excess))
        return max(min_factor, min(max_factor, raw_factor))

    def compute_relative_factor(
        self,
        token_count: int,
        group_mean: float,
        group_std: float,
    ) -> float:
        """Compute factor using group-relative rescaled sigmoid.

        Returns ~max_factor at or below mean, decays toward min_factor above mean.

        The formula is designed so that:
        - z_score <= 0 (at or below mean) -> factor ~ max_factor
        - z_score > 0 (above mean) -> factor in (min_factor, max_factor)

        Note:
            If max_factor < 1.0, samples at the mean will still receive a penalty.
            Set max_factor=1.0 if you want no penalty for at-or-below-mean samples.

        Args:
            token_count: Number of tokens for this sample.
            group_mean: Mean token count across the group.
            group_std: Standard deviation of token counts across the group.

        Returns:
            Verbosity factor in [min_factor, max_factor].
        """
        max_factor = float(self._config.max_factor)
        min_factor = float(self._config.min_factor)
        temperature = float(self._config.temperature)
        z_score = (float(token_count) - group_mean) / group_std
        raw_sigmoid = _stable_sigmoid(-z_score / temperature)
        raw_factor = min(1.0, 2.0 * raw_sigmoid)  # rescale so z<=0 -> ~1.0
        factor = min_factor + (max_factor - min_factor) * raw_factor
        return max(min_factor, min(max_factor, factor))

    def apply_absolute(
        self,
        aggregated_reward: float,
        weighted_terms: dict[str, float],
        cache: reward_types.TokenCountCache,
    ) -> tuple[float, dict[str, float], dict[str, reward_types.MetricValue]]:
        """Apply absolute scaling to a single sample.

        Args:
            aggregated_reward: Pre-scaling total reward.
            weighted_terms: Pre-scaling weighted term breakdown.
            cache: Token count cache from parsing stats.

        Returns:
            Tuple of (scaled_total, scaled_weighted_terms, metrics).

        Raises:
            ValueError: If the configured length source is not available in the cache.
        """
        metrics: dict[str, reward_types.MetricValue] = {}
        token_count = self.get_token_count(cache)
        if token_count is None:
            raise ValueError(
                f"verbosity_scaling requires token count for source={self._config.length_source!r} "
                "but it is not available in cache. For parsed_reasoning or parsed_final_answer, "
                "ensure parsing is enabled and the respective field is non-None in the parsed output."
            )
        # skip scaling for negative rewards if configured
        if self._config.skip_negative_rewards and aggregated_reward < 0:
            factor = 1.0
            skipped = True
        else:
            factor = self.compute_absolute_factor(token_count)
            skipped = False
        scaled_total = aggregated_reward * factor
        scaled_terms = {name: val * factor for name, val in weighted_terms.items()}
        if self._config.emit_metrics:
            metrics["verbosity/token_count"] = token_count
            metrics["verbosity/factor"] = factor
            metrics["verbosity/pre_scaling_reward"] = aggregated_reward
            metrics["verbosity/post_scaling_reward"] = scaled_total
            if self._config.skip_negative_rewards:
                metrics["verbosity/skipped_negative"] = skipped
        return scaled_total, scaled_terms, metrics

    def apply_to_group(
        self,
        aggregated_rewards: collections.abc.Sequence[float],
        all_weighted_terms: collections.abc.Sequence[dict[str, float]],
        caches: collections.abc.Sequence[reward_types.TokenCountCache],
    ) -> tuple[list[float], list[dict[str, float]], list[dict[str, reward_types.MetricValue]]]:
        """Apply relative scaling to a group of samples sharing the same prompt.

        Args:
            aggregated_rewards: Pre-scaling total rewards for each sample.
            all_weighted_terms: Pre-scaling weighted term breakdowns for each sample.
            caches: Token count caches from parsing stats for each sample.

        Returns:
            Tuple of (scaled_totals, scaled_weighted_terms, metrics_per_sample).

        Raises:
            ValueError: If any sample is missing the required token count in its cache.
        """
        source = self._config.length_source
        token_counts: list[int] = []
        for sample_idx, cache in enumerate(caches):
            count = self.get_token_count(cache)
            if count is None:
                raise ValueError(
                    f"verbosity_scaling requires token count for source={source!r} "
                    f"but sample {sample_idx} is missing it. For parsed_reasoning or parsed_final_answer, "
                    "ensure parsing is enabled and the respective field is non-None."
                )
            token_counts.append(count)
        # check for std=0 condition and warn
        if len(token_counts) < 2:
            if not self._warned_std_zero:
                warnings.warn(
                    "verbosity_scaling mode='relative' received group with < 2 samples; "
                    "skipping scaling (factor=1.0). This is common in eval (single generation per prompt).",
                    stacklevel=2,  # caller -> this method
                )
                self._warned_std_zero = True
            factors = [1.0] * len(caches)
            group_mean = float(token_counts[0]) if token_counts else 0.0
            group_std = 0.0
        else:
            group_mean = statistics.mean(token_counts)
            group_std = statistics.stdev(token_counts)
            if group_std == 0:
                if not self._warned_std_zero:
                    warnings.warn(
                        "verbosity_scaling mode='relative' received group with std=0 "
                        "(all identical lengths); skipping scaling (factor=1.0)",
                        stacklevel=2,  # caller -> this method
                    )
                    self._warned_std_zero = True
                factors = [1.0] * len(caches)
            else:
                factors: list[float] = []
                for count in token_counts:
                    factors.append(self.compute_relative_factor(count, group_mean, group_std))
        # apply scaling
        scaled_totals: list[float] = []
        scaled_terms_list: list[dict[str, float]] = []
        all_metrics: list[dict[str, reward_types.MetricValue]] = []
        for reward, terms, factor, count in zip(
            aggregated_rewards, all_weighted_terms, factors, token_counts, strict=True
        ):
            # skip scaling for negative rewards if configured
            if self._config.skip_negative_rewards and reward < 0:
                effective_factor = 1.0
                skipped = True
            else:
                effective_factor = factor
                skipped = False
            scaled_total = reward * effective_factor
            scaled_terms = {name: val * effective_factor for name, val in terms.items()}
            scaled_totals.append(scaled_total)
            scaled_terms_list.append(scaled_terms)
            metrics: dict[str, reward_types.MetricValue] = {}
            if self._config.emit_metrics:
                metrics["verbosity/token_count"] = count
                metrics["verbosity/group_mean"] = group_mean
                metrics["verbosity/group_std"] = group_std
                if group_std > 0:
                    metrics["verbosity/z_score"] = (float(count) - group_mean) / group_std
                metrics["verbosity/factor"] = effective_factor
                metrics["verbosity/pre_scaling_reward"] = reward
                metrics["verbosity/post_scaling_reward"] = scaled_total
                if self._config.skip_negative_rewards:
                    metrics["verbosity/skipped_negative"] = skipped
            all_metrics.append(metrics)
        return scaled_totals, scaled_terms_list, all_metrics
