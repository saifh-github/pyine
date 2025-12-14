"""Aggregation strategies for term outputs.

Aggregation combines per-term scalar values into a single scalar reward per sample. The default
strategy is a weighted sum with optional clipping.
"""

import collections.abc

import pyine.organisms.models.rewards.core.configs


class WeightedSumAggregator:
    """Weighted sum aggregator with optional clipping.

    This implementation expects:
    - one scalar value per enabled term (per sample)
    - one scalar weight per term

    Clipping is applied in the following order:
    1. Per-term clipping (applied to unweighted values before weighting)
    2. Weighting (clipped_value * weight)
    3. Summation
    4. Total clipping (applied to final sum)
    """

    def __init__(
        self,
        config: pyine.organisms.models.rewards.core.configs.AggregationConfig,
    ) -> None:
        """Create an aggregator from config.

        Args:
            config: Aggregation configuration (strategy + optional clipping).
        """
        if config.strategy != "weighted_sum":
            raise ValueError(f"unsupported aggregation strategy: {config.strategy}")
        self._config = config

    def aggregate(
        self,
        *,
        values: collections.abc.Mapping[str, float],
        weights: collections.abc.Mapping[str, float],
    ) -> tuple[float, dict[str, float], dict[str, float] | None]:
        """Aggregate term values into a total reward.

        Args:
            values: Mapping of term name to unweighted term value.
            weights: Mapping of term name to scalar weight.

        Returns:
            A tuple of `(total, weighted_terms, unweighted_terms)`.
        """
        unweighted_terms: dict[str, float] = dict(values)
        weighted_terms: dict[str, float] = {}
        total = 0.0
        for term_name, value in values.items():
            clipped = self._clip_term(value)
            weight = float(weights.get(term_name, 1.0))
            weighted = weight * clipped
            weighted_terms[term_name] = weighted
            total += weighted
        total = self._clip_total(total)
        return total, weighted_terms, unweighted_terms

    def _clip_term(
        self,
        value: float,
    ) -> float:
        """Clip an individual term value if configured."""
        clipped = float(value)
        if self._config.clip_term_min is not None:
            clipped = max(clipped, float(self._config.clip_term_min))
        if self._config.clip_term_max is not None:
            clipped = min(clipped, float(self._config.clip_term_max))
        return clipped

    def _clip_total(
        self,
        total: float,
    ) -> float:
        """Clip the aggregated total if configured."""
        clipped = float(total)
        if self._config.clip_total_min is not None:
            clipped = max(clipped, float(self._config.clip_total_min))
        if self._config.clip_total_max is not None:
            clipped = min(clipped, float(self._config.clip_total_max))
        return clipped
