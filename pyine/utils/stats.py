"""Streaming statistics utilities.

This module provides small, dependency-light helpers for tracking scalar statistics in a streaming
fashion. These are useful across training, reward computation, and evaluation code paths.
"""

import collections.abc
import dataclasses
import math


@dataclasses.dataclass(slots=True)
class RunningStats:
    """Streaming numeric stats accumulator for scalar values.

    Tracks mean/std/min/max without storing all observations. Supports merging and
    serialization/deserialization to a picklable state dict (useful for distributed aggregation).
    """

    count: int = 0
    """Number of observations."""
    sum: float = 0.0
    """Sum of observed values."""
    sumsq: float = 0.0
    """Sum of squares of observed values."""
    min: float | None = None
    """Minimum observed value."""
    max: float | None = None
    """Maximum observed value."""

    def update(
        self,
        value: float,
    ) -> None:
        """Update the stats with a new observation."""
        value_float = float(value)
        self.count += 1
        self.sum += value_float
        self.sumsq += value_float * value_float
        self.min = value_float if self.min is None else min(self.min, value_float)
        self.max = value_float if self.max is None else max(self.max, value_float)

    def merge(
        self,
        other: "RunningStats",
    ) -> None:
        """Merge another stats object into this one."""
        if other.count == 0:
            return
        self.count += other.count
        self.sum += other.sum
        self.sumsq += other.sumsq
        self.min = other.min if self.min is None else self.min if other.min is None else min(self.min, other.min)
        self.max = other.max if self.max is None else self.max if other.max is None else max(self.max, other.max)

    def mean(self) -> float:
        """Return the mean of observed values (0.0 if empty)."""
        if self.count == 0:
            return 0.0
        return self.sum / float(self.count)

    def std(self) -> float:
        """Return the population std of observed values (0.0 if empty)."""
        if self.count == 0:
            return 0.0
        mean = self.mean()
        variance = max(0.0, (self.sumsq / float(self.count)) - (mean * mean))
        return math.sqrt(variance)

    def as_state(self) -> dict[str, int | float]:
        """Serialize the running stats to a picklable state dict.

        Notes:
            `min` and `max` are represented as NaN when unset to keep the payload numeric.
        """
        return {
            "count": int(self.count),
            "sum": float(self.sum),
            "sumsq": float(self.sumsq),
            "min": float(self.min) if self.min is not None else float("nan"),
            "max": float(self.max) if self.max is not None else float("nan"),
        }

    @classmethod
    def from_state(
        cls,
        state: collections.abc.Mapping[str, int | float],
    ) -> "RunningStats":
        """Deserialize running stats from a state dict produced by `as_state()`."""
        stats = cls()
        stats.count = int(state.get("count", 0))
        stats.sum = float(state.get("sum", 0.0))
        stats.sumsq = float(state.get("sumsq", 0.0))
        min_val = float(state.get("min", float("nan")))
        max_val = float(state.get("max", float("nan")))
        stats.min = None if math.isnan(min_val) else min_val
        stats.max = None if math.isnan(max_val) else max_val
        return stats
