"""Streaming statistics utilities.

This module provides small, dependency-light helpers for tracking scalar statistics in a streaming
fashion. These are useful across training, reward computation, and evaluation code paths.

Classes:
    RunningStats: Tracks mean/std/min/max for a single variable.
    RunningCorrStats: Tracks correlation and regression slope for paired (x, y) observations.
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

    def to_metrics(
        self,
        prefix: str = "",
    ) -> dict[str, int | float]:
        """Convert stats to a metrics dict suitable for logging.

        Args:
            prefix: Optional prefix for metric keys (e.g., "reward/total" produces
                "reward/total/mean", "reward/total/std", etc.). If empty, keys are
                bare (e.g., "mean", "std"). Trailing slashes in the prefix are handled.

        Returns:
            Dict with mean/std/min/max/count metrics, or empty dict if no observations.
        """
        if self.count == 0:
            return {}
        assert self.min is not None and self.max is not None
        key_prefix = f"{prefix.rstrip('/')}/" if prefix else ""
        return {
            f"{key_prefix}mean": self.mean(),
            f"{key_prefix}std": self.std(),
            f"{key_prefix}min": self.min,
            f"{key_prefix}max": self.max,
            f"{key_prefix}count": self.count,
        }


@dataclasses.dataclass(slots=True)
class RunningCorrStats:
    """Streaming correlation and regression statistics for paired (x, y) observations.

    Computes Pearson correlation coefficient and linear regression slope using running sums,
    without storing individual observations. This is memory-efficient for large datasets.

    The formulas used are:
        - Pearson r = (n*sum_xy - sum_x*sum_y) / sqrt((n*sum_x2 - sum_x^2) * (n*sum_y2 - sum_y^2))
        - Slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)

    Example:
        >>> stats = RunningCorrStats()
        >>> for x, y in [(1, 2), (2, 4), (3, 6)]:
        ...     stats.update(x, y)
        >>> stats.correlation()  # perfect positive correlation
        1.0
        >>> stats.slope()  # y = 2x, so slope is 2
        2.0
    """

    count: int = 0
    """Number of (x, y) pairs observed."""
    sum_x: float = 0.0
    """Sum of x values."""
    sum_y: float = 0.0
    """Sum of y values."""
    sum_xy: float = 0.0
    """Sum of x*y products."""
    sum_x2: float = 0.0
    """Sum of x^2 values."""
    sum_y2: float = 0.0
    """Sum of y^2 values."""

    def update(
        self,
        x: float,
        y: float,
    ) -> None:
        """Add a new (x, y) observation pair."""
        self.count += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_xy += x * y
        self.sum_x2 += x * x
        self.sum_y2 += y * y

    def merge(
        self,
        other: "RunningCorrStats",
    ) -> None:
        """Merge another correlation stats object into this one."""
        self.count += other.count
        self.sum_x += other.sum_x
        self.sum_y += other.sum_y
        self.sum_xy += other.sum_xy
        self.sum_x2 += other.sum_x2
        self.sum_y2 += other.sum_y2

    def correlation(self) -> float | None:
        """Compute Pearson correlation coefficient.

        Returns:
            Correlation coefficient in [-1, 1], or None if fewer than 2 observations
            or if either variable has zero variance.
        """
        if self.count < 2:
            return None
        n = self.count
        var_x = n * self.sum_x2 - self.sum_x**2
        var_y = n * self.sum_y2 - self.sum_y**2
        if var_x <= 0 or var_y <= 0:
            return None
        denom = math.sqrt(var_x * var_y)
        return (n * self.sum_xy - self.sum_x * self.sum_y) / denom

    def slope(self) -> float | None:
        """Compute linear regression slope (y = slope * x + intercept).

        Returns:
            Slope coefficient, or None if fewer than 2 observations or if x has zero variance.
        """
        if self.count < 2:
            return None
        denom = self.count * self.sum_x2 - self.sum_x**2
        if denom == 0:
            return None
        return (self.count * self.sum_xy - self.sum_x * self.sum_y) / denom

    def intercept(self) -> float | None:
        """Compute linear regression intercept (y = slope * x + intercept).

        Returns:
            Intercept coefficient, or None if slope cannot be computed.
        """
        slope = self.slope()
        if slope is None:
            return None
        mean_x = self.sum_x / self.count
        mean_y = self.sum_y / self.count
        return mean_y - slope * mean_x

    def as_state(self) -> dict[str, float | int]:
        """Serialize to a picklable state dict."""
        return {
            "count": self.count,
            "sum_x": self.sum_x,
            "sum_y": self.sum_y,
            "sum_xy": self.sum_xy,
            "sum_x2": self.sum_x2,
            "sum_y2": self.sum_y2,
        }

    @classmethod
    def from_state(
        cls,
        state: collections.abc.Mapping[str, float | int],
    ) -> "RunningCorrStats":
        """Deserialize from a state dict produced by `as_state()`."""
        return cls(
            count=int(state.get("count", 0)),
            sum_x=float(state.get("sum_x", 0.0)),
            sum_y=float(state.get("sum_y", 0.0)),
            sum_xy=float(state.get("sum_xy", 0.0)),
            sum_x2=float(state.get("sum_x2", 0.0)),
            sum_y2=float(state.get("sum_y2", 0.0)),
        )

    def to_metrics(
        self,
        prefix: str = "",
    ) -> dict[str, float]:
        """Convert stats to a metrics dict suitable for logging.

        Args:
            prefix: Optional prefix for metric keys (e.g., "reward" produces
                "reward/correlation", "reward/slope"). If empty, keys are bare.

        Returns:
            Dict with correlation and slope metrics. Only includes values that
            are computable (i.e., non-None). Returns empty dict if fewer than
            2 observations.
        """
        metrics: dict[str, float] = {}
        key_prefix = f"{prefix.rstrip('/')}/" if prefix else ""
        corr = self.correlation()
        if corr is not None:
            metrics[f"{key_prefix}correlation"] = corr
        slope = self.slope()
        if slope is not None:
            metrics[f"{key_prefix}slope"] = slope
        return metrics
