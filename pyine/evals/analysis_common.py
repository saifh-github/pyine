"""Shared types, matplotlib helpers, and W&B helpers for evaluation analysis modules.

This module provides common building blocks used by both ``code_exec.analysis`` and
``correctness.analysis``. Persistence utilities live in ``persistence.py``.
"""
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

import datetime
import re
import typing

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import pydantic

import pyine.utils.wandb_utils

if typing.TYPE_CHECKING:
    import numpy as np
    import wandb.apis.public
    from numpy.typing import NDArray


class MetricWithCI(typing.NamedTuple):
    """A metric value with optional confidence interval bounds.

    The CI method depends on the source metric because the underlying data differs:

    - **Accuracy** (per-attempt proportion): Wilson score interval. Accuracy is a ratio of
      binary counts (correct / total), and Wilson is designed for this; it handles small
      samples and extreme proportions better than normal approximations.
    - **Pass@K** (mean of per-sample real-valued estimates): SEM-based normal approximation.
      Per-sample pass@k values are continuous (e.g. 0.695), not binary counts, so Wilson
      does not apply. SEM on the mean is the standard approach here.
    - **Pass@1 when K=1**: Wilson (same as accuracy). With one attempt per sample, each
      per-sample estimate is binary (0 or 1), so Wilson applies and keeps the CIs identical
      to the corresponding accuracy CIs.
    - **Correctness metrics** (AUROC, TPR, etc.): cross-run and/or bootstrap CIs from
      ``correctness.types.AggregatedResult``.

    See ``OutcomeEvaluator.compute_metrics`` and ``_compute_pass_at_k_metrics`` for details.
    """

    value: float | None = None
    """Point estimate of the metric, or None if not available."""
    ci_lower: float | None = None
    """Lower bound of the confidence interval, or None if not available."""
    ci_upper: float | None = None
    """Upper bound of the confidence interval, or None if not available."""


class BaseRunInfo(pydantic.BaseModel):
    """Shared W&B run fields common to all evaluation analysis modules."""

    model_config = pydantic.ConfigDict(frozen=True)

    run_id: str
    """Unique identifier for the wandb run."""
    run_name: str
    """Human-readable name of the run."""
    run_group: str
    """Name of the run group (optional)."""
    project: str
    """Wandb project name."""
    entity: str | None
    """Wandb entity (team or user)."""
    created_at: str
    """ISO timestamp when the run was created."""
    subset_name: str
    """Name of the evaluation subset (e.g., 'test', 'valid')."""


# -- W&B helpers (covered by tests) --

fetch_runs = pyine.utils.wandb_utils.fetch_runs
"""Re-export from centralized wandb utilities for convenient import."""


def filter_runs_by_date(
    runs: list[wandb.apis.public.Run],
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[wandb.apis.public.Run]:
    """Filters runs by creation date range.

    Args:
        runs: List of wandb Run objects.
        start_date: ISO format start date (inclusive), e.g. "2024-01-15".
        end_date: ISO format end date (inclusive), e.g. "2024-12-31".

    Returns:
        Filtered list of runs.

    Note:
        Naive dates (without timezone) are interpreted as UTC. W&B timestamps are
        normalized to UTC for comparison.
    """

    def _normalize_to_utc(dt: datetime.datetime) -> datetime.datetime:
        """Normalize a datetime to UTC. Naive datetimes are assumed to be UTC."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=datetime.UTC)
        return dt.astimezone(datetime.UTC)

    def _parse_created_at(created_at: str | datetime.datetime) -> datetime.datetime:
        """Parse created_at which may be string or datetime depending on wandb version."""
        if isinstance(created_at, datetime.datetime):
            return _normalize_to_utc(created_at)
        # wandb uses "Z" suffix for UTC, convert to +00:00 for fromisoformat
        parsed = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return _normalize_to_utc(parsed)

    # parse ISO date strings to datetime for comparison with run.created_at
    start_dt = datetime.datetime.fromisoformat(start_date) if start_date else None
    end_dt = datetime.datetime.fromisoformat(end_date) if end_date else None
    # normalize to UTC for consistent comparison with wandb timestamps
    if start_dt is not None:
        start_dt = _normalize_to_utc(start_dt)
    if end_dt is not None:
        end_dt = _normalize_to_utc(end_dt)
        assert end_date is not None  # end_dt is only set when end_date is truthy
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", end_date):
            end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)

    return [
        run
        for run in runs
        if (start_dt is None or _parse_created_at(run.created_at) >= start_dt)
        and (end_dt is None or _parse_created_at(run.created_at) <= end_dt)
    ]


# -- Matplotlib helpers (plotting only, excluded from coverage) --


def get_or_create_axes(
    ax: matplotlib.axes.Axes | None,
    figsize: tuple[int, int] = (10, 6),
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:  # pragma: no cover
    """Helper to get or create matplotlib figure and axes."""
    if ax is None:
        return plt.subplots(figsize=figsize)
    return typing.cast("matplotlib.figure.Figure", ax.get_figure()), ax


def configure_bar_chart(
    ax: matplotlib.axes.Axes,
    x: NDArray[np.integer],
    labels: list[str],
    title: str,
    ylabel: str = "Accuracy",
    ylim: tuple[float, float] | None = (0, 1.12),
) -> None:  # pragma: no cover
    """Helper to configure common bar chart properties.

    Note: ylim upper bound is set to 1.12 (instead of 1.0) to provide headroom
    for sample count labels placed above CI whiskers.
    """
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
