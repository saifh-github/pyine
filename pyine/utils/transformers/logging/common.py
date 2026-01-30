"""Shared helpers for direct wandb logging."""

import typing

import wandb

DEFAULT_STEP_METRIC_KEY = "train/global_step"
"""Default x-axis metric for wandb charts. Matches HuggingFace Trainer convention."""


def deferred_define_metric(
    wandb_run: typing.Any,
    metric_name: str,
    step_metric: str,
    summary: str = "last",
) -> None:
    """Define a wandb metric with custom step axis (deferred until after WandbCallback).

    This should be called lazily on first log to ensure it runs AFTER HuggingFace's
    WandbCallback calls wandb.define_metric("*", step_metric="train/global_step").

    Args:
        wandb_run: The wandb run object (from runtime.wandb_run)
        metric_name: Name of the metric (can include wildcards like "train/throughput/*")
        step_metric: Name of the step metric to use as x-axis (e.g., "train/global_step")
        summary: Summary statistic for this metric (default: "last")
    """
    if wandb.run is None:
        return
    wandb_run.define_metric(
        metric_name,
        step_metric=step_metric,
        summary=summary,
    )
