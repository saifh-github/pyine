"""Shared W&B helpers for integration tests that create and clean up real runs."""

from __future__ import annotations

import logging
import time

import pandas as pd  # noqa: TC002
import pytest
import wandb

import pyine.evals.code_exec.analysis as code_exec_analysis

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 2.0
POLL_TIMEOUT_SECONDS = 60.0


def wait_for_run(
    project: str,
    entity: str | None,
    run_name: str,
    timeout: float = POLL_TIMEOUT_SECONDS,
) -> wandb.apis.public.Run:
    """Poll wandb until the newly-created run is visible via the public API."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        runs = code_exec_analysis.fetch_runs(
            project=project,
            entity=entity,
            filters={"displayName": run_name},
            per_page=5,
        )
        if runs:
            return runs[0]
        time.sleep(POLL_INTERVAL_SECONDS)
    pytest.fail(f"timed out waiting for wandb run '{run_name}' to materialize")


def delete_run(run: wandb.apis.public.Run) -> None:
    """Attempt to delete a test run to keep the project tidy."""
    try:
        api = wandb.Api()
        api_run = api.run(f"{run.entity}/{run.project}/{run.id}")
        api_run.delete()
    except Exception as exc:
        logger.debug("unable to delete test wandb run %s/%s/%s: %s", run.entity, run.project, run.id, exc)


def wait_for_sample_metrics_table(
    run: wandb.apis.public.Run,
    subset_name: str,
    timeout: float = POLL_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Poll until ``fetch_sample_metrics_table`` returns a non-None DataFrame."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        df = code_exec_analysis.fetch_sample_metrics_table(run, subset_name)
        if df is not None:
            return df
        time.sleep(POLL_INTERVAL_SECONDS)
    pytest.fail(f"timed out waiting for sample metrics table for subset '{subset_name}'")
