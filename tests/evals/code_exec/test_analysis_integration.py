import logging
import os
import pathlib
import time
import uuid

import pytest
import wandb

import pyine.evals.code_exec.analysis as analysis
import tests.env_checks

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 2.0
_POLL_TIMEOUT_SECONDS = 60.0


def _wait_for_run(
    project: str,
    entity: str | None,
    run_name: str,
) -> wandb.apis.public.Run:
    """Polls wandb until the newly-created run is visible via the public API."""
    deadline = time.time() + _POLL_TIMEOUT_SECONDS
    while time.time() < deadline:
        runs = analysis.fetch_runs(
            project=project,
            entity=entity,
            filters={"displayName": run_name},
            per_page=5,
        )
        if runs:
            return runs[0]
        time.sleep(_POLL_INTERVAL_SECONDS)
    pytest.fail(f"timed out waiting for wandb run '{run_name}' to materialize")


def _delete_run(run: wandb.apis.public.Run) -> None:
    """Attempts to delete the test run to keep the project tidy."""
    try:
        api = wandb.Api()
        api_run = api.run(f"{run.entity}/{run.project}/{run.id}")
        api_run.delete()
    except Exception as exc:
        logger.debug("unable to delete test wandb run %s/%s/%s: %s", run.entity, run.project, run.id, exc)


@pytest.mark.integration
@pytest.mark.skipif(
    tests.env_checks.WANDB_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="requires WANDB credentials and network access",
)
def test_analysis_helpers_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Creates a real wandb run and ensures analysis helpers can read it back."""
    if os.getenv("WANDB_MODE", "").lower() == "offline":
        pytest.skip("wandb offline mode cannot create remote runs for analysis integration test")
    monkeypatch.setenv("WANDB_PROJECT", "pyine-tests")
    project = os.getenv("WANDB_PROJECT")
    entity = os.getenv("WANDB_ENTITY")
    run_name = f"analysis-itest-{uuid.uuid4().hex[:8]}"
    summary_payload = {
        "predict/test/accuracy_hard": 0.6,
        "predict/test/accuracy_soft": 0.75,
        "predict/test/code_type/python/accuracy_hard": 0.9,
        "predict/test/code_type/python/sample_count": 10,
        "predict/test/predict_type/output/accuracy_hard": 0.5,
        "predict/test/predict_type/output/sample_count": 4,
        "predict/test/complexity/loc_mean": 42.0,
        "predict/test/complexity/loc_median": 40.0,
        "predict/test/complexity/loc_std": 7.0,
        "predict/test/complexity/loc_min": 10.0,
        "predict/test/complexity/loc_max": 90.0,
    }
    settings = wandb.Settings(start_method="thread", _disable_stats=True)
    with wandb.init(
        project=project,
        entity=entity,
        dir=str(tmp_path),
        name=run_name,
        reinit=True,
        settings=settings,
    ) as run:
        run.summary.update(summary_payload)
        run.log({"predict/test/metrics_table": 1})
    fetched_run = _wait_for_run(project, entity, run_name)
    try:
        summary = analysis.fetch_eval_summary(fetched_run, subset_name="test")
        assert summary.run_info.run_name == run_name
        assert summary.run_info.accuracy_hard == pytest.approx(0.6)
        assert summary.run_info.accuracy_soft == pytest.approx(0.75)
        categories = {c.category: c for c in summary.category_metrics}
        assert categories["code_type/python"].accuracy_hard == pytest.approx(0.9)
        assert categories["code_type/python"].count == 10
        assert categories["predict_type/output"].accuracy_hard == pytest.approx(0.5)
        assert categories["predict_type/output"].count == 4
        assert summary.complexity_metrics is not None
        complexity_by_name = {m.metric_name: m for m in summary.complexity_metrics.metrics}
        loc_stats = complexity_by_name["loc"]
        assert loc_stats.mean == pytest.approx(42.0)
        assert loc_stats.median == pytest.approx(40.0)
        assert loc_stats.std == pytest.approx(7.0)
        assert loc_stats.min == pytest.approx(10.0)
        assert loc_stats.max == pytest.approx(90.0)
        df = analysis.summarize_runs_to_dataframe([summary])
        assert float(df.loc[0, "accuracy_hard"]) == pytest.approx(0.6)
        assert float(df.loc[0, "accuracy_soft"]) == pytest.approx(0.75)
    finally:
        _delete_run(fetched_run)
