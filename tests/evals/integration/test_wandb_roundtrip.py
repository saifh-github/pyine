"""End-to-end W&B round-trip tests for code exec and correctness pipelines."""

from __future__ import annotations

import asyncio
import os
import typing
import uuid

import pytest
import wandb

import pyine.data.utils.splits
import pyine.evals.code_exec.analysis as code_exec_analysis
import pyine.evals.code_exec.configs as code_exec_configs
import pyine.evals.correctness._impl as correctness_impl
import pyine.evals.correctness.analysis as correctness_analysis
import pyine.evals.correctness.configs as correctness_configs
import pyine.evals.correctness.data_loading as correctness_data_loading
import pyine.evals.correctness.datamodule as correctness_datamodule
import pyine.evals.correctness.datamodule_configs as correctness_datamodule_configs
import pyine.evals.correctness.types as correctness_types
import tests.env_checks
import tests.evals.integration.conftest as integration_conftest
import tests.evals.integration.fake_models as fake_models
import tests.evals.integration.wandb_helpers

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        tests.env_checks.WANDB_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
        reason="wandb API key or network unavailable",
    ),
]


def _skip_if_offline() -> None:
    if os.getenv("WANDB_MODE", "").lower() == "offline":
        pytest.skip("wandb offline mode cannot create remote runs")


# ---------------------------------------------------------------------------
# Code exec W&B round-trip
# ---------------------------------------------------------------------------


class TestCodeExecWandbRoundTrip:
    def test_log_and_fetch_summary_metrics_match(
        self,
        tmp_path: typing.Any,
        oracle_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        _skip_if_offline()
        project = "pyine-tests"
        entity = os.getenv("WANDB_ENTITY")
        unique_name = f"test-ce-summary-{uuid.uuid4().hex[:8]}"
        eval_config = code_exec_configs.CodeExecEvalsConfig()
        result = oracle_code_exec_output.result
        settings = wandb.Settings(start_method="thread", _disable_stats=True)
        api_run: wandb.apis.public.Run | None = None
        run = wandb.init(
            project=project,
            entity=entity,
            dir=str(tmp_path),
            name=unique_name,
            reinit=True,
            settings=settings,
        )
        try:
            assert run is not None
            eval_config.log_metrics(run, {"test": result})
            run.finish()
            api_run = tests.evals.integration.wandb_helpers.wait_for_run(project, entity, unique_name)
            summary = code_exec_analysis.fetch_eval_summary(api_run, subset_name="test")
            assert summary.run_info.accuracy["hard"].value == pytest.approx(
                result.metrics["accuracy_hard"],
                abs=1e-4,
            )
            assert summary.run_info.accuracy["soft"].value == pytest.approx(
                result.metrics["accuracy_soft"],
                abs=1e-4,
            )
        finally:
            if api_run is not None:
                tests.evals.integration.wandb_helpers.delete_run(api_run)

    def test_log_and_fetch_sample_metrics_table(
        self,
        tmp_path: typing.Any,
        oracle_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        _skip_if_offline()
        project = "pyine-tests"
        entity = os.getenv("WANDB_ENTITY")
        unique_name = f"test-ce-samples-{uuid.uuid4().hex[:8]}"
        eval_config = code_exec_configs.CodeExecEvalsConfig()
        result = oracle_code_exec_output.result
        settings = wandb.Settings(start_method="thread", _disable_stats=True)
        api_run: wandb.apis.public.Run | None = None
        run = wandb.init(
            project=project,
            entity=entity,
            dir=str(tmp_path),
            name=unique_name,
            reinit=True,
            settings=settings,
        )
        try:
            assert run is not None
            eval_config.log_sample_metrics(run, "test", result)
            run.finish()
            api_run = tests.evals.integration.wandb_helpers.wait_for_run(project, entity, unique_name)
            df = tests.evals.integration.wandb_helpers.wait_for_sample_metrics_table(api_run, "test")
            assert len(df) == len(result.artifacts)
        finally:
            if api_run is not None:
                tests.evals.integration.wandb_helpers.delete_run(api_run)


# ---------------------------------------------------------------------------
# Correctness W&B round-trip
# ---------------------------------------------------------------------------


class TestCorrectnessWandbRoundTrip:
    def test_correctness_log_and_fetch_metrics_match(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: typing.Any,
        random_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        _skip_if_offline()
        # build correctness result from the random code exec LMDB
        records = correctness_data_loading.load_records_from_lmdb(
            [random_code_exec_output.lmdb_path],
            label_type=correctness_types.LabelType.HARD_MATCH,
        )
        problem_ids = sorted({rec.problem_id for rec in records})
        valid_count = len(problem_ids) * 7 // 10
        split_result = pyine.data.utils.splits.SplitResult(
            source_dataset_name="TACO",
            source_dataset_hash="fake_hash",
            identifiers=problem_ids,
            tag_lists=[[] for _ in problem_ids],
            source_data_hashes=[f"hash_{idx}" for idx in range(len(problem_ids))],
            subset_assignments={pid: ("valid" if idx < valid_count else "test") for idx, pid in enumerate(problem_ids)},
            creation_metadata={},
            config=pyine.data.utils.splits.SplitConfig(
                subset_names=["valid", "test"],
                subset_assign_prob_map={"valid": 0.5, "test": 0.5},
            ),
        )
        monkeypatch.setattr(
            "pyine.evals.correctness.splits.pyine.data.utils.splits.get_dataset_split_result",
            lambda _source: split_result,
        )
        dm_config = correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(random_code_exec_output.lmdb_path,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
        )
        dm = correctness_datamodule.CorrectnessDataModule(dm_config)
        dm.prepare_data()
        dm.setup()
        eval_config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=dm_config,
            target_fpr_values=[0.05],
            num_bootstrap_replicates=10,
            roc_fpr_grid_size=20,
        )
        result = asyncio.run(
            correctness_impl.evaluate_guardrail_replicas(
                config=eval_config,
                guardrails=[fake_models.OracleGuardrailScorer()],
                datamodule=dm,
                eval_subset_name="guardrail_test",
            )
        )
        # log to wandb and fetch back
        project = "pyine-tests"
        entity = os.getenv("WANDB_ENTITY")
        subset_name = "guardrail_test"
        unique_name = f"test-corr-summary-{uuid.uuid4().hex[:8]}"
        settings = wandb.Settings(start_method="thread", _disable_stats=True)
        api_run: wandb.apis.public.Run | None = None
        run = wandb.init(
            project=project,
            entity=entity,
            dir=str(tmp_path),
            name=unique_name,
            reinit=True,
            settings=settings,
        )
        try:
            assert run is not None
            eval_config.log_metrics(run, {subset_name: result})
            run.finish()
            api_run = tests.evals.integration.wandb_helpers.wait_for_run(project, entity, unique_name)
            summary = correctness_analysis.fetch_correctness_eval_summary(api_run, subset_name)
            original_flat = result.aggregated.to_flat_dict()
            assert summary.run_info.auroc.value == pytest.approx(original_flat["auroc/mean"], abs=1e-4)
            if summary.run_info.sample_count is not None:
                assert summary.run_info.sample_count == original_flat["sample_count"]
        finally:
            if api_run is not None:
                tests.evals.integration.wandb_helpers.delete_run(api_run)
