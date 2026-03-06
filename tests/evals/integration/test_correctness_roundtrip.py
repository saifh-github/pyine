"""End-to-end round-trip tests for correctness LMDB loading, evaluation, and pickle persistence."""

from __future__ import annotations

import pathlib  # noqa: TC003
import re

import pytest

import pyine.data.utils.lmdb_io
import pyine.data.utils.splits
import pyine.evals.correctness._impl as correctness_impl
import pyine.evals.correctness.configs as correctness_configs
import pyine.evals.correctness.data_loading as correctness_data_loading
import pyine.evals.correctness.datamodule as correctness_datamodule
import pyine.evals.correctness.datamodule_configs as correctness_datamodule_configs
import pyine.evals.correctness.types as correctness_types
import pyine.evals.persistence
import tests.evals.integration.conftest as integration_conftest
import tests.evals.integration.fake_models as fake_models

pytestmark = pytest.mark.integration

_NUM_SAMPLES = integration_conftest._NUM_TACO_SAMPLES
_NUM_ATTEMPTS = integration_conftest._NUM_ATTEMPTS
_TOTAL_ATTEMPTS = _NUM_SAMPLES * _NUM_ATTEMPTS  # 60


def _build_synthetic_split_result(
    problem_ids: list[str],
    valid_count: int,
) -> pyine.data.utils.splits.SplitResult:
    """Build a synthetic SplitResult mapping problem IDs to "valid" and "test" subsets."""
    subset_assignments: dict[str, str] = {}
    for idx, pid in enumerate(problem_ids):
        subset_assignments[pid] = "valid" if idx < valid_count else "test"
    return pyine.data.utils.splits.SplitResult(
        source_dataset_name="TACO",
        source_dataset_hash="fake_hash",
        identifiers=problem_ids,
        tag_lists=[[] for _ in problem_ids],
        source_data_hashes=[f"hash_{idx}" for idx in range(len(problem_ids))],
        subset_assignments=subset_assignments,
        creation_metadata={},
        config=pyine.data.utils.splits.SplitConfig(
            subset_names=["valid", "test"],
            subset_assign_prob_map={"valid": 0.5, "test": 0.5},
        ),
    )


# ---------------------------------------------------------------------------
# LMDB loading (happy path)
# ---------------------------------------------------------------------------


class TestCorrectnessLmdbLoading:
    def test_lmdb_records_loadable(
        self,
        random_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        records = correctness_data_loading.load_records_from_lmdb(
            [random_code_exec_output.lmdb_path],
            label_type=correctness_types.LabelType.HARD_MATCH,
        )
        assert len(records) == _TOTAL_ATTEMPTS

    def test_records_have_both_label_classes(
        self,
        random_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        records = correctness_data_loading.load_records_from_lmdb(
            [random_code_exec_output.lmdb_path],
            label_type=correctness_types.LabelType.HARD_MATCH,
        )
        labels = {rec.label for rec in records}
        assert True in labels, "expected at least one True label"
        assert False in labels, "expected at least one False label"

    def test_record_fields_populated(
        self,
        random_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        records = correctness_data_loading.load_records_from_lmdb(
            [random_code_exec_output.lmdb_path],
            label_type=correctness_types.LabelType.HARD_MATCH,
        )
        for rec in records:
            assert rec.problem_id, "problem_id should be non-empty"
            assert rec.sample_id, "sample_id should be non-empty"
            assert rec.attempt_index >= 0
            assert rec.model_output, "model_output should be non-empty"
            assert rec.expected_output, "expected_output should be non-empty"

    def test_problem_ids_are_valid_taco_format(
        self,
        random_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        records = correctness_data_loading.load_records_from_lmdb(
            [random_code_exec_output.lmdb_path],
            label_type=correctness_types.LabelType.HARD_MATCH,
        )
        pattern = re.compile(r"^TACO/TRAIN/p\d{6}$")
        for rec in records:
            assert pattern.match(rec.problem_id), f"unexpected problem_id format: {rec.problem_id}"


# ---------------------------------------------------------------------------
# LMDB error paths (E2E-seam)
# ---------------------------------------------------------------------------


class TestCorrectnessLmdbErrorPaths:
    def test_load_records_malformed_sample_id_raises(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        lmdb_path = tmp_path / "malformed_lmdb"
        with pyine.data.utils.lmdb_io.LMDBWriter(lmdb_path) as writer:
            writer.put(
                "test/0",
                {
                    "sample_id": "foobar",
                    "attempt_index": 0,
                    "hard_match": True,
                    "soft_match": True,
                    "model_output": "out",
                    "final_answer": None,
                    "expected_output": "expected",
                    "code_type": "original",
                    "tags": [],
                    "difficulty_score": None,
                },
            )
            writer.write_metadata({"record_type": "benchmark"})
        with pytest.raises(ValueError):
            correctness_data_loading.load_records_from_lmdb(
                [lmdb_path],
                label_type=correctness_types.LabelType.HARD_MATCH,
            )


# ---------------------------------------------------------------------------
# Correctness eval pipeline (happy path)
# ---------------------------------------------------------------------------


class TestCorrectnessEvalPipeline:
    def _build_datamodule(
        self,
        monkeypatch: pytest.MonkeyPatch,
        lmdb_path: pathlib.Path,
    ) -> correctness_datamodule.CorrectnessDataModule:
        """Build a real CorrectnessDataModule using the code exec LMDB, with a synthetic split."""
        records = correctness_data_loading.load_records_from_lmdb(
            [lmdb_path],
            label_type=correctness_types.LabelType.HARD_MATCH,
        )
        problem_ids = sorted({rec.problem_id for rec in records})
        valid_count = len(problem_ids) * 7 // 10  # ~70% valid, ~30% test
        split_result = _build_synthetic_split_result(problem_ids, valid_count)
        monkeypatch.setattr(
            "pyine.evals.correctness.splits.pyine.data.utils.splits.get_dataset_split_result",
            lambda _source: split_result,
        )
        dm_config = correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(lmdb_path,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
        )
        dm = correctness_datamodule.CorrectnessDataModule(dm_config)
        dm.prepare_data()
        dm.setup()
        return dm

    @pytest.mark.asyncio
    async def test_correctness_eval_produces_valid_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
        random_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        dm = self._build_datamodule(monkeypatch, random_code_exec_output.lmdb_path)
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=dm.config,
            target_fpr_values=[0.05],
            num_bootstrap_replicates=10,
            roc_fpr_grid_size=20,
        )
        result = await correctness_impl.evaluate_guardrail_replicas(
            config=config,
            guardrails=[fake_models.OracleGuardrailScorer()],
            datamodule=dm,
            eval_subset_name="guardrail_test",
        )
        assert result.aggregated is not None
        single_run = result.aggregated.per_run[0]
        assert single_run.threshold_free.auroc == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_correctness_flat_dict_has_expected_keys(
        self,
        monkeypatch: pytest.MonkeyPatch,
        random_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        dm = self._build_datamodule(monkeypatch, random_code_exec_output.lmdb_path)
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=dm.config,
            target_fpr_values=[0.05],
            num_bootstrap_replicates=10,
            roc_fpr_grid_size=20,
        )
        result = await correctness_impl.evaluate_guardrail_replicas(
            config=config,
            guardrails=[fake_models.OracleGuardrailScorer()],
            datamodule=dm,
            eval_subset_name="guardrail_test",
        )
        flat = result.aggregated.to_flat_dict()
        assert "auroc/mean" in flat
        assert "sample_count" in flat
        assert "record_count" in flat

    @pytest.mark.asyncio
    async def test_correctness_pickle_round_trip(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
        random_code_exec_output: integration_conftest.CodeExecPipelineOutput,
    ) -> None:
        dm = self._build_datamodule(monkeypatch, random_code_exec_output.lmdb_path)
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=dm.config,
            target_fpr_values=[0.05],
            num_bootstrap_replicates=10,
            roc_fpr_grid_size=20,
        )
        result = await correctness_impl.evaluate_guardrail_replicas(
            config=config,
            guardrails=[fake_models.OracleGuardrailScorer()],
            datamodule=dm,
            eval_subset_name="guardrail_test",
        )
        pickle_path = tmp_path / "correctness_result.pkl"
        pyine.evals.persistence.save_eval_result(result, pickle_path)
        loaded = pyine.evals.persistence.load_eval_result(
            pickle_path,
            expected_type=correctness_impl.CorrectnessEvalResult,
        )
        original_flat = result.aggregated.to_flat_dict()
        loaded_flat = loaded.aggregated.to_flat_dict()
        for key in ("auroc/mean", "sample_count", "record_count"):
            assert loaded_flat[key] == pytest.approx(original_flat[key]), f"mismatch on {key}"
