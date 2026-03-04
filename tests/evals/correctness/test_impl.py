"""End-to-end tests for pyine.evals.correctness._impl."""

from __future__ import annotations

import pathlib
import typing
import unittest.mock

import numpy as np
import pytest

import pyine.evals.correctness._impl as correctness_impl
import pyine.evals.correctness.configs as correctness_configs
import pyine.evals.correctness.datamodule as correctness_datamodule
import pyine.evals.correctness.datamodule_configs as correctness_datamodule_configs
import pyine.evals.correctness.splits as correctness_splits
import pyine.evals.correctness.types as correctness_types
import tests.evals.correctness.conftest as correctness_conftest

_FAKE_LMDB_PATH = pathlib.Path("/fake/lmdb")


class _FakeLMDBReader:
    def __init__(
        self,
        records: list[dict[str, typing.Any]],
        metadata: dict[str, typing.Any],
    ) -> None:
        self._records = records
        self._metadata = metadata
        self.sample_count = len(records)

    def get_metadata(self) -> dict[str, typing.Any]:
        return self._metadata

    def get(self, idx: int) -> dict[str, typing.Any]:
        return self._records[idx]

    def close(self) -> None:
        pass


@pytest.mark.slow
class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_pipeline_with_mock_scorer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end: synthetic data -> mock scorer -> verify AggregatedResult structure."""
        rng = np.random.default_rng(42)
        lmdb_records: list[dict[str, typing.Any]] = []
        for prob_idx in range(20):
            for attempt_idx in range(3):
                label = bool(rng.random() > 0.4)
                lmdb_records.append(
                    {
                        "sample_id": f"TACO/TRAIN/p{prob_idx:06d}/s0000/t0000",
                        "attempt_index": attempt_idx,
                        "hard_match": label,
                        "soft_match": label,
                        "model_output": "output",
                        "final_answer": None,
                        "expected_output": "expected",
                        "code_type": "original",
                        "tags": [],
                    }
                )
        reader = _FakeLMDBReader(lmdb_records, {"record_type": "benchmark"})
        monkeypatch.setattr(
            "pyine.evals.correctness.data_loading.pyine.data.utils.lmdb_io.LMDBReader",
            lambda path: reader,
        )
        subset_map = {
            "valid": [f"TACO/TRAIN/p{idx:06d}" for idx in range(10)],
            "test": [f"TACO/TRAIN/p{idx:06d}" for idx in range(10, 20)],
        }
        mock_split_result = unittest.mock.MagicMock()
        mock_split_result.get_subset_to_ids_map.return_value = subset_map
        mock_split_result.source_dataset_name = "TACO"
        monkeypatch.setattr(
            "pyine.evals.correctness.splits.pyine.data.utils.splits.get_dataset_split_result",
            lambda _source: mock_split_result,
        )
        monkeypatch.setattr(
            "pyine.data.utils.lmdb_io.resolve_lmdb_paths",
            lambda raw_paths: list(raw_paths),
        )
        dm_config = correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(
                split_source="TACO",
                guardrail_valid_fraction=0.5,
                seed=42,
            ),
        )
        dm = correctness_datamodule.CorrectnessDataModule(dm_config)
        dm.prepare_data()
        dm.setup()
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=dm_config,
            target_fpr_values=[0.01, 0.05],
            num_bootstrap_replicates=20,
            roc_fpr_grid_size=50,
        )
        scorer = correctness_conftest.MockGuardrailScorer(noise_seed=0)
        result = await correctness_impl.evaluate_guardrail_replicas(
            config=config,
            guardrails=[scorer],
            datamodule=dm,
            eval_subset_name="guardrail_valid",
            verbose=True,
        )
        assert isinstance(result, correctness_impl.CorrectnessEvalResult)
        agg = result.aggregated
        assert len(agg.per_run) == 1
        assert agg.class_balance.overall_positive_rate > 0.0
        single_run = agg.per_run[0]
        assert single_run.threshold_free.auroc is not None
        assert 0.01 in single_run.attempt_metrics
        assert 0.05 in single_run.attempt_metrics
        assert 0.01 in single_run.sample_metrics
        flat = agg.to_flat_dict()
        assert "auroc/mean" in flat
        assert "class_balance/overall_positive_rate" in flat
        assert "record_count" in flat
        assert "auroc/num_valid_runs" in flat
        assert any(key.endswith("/base_pass_rate/mean") for key in flat)
        assert single_run.guardrail_metadata["name"] == "mock"
        assert result.eval_metadata["eval_subset_name"] == "guardrail_valid"
        assert "eval_config" in result.eval_metadata
        assert "datamodule_config" in result.eval_metadata
        assert "reprod_metadata" in result.eval_metadata
        assert len(result.eval_metadata["guardrail_metadata_by_run"]) == 1
        assert result.aggregated.attempt_records_by_key is not None
        assert len(result.aggregated.attempt_records_by_key) == len(dm.get_records_for_subset("guardrail_valid"))

    @pytest.mark.asyncio
    async def test_empty_guardrails_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        valid_records = [
            correctness_conftest.make_record("v0/s0/t0", "v0", 0, True),
            correctness_conftest.make_record("v0/s0/t0", "v0", 1, False),
        ]
        dm = correctness_conftest.build_mock_datamodule(monkeypatch, valid_records=valid_records, test_records=[])
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_conftest.make_dm_config(),
        )
        with pytest.raises(ValueError, match="guardrails must not be empty"):
            await correctness_impl.evaluate_guardrail_replicas(
                config=config,
                guardrails=[],
                datamodule=dm,
                eval_subset_name="guardrail_valid",
            )

    @pytest.mark.asyncio
    async def test_perfect_scorer_sanity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Perfect scorer -> AUROC=1.0, TPR=1.0, FPR=0.0."""

        class _PerfectScorer:
            def score_records(
                self,
                records: list[correctness_types.EvalRecord],
            ) -> correctness_types.ScoringResult:
                return correctness_types.ScoringResult(
                    scores=[1.0 if rec.label else 0.0 for rec in records],
                )

            def get_metadata(self) -> dict[str, typing.Any]:
                return {"name": "perfect"}

            def get_verification_cost_unit(self) -> str | None:
                return None

        lmdb_records: list[dict[str, typing.Any]] = []
        for prob_idx in range(40):
            for attempt_idx in range(4):
                label = attempt_idx < 2
                lmdb_records.append(
                    {
                        "sample_id": f"TACO/TRAIN/p{prob_idx:06d}/s0000/t0000",
                        "attempt_index": attempt_idx,
                        "hard_match": label,
                        "soft_match": label,
                        "model_output": "output",
                        "final_answer": None,
                        "expected_output": "expected",
                        "code_type": "original",
                        "tags": [],
                    }
                )
        reader = _FakeLMDBReader(lmdb_records, {"record_type": "benchmark"})
        monkeypatch.setattr(
            "pyine.evals.correctness.data_loading.pyine.data.utils.lmdb_io.LMDBReader",
            lambda path: reader,
        )
        subset_map = {
            "valid": [f"TACO/TRAIN/p{idx:06d}" for idx in range(20)],
            "test": [f"TACO/TRAIN/p{idx:06d}" for idx in range(20, 40)],
        }
        mock_split_result = unittest.mock.MagicMock()
        mock_split_result.get_subset_to_ids_map.return_value = subset_map
        mock_split_result.source_dataset_name = "TACO"
        monkeypatch.setattr(
            "pyine.evals.correctness.splits.pyine.data.utils.splits.get_dataset_split_result",
            lambda _source: mock_split_result,
        )
        monkeypatch.setattr(
            "pyine.data.utils.lmdb_io.resolve_lmdb_paths",
            lambda raw_paths: list(raw_paths),
        )
        dm_config = correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(
                split_source="TACO",
                guardrail_valid_fraction=0.5,
                seed=42,
            ),
            eval_subset_names=("guardrail_test",),
        )
        dm = correctness_datamodule.CorrectnessDataModule(dm_config)
        dm.prepare_data()
        dm.setup()
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=dm_config,
            target_fpr_values=[0.1],
            num_bootstrap_replicates=10,
            roc_fpr_grid_size=20,
        )
        result = await correctness_impl.evaluate_guardrail_replicas(
            config=config,
            guardrails=[_PerfectScorer()],
            datamodule=dm,
            eval_subset_name="guardrail_test",
        )
        agg = result.aggregated
        single_run = agg.per_run[0]
        assert single_run.threshold_free.auroc == pytest.approx(1.0)
        att = single_run.attempt_metrics[0.1]
        assert att.tpr == pytest.approx(1.0)
        assert att.fpr == pytest.approx(0.0)


class TestEvalSubsetValidation:
    @pytest.mark.asyncio
    async def test_guardrail_train_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        valid_records = [
            correctness_conftest.make_record("v0/s0/t0", "v0", 0, True),
            correctness_conftest.make_record("v0/s0/t0", "v0", 1, False),
        ]
        dm = correctness_conftest.build_mock_datamodule(monkeypatch, valid_records=valid_records, test_records=[])
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_conftest.make_dm_config(),
        )
        with pytest.raises(ValueError, match="guardrail_train is not supported"):
            await correctness_impl.evaluate_guardrail_replicas(
                config=config,
                guardrails=[correctness_conftest.MockGuardrailScorer()],
                datamodule=dm,
                eval_subset_name="guardrail_train",
            )


class TestFastEndToEnd:
    """Fast E2E test that runs in the default suite (not slow)."""

    @pytest.mark.asyncio
    async def test_mini_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Minimal E2E: mocked LMDB/splits, tiny bootstrap, verify result structure."""
        valid_records = []
        test_records = []
        for prob_idx in range(3):
            for attempt_idx in range(2):
                valid_records.append(
                    correctness_conftest.make_record(
                        f"TACO/TRAIN/p{prob_idx:06d}/s0000/t0000",
                        f"TACO/TRAIN/p{prob_idx:06d}",
                        attempt_idx,
                        label=(attempt_idx == 0),
                    )
                )
        for prob_idx in range(3, 6):
            for attempt_idx in range(2):
                test_records.append(
                    correctness_conftest.make_record(
                        f"TACO/TRAIN/p{prob_idx:06d}/s0000/t0000",
                        f"TACO/TRAIN/p{prob_idx:06d}",
                        attempt_idx,
                        label=(attempt_idx == 0),
                    )
                )
        dm = correctness_conftest.build_mock_datamodule(
            monkeypatch, valid_records=valid_records, test_records=test_records
        )
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_conftest.make_dm_config(),
            target_fpr_values=[0.05],
            num_bootstrap_replicates=5,
            roc_fpr_grid_size=10,
        )
        scorer = correctness_conftest.MockGuardrailScorer(noise_seed=0)
        result = await correctness_impl.evaluate_guardrail_replicas(
            config=config,
            guardrails=[scorer],
            datamodule=dm,
            eval_subset_name="guardrail_valid",
        )
        assert isinstance(result, correctness_impl.CorrectnessEvalResult)
        flat = result.aggregated.to_flat_dict()
        assert "auroc/mean" in flat
        assert "sample_count" in flat
        assert flat["sample_count"] == 3  # 3 valid problems
        assert "category/regular/auroc/mean" in flat
        assert "category/regular/fpr_0_05/guarded_pass_rate/mean" in flat
        assert not any(key.startswith("category/") and "average_precision" in key for key in flat)


class TestAggregateCostStats:
    def _make_cost_stats(
        self,
        target_fpr: float,
        cost_unit: str | None,
    ) -> correctness_types.VerificationCostStats:
        return correctness_types.VerificationCostStats(
            target_fpr=target_fpr,
            cost_unit=cost_unit,
            total_cost=10.0,
            mean_cost_per_record=2.0,
            median_cost_per_record=1.5,
            std_cost_per_record=0.5,
            cost_per_correct_acceptance=1.0,
            cost_per_incorrect_block=3.0,
            cost_accuracy_rank_correlation=None,
            cost_difficulty_rank_correlation=None,
        )

    def _make_single_run(
        self,
        cost_stats: dict[float, correctness_types.VerificationCostStats] | None,
    ) -> correctness_types.SingleRunResult:
        empty_grid = np.array([], dtype=np.float64)
        return correctness_types.SingleRunResult(
            guardrail_metadata={},
            threshold_free=correctness_types.ThresholdFreeMetrics(
                auroc=None,
                average_precision=None,
                tpr_at_fpr=None,
                fpr_grid=empty_grid,
                tpr_grid=empty_grid,
                precision_grid=empty_grid,
                recall_grid=empty_grid,
            ),
            attempt_metrics={},
            sample_metrics={},
            category_results={},
            bootstrap_cis={},
            difficulty_stats=None,
            verification_cost_stats=cost_stats,
        )

    def test_matching_cost_units_succeeds(self) -> None:
        runs = [
            self._make_single_run({0.05: self._make_cost_stats(0.05, "tokens")}),
            self._make_single_run({0.05: self._make_cost_stats(0.05, "tokens")}),
        ]
        result = correctness_impl._aggregate_cost_stats(runs, [0.05])
        assert result is not None
        assert result[0.05].cost_unit == "tokens"

    def test_mismatched_cost_units_raises(self) -> None:
        runs = [
            self._make_single_run({0.05: self._make_cost_stats(0.05, "tokens")}),
            self._make_single_run({0.05: self._make_cost_stats(0.05, "FLOPs")}),
        ]
        with pytest.raises(ValueError, match="cost_unit mismatch"):
            correctness_impl._aggregate_cost_stats(runs, [0.05])

    def test_none_vs_string_cost_units_raises(self) -> None:
        runs = [
            self._make_single_run({0.05: self._make_cost_stats(0.05, None)}),
            self._make_single_run({0.05: self._make_cost_stats(0.05, "tokens")}),
        ]
        with pytest.raises(ValueError, match="cost_unit mismatch"):
            correctness_impl._aggregate_cost_stats(runs, [0.05])


class TestCostCorrelationCollection:
    """Verify that cost correlation metrics are collected during cross-run aggregation."""

    def test_cost_correlations_appear_in_cross_run_mean(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        empty_grid = np.array([], dtype=np.float64)
        cost_stats = correctness_types.VerificationCostStats(
            target_fpr=0.05,
            cost_unit="tokens",
            total_cost=10.0,
            mean_cost_per_record=2.0,
            median_cost_per_record=1.5,
            std_cost_per_record=0.5,
            cost_per_correct_acceptance=1.0,
            cost_per_incorrect_block=3.0,
            cost_accuracy_rank_correlation=0.42,
            cost_difficulty_rank_correlation=-0.15,
        )
        run_result = correctness_types.SingleRunResult(
            guardrail_metadata={},
            threshold_free=correctness_types.ThresholdFreeMetrics(
                auroc=None,
                average_precision=None,
                tpr_at_fpr=None,
                fpr_grid=empty_grid,
                tpr_grid=empty_grid,
                precision_grid=empty_grid,
                recall_grid=empty_grid,
            ),
            attempt_metrics={},
            sample_metrics={},
            category_results={},
            bootstrap_cis={},
            difficulty_stats=None,
            verification_cost_stats={0.05: cost_stats},
        )
        test_records = [
            correctness_conftest.make_record("s1", "p1", 0, True),
            correctness_conftest.make_record("s2", "p2", 0, False),
        ]
        test_scores = np.array([0.9, 0.1], dtype=np.float64)
        splits = correctness_splits.GuardrailSplits(
            guardrail_train=[],
            guardrail_valid=[],
            guardrail_test=test_records,
            train_problem_ids=frozenset(),
            valid_problem_ids=frozenset(),
            test_problem_ids=frozenset({"p1", "p2"}),
        )
        monkeypatch.setattr(
            "pyine.evals.correctness._impl.correctness_metrics.compute_hierarchical_bootstrap_cis",
            lambda **_kwargs: {},
        )
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_conftest.make_dm_config(),
            target_fpr_values=[0.05],
        )
        eval_class_balance = correctness_types.ClassBalanceStats(
            overall_positive_rate=0.5,
            per_sample_positive_rates=[1.0, 0.0],
            num_all_correct_samples=1,
            num_all_incorrect_samples=1,
            code_type_proportions={"original": 1.0},
            predict_type_proportions={},
        )
        aggregated = correctness_impl._aggregate_runs(
            per_run_results=[run_result],
            per_run_records=[test_records],
            per_run_scores=[test_scores],
            per_run_thresholds={0.05: [0.5]},
            guardrail_splits=splits,
            eval_class_balance=eval_class_balance,
            config=config,
        )
        assert "fpr_0_05/cost_accuracy_rank_correlation" in aggregated.cross_run_mean
        assert aggregated.cross_run_mean["fpr_0_05/cost_accuracy_rank_correlation"] == pytest.approx(0.42)
        assert "fpr_0_05/cost_difficulty_rank_correlation" in aggregated.cross_run_mean
        assert aggregated.cross_run_mean["fpr_0_05/cost_difficulty_rank_correlation"] == pytest.approx(-0.15)


class TestAttemptMetadata:
    """Tests for scorer per-attempt metadata alignment and persistence in run results."""

    @pytest.mark.asyncio
    async def test_attempt_metadata_propagates_to_single_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        valid_records = [
            correctness_conftest.make_record("v0/s0/t0", "v0", 0, True),
            correctness_conftest.make_record("v0/s0/t0", "v0", 1, False),
            correctness_conftest.make_record("v1/s0/t0", "v1", 0, True),
            correctness_conftest.make_record("v1/s0/t0", "v1", 1, False),
        ]
        dm = correctness_conftest.build_mock_datamodule(monkeypatch, valid_records=valid_records, test_records=[])
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_conftest.make_dm_config(),
            target_fpr_values=[0.05],
            num_bootstrap_replicates=5,
            roc_fpr_grid_size=10,
        )

        class _MetadataScorer:
            def score_records(
                self,
                records: list[correctness_types.EvalRecord],
            ) -> correctness_types.ScoringResult:
                return correctness_types.ScoringResult(
                    scores=[0.9 if record.label else 0.1 for record in records],
                    attempt_metadata={
                        (record.sample_id, record.attempt_index, draw_index): {"seen": True}
                        for draw_index, record in enumerate(records)
                    },
                )

            def get_metadata(self) -> dict[str, typing.Any]:
                return {"name": "metadata"}

            def get_verification_cost_unit(self) -> str | None:
                return None

        result = await correctness_impl.evaluate_guardrail_replicas(
            config=config,
            guardrails=[_MetadataScorer()],
            datamodule=dm,
            eval_subset_name="guardrail_valid",
        )
        single_run = result.aggregated.per_run[0]
        assert single_run.attempt_metadata is not None
        expected_keys = {
            (record.sample_id, record.attempt_index, draw_index) for draw_index, record in enumerate(valid_records)
        }
        assert set(single_run.attempt_metadata.keys()) == expected_keys
        assert len(single_run.attempt_records) == len(valid_records)
        first_attempt = single_run.attempt_records[0]
        assert first_attempt.sample_id == valid_records[0].sample_id
        assert first_attempt.draw_index == 0
        assert first_attempt.score == pytest.approx(0.9)
        assert first_attempt.attempt_metadata is not None
        assert first_attempt.attempt_metadata["seen"] is True
        assert result.aggregated.attempt_records_by_key is not None
        first_key = (valid_records[0].sample_id, valid_records[0].attempt_index, 0)
        assert first_key in result.aggregated.attempt_records_by_key

    @pytest.mark.asyncio
    async def test_attempt_metadata_mismatch_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        valid_records = [
            correctness_conftest.make_record("v0/s0/t0", "v0", 0, True),
            correctness_conftest.make_record("v0/s0/t0", "v0", 1, False),
        ]
        dm = correctness_conftest.build_mock_datamodule(monkeypatch, valid_records=valid_records, test_records=[])
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_conftest.make_dm_config(),
            target_fpr_values=[0.05],
            num_bootstrap_replicates=5,
            roc_fpr_grid_size=10,
        )

        class _BadMetadataScorer:
            def score_records(
                self,
                records: list[correctness_types.EvalRecord],
            ) -> correctness_types.ScoringResult:
                return correctness_types.ScoringResult(
                    scores=[0.5 for _ in records],
                    attempt_metadata={
                        (record.sample_id, record.attempt_index, draw_index + 100): {"bad": True}
                        for draw_index, record in enumerate(records)
                    },
                )

            def get_metadata(self) -> dict[str, typing.Any]:
                return {"name": "bad_metadata"}

            def get_verification_cost_unit(self) -> str | None:
                return None

        with pytest.raises(ValueError, match="attempt_metadata keys do not align"):
            await correctness_impl.evaluate_guardrail_replicas(
                config=config,
                guardrails=[_BadMetadataScorer()],
                datamodule=dm,
                eval_subset_name="guardrail_valid",
            )

    @pytest.mark.asyncio
    async def test_attempt_records_populated_without_attempt_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        valid_records = [
            correctness_conftest.make_record("v0/s0/t0", "v0", 0, True),
            correctness_conftest.make_record("v0/s0/t0", "v0", 1, False),
        ]
        dm = correctness_conftest.build_mock_datamodule(monkeypatch, valid_records=valid_records, test_records=[])
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_conftest.make_dm_config(),
            target_fpr_values=[0.05],
            num_bootstrap_replicates=5,
            roc_fpr_grid_size=10,
        )

        class _ScoreOnlyScorer:
            def score_records(
                self,
                records: list[correctness_types.EvalRecord],
            ) -> correctness_types.ScoringResult:
                return correctness_types.ScoringResult(scores=[0.8 if record.label else 0.2 for record in records])

            def get_metadata(self) -> dict[str, typing.Any]:
                return {"name": "score_only"}

            def get_verification_cost_unit(self) -> str | None:
                return None

        result = await correctness_impl.evaluate_guardrail_replicas(
            config=config,
            guardrails=[_ScoreOnlyScorer()],
            datamodule=dm,
            eval_subset_name="guardrail_valid",
        )
        single_run = result.aggregated.per_run[0]
        assert len(single_run.attempt_records) == len(valid_records)
        assert all(record.attempt_metadata is None for record in single_run.attempt_records)
        assert single_run.attempt_records[0].score == pytest.approx(0.8)
        assert single_run.attempt_records[1].score == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_calibration_resampling_with_replacement_supports_attempt_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        valid_records = [
            correctness_conftest.make_record("v0/s0/t0", "v0", 0, True),
            correctness_conftest.make_record("v1/s0/t0", "v1", 0, False),
        ]
        dm = correctness_conftest.build_mock_datamodule(monkeypatch, valid_records=valid_records, test_records=[])
        resampling_config = correctness_types.RecordResamplingConfig(
            target_positive_ratio=0.9,
            strategy="oversample",
            max_records=4,
            min_records_per_label=1,
            seed=0,
        )
        calibration_records = dm.get_records_for_calibration(resampling_config=resampling_config)
        base_keys = [(record.sample_id, record.attempt_index) for record in calibration_records]
        assert len(base_keys) > len(set(base_keys))  # duplicate base attempts due to replacement
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_conftest.make_dm_config(),
            target_fpr_values=[0.05],
            num_bootstrap_replicates=5,
            roc_fpr_grid_size=10,
            calibration_resampling=resampling_config,
        )

        class _MetadataScorer:
            def score_records(
                self,
                records: list[correctness_types.EvalRecord],
            ) -> correctness_types.ScoringResult:
                return correctness_types.ScoringResult(
                    scores=[0.8 if record.label else 0.2 for record in records],
                    attempt_metadata={
                        (record.sample_id, record.attempt_index, draw_index): {"seen": True}
                        for draw_index, record in enumerate(records)
                    },
                )

            def get_metadata(self) -> dict[str, typing.Any]:
                return {"name": "metadata"}

            def get_verification_cost_unit(self) -> str | None:
                return None

        result = await correctness_impl.evaluate_guardrail_replicas(
            config=config,
            guardrails=[_MetadataScorer()],
            datamodule=dm,
            eval_subset_name="guardrail_valid",
        )
        assert result.aggregated.per_run[0].attempt_metadata is not None


class TestAggregateDifficultyStats:
    """Tests for _aggregate_difficulty_stats cross-run averaging."""

    def _make_difficulty_stats(
        self,
        per_bucket_auroc: dict[str, float | None] | None = None,
        correlation: float | None = None,
    ) -> correctness_types.DifficultyStats:
        return correctness_types.DifficultyStats(
            bucket_boundaries=(0.33, 0.67),
            per_bucket_auroc=per_bucket_auroc,
            per_bucket_tpr=None,
            per_bucket_sample_count={"easy": 10, "medium": 10, "hard": 10},
            difficulty_accuracy_rank_correlation=correlation,
        )

    def _make_single_run(
        self,
        difficulty_stats: correctness_types.DifficultyStats | None,
    ) -> correctness_types.SingleRunResult:
        empty_grid = np.array([], dtype=np.float64)
        return correctness_types.SingleRunResult(
            guardrail_metadata={},
            threshold_free=correctness_types.ThresholdFreeMetrics(
                auroc=None,
                average_precision=None,
                tpr_at_fpr=None,
                fpr_grid=empty_grid,
                tpr_grid=empty_grid,
                precision_grid=empty_grid,
                recall_grid=empty_grid,
            ),
            attempt_metrics={},
            sample_metrics={},
            category_results={},
            bootstrap_cis={},
            difficulty_stats=difficulty_stats,
            verification_cost_stats=None,
        )

    def test_normal_averaging(self) -> None:
        stats_a = self._make_difficulty_stats(
            per_bucket_auroc={"easy": 0.9, "medium": 0.8, "hard": 0.7},
            correlation=0.5,
        )
        stats_b = self._make_difficulty_stats(
            per_bucket_auroc={"easy": 0.8, "medium": 0.7, "hard": 0.6},
            correlation=0.3,
        )
        runs = [self._make_single_run(stats_a), self._make_single_run(stats_b)]
        result = correctness_impl._aggregate_difficulty_stats(runs)
        assert result is not None
        assert result.per_bucket_auroc is not None
        assert result.per_bucket_auroc["easy"] == pytest.approx(0.85)
        assert result.per_bucket_auroc["medium"] == pytest.approx(0.75)
        assert result.per_bucket_auroc["hard"] == pytest.approx(0.65)
        assert result.difficulty_accuracy_rank_correlation == pytest.approx(0.4)

    def test_runs_missing_data_skipped(self) -> None:
        stats_with = self._make_difficulty_stats(
            per_bucket_auroc={"easy": 0.9, "medium": 0.8, "hard": 0.7},
            correlation=0.5,
        )
        runs = [self._make_single_run(stats_with), self._make_single_run(None)]
        result = correctness_impl._aggregate_difficulty_stats(runs)
        assert result is not None
        assert result.per_bucket_auroc is not None
        assert result.per_bucket_auroc["easy"] == pytest.approx(0.9)
        assert result.difficulty_accuracy_rank_correlation == pytest.approx(0.5)

    def test_all_none_returns_none(self) -> None:
        runs = [self._make_single_run(None), self._make_single_run(None)]
        result = correctness_impl._aggregate_difficulty_stats(runs)
        assert result is None

    def test_none_bucket_values_handled(self) -> None:
        stats_a = self._make_difficulty_stats(
            per_bucket_auroc={"easy": 0.9, "medium": None, "hard": 0.7},
            correlation=None,
        )
        stats_b = self._make_difficulty_stats(
            per_bucket_auroc={"easy": 0.8, "medium": 0.6, "hard": None},
            correlation=None,
        )
        runs = [self._make_single_run(stats_a), self._make_single_run(stats_b)]
        result = correctness_impl._aggregate_difficulty_stats(runs)
        assert result is not None
        assert result.per_bucket_auroc is not None
        assert result.per_bucket_auroc["easy"] == pytest.approx(0.85)
        assert result.per_bucket_auroc["medium"] == pytest.approx(0.6)
        assert result.per_bucket_auroc["hard"] == pytest.approx(0.7)
        assert result.difficulty_accuracy_rank_correlation is None
