"""End-to-end tests for pyine.evals.correctness._impl."""

from __future__ import annotations

import pathlib
import typing
import unittest.mock

import numpy as np
import pytest

import pyine.evals.correctness._impl as correctness_impl
import pyine.evals.correctness.configs as correctness_configs
import pyine.evals.correctness.types as correctness_types


class _MockGuardrailScorer:
    """Scorer that returns predetermined scores based on labels."""

    def __init__(
        self,
        noise_seed: int = 0,
    ) -> None:
        self._noise_seed = noise_seed

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        rng = np.random.default_rng(self._noise_seed)
        scores: list[float] = []
        for rec in records:
            base = 0.8 if rec.label else 0.2
            scores.append(float(np.clip(base + rng.normal(0, 0.1), 0.0, 1.0)))
        return correctness_types.ScoringResult(scores=scores)

    def get_metadata(self) -> dict[str, typing.Any]:
        return {"name": "mock", "seed": self._noise_seed}

    def get_verification_cost_unit(self) -> str | None:
        return None


def _make_record(
    sample_id: str,
    problem_id: str,
    attempt_index: int,
    label: bool,
    code_type: str = "original",
) -> correctness_types.EvalRecord:
    return correctness_types.EvalRecord(
        sample_id=sample_id,
        problem_id=problem_id,
        attempt_index=attempt_index,
        model_output="output",
        final_answer=None,
        expected_output="expected",
        label=label,
        code_type=code_type,
        tags=[],
        record={},
        difficulty_score=None,
    )


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
        # build synthetic LMDB records: 20 problems, 3 attempts each
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
        # mock split result: problems 0-9 valid, 10-19 test (no train in LMDB)
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
        config = correctness_configs.CorrectnessEvalsConfig(
            lmdb_paths=[pathlib.Path("/fake/path")],
            split_config=correctness_configs.GuardrailSplitConfig(
                split_source="TACO",
                guardrail_valid_fraction=0.5,
                seed=42,
            ),
            target_fpr_values=[0.01, 0.05],
            num_bootstrap_replicates=20,  # small for speed
            roc_fpr_grid_size=50,
        )
        scorer = _MockGuardrailScorer(noise_seed=0)
        result = await correctness_impl.evaluate_wrapped_model_impl(
            config=config,
            guardrails=[scorer],
            verbose=True,
        )
        # verify structure
        assert isinstance(result, correctness_impl.CorrectnessEvalResult)
        agg = result.aggregated
        assert len(agg.per_run) == 1
        assert agg.class_balance.overall_positive_rate > 0.0
        # verify metrics exist
        single_run = agg.per_run[0]
        assert single_run.threshold_free.auroc is not None
        assert 0.01 in single_run.attempt_metrics
        assert 0.05 in single_run.attempt_metrics
        assert 0.01 in single_run.sample_metrics
        # verify flat dict has expected keys
        flat = agg.to_flat_dict()
        assert "auroc/mean" in flat
        assert "class_balance/overall_positive_rate" in flat
        assert "record_count" in flat
        assert "auroc/num_valid_runs" in flat
        # verify base_pass_rate is aggregated
        assert any(key.endswith("/base_pass_rate/mean") for key in flat)
        # verify guardrail metadata
        assert single_run.guardrail_metadata["name"] == "mock"

    @pytest.mark.asyncio
    async def test_empty_guardrails_raises(self) -> None:
        config = correctness_configs.CorrectnessEvalsConfig(
            lmdb_paths=[],
            split_config=correctness_configs.GuardrailSplitConfig(split_source="TACO"),
        )
        with pytest.raises(ValueError, match="guardrails must not be empty"):
            await correctness_impl.evaluate_wrapped_model_impl(config=config, guardrails=[])

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

        # use 20 valid + 20 test problems with 4 attempts each for enough calibration data
        lmdb_records: list[dict[str, typing.Any]] = []
        for prob_idx in range(40):
            for attempt_idx in range(4):
                label = attempt_idx < 2  # 50% correct
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
        config = correctness_configs.CorrectnessEvalsConfig(
            lmdb_paths=[pathlib.Path("/fake/path")],
            split_config=correctness_configs.GuardrailSplitConfig(
                split_source="TACO",
                guardrail_valid_fraction=0.5,
                seed=42,
            ),
            target_fpr_values=[0.1],  # relaxed FPR to allow some through
            num_bootstrap_replicates=10,
            roc_fpr_grid_size=20,
        )
        result = await correctness_impl.evaluate_wrapped_model_impl(
            config=config,
            guardrails=[_PerfectScorer()],
        )
        agg = result.aggregated
        single_run = agg.per_run[0]
        assert single_run.threshold_free.auroc == pytest.approx(1.0)
        att = single_run.attempt_metrics[0.1]
        assert att.tpr == pytest.approx(1.0)
        assert att.fpr == pytest.approx(0.0)


class TestSplitValidation:
    """Tests for split validation checks in evaluate_wrapped_model_impl."""

    @pytest.mark.asyncio
    async def test_empty_records_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "pyine.evals.correctness._impl.correctness_data_loading.load_records_from_lmdb",
            lambda *_args, **_kwargs: [],
        )
        config = correctness_configs.CorrectnessEvalsConfig(
            lmdb_paths=[pathlib.Path("/fake")],
            split_config=correctness_configs.GuardrailSplitConfig(split_source="TACO"),
        )
        with pytest.raises(ValueError, match="no records loaded"):
            await correctness_impl.evaluate_wrapped_model_impl(
                config=config,
                guardrails=[_MockGuardrailScorer()],
            )

    @pytest.mark.asyncio
    async def test_empty_valid_split_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        splits = correctness_types.GuardrailSplits(
            guardrail_train=[],
            guardrail_valid=[],
            guardrail_test=[_make_record("s1", "p1", 0, True)],
            train_problem_ids=frozenset(),
            valid_problem_ids=frozenset(),
            test_problem_ids=frozenset({"p1"}),
        )
        monkeypatch.setattr(
            "pyine.evals.correctness._impl.correctness_data_loading.load_records_from_lmdb",
            lambda *_args, **_kwargs: [_make_record("s1", "p1", 0, True)],
        )
        monkeypatch.setattr(
            "pyine.evals.correctness._impl.correctness_splits.build_guardrail_splits",
            lambda *_args, **_kwargs: splits,
        )
        config = correctness_configs.CorrectnessEvalsConfig(
            lmdb_paths=[pathlib.Path("/fake")],
            split_config=correctness_configs.GuardrailSplitConfig(split_source="TACO"),
        )
        scorer = _MockGuardrailScorer()
        with pytest.raises(ValueError, match="guardrail_valid split is empty"):
            await correctness_impl.evaluate_wrapped_model_impl(config=config, guardrails=[scorer])

    @pytest.mark.asyncio
    async def test_empty_test_split_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        splits = correctness_types.GuardrailSplits(
            guardrail_train=[],
            guardrail_valid=[_make_record("s1", "p1", 0, True)],
            guardrail_test=[],
            train_problem_ids=frozenset(),
            valid_problem_ids=frozenset({"p1"}),
            test_problem_ids=frozenset(),
        )
        monkeypatch.setattr(
            "pyine.evals.correctness._impl.correctness_data_loading.load_records_from_lmdb",
            lambda *_args, **_kwargs: [_make_record("s1", "p1", 0, True)],
        )
        monkeypatch.setattr(
            "pyine.evals.correctness._impl.correctness_splits.build_guardrail_splits",
            lambda *_args, **_kwargs: splits,
        )
        config = correctness_configs.CorrectnessEvalsConfig(
            lmdb_paths=[pathlib.Path("/fake")],
            split_config=correctness_configs.GuardrailSplitConfig(split_source="TACO"),
        )
        scorer = _MockGuardrailScorer()
        with pytest.raises(ValueError, match="guardrail_test split is empty"):
            await correctness_impl.evaluate_wrapped_model_impl(config=config, guardrails=[scorer])

    @pytest.mark.asyncio
    async def test_single_class_valid_split_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # all valid records have label=True (no negatives for calibration)
        valid_records = [
            _make_record("s1", "p1", 0, True),
            _make_record("s1", "p1", 1, True),
        ]
        splits = correctness_types.GuardrailSplits(
            guardrail_train=[],
            guardrail_valid=valid_records,
            guardrail_test=[_make_record("s2", "p2", 0, False)],
            train_problem_ids=frozenset(),
            valid_problem_ids=frozenset({"p1"}),
            test_problem_ids=frozenset({"p2"}),
        )
        monkeypatch.setattr(
            "pyine.evals.correctness._impl.correctness_data_loading.load_records_from_lmdb",
            lambda *_args, **_kwargs: valid_records,
        )
        monkeypatch.setattr(
            "pyine.evals.correctness._impl.correctness_splits.build_guardrail_splits",
            lambda *_args, **_kwargs: splits,
        )
        config = correctness_configs.CorrectnessEvalsConfig(
            lmdb_paths=[pathlib.Path("/fake")],
            split_config=correctness_configs.GuardrailSplitConfig(split_source="TACO"),
        )
        scorer = _MockGuardrailScorer()
        with pytest.raises(ValueError, match="only one label class"):
            await correctness_impl.evaluate_wrapped_model_impl(config=config, guardrails=[scorer])


class TestFastEndToEnd:
    """Fast E2E test that runs in the default suite (not slow)."""

    @pytest.mark.asyncio
    async def test_mini_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Minimal E2E: mocked LMDB/splits, tiny bootstrap, verify result structure."""
        # build minimal records: 3 valid problems + 3 test problems, 2 attempts each
        valid_records = []
        test_records = []
        for prob_idx in range(3):
            for attempt_idx in range(2):
                valid_records.append(
                    _make_record(
                        f"TACO/TRAIN/p{prob_idx:06d}/s0000/t0000",
                        f"TACO/TRAIN/p{prob_idx:06d}",
                        attempt_idx,
                        label=(attempt_idx == 0),
                    )
                )
        for prob_idx in range(3, 6):
            for attempt_idx in range(2):
                test_records.append(
                    _make_record(
                        f"TACO/TRAIN/p{prob_idx:06d}/s0000/t0000",
                        f"TACO/TRAIN/p{prob_idx:06d}",
                        attempt_idx,
                        label=(attempt_idx == 0),
                    )
                )
        splits = correctness_types.GuardrailSplits(
            guardrail_train=[],
            guardrail_valid=valid_records,
            guardrail_test=test_records,
            train_problem_ids=frozenset(),
            valid_problem_ids=frozenset(f"TACO/TRAIN/p{idx:06d}" for idx in range(3)),
            test_problem_ids=frozenset(f"TACO/TRAIN/p{idx:06d}" for idx in range(3, 6)),
        )
        monkeypatch.setattr(
            "pyine.evals.correctness._impl.correctness_data_loading.load_records_from_lmdb",
            lambda *_args, **_kwargs: valid_records + test_records,
        )
        monkeypatch.setattr(
            "pyine.evals.correctness._impl.correctness_splits.build_guardrail_splits",
            lambda *_args, **_kwargs: splits,
        )
        config = correctness_configs.CorrectnessEvalsConfig(
            lmdb_paths=[pathlib.Path("/fake")],
            split_config=correctness_configs.GuardrailSplitConfig(split_source="TACO"),
            target_fpr_values=[0.05],
            num_bootstrap_replicates=5,  # minimal for speed
            roc_fpr_grid_size=10,
        )
        scorer = _MockGuardrailScorer(noise_seed=0)
        result = await correctness_impl.evaluate_wrapped_model_impl(
            config=config,
            guardrails=[scorer],
        )
        assert isinstance(result, correctness_impl.CorrectnessEvalResult)
        flat = result.aggregated.to_flat_dict()
        assert "auroc/mean" in flat
        assert "sample_count" in flat
        assert flat["sample_count"] == 3  # 3 test problems
        # verify category metrics are aggregated (not just first-run)
        assert "category/regular/auroc/mean" in flat
        # verify category sample-level metrics are surfaced
        assert "category/regular/fpr_0_05/guarded_pass_rate/mean" in flat


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
            _make_record("s1", "p1", 0, True),
            _make_record("s2", "p2", 0, False),
        ]
        test_scores = np.array([0.9, 0.1], dtype=np.float64)
        splits = correctness_types.GuardrailSplits(
            guardrail_train=[],
            guardrail_valid=[],
            guardrail_test=test_records,
            train_problem_ids=frozenset(),
            valid_problem_ids=frozenset(),
            test_problem_ids=frozenset({"p1", "p2"}),
        )
        # skip expensive bootstrap and class balance by mocking
        monkeypatch.setattr(
            "pyine.evals.correctness._impl.correctness_metrics.compute_hierarchical_bootstrap_cis",
            lambda **_kwargs: {},
        )
        config = correctness_configs.CorrectnessEvalsConfig(
            lmdb_paths=[],
            split_config=correctness_configs.GuardrailSplitConfig(split_source="TACO"),
            target_fpr_values=[0.05],
        )
        test_class_balance = correctness_types.ClassBalanceStats(
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
            test_class_balance=test_class_balance,
            config=config,
        )
        assert "fpr_0_05/cost_accuracy_rank_correlation" in aggregated.cross_run_mean
        assert aggregated.cross_run_mean["fpr_0_05/cost_accuracy_rank_correlation"] == pytest.approx(0.42)
        assert "fpr_0_05/cost_difficulty_rank_correlation" in aggregated.cross_run_mean
        assert aggregated.cross_run_mean["fpr_0_05/cost_difficulty_rank_correlation"] == pytest.approx(-0.15)
