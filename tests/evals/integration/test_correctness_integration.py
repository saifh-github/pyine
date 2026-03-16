"""Integration tests for the CORRECTNESS evaluation pipeline."""

from __future__ import annotations

import pytest

import pyine.evals.correctness._impl as correctness_impl
import pyine.evals.correctness.configs as correctness_configs
import pyine.evals.correctness.datamodule as correctness_datamodule
import pyine.evals.correctness.types as correctness_types
import tests.evals.integration.conftest as integration_conftest
import tests.evals.integration.fake_models as fake_models


def _make_eval_config() -> correctness_configs.CorrectnessEvalsConfig:
    return correctness_configs.CorrectnessEvalsConfig(
        datamodule_config=integration_conftest.make_dm_config(),
        target_fpr_values=[0.05, 0.1],
        num_bootstrap_replicates=10,
        roc_fpr_grid_size=20,
        bootstrap_num_workers=1,  # sequential; avoids process pool overhead for tiny workloads
    )


async def _run_scorer(
    scorer: correctness_types.GuardrailScorer,
    dm: correctness_datamodule.CorrectnessDataModule,
    eval_subset_name: str = "guardrail_valid",
) -> correctness_impl.CorrectnessEvalResult:
    return await correctness_impl.evaluate_guardrail_replicas(
        config=_make_eval_config(),
        guardrails=[scorer],
        datamodule=dm,
        eval_subset_name=eval_subset_name,
    )


class TestCorrectnessOracleScorer:
    @pytest.mark.asyncio
    async def test_oracle_achieves_perfect_auroc(
        self,
        monkeypatch: pytest.MonkeyPatch,
        correctness_eval_records: tuple[
            list[correctness_types.EvalRecord],
            list[correctness_types.EvalRecord],
        ],
    ) -> None:
        valid_records, test_records = correctness_eval_records
        dm = integration_conftest.build_mock_correctness_datamodule(monkeypatch, valid_records, test_records)
        result = await _run_scorer(fake_models.OracleGuardrailScorer(), dm)
        agg = result.aggregated
        single_run = agg.per_run[0]
        assert single_run.threshold_free.auroc == pytest.approx(1.0)
        assert single_run.threshold_free.average_precision == pytest.approx(1.0)
        for target_fpr in [0.05, 0.1]:
            att = single_run.attempt_metrics[target_fpr]
            assert att.tpr == pytest.approx(1.0)
            assert att.fpr == pytest.approx(0.0)
            samp = single_run.sample_metrics[target_fpr]
            assert samp.guarded_pass_rate == pytest.approx(samp.base_pass_rate)
        # verification cost metrics (oracle uses fixed cost of 1.0 per record)
        assert single_run.verification_cost_stats is not None
        for target_fpr in [0.05, 0.1]:
            cost_stats = single_run.verification_cost_stats[target_fpr]
            assert cost_stats.cost_unit == "tokens"
            assert cost_stats.total_cost == pytest.approx(60.0)  # 60 records x 1.0
            assert cost_stats.mean_cost_per_record == pytest.approx(1.0)
            assert cost_stats.std_cost_per_record == pytest.approx(0.0)
        # bootstrap CIs: perfect oracle should have tight CI around 1.0
        assert "auroc" in single_run.bootstrap_cis
        auroc_ci = single_run.bootstrap_cis["auroc"]
        assert auroc_ci.point_estimate == pytest.approx(1.0)
        assert auroc_ci.lower_bound == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_oracle_flat_dict_structure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        correctness_eval_records: tuple[
            list[correctness_types.EvalRecord],
            list[correctness_types.EvalRecord],
        ],
    ) -> None:
        valid_records, test_records = correctness_eval_records
        dm = integration_conftest.build_mock_correctness_datamodule(monkeypatch, valid_records, test_records)
        result = await _run_scorer(fake_models.OracleGuardrailScorer(), dm)
        flat = result.aggregated.to_flat_dict()
        assert "auroc/mean" in flat
        assert "average_precision/mean" in flat
        assert "class_balance/overall_positive_rate" in flat
        assert "sample_count" in flat
        assert "record_count" in flat
        # verification cost keys appear in the flat dict via cross_run_mean
        assert "fpr_0_05/cost_total/mean" in flat
        assert "fpr_0_05/cost_mean/mean" in flat
        assert "fpr_0_1/cost_total/mean" in flat
        # additional cost keys
        assert "fpr_0_05/cost_std/mean" in flat
        assert "fpr_0_05/cost_median/mean" in flat
        # bootstrap CI keys
        assert "auroc/bootstrap_ci_lower" in flat
        assert "auroc/bootstrap_ci_upper" in flat
        assert "auroc/bootstrap_ci_point" in flat
        # cross-run aggregation keys for sample-level metrics
        assert "fpr_0_05/guarded_pass_rate/mean" in flat
        assert "fpr_0_1/tpr/mean" in flat
        # run validity
        assert "auroc/num_valid_runs" in flat


class TestCorrectnessInverseOracleScorer:
    @pytest.mark.asyncio
    async def test_inverse_oracle_achieves_worst_auroc(
        self,
        monkeypatch: pytest.MonkeyPatch,
        correctness_eval_records: tuple[
            list[correctness_types.EvalRecord],
            list[correctness_types.EvalRecord],
        ],
    ) -> None:
        valid_records, test_records = correctness_eval_records
        dm = integration_conftest.build_mock_correctness_datamodule(monkeypatch, valid_records, test_records)
        result = await _run_scorer(fake_models.InverseOracleGuardrailScorer(), dm)
        agg = result.aggregated
        single_run = agg.per_run[0]
        assert single_run.threshold_free.auroc == pytest.approx(0.0)
        for target_fpr in [0.05, 0.1]:
            att = single_run.attempt_metrics[target_fpr]
            assert att.tpr == pytest.approx(0.0)
            samp = single_run.sample_metrics[target_fpr]
            assert samp.total_block_rate == pytest.approx(1.0)
            assert samp.guarded_pass_rate == pytest.approx(0.0)
        # verification cost metrics (inverse oracle uses fixed cost of 2.0 per record)
        assert single_run.verification_cost_stats is not None
        for target_fpr in [0.05, 0.1]:
            cost_stats = single_run.verification_cost_stats[target_fpr]
            assert cost_stats.cost_unit == "tokens"
            assert cost_stats.total_cost == pytest.approx(120.0)  # 60 records x 2.0
            assert cost_stats.mean_cost_per_record == pytest.approx(2.0)
            assert cost_stats.std_cost_per_record == pytest.approx(0.0)
        # bootstrap CIs: inverse oracle should have CI tight around 0.0
        assert "auroc" in single_run.bootstrap_cis
        auroc_ci = single_run.bootstrap_cis["auroc"]
        assert auroc_ci.point_estimate == pytest.approx(0.0)
        assert auroc_ci.upper_bound == pytest.approx(0.0)


class TestCorrectnessRandomScorer:
    @pytest.mark.asyncio
    async def test_random_scorer_achieves_approximately_chance_auroc(
        self,
        monkeypatch: pytest.MonkeyPatch,
        correctness_eval_records: tuple[
            list[correctness_types.EvalRecord],
            list[correctness_types.EvalRecord],
        ],
    ) -> None:
        valid_records, test_records = correctness_eval_records
        dm = integration_conftest.build_mock_correctness_datamodule(monkeypatch, valid_records, test_records)
        result = await _run_scorer(fake_models.RandomGuardrailScorer(seed=42), dm)
        agg = result.aggregated
        single_run = agg.per_run[0]
        assert single_run.threshold_free.auroc is not None
        assert 0.3 <= single_run.threshold_free.auroc <= 0.7
        # verification costs are random in [0, 5]; just check they are present and positive
        assert single_run.verification_cost_stats is not None
        for target_fpr in [0.05, 0.1]:
            cost_stats = single_run.verification_cost_stats[target_fpr]
            assert cost_stats.cost_unit == "tokens"
            assert cost_stats.total_cost is not None and cost_stats.total_cost > 0
            assert cost_stats.mean_cost_per_record is not None and cost_stats.mean_cost_per_record > 0
        # cost keys should appear in the flat dict and be positive
        flat = result.aggregated.to_flat_dict()
        assert flat["fpr_0_05/cost_total/mean"] > 0
        assert flat["fpr_0_1/cost_mean/mean"] > 0


class TestCorrectnessTestSubset:
    @pytest.mark.asyncio
    async def test_guardrail_test_subset_produces_valid_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        correctness_eval_records: tuple[
            list[correctness_types.EvalRecord],
            list[correctness_types.EvalRecord],
        ],
    ) -> None:
        """Eval on guardrail_test (calibrated on guardrail_valid) produces consistent counts."""
        valid_records, test_records = correctness_eval_records
        dm = integration_conftest.build_mock_correctness_datamodule(monkeypatch, valid_records, test_records)
        result = await _run_scorer(fake_models.OracleGuardrailScorer(), dm, eval_subset_name="guardrail_test")
        flat = result.aggregated.to_flat_dict()
        assert flat["sample_count"] == 20  # 20 test problems
        assert flat["record_count"] == 60  # 20 problems x 3 attempts


class TestCorrectnessMultiReplica:
    @pytest.mark.asyncio
    async def test_two_identical_replicas_have_zero_std(
        self,
        monkeypatch: pytest.MonkeyPatch,
        correctness_eval_records: tuple[
            list[correctness_types.EvalRecord],
            list[correctness_types.EvalRecord],
        ],
    ) -> None:
        valid_records, test_records = correctness_eval_records
        dm = integration_conftest.build_mock_correctness_datamodule(monkeypatch, valid_records, test_records)
        config = _make_eval_config()
        result = await correctness_impl.evaluate_guardrail_replicas(
            config=config,
            guardrails=[fake_models.OracleGuardrailScorer(), fake_models.OracleGuardrailScorer()],
            datamodule=dm,
            eval_subset_name="guardrail_valid",
        )
        agg = result.aggregated
        assert len(agg.per_run) == 2
        assert agg.cross_run_std["auroc"] == pytest.approx(0.0)
        # zero std for cost metrics across identical replicas
        assert agg.cross_run_std["fpr_0_05/cost_total"] == pytest.approx(0.0)
        # flat dict should include bootstrap CI keys
        flat = agg.to_flat_dict()
        assert "auroc/bootstrap_ci_lower" in flat


class TestCorrectnessPromptedLLMScorer:
    """Integration tests for the FakePromptedLLMScorer through the full eval pipeline."""

    @pytest.mark.asyncio
    async def test_prompted_llm_scorer_produces_valid_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        correctness_eval_records: tuple[
            list[correctness_types.EvalRecord],
            list[correctness_types.EvalRecord],
        ],
    ) -> None:
        """Run through full eval pipeline and verify AggregatedResult structure."""
        valid_records, test_records = correctness_eval_records
        dm = integration_conftest.build_mock_correctness_datamodule(monkeypatch, valid_records, test_records)
        result = await _run_scorer(fake_models.FakePromptedLLMScorer(), dm)
        agg = result.aggregated
        single_run = agg.per_run[0]
        # FakePromptedLLMScorer behaves like an oracle, so expect perfect AUROC
        assert single_run.threshold_free.auroc == pytest.approx(1.0)
        assert single_run.threshold_free.average_precision == pytest.approx(1.0)
        # Verify the flat dict has all expected keys
        flat = agg.to_flat_dict()
        assert "auroc/mean" in flat
        assert "sample_count" in flat
        assert "record_count" in flat

    @pytest.mark.asyncio
    async def test_prompted_llm_scorer_cost_unit_is_tokens(
        self,
        monkeypatch: pytest.MonkeyPatch,
        correctness_eval_records: tuple[
            list[correctness_types.EvalRecord],
            list[correctness_types.EvalRecord],
        ],
    ) -> None:
        """Verify cost_unit in VerificationCostStats is 'tokens'."""
        valid_records, test_records = correctness_eval_records
        dm = integration_conftest.build_mock_correctness_datamodule(monkeypatch, valid_records, test_records)
        result = await _run_scorer(fake_models.FakePromptedLLMScorer(), dm)
        single_run = result.aggregated.per_run[0]
        assert single_run.verification_cost_stats is not None
        for target_fpr in [0.05, 0.1]:
            cost_stats = single_run.verification_cost_stats[target_fpr]
            assert cost_stats.cost_unit == "tokens"
            # FakePromptedLLMScorer uses 150.0 per record, 60 records total
            assert cost_stats.total_cost == pytest.approx(9000.0)  # 60 x 150
            assert cost_stats.mean_cost_per_record == pytest.approx(150.0)
            assert cost_stats.std_cost_per_record == pytest.approx(0.0)
