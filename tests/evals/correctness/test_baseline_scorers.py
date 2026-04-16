"""Tests for baseline (sanity-check) guardrail scorers and their config/integration."""

from __future__ import annotations

import pathlib
import typing

import pydantic
import pytest

import pyine.evals.correctness._impl as correctness_impl
import pyine.evals.correctness.baseline_scorers as baseline_scorers
import pyine.evals.correctness.configs as correctness_configs
import pyine.evals.correctness.datamodule_configs as correctness_datamodule_configs
import pyine.evals.correctness.types as correctness_types
import tests.evals.correctness.conftest as correctness_conftest
from pyine.apps.guardrail_eval.baseline_eval_configs import (
    BaselineEvalAppConfig,
    ConstantBaselineConfig,
    UniformRandomBaselineConfig,
)

_FAKE_LMDB_PATH = pathlib.Path("/fake/lmdb")


def _make_records(count: int = 5) -> list[correctness_types.EvalRecord]:
    return [
        correctness_conftest.make_record(
            sample_id=f"TACO/TRAIN/p{idx:06d}/s0000/t0000",
            problem_id=f"TACO/TRAIN/p{idx:06d}",
            attempt_index=idx % 2,
            label=(idx % 2 == 0),
        )
        for idx in range(count)
    ]


class TestConstantScorer:
    def test_scores_are_constant(self) -> None:
        scorer = baseline_scorers.ConstantScorer(score_value=0.5)
        records = _make_records(10)
        result = scorer.score_records(records)
        assert all(score == 0.5 for score in result.scores)
        assert len(result.scores) == 10

    def test_custom_score_value(self) -> None:
        scorer = baseline_scorers.ConstantScorer(score_value=0.3)
        result = scorer.score_records(_make_records(3))
        assert all(score == 0.3 for score in result.scores)

    def test_no_verification_costs(self) -> None:
        scorer = baseline_scorers.ConstantScorer()
        result = scorer.score_records(_make_records(2))
        assert result.verification_costs is None

    def test_no_verification_cost_unit(self) -> None:
        assert baseline_scorers.ConstantScorer().get_verification_cost_unit() is None

    def test_metadata(self) -> None:
        scorer = baseline_scorers.ConstantScorer(score_value=0.7)
        metadata = scorer.get_metadata()
        assert metadata["scorer_type"] == "constant"
        assert metadata["score_value"] == 0.7


class TestUniformRandomScorer:
    def test_scores_in_unit_interval(self) -> None:
        scorer = baseline_scorers.UniformRandomScorer(seed=42)
        result = scorer.score_records(_make_records(100))
        assert all(0.0 <= score <= 1.0 for score in result.scores)

    def test_order_independence(self) -> None:
        """Same records in different orders produce the same per-record scores."""
        records = _make_records(10)
        reversed_records = list(reversed(records))
        scorer = baseline_scorers.UniformRandomScorer(seed=99)
        result_forward = scorer.score_records(records)
        result_reversed = scorer.score_records(reversed_records)
        forward_by_id = {
            (records[idx].sample_id, records[idx].attempt_index): result_forward.scores[idx]
            for idx in range(len(records))
        }
        reversed_by_id = {
            (reversed_records[idx].sample_id, reversed_records[idx].attempt_index): result_reversed.scores[idx]
            for idx in range(len(reversed_records))
        }
        assert forward_by_id == reversed_by_id

    def test_cross_call_stability(self) -> None:
        """Calling score_records twice on the same records returns identical scores."""
        records = _make_records(10)
        scorer = baseline_scorers.UniformRandomScorer(seed=42)
        result_first = scorer.score_records(records)
        result_second = scorer.score_records(records)
        assert result_first.scores == result_second.scores

    def test_seed_variation(self) -> None:
        """Different seeds produce different scores for the same records."""
        records = _make_records(10)
        scorer_a = baseline_scorers.UniformRandomScorer(seed=1)
        scorer_b = baseline_scorers.UniformRandomScorer(seed=2)
        assert scorer_a.score_records(records).scores != scorer_b.score_records(records).scores

    def test_replica_independence(self) -> None:
        """Adjacent seeds (simulating replicas) produce different scores."""
        records = _make_records(10)
        scorer_0 = baseline_scorers.UniformRandomScorer(seed=42)
        scorer_1 = baseline_scorers.UniformRandomScorer(seed=43)
        assert scorer_0.score_records(records).scores != scorer_1.score_records(records).scores

    def test_no_verification_costs(self) -> None:
        scorer = baseline_scorers.UniformRandomScorer(seed=0)
        result = scorer.score_records(_make_records(2))
        assert result.verification_costs is None

    def test_no_verification_cost_unit(self) -> None:
        assert baseline_scorers.UniformRandomScorer(seed=0).get_verification_cost_unit() is None

    def test_metadata(self) -> None:
        scorer = baseline_scorers.UniformRandomScorer(seed=123)
        metadata = scorer.get_metadata()
        assert metadata["scorer_type"] == "uniform_random"
        assert metadata["seed"] == 123


class TestBaselineConfigValidation:
    def test_constant_config_roundtrip(self) -> None:
        config = ConstantBaselineConfig(name="test_const", score_value=0.3, num_replicas=2)
        assert config.scorer_type == "constant"
        assert config.name == "test_const"
        scorers = config.build_scorers()
        assert len(scorers) == 2

    def test_uniform_random_config_roundtrip(self) -> None:
        config = UniformRandomBaselineConfig(name="test_rng", seed=7, num_replicas=3)
        assert config.scorer_type == "uniform_random"
        scorers = config.build_scorers()
        assert len(scorers) == 3

    def test_discriminated_union_constant(self) -> None:
        """Pydantic discriminated union correctly deserializes a constant config."""
        raw = {"scorer_type": "constant", "name": "c", "score_value": 0.1, "num_replicas": 1}
        app = BaselineEvalAppConfig(
            baseline_configs=[raw],
            evals_config=_make_evals_config(),
        )
        assert isinstance(app.baseline_configs[0], ConstantBaselineConfig)

    def test_discriminated_union_uniform_random(self) -> None:
        """Pydantic discriminated union correctly deserializes a uniform_random config."""
        raw = {"scorer_type": "uniform_random", "name": "r", "seed": 10, "num_replicas": 1}
        app = BaselineEvalAppConfig(
            baseline_configs=[raw],
            evals_config=_make_evals_config(),
        )
        assert isinstance(app.baseline_configs[0], UniformRandomBaselineConfig)

    def test_duplicate_names_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="duplicate baseline names"):
            BaselineEvalAppConfig(
                baseline_configs=[
                    {"scorer_type": "constant", "name": "dup"},
                    {"scorer_type": "uniform_random", "name": "dup"},
                ],
                evals_config=_make_evals_config(),
            )

    def test_extra_fields_rejected_constant(self) -> None:
        """Constant config rejects fields belonging to uniform_random."""
        with pytest.raises(pydantic.ValidationError):
            ConstantBaselineConfig(name="c", seed=42)  # type: ignore[call-arg]

    def test_extra_fields_rejected_uniform_random(self) -> None:
        """Uniform random config rejects fields belonging to constant."""
        with pytest.raises(pydantic.ValidationError):
            UniformRandomBaselineConfig(name="r", score_value=0.5)  # type: ignore[call-arg]


class TestBaselineEvalIntegration:
    def test_evaluate_both_baselines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run both baseline types through evaluate_guardrail_types with mock data."""
        valid_records: list[correctness_types.EvalRecord] = []
        test_records: list[correctness_types.EvalRecord] = []
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
        datamodule = correctness_conftest.build_mock_datamodule(
            monkeypatch, valid_records=valid_records, test_records=test_records
        )
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_datamodule_configs.CorrectnessDataModuleConfig(
                lmdb_paths=(_FAKE_LMDB_PATH,),
                split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
            ),
            target_fpr_values=[0.05],
            num_bootstrap_replicates=5,
            roc_fpr_grid_size=10,
        )
        constant_scorers = ConstantBaselineConfig(name="constant_0.5", num_replicas=2).build_scorers()
        random_scorers = UniformRandomBaselineConfig(name="uniform_random_s42", seed=42, num_replicas=2).build_scorers()
        guardrails_by_type: dict[str, list[correctness_types.GuardrailScorer]] = {
            "constant_0.5": constant_scorers,
            "uniform_random_s42": random_scorers,
        }
        results = correctness_impl.evaluate_guardrail_types(
            config=config,
            guardrails_by_type=guardrails_by_type,
            datamodule=datamodule,
            eval_subset_name="guardrail_valid",
        )
        assert "constant_0.5" in results
        assert "uniform_random_s42" in results
        assert isinstance(results["constant_0.5"], correctness_impl.CorrectnessEvalResult)
        assert isinstance(results["uniform_random_s42"], correctness_impl.CorrectnessEvalResult)

    def test_constant_baseline_auroc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constant scorer with mixed labels should produce AUROC = 0.5."""
        valid_records = [
            correctness_conftest.make_record(
                f"TACO/TRAIN/p{idx:06d}/s0000/t0000",
                f"TACO/TRAIN/p{idx:06d}",
                attempt_idx,
                label=(attempt_idx == 0),
            )
            for idx in range(5)
            for attempt_idx in range(2)
        ]
        test_records = [
            correctness_conftest.make_record(
                f"TACO/TRAIN/p{idx:06d}/s0000/t0000",
                f"TACO/TRAIN/p{idx:06d}",
                attempt_idx,
                label=(attempt_idx == 0),
            )
            for idx in range(5, 10)
            for attempt_idx in range(2)
        ]
        datamodule = correctness_conftest.build_mock_datamodule(
            monkeypatch, valid_records=valid_records, test_records=test_records
        )
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_datamodule_configs.CorrectnessDataModuleConfig(
                lmdb_paths=(_FAKE_LMDB_PATH,),
                split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
            ),
            target_fpr_values=[0.05],
            num_bootstrap_replicates=5,
            roc_fpr_grid_size=10,
        )
        guardrails_by_type: dict[str, list[typing.Any]] = {
            "constant": [baseline_scorers.ConstantScorer(score_value=0.5)],
        }
        results = correctness_impl.evaluate_guardrail_types(
            config=config,
            guardrails_by_type=guardrails_by_type,
            datamodule=datamodule,
            eval_subset_name="guardrail_valid",
        )
        single_run = results["constant"].aggregated.per_run[0]
        assert single_run.threshold_free.auroc == pytest.approx(0.5)


def _make_evals_config() -> correctness_configs.CorrectnessEvalsConfig:
    """Minimal evals config for config-validation tests (not used for actual evaluation)."""
    return correctness_configs.CorrectnessEvalsConfig(
        datamodule_config=correctness_datamodule_configs.CorrectnessDataModuleConfig(
            lmdb_paths=(_FAKE_LMDB_PATH,),
            split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
        ),
        target_fpr_values=[0.05],
    )
