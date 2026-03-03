"""Tests for evaluate_guardrail_types() in pyine.evals.correctness._impl."""

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

_FAKE_LMDB_PATH = pathlib.Path("/fake/lmdb")


class _MockGuardrailScorer:
    def __init__(self, noise_seed: int = 0) -> None:
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
) -> correctness_types.EvalRecord:
    return correctness_types.EvalRecord(
        sample_id=sample_id,
        problem_id=problem_id,
        attempt_index=attempt_index,
        model_output="output",
        final_answer=None,
        expected_output="expected",
        label=label,
        code_type="original",
        tags=[],
        record={},
        difficulty_score=None,
    )


def _build_mock_datamodule(monkeypatch: pytest.MonkeyPatch) -> correctness_datamodule.CorrectnessDataModule:
    """Build a mock datamodule with minimal data."""
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
    splits = correctness_splits.GuardrailSplits(
        guardrail_train=[],
        guardrail_valid=valid_records,
        guardrail_test=test_records,
        train_problem_ids=frozenset(),
        valid_problem_ids=frozenset(f"TACO/TRAIN/p{idx:06d}" for idx in range(3)),
        test_problem_ids=frozenset(f"TACO/TRAIN/p{idx:06d}" for idx in range(3, 6)),
    )
    monkeypatch.setattr(
        "pyine.data.utils.lmdb_io.resolve_lmdb_paths",
        lambda raw_paths: list(raw_paths),
    )
    monkeypatch.setattr(
        "pyine.evals.correctness.datamodule.correctness_data_loading.load_records_from_lmdb",
        lambda *_args, **_kwargs: valid_records + test_records,
    )
    monkeypatch.setattr(
        "pyine.evals.correctness.datamodule.correctness_splits.build_guardrail_splits",
        lambda *_args, **_kwargs: splits,
    )
    dm_config = correctness_datamodule_configs.CorrectnessDataModuleConfig(
        lmdb_paths=(_FAKE_LMDB_PATH,),
        split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
    )
    dm = correctness_datamodule.CorrectnessDataModule(dm_config)
    dm.prepare_data()
    dm.setup()
    return dm


class TestEvaluateGuardrailTypes:
    @pytest.mark.asyncio
    async def test_multiple_types_returns_independent_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dm = _build_mock_datamodule(monkeypatch)
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_datamodule_configs.CorrectnessDataModuleConfig(
                lmdb_paths=(_FAKE_LMDB_PATH,),
                split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
            ),
            target_fpr_values=[0.05],
            num_bootstrap_replicates=5,
            roc_fpr_grid_size=10,
        )
        results = await correctness_impl.evaluate_guardrail_types(
            config=config,
            guardrails_by_type={
                "type_a": [_MockGuardrailScorer(noise_seed=0)],
                "type_b": [_MockGuardrailScorer(noise_seed=1)],
            },
            datamodule=dm,
            eval_subset_name="guardrail_valid",
        )
        assert "type_a" in results
        assert "type_b" in results
        assert isinstance(results["type_a"], correctness_impl.CorrectnessEvalResult)
        assert isinstance(results["type_b"], correctness_impl.CorrectnessEvalResult)

    @pytest.mark.asyncio
    async def test_single_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dm = _build_mock_datamodule(monkeypatch)
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_datamodule_configs.CorrectnessDataModuleConfig(
                lmdb_paths=(_FAKE_LMDB_PATH,),
                split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
            ),
            target_fpr_values=[0.05],
            num_bootstrap_replicates=5,
            roc_fpr_grid_size=10,
        )
        results = await correctness_impl.evaluate_guardrail_types(
            config=config,
            guardrails_by_type={"only_type": [_MockGuardrailScorer()]},
            datamodule=dm,
            eval_subset_name="guardrail_valid",
        )
        assert len(results) == 1
        assert "only_type" in results

    @pytest.mark.asyncio
    async def test_empty_replicas_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dm = _build_mock_datamodule(monkeypatch)
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_datamodule_configs.CorrectnessDataModuleConfig(
                lmdb_paths=(_FAKE_LMDB_PATH,),
                split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
            ),
            target_fpr_values=[0.05],
            num_bootstrap_replicates=5,
            roc_fpr_grid_size=10,
        )
        with pytest.raises(ValueError, match="empty replicas list"):
            await correctness_impl.evaluate_guardrail_types(
                config=config,
                guardrails_by_type={"bad_type": []},
                datamodule=dm,
                eval_subset_name="guardrail_valid",
            )

    @pytest.mark.asyncio
    async def test_wandb_metric_prefixing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dm = _build_mock_datamodule(monkeypatch)
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_datamodule_configs.CorrectnessDataModuleConfig(
                lmdb_paths=(_FAKE_LMDB_PATH,),
                split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
            ),
            target_fpr_values=[0.05],
            num_bootstrap_replicates=5,
            roc_fpr_grid_size=10,
        )
        mock_wandb_run = unittest.mock.MagicMock()
        mock_wandb_run.summary = {}
        await correctness_impl.evaluate_guardrail_types(
            config=config,
            guardrails_by_type={"my_type": [_MockGuardrailScorer()]},
            datamodule=dm,
            eval_subset_name="guardrail_valid",
            wandb_run=mock_wandb_run,
        )
        # check that metrics were logged with the type name prefix
        assert any(key.startswith("benchmark/guardrail_valid/my_type/") for key in mock_wandb_run.summary)
