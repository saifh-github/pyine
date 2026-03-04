"""Shared fixtures for pyine.evals.correctness tests."""

from __future__ import annotations

import pathlib
import typing

import numpy as np

import pyine.evals.correctness.datamodule as correctness_datamodule
import pyine.evals.correctness.datamodule_configs as correctness_datamodule_configs
import pyine.evals.correctness.splits as correctness_splits
import pyine.evals.correctness.types as correctness_types

if typing.TYPE_CHECKING:
    import pytest

_FAKE_LMDB_PATH = pathlib.Path("/fake/lmdb")


class MockGuardrailScorer:
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
        for record in records:
            base = 0.8 if record.label else 0.2
            scores.append(float(np.clip(base + rng.normal(0, 0.1), 0.0, 1.0)))
        return correctness_types.ScoringResult(scores=scores)

    def get_metadata(self) -> dict[str, typing.Any]:
        return {"name": "mock", "seed": self._noise_seed}

    def get_verification_cost_unit(self) -> str | None:
        return None


def make_record(
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


def make_dm_config() -> correctness_datamodule_configs.CorrectnessDataModuleConfig:
    return correctness_datamodule_configs.CorrectnessDataModuleConfig(
        lmdb_paths=(_FAKE_LMDB_PATH,),
        split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
    )


def build_mock_datamodule(
    monkeypatch: pytest.MonkeyPatch,
    valid_records: list[correctness_types.EvalRecord],
    test_records: list[correctness_types.EvalRecord],
    train_records: list[correctness_types.EvalRecord] | None = None,
) -> correctness_datamodule.CorrectnessDataModule:
    """Build a datamodule with mocked LMDB/split loading."""
    if train_records is None:
        train_records = []
    all_records = train_records + valid_records + test_records
    valid_pids = frozenset({record.problem_id for record in valid_records})
    test_pids = frozenset({record.problem_id for record in test_records})
    train_pids = frozenset({record.problem_id for record in train_records})
    splits = correctness_splits.GuardrailSplits(
        guardrail_train=train_records,
        guardrail_valid=valid_records,
        guardrail_test=test_records,
        train_problem_ids=train_pids,
        valid_problem_ids=valid_pids,
        test_problem_ids=test_pids,
    )
    monkeypatch.setattr(
        "pyine.data.utils.lmdb_io.resolve_lmdb_paths",
        lambda raw_paths: list(raw_paths),
    )
    monkeypatch.setattr(
        "pyine.evals.correctness.datamodule.correctness_data_loading.load_records_from_lmdb",
        lambda *_args, **_kwargs: all_records,
    )
    monkeypatch.setattr(
        "pyine.evals.correctness.datamodule.correctness_splits.build_guardrail_splits",
        lambda *_args, **_kwargs: splits,
    )
    dm = correctness_datamodule.CorrectnessDataModule(make_dm_config())
    dm.prepare_data()
    dm.setup()
    return dm
