"""Shared fixtures and data factories for integration tests of evaluation pipelines."""

from __future__ import annotations

import pathlib
import typing

import pytest

import pyine.evals.correctness.datamodule as correctness_datamodule
import pyine.evals.correctness.datamodule_configs as correctness_datamodule_configs
import pyine.evals.correctness.splits as correctness_splits
import pyine.evals.correctness.types as correctness_types
import pyine.organisms.datamodules.samples.common as samples_common
import pyine.utils.code.complexity_metrics

_FAKE_LMDB_PATH = pathlib.Path("/fake/lmdb")
_ZERO_COMPLEXITY_METRICS: dict[str, float | int] = dict.fromkeys(
    pyine.utils.code.complexity_metrics.COMPLEXITY_METRICS, 0.0
)


# ---------------------------------------------------------------------------
# Data factories
# ---------------------------------------------------------------------------


def make_sample_data(
    identifier: str,
    expected_output: str,
    code: str = "x = 1",
    predict_type: samples_common.SamplePredictType = samples_common.SamplePredictType.program_output,
    code_type: str = "original",
    comma_separated_tags: str = "",
) -> samples_common.SampleData:
    """Build a minimal ``SampleData`` namedtuple with sensible defaults."""
    return samples_common.SampleData(
        identifier=identifier,
        code=code,
        description="",
        entrypoint="",
        first_line=0,
        last_line=1,
        inputs="",
        expected_output=expected_output,
        predict_type=predict_type,
        code_type=code_type,
        trace_step_count=0,
        comma_separated_tags=comma_separated_tags,
        has_code_override=False,
        complexity_metrics=dict(_ZERO_COMPLEXITY_METRICS),
    )


def make_eval_record(
    sample_id: str,
    problem_id: str,
    attempt_index: int,
    label: bool,
    code_type: str = "original",
) -> correctness_types.EvalRecord:
    """Build an ``EvalRecord`` with minimal defaults."""
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


# ---------------------------------------------------------------------------
# Fake ConversationDataModule (for CODE_EXEC)
# ---------------------------------------------------------------------------


class FakeConversationDataModule:
    """Minimal duck-typed ConversationDataModule wrapping a list of ``SampleData``."""

    def __init__(self, samples: list[samples_common.SampleData]) -> None:
        self._samples = samples
        self.config: typing.Any = {"fake": True}  # only accessed when disk export is enabled

    def get_parser(
        self,
        subset_name: str,
    ) -> list[samples_common.SampleData]:
        return self._samples


# ---------------------------------------------------------------------------
# Mock datamodule builder (for CORRECTNESS)
# ---------------------------------------------------------------------------


def make_dm_config() -> correctness_datamodule_configs.CorrectnessDataModuleConfig:
    return correctness_datamodule_configs.CorrectnessDataModuleConfig(
        lmdb_paths=(_FAKE_LMDB_PATH,),
        split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
    )


def build_mock_correctness_datamodule(
    monkeypatch: pytest.MonkeyPatch,
    valid_records: list[correctness_types.EvalRecord],
    test_records: list[correctness_types.EvalRecord],
    train_records: list[correctness_types.EvalRecord] | None = None,
) -> correctness_datamodule.CorrectnessDataModule:
    """Build a ``CorrectnessDataModule`` with mocked LMDB/split loading."""
    if train_records is None:
        train_records = []
    all_records = train_records + valid_records + test_records
    splits = correctness_splits.GuardrailSplits(
        guardrail_train=train_records,
        guardrail_valid=valid_records,
        guardrail_test=test_records,
        train_problem_ids=frozenset({rec.problem_id for rec in train_records}),
        valid_problem_ids=frozenset({rec.problem_id for rec in valid_records}),
        test_problem_ids=frozenset({rec.problem_id for rec in test_records}),
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


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def code_exec_samples() -> list[samples_common.SampleData]:
    """20 ``SampleData`` instances with distinct expected outputs."""
    return [
        make_sample_data(
            identifier=f"test/p{idx:04d}/s0000/t0000",
            expected_output=f"result_{idx}",
        )
        for idx in range(20)
    ]


@pytest.fixture
def correctness_eval_records() -> tuple[list[correctness_types.EvalRecord], list[correctness_types.EvalRecord]]:
    """Two lists (valid, test) of 20 problems x 3 attempts = 60 records each.

    Label pattern: ``(prob_idx + attempt_idx) % 3 != 0`` -> ~67% positive.
    """
    valid_records: list[correctness_types.EvalRecord] = []
    test_records: list[correctness_types.EvalRecord] = []
    for prob_idx in range(20):
        for attempt_idx in range(3):
            label = (prob_idx + attempt_idx) % 3 != 0
            valid_records.append(
                make_eval_record(
                    sample_id=f"TACO/TRAIN/p{prob_idx:06d}/s0000/t0000",
                    problem_id=f"TACO/TRAIN/p{prob_idx:06d}",
                    attempt_index=attempt_idx,
                    label=label,
                )
            )
    for prob_idx in range(20, 40):
        for attempt_idx in range(3):
            label = (prob_idx + attempt_idx) % 3 != 0
            test_records.append(
                make_eval_record(
                    sample_id=f"TACO/TRAIN/p{prob_idx:06d}/s0000/t0000",
                    problem_id=f"TACO/TRAIN/p{prob_idx:06d}",
                    attempt_index=attempt_idx,
                    label=label,
                )
            )
    return valid_records, test_records
