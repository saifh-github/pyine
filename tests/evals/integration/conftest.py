"""Shared fixtures and data factories for integration tests of evaluation pipelines."""

from __future__ import annotations

import asyncio
import dataclasses
import pathlib
import shutil
import tempfile
import typing

import pytest

import pyine.evals.code_exec._impl as code_exec_impl
import pyine.evals.code_exec.configs as code_exec_configs
import pyine.evals.code_exec.utils as code_exec_utils
import pyine.evals.common
import pyine.evals.correctness.datamodule as correctness_datamodule
import pyine.evals.correctness.datamodule_configs as correctness_datamodule_configs
import pyine.evals.correctness.splits as correctness_splits
import pyine.evals.correctness.types as correctness_types
import pyine.organisms.datamodules.samples.common as samples_common
import pyine.utils.code.complexity_metrics
import tests.evals.integration.fake_models as fake_models

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


# ---------------------------------------------------------------------------
# TACO-format sample factory (for round-trip tests)
# ---------------------------------------------------------------------------

_NUM_TACO_SAMPLES = 20
_NUM_ATTEMPTS = 3
_PASS_AT_K_VALUES = [1, 3]


def make_taco_sample_data(
    count: int = _NUM_TACO_SAMPLES,
) -> list[samples_common.SampleData]:
    """Build ``SampleData`` instances with TACO-format identifiers.

    These identifiers are parseable by ``extract_problem_id()`` ->
    ``TraceIdentifier.from_string()``, enabling end-to-end round-trips
    through the correctness pipeline without monkeypatching the parser.
    """
    return [
        make_sample_data(
            identifier=f"TACO/TRAIN/p{idx:06d}/s0000/t0000",
            expected_output=f"result_{idx}",
        )
        for idx in range(count)
    ]


# ---------------------------------------------------------------------------
# Pipeline output containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CodeExecPipelineOutput:
    """Holds the output of a full code exec pipeline run for round-trip tests."""

    result: code_exec_utils.CodeExecEvalResult
    lmdb_path: pathlib.Path
    pickle_path: pathlib.Path | None
    samples: list[samples_common.SampleData]
    tmp_dir: pathlib.Path


def _run_code_exec_pipeline(
    chain: typing.Any,
    samples: list[samples_common.SampleData],
    *,
    enable_pickle: bool = True,
    store_aggregated_metrics: bool = True,
) -> CodeExecPipelineOutput:
    """Run the code exec pipeline synchronously with real ``disk_export_config``.

    Fake chains in ``fake_models`` fire ``on_chat_model_start`` on any callbacks passed via the
    LangChain ``config`` dict, so the real prompt-capture + LMDB-export path inside
    ``evaluate_runnable_model`` works end-to-end.
    """
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="roundtrip_"))
    lmdb_base = tmp_dir / "lmdb_export"
    pickle_dir = tmp_dir / "pickle" if enable_pickle else None
    eval_config = code_exec_configs.CodeExecEvalsConfig(
        eval_runnable_config=pyine.evals.common.RunnableEvalConfig(parallel=False),
        num_attempts_per_sample=_NUM_ATTEMPTS,
        pass_at_k_values=_PASS_AT_K_VALUES,
        result_dump_dir=pickle_dir,
        disk_export_config=pyine.evals.common.EvalExportConfig(
            output_path=lmdb_base,
            store_aggregated_metrics=store_aggregated_metrics,
        ),
    )
    result = asyncio.run(
        code_exec_impl.evaluate_runnable_model(
            eval_config=eval_config,
            chain=chain,
            datamodule=FakeConversationDataModule(samples),  # type: ignore[arg-type]
            eval_subset_name="test",
        )
    )
    lmdb_path = lmdb_base / "test"
    pickle_path: pathlib.Path | None = None
    if pickle_dir is not None:
        pickle_path = pickle_dir / "test.pkl"
    return CodeExecPipelineOutput(
        result=result,
        lmdb_path=lmdb_path,
        pickle_path=pickle_path,
        samples=samples,
        tmp_dir=tmp_dir,
    )


# ---------------------------------------------------------------------------
# Module-scoped pipeline fixtures (run once, shared across tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oracle_code_exec_output() -> typing.Generator[CodeExecPipelineOutput]:
    """Run the code exec pipeline with ``OracleCodeExecChain`` (deterministic, 100% accuracy)."""
    samples = make_taco_sample_data()
    output = _run_code_exec_pipeline(fake_models.OracleCodeExecChain(), samples)
    yield output
    shutil.rmtree(output.tmp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def random_code_exec_output() -> typing.Generator[CodeExecPipelineOutput]:
    """Run the code exec pipeline with ``RandomCodeExecChain`` (seeded ~50% accuracy).

    Only LMDB export is enabled (no pickle); this fixture exists to provide mixed-label
    data for correctness pipeline tests.
    """
    samples = make_taco_sample_data()
    output = _run_code_exec_pipeline(
        fake_models.RandomCodeExecChain(seed=42),
        samples,
        enable_pickle=False,
    )
    yield output
    shutil.rmtree(output.tmp_dir, ignore_errors=True)
