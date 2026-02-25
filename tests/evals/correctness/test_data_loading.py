"""Tests for pyine.evals.correctness.data_loading."""

from __future__ import annotations

import pathlib
import typing

import pytest

import pyine.evals.correctness.configs as correctness_configs
import pyine.evals.correctness.data_loading as correctness_data_loading


class _FakeLMDBReader:
    """Minimal mock of LMDBReader for testing data_loading without real LMDB files."""

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


def _make_lmdb_record(
    sample_id: str = "TACO/TRAIN/p000001/s0000/t0000",
    attempt_index: int = 0,
    hard_match: bool = True,
    soft_match: bool = True,
    model_output: str = "42",
    final_answer: str | None = "42",
    expected_output: str = "42",
    code_type: str = "original",
    tags: list[str] | None = None,
) -> dict[str, typing.Any]:
    return {
        "sample_id": sample_id,
        "attempt_index": attempt_index,
        "hard_match": hard_match,
        "soft_match": soft_match,
        "model_output": model_output,
        "final_answer": final_answer,
        "expected_output": expected_output,
        "code_type": code_type,
        "tags": tags or [],
    }


class TestLoadRecordsFromLmdb:
    def test_basic_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        records = [
            _make_lmdb_record(attempt_index=0, hard_match=True),
            _make_lmdb_record(attempt_index=1, hard_match=False),
        ]
        reader = _FakeLMDBReader(records, {"record_type": "benchmark"})
        monkeypatch.setattr(
            "pyine.evals.correctness.data_loading.pyine.data.utils.lmdb_io.LMDBReader",
            lambda path: reader,
        )
        result = correctness_data_loading.load_records_from_lmdb(
            [pathlib.Path("/fake/path")],
            correctness_configs.LabelType.HARD_MATCH,
        )
        assert len(result) == 2
        assert result[0].label is True
        assert result[1].label is False
        assert result[0].model_output == "42"
        assert result[0].final_answer == "42"
        assert result[0].problem_id == "TACO/TRAIN/p000001"

    def test_soft_match_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        records = [_make_lmdb_record(hard_match=False, soft_match=True)]
        reader = _FakeLMDBReader(records, {"record_type": "benchmark"})
        monkeypatch.setattr(
            "pyine.evals.correctness.data_loading.pyine.data.utils.lmdb_io.LMDBReader",
            lambda path: reader,
        )
        result = correctness_data_loading.load_records_from_lmdb(
            [pathlib.Path("/fake/path")],
            correctness_configs.LabelType.SOFT_MATCH,
        )
        assert result[0].label is True

    def test_bad_record_type_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reader = _FakeLMDBReader([], {"record_type": "other"})
        monkeypatch.setattr(
            "pyine.evals.correctness.data_loading.pyine.data.utils.lmdb_io.LMDBReader",
            lambda path: reader,
        )
        with pytest.raises(ValueError, match="record_type='other'"):
            correctness_data_loading.load_records_from_lmdb(
                [pathlib.Path("/fake/path")],
                correctness_configs.LabelType.HARD_MATCH,
            )

    def test_duplicate_keys_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        records = [_make_lmdb_record(attempt_index=0), _make_lmdb_record(attempt_index=0)]
        reader = _FakeLMDBReader(records, {"record_type": "benchmark"})
        monkeypatch.setattr(
            "pyine.evals.correctness.data_loading.pyine.data.utils.lmdb_io.LMDBReader",
            lambda path: reader,
        )
        with pytest.raises(ValueError, match="duplicate"):
            correctness_data_loading.load_records_from_lmdb(
                [pathlib.Path("/fake/path")],
                correctness_configs.LabelType.HARD_MATCH,
            )

    def test_missing_code_type_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record_data = _make_lmdb_record()
        del record_data["code_type"]
        reader = _FakeLMDBReader([record_data], {"record_type": "benchmark"})
        monkeypatch.setattr(
            "pyine.evals.correctness.data_loading.pyine.data.utils.lmdb_io.LMDBReader",
            lambda path: reader,
        )
        with pytest.raises(ValueError, match="missing or empty code_type"):
            correctness_data_loading.load_records_from_lmdb(
                [pathlib.Path("/fake/path")],
                correctness_configs.LabelType.HARD_MATCH,
            )

    def test_empty_code_type_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record_data = _make_lmdb_record(code_type="")
        reader = _FakeLMDBReader([record_data], {"record_type": "benchmark"})
        monkeypatch.setattr(
            "pyine.evals.correctness.data_loading.pyine.data.utils.lmdb_io.LMDBReader",
            lambda path: reader,
        )
        with pytest.raises(ValueError, match="missing or empty code_type"):
            correctness_data_loading.load_records_from_lmdb(
                [pathlib.Path("/fake/path")],
                correctness_configs.LabelType.HARD_MATCH,
            )

    def test_invalid_label_type_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        records = [_make_lmdb_record()]
        reader = _FakeLMDBReader(records, {"record_type": "benchmark"})
        monkeypatch.setattr(
            "pyine.evals.correctness.data_loading.pyine.data.utils.lmdb_io.LMDBReader",
            lambda path: reader,
        )
        with pytest.raises(ValueError, match="unsupported label_type"):
            correctness_data_loading.load_records_from_lmdb(
                [pathlib.Path("/fake/path")],
                "not_a_real_label_type",  # type: ignore[arg-type]
            )

    def test_difficulty_score_read_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record_data = _make_lmdb_record()
        record_data["difficulty_score"] = 0.42
        reader = _FakeLMDBReader([record_data], {"record_type": "benchmark"})
        monkeypatch.setattr(
            "pyine.evals.correctness.data_loading.pyine.data.utils.lmdb_io.LMDBReader",
            lambda path: reader,
        )
        result = correctness_data_loading.load_records_from_lmdb(
            [pathlib.Path("/fake/path")],
            correctness_configs.LabelType.HARD_MATCH,
        )
        assert result[0].difficulty_score == pytest.approx(0.42)

    def test_difficulty_score_none_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        records = [_make_lmdb_record()]  # no difficulty_score key
        reader = _FakeLMDBReader(records, {"record_type": "benchmark"})
        monkeypatch.setattr(
            "pyine.evals.correctness.data_loading.pyine.data.utils.lmdb_io.LMDBReader",
            lambda path: reader,
        )
        result = correctness_data_loading.load_records_from_lmdb(
            [pathlib.Path("/fake/path")],
            correctness_configs.LabelType.HARD_MATCH,
        )
        assert result[0].difficulty_score is None

    def test_final_answer_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        records = [_make_lmdb_record(final_answer=None)]
        reader = _FakeLMDBReader(records, {"record_type": "benchmark"})
        monkeypatch.setattr(
            "pyine.evals.correctness.data_loading.pyine.data.utils.lmdb_io.LMDBReader",
            lambda path: reader,
        )
        result = correctness_data_loading.load_records_from_lmdb(
            [pathlib.Path("/fake/path")],
            correctness_configs.LabelType.HARD_MATCH,
        )
        assert result[0].final_answer is None
