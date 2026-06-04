"""Regression tests for CueFlip aggregation."""

from __future__ import annotations

import json
import typing

import analyze
import code_eval

if typing.TYPE_CHECKING:
    import pathlib

    import pytest


class TestComputeCell:
    def test_excludes_unparseable_cues_from_denominator(self) -> None:
        baselines = {"qid": {"parsed_answer": "A", "correct": True}}
        cues = [
            {"qid": "qid", "parsed_answer": "B", "suggested_value": "B"},
            {"qid": "qid", "parsed_answer": None, "suggested_value": "B"},
        ]
        cell = analyze.compute_cell("shortcut", "bench", "authority", 0, None, baselines, cues)
        assert cell.n_cue == 1
        assert cell.switch_rate.value == 1.0
        assert cell.uptake_rate.value == 1.0

    def test_excludes_unparseable_baselines_from_denominator(self) -> None:
        baselines = {"qid": {"parsed_answer": None, "correct": False}}
        cues = [{"qid": "qid", "parsed_answer": "B", "suggested_value": "B"}]
        cell = analyze.compute_cell("shortcut", "bench", "authority", 0, None, baselines, cues)
        assert cell.n_cue == 0
        assert cell.switches_total == 0
        assert cell.switch_rate.value is None


class TestLoadRuns:
    def test_ignores_stale_humaneval_records(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(analyze, "ROOT", tmp_path)
        path = tmp_path / "shortcut" / "humaneval" / "runs.jsonl"
        path.parent.mkdir(parents=True)
        stale = {
            "kind": "code",
            "qid": "old",
            "phase": "baseline",
            "code_eval_version": code_eval.EVAL_VERSION - 1,
        }
        current = {
            "kind": "code",
            "qid": "new",
            "phase": "baseline",
            "code_eval_version": code_eval.EVAL_VERSION,
        }
        path.write_text(f"{json.dumps(stale)}\n{json.dumps(current)}\n")
        baselines, _ = analyze.load_runs("shortcut", "humaneval")
        assert set(baselines) == {"new"}


class TestCollectCrossModelCells:
    def test_tracks_different_model_specific_suggestions(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(analyze, "ROOT", tmp_path)
        base_record = {
            "benchmark": "bench",
            "qid": "qid",
            "cue_family": None,
            "cue_paraphrase_idx": None,
            "perturbation_strategy": None,
            "phase": "baseline",
            "parsed_answer": "A",
        }
        cue_common = {
            "benchmark": "bench",
            "qid": "qid",
            "cue_family": "authority",
            "cue_paraphrase_idx": 0,
            "perturbation_strategy": None,
            "phase": "cue",
        }
        shortcut_records = [
            {**base_record, "model_tag": "shortcut"},
            {**cue_common, "model_tag": "shortcut", "parsed_answer": "B", "suggested_value": "B"},
        ]
        base_records = [
            {**base_record, "model_tag": "base"},
            {**cue_common, "model_tag": "base", "parsed_answer": "C", "suggested_value": "C"},
        ]
        for model, records in (("shortcut", shortcut_records), ("base", base_records)):
            path = tmp_path / model / "bench" / "runs.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("".join(f"{json.dumps(record)}\n" for record in records))

        cells = analyze.collect_cross_model_cells()
        assert len(cells) == 1
        assert cells[0].both_took_different == 1
        assert cells[0].neither_took == 0
