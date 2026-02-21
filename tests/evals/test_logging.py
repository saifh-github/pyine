"""Tests for DiskEvalLogger."""

import pathlib

import pytest

import pyine.data.utils.generation_record
import pyine.data.utils.lmdb_io
import pyine.evals.code_exec.utils
import pyine.evals.logging
import pyine.evals.utils
import pyine.organisms.datamodules.samples
import pyine.utils.code.output_compare
import pyine.utils.parsing

_DEFAULT_COMPLEXITY_METRICS: dict[str, float | int] = {
    "cyclomatic_complexity_avg": 1.0,
    "cyclomatic_complexity_max": 1,
    "cyclomatic_complexity_sum": 1,
    "loc": 1,
    "lloc": 1,
    "sloc": 1,
    "comments": 0,
    "multi": 0,
    "blank": 0,
    "halstead_volume": 1.0,
    "halstead_difficulty": 1.0,
    "halstead_effort": 1.0,
    "maintainability_index": 100.0,
}


def _make_sample_data(
    identifier: str = "test_sample",
    code: str = "print(1)",
    inputs: str = "",
    expected_output: str = "1",
    predict_type: str = "program_output",
    code_type: str = "original",
    pregenerated_output: str | None = None,
    pregenerated_output_lmdb_path: str | None = None,
    pregenerated_output_lmdb_key: str | None = None,
) -> pyine.organisms.datamodules.samples.SampleData:
    return pyine.organisms.datamodules.samples.SampleData(
        identifier=identifier,
        code=code,
        description="test description",
        entrypoint="main",
        first_line=0,
        last_line=1,
        inputs=inputs,
        expected_output=expected_output,
        predict_type=predict_type,
        code_type=code_type,
        trace_step_count=1,
        comma_separated_tags="tag1,tag2",
        has_code_override=False,
        complexity_metrics=dict(_DEFAULT_COMPLEXITY_METRICS),
        first_line_hit=0,
        last_line_hit=1,
        first_step_idx=0,
        last_step_idx=1,
        pregenerated_output=pregenerated_output,
        pregenerated_output_lmdb_path=pregenerated_output_lmdb_path,
        pregenerated_output_lmdb_key=pregenerated_output_lmdb_key,
    )


def _make_sample_eval(
    identifier: str = "test_sample",
    expected: str = "1",
    predicted: str = "1",
    hard_match: bool = True,
    attempt_index: int = 0,
) -> pyine.evals.code_exec.utils.SampleEval:
    return pyine.evals.code_exec.utils.SampleEval(
        identifier=identifier,
        expected=expected,
        predicted=predicted,
        hard_match=hard_match,
        soft_match=pyine.utils.code.output_compare.CompareResult(
            equal=hard_match, reason="" if hard_match else "mismatch", path=""
        ),
        _llm_score=None,
        tags=["tag1", "tag2"],
        attempt_index=attempt_index,
        predict_type="program_output",
    )


def _make_artifact(
    identifier: str = "test_sample",
    predicted: str = "1",
    hard_match: bool = True,
    attempt_index: int = 0,
    parsed_output: pyine.utils.parsing.ParsedOutput | None = None,
    pregenerated_output_lmdb_path: str | None = None,
    pregenerated_output_lmdb_key: str | None = None,
) -> pyine.evals.code_exec.utils.CodeExecEvalArtifact:
    sample = _make_sample_data(
        identifier=identifier,
        pregenerated_output_lmdb_path=pregenerated_output_lmdb_path,
        pregenerated_output_lmdb_key=pregenerated_output_lmdb_key,
    )
    eval_result = _make_sample_eval(
        identifier=identifier,
        predicted=predicted,
        hard_match=hard_match,
        attempt_index=attempt_index,
    )
    token_usage = pyine.evals.utils.TokenUsageInfo(
        total_tokens=100,
        prompt_tokens=80,
        cached_tokens=0,
        reasoning_tokens=0,
        completion_tokens=20,
    )
    return pyine.evals.code_exec.utils.CodeExecEvalArtifact(
        sample=sample,
        token_usage=token_usage,
        eval_result=eval_result,
        parsed_output=parsed_output,
    )


class TestDiskEvalLogger:
    def test_rejects_nonempty_output_path(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "existing_file.txt").write_text("data")
        with pytest.raises(ValueError, match="non-empty"):
            pyine.evals.logging.DiskEvalLogger(output_path=tmp_path)

    def test_export_round_trip(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        artifacts = [_make_artifact()]
        logger.export_results(
            artifacts=artifacts,
            metrics={"accuracy_hard": 1.0},
            category_to_identifiers={},
        )
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        assert reader.sample_count == 1
        record = reader.get(0)
        assert record["sample_id"] == "test_sample"
        assert record["hard_match"] is True
        assert record["model_output"] == "1"
        reader.close()

    def test_lmdb_key_format(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        artifacts = [_make_artifact(identifier="sample_a", attempt_index=0)]
        logger.export_results(artifacts=artifacts, metrics={}, category_to_identifiers={})
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        assert "sample_a/0" in reader.key_map
        reader.close()

    def test_lmdb_key_format_with_subset_prefix(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        artifacts = [_make_artifact(identifier="sample_a", attempt_index=0)]
        logger.export_results(
            artifacts=artifacts,
            metrics={},
            category_to_identifiers={},
            key_prefix="valid",
        )
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        assert "valid/sample_a/0" in reader.key_map
        reader.close()

    def test_aggregated_metrics_in_metadata(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        metrics = {"accuracy_hard": 0.85, "accuracy_soft": 0.90}
        logger.export_results(
            artifacts=[_make_artifact()],
            metrics=metrics,
            category_to_identifiers={},
            store_aggregated_metrics=True,
        )
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        metadata = reader.get_metadata()
        assert "aggregated_metrics" in metadata
        assert metadata["aggregated_metrics"]["accuracy_hard"] == 0.85
        reader.close()

    def test_skips_metrics_when_disabled(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        logger.export_results(
            artifacts=[_make_artifact()],
            metrics={"accuracy_hard": 0.85},
            category_to_identifiers={},
            store_aggregated_metrics=False,
        )
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        metadata = reader.get_metadata()
        assert "aggregated_metrics" not in metadata
        reader.close()

    def test_multiple_attempts(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        artifacts = [
            _make_artifact(identifier="s1", attempt_index=0),
            _make_artifact(identifier="s1", attempt_index=1, hard_match=False),
            _make_artifact(identifier="s1", attempt_index=2),
        ]
        logger.export_results(artifacts=artifacts, metrics={}, category_to_identifiers={})
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        assert reader.sample_count == 3
        record_0 = reader.get("s1/0")
        record_1 = reader.get("s1/1")
        record_2 = reader.get("s1/2")
        assert record_0["attempt_index"] == 0
        assert record_0["hard_match"] is True
        assert record_1["attempt_index"] == 1
        assert record_1["hard_match"] is False
        assert record_2["attempt_index"] == 2
        reader.close()

    def test_export_metadata_serialization(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        export_metadata = {
            "model_name": "test-model",
            "checkpoint_path": tmp_path / "checkpoint",
            "gen_config": {"temperature": 0.7},
        }
        logger.export_results(
            artifacts=[_make_artifact()],
            metrics={},
            category_to_identifiers={},
            export_metadata=export_metadata,
        )
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        metadata = reader.get_metadata()
        assert metadata["export_metadata"]["model_name"] == "test-model"
        assert metadata["export_metadata"]["checkpoint_path"] == str(tmp_path / "checkpoint")
        reader.close()

    def test_empty_artifacts(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        logger.export_results(artifacts=[], metrics={}, category_to_identifiers={})
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        assert reader.sample_count == 0
        reader.close()

    def test_record_type_metadata(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        logger.export_results(artifacts=[_make_artifact()], metrics={}, category_to_identifiers={})
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        metadata = reader.get_metadata()
        assert metadata["record_type"] == "benchmark"
        reader.close()

    def test_key_prefix_normalization(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        logger.export_results(
            artifacts=[_make_artifact(identifier="s1")],
            metrics={},
            category_to_identifiers={},
            key_prefix="test",  # no trailing slash
        )
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        assert "test/s1/0" in reader.key_map  # should be normalized
        record = reader.get("test/s1/0")
        assert record["key_prefix"] == "test/"
        reader.close()

    def test_categories_per_record(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        artifacts = [
            _make_artifact(identifier="s1"),
            _make_artifact(identifier="s2"),
        ]
        category_to_identifiers = {"cat_a": ["s1"], "cat_b": ["s1", "s2"]}
        logger.export_results(
            artifacts=artifacts,
            metrics={},
            category_to_identifiers=category_to_identifiers,
        )
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        rec_s1 = reader.get("s1/0")
        rec_s2 = reader.get("s2/0")
        assert sorted(rec_s1["categories"]) == ["cat_a", "cat_b"]
        assert rec_s2["categories"] == ["cat_b"]
        reader.close()

    def test_categories_empty_for_unmatched_identifier(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        artifacts = [_make_artifact(identifier="s1")]
        logger.export_results(
            artifacts=artifacts,
            metrics={},
            category_to_identifiers={"cat_a": ["s_other"]},
        )
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        record = reader.get("s1/0")
        assert record["categories"] == []
        reader.close()

    def test_category_to_identifiers_in_metadata(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        cat_map = {"cat_a": ["s1", "s2"]}
        logger.export_results(
            artifacts=[_make_artifact(identifier="s1")],
            metrics={},
            category_to_identifiers=cat_map,
        )
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        metadata = reader.get_metadata()
        assert metadata["category_to_identifiers"] == {"cat_a": ["s1", "s2"]}
        reader.close()

    def test_prompt_text_stored(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        logger.export_results(
            artifacts=[_make_artifact(identifier="s1")],
            metrics={},
            category_to_identifiers={},
            prompt_text_store={"s1": "What is 1+1?"},
        )
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        record = reader.get("s1/0")
        assert record["prompt"] == "What is 1+1?"
        reader.close()

    def test_prompt_messages_stored(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        messages = [{"role": "user", "content": "hello"}]
        logger.export_results(
            artifacts=[_make_artifact(identifier="s1")],
            metrics={},
            category_to_identifiers={},
            prompt_messages_store={"s1": messages},
        )
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        record = reader.get("s1/0")
        assert record["prompt_messages"] == [{"role": "user", "content": "hello"}]
        reader.close()

    def test_eval_subset_name_in_metadata_and_key_prefix_in_records(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        logger.export_results(
            artifacts=[_make_artifact(identifier="s1")],
            metrics={},
            category_to_identifiers={},
            key_prefix="valid",
            eval_subset_name="valid",
        )
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        metadata = reader.get_metadata()
        assert metadata["eval_subset_name"] == "valid"
        record = reader.get("valid/s1/0")
        assert record["key_prefix"] == "valid/"
        reader.close()

    def test_multi_subset_separate_lmdbs(self, tmp_path: pathlib.Path) -> None:
        base_path = tmp_path / "export"
        for subset_name in ("valid", "test"):
            subset_path = base_path / subset_name
            logger = pyine.evals.logging.DiskEvalLogger(output_path=subset_path)
            logger.export_results(
                artifacts=[_make_artifact(identifier=f"s_{subset_name}")],
                metrics={},
                category_to_identifiers={},
                key_prefix=subset_name,
                eval_subset_name=subset_name,
            )
            logger.close()
        # verify separate LMDBs exist
        valid_reader = pyine.data.utils.lmdb_io.LMDBReader(base_path / "valid")
        test_reader = pyine.data.utils.lmdb_io.LMDBReader(base_path / "test")
        assert valid_reader.sample_count == 1
        assert test_reader.sample_count == 1
        assert valid_reader.get(0)["sample_id"] == "s_valid"
        assert test_reader.get(0)["sample_id"] == "s_test"
        valid_reader.close()
        test_reader.close()

    def test_provenance_fields_stored(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        artifacts = [
            _make_artifact(
                identifier="s1",
                pregenerated_output_lmdb_path="/data/pregen.lmdb",
                pregenerated_output_lmdb_key="key_123",
            )
        ]
        logger.export_results(artifacts=artifacts, metrics={}, category_to_identifiers={})
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        record = reader.get("s1/0")
        assert record["pregenerated_output_lmdb_path"] == "/data/pregen.lmdb"
        assert record["pregenerated_output_lmdb_key"] == "key_123"
        reader.close()

    def test_all_shared_fields_present(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        disk_logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        disk_logger.export_results(artifacts=[_make_artifact()], metrics={}, category_to_identifiers={})
        disk_logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        record = reader.get(0)
        shared_keys = set(pyine.data.utils.generation_record.SharedGenerationRecordFields.__annotations__.keys())
        for key in shared_keys:
            assert key in record, f"missing shared key: {key}"
        reader.close()

    def test_parsed_output_raw_vs_final(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        parsed = pyine.utils.parsing.ParsedOutput(raw="raw text", final_answer="parsed answer", reasoning="thought")
        artifact_with_parsed = _make_artifact(identifier="s1", parsed_output=parsed)
        artifact_without_parsed = _make_artifact(identifier="s2", predicted="direct output")
        logger.export_results(
            artifacts=[artifact_with_parsed, artifact_without_parsed],
            metrics={},
            category_to_identifiers={},
        )
        logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        rec_parsed = reader.get("s1/0")
        assert rec_parsed["model_output"] == "raw text"
        assert rec_parsed["final_answer"] == "parsed answer"
        assert rec_parsed["reasoning"] == "thought"
        rec_no_parsed = reader.get("s2/0")
        assert rec_no_parsed["model_output"] == "direct output"
        assert rec_no_parsed["final_answer"] is None
        assert rec_no_parsed["reasoning"] is None
        reader.close()

    def test_prompt_capture_validation_at_export(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "lmdb"
        logger = pyine.evals.logging.DiskEvalLogger(output_path=output_dir)
        artifacts = [_make_artifact(identifier="s1"), _make_artifact(identifier="s2")]
        with pytest.raises(ValueError, match="missing prompt data"):
            logger.export_results(
                artifacts=artifacts,
                metrics={},
                category_to_identifiers={},
                prompt_text_store={"s1": "prompt for s1"},  # s2 missing
            )
        logger.close()
