"""Tests for reevaluate_from_lmdb."""

import asyncio
import pathlib

import pytest

import pyine.data.utils.lmdb_io
import pyine.evals.code_exec.reeval
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
    expected_output: str = "1",
) -> pyine.organisms.datamodules.samples.SampleData:
    return pyine.organisms.datamodules.samples.SampleData(
        identifier=identifier,
        code=code,
        description="desc",
        entrypoint="main",
        first_line=0,
        last_line=1,
        inputs="",
        expected_output=expected_output,
        predict_type="program_output",
        code_type="original",
        trace_step_count=1,
        comma_separated_tags="tag1",
        has_code_override=False,
        complexity_metrics=dict(_DEFAULT_COMPLEXITY_METRICS),
    )


def _make_artifact(
    identifier: str = "test_sample",
    predicted: str = "1",
    expected: str = "1",
    hard_match: bool = True,
    attempt_index: int = 0,
    parsed_output: pyine.utils.parsing.ParsedOutput | None = None,
) -> pyine.evals.code_exec.utils.CodeExecEvalArtifact:
    sample = _make_sample_data(identifier=identifier, expected_output=expected)
    eval_result = pyine.evals.code_exec.utils.SampleEval(
        identifier=identifier,
        expected=expected,
        predicted=predicted,
        hard_match=hard_match,
        soft_match=pyine.utils.code.output_compare.CompareResult(equal=hard_match, reason="", path=""),
        _llm_score=None,
        tags=["tag1"],
        attempt_index=attempt_index,
        predict_type="program_output",
    )
    token_usage = pyine.evals.utils.TokenUsageInfo(
        total_tokens=100, prompt_tokens=80, cached_tokens=0, reasoning_tokens=0, completion_tokens=20
    )
    return pyine.evals.code_exec.utils.CodeExecEvalArtifact(
        sample=sample, token_usage=token_usage, eval_result=eval_result, parsed_output=parsed_output
    )


def _export_artifacts_to_lmdb(
    output_path: pathlib.Path,
    artifacts: list[pyine.evals.code_exec.utils.CodeExecEvalArtifact],
    key_prefix: str = "",
    eval_subset_name: str | None = None,
    prompt_text_store: dict[str, str] | None = None,
) -> None:
    logger = pyine.evals.logging.DiskEvalLogger(output_path=output_path)
    logger.export_results(
        artifacts=artifacts,
        metrics={"accuracy_hard": 1.0},
        category_to_identifiers={"cat_a": [a.sample_identifier for a in artifacts]},
        key_prefix=key_prefix,
        eval_subset_name=eval_subset_name,
        prompt_text_store=prompt_text_store,
    )
    logger.close()


class TestReevaluateFromLmdb:
    def test_round_trip(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "lmdb"
        artifacts = [_make_artifact(identifier="s1"), _make_artifact(identifier="s2")]
        _export_artifacts_to_lmdb(lmdb_path, artifacts)
        result = asyncio.run(pyine.evals.code_exec.reeval.reevaluate_from_lmdb(lmdb_paths=[lmdb_path]))
        assert result.num_samples == 2
        assert "accuracy_hard" in result.metrics

    def test_multiple_lmdbs(self, tmp_path: pathlib.Path) -> None:
        lmdb1 = tmp_path / "lmdb1"
        lmdb2 = tmp_path / "lmdb2"
        _export_artifacts_to_lmdb(lmdb1, [_make_artifact(identifier="s1")])
        _export_artifacts_to_lmdb(lmdb2, [_make_artifact(identifier="s2")])
        result = asyncio.run(pyine.evals.code_exec.reeval.reevaluate_from_lmdb(lmdb_paths=[lmdb1, lmdb2]))
        assert result.num_samples == 2

    def test_multi_attempt(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "lmdb"
        artifacts = [
            _make_artifact(identifier="s1", attempt_index=0),
            _make_artifact(identifier="s1", attempt_index=1, hard_match=False, predicted="wrong"),
            _make_artifact(identifier="s1", attempt_index=2),
        ]
        _export_artifacts_to_lmdb(lmdb_path, artifacts)
        result = asyncio.run(pyine.evals.code_exec.reeval.reevaluate_from_lmdb(lmdb_paths=[lmdb_path]))
        assert result.num_attempts == 3
        assert result.num_samples == 1

    def test_nonuniform_attempts_raises(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "lmdb"
        artifacts = [
            _make_artifact(identifier="s1", attempt_index=0),
            _make_artifact(identifier="s1", attempt_index=1),
            _make_artifact(identifier="s2", attempt_index=0),
        ]
        _export_artifacts_to_lmdb(lmdb_path, artifacts)
        with pytest.raises(ValueError, match="non-uniform"):
            asyncio.run(pyine.evals.code_exec.reeval.reevaluate_from_lmdb(lmdb_paths=[lmdb_path]))

    def test_token_usage_preserved(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "lmdb"
        _export_artifacts_to_lmdb(lmdb_path, [_make_artifact(identifier="s1")])
        result = asyncio.run(pyine.evals.code_exec.reeval.reevaluate_from_lmdb(lmdb_paths=[lmdb_path]))
        artifact = result.artifacts[0]
        assert artifact.token_usage.total_tokens == 100
        assert artifact.token_usage.prompt_tokens == 80
        assert artifact.token_usage.completion_tokens == 20

    def test_categories_from_records(self, tmp_path: pathlib.Path) -> None:
        lmdb_path = tmp_path / "lmdb"
        _export_artifacts_to_lmdb(lmdb_path, [_make_artifact(identifier="s1")])
        result = asyncio.run(pyine.evals.code_exec.reeval.reevaluate_from_lmdb(lmdb_paths=[lmdb_path]))
        assert "cat_a" in result.category_to_identifiers
        assert "s1" in result.category_to_identifiers["cat_a"]

    def test_parsed_output_preserved_in_reeval(self, tmp_path: pathlib.Path) -> None:
        """When records have final_answer, reeval uses it as predicted and preserves parsed output."""
        lmdb_path = tmp_path / "lmdb"
        parsed = pyine.utils.parsing.ParsedOutput(raw="<final>42</final>", final_answer="42", reasoning="think")
        artifact = _make_artifact(identifier="s1", predicted="42", expected="1", hard_match=False, parsed_output=parsed)
        _export_artifacts_to_lmdb(lmdb_path, [artifact])
        # verify the exported record has final_answer and model_output
        reader = pyine.data.utils.lmdb_io.LMDBReader(lmdb_path)
        exported_record = reader.get(0)
        assert exported_record["model_output"] == "<final>42</final>"
        assert exported_record["final_answer"] == "42"
        assert exported_record["reasoning"] == "think"
        reader.close()
        # re-evaluate
        result = asyncio.run(pyine.evals.code_exec.reeval.reevaluate_from_lmdb(lmdb_paths=[lmdb_path]))
        # reeval should use final_answer="42" as predicted (matching original behavior)
        reeval_artifact = result.artifacts[0]
        assert reeval_artifact.eval_result.predicted == "42"
        # parsed_output should be reconstructed on the artifact
        assert reeval_artifact.parsed_output is not None
        assert reeval_artifact.parsed_output.raw == "<final>42</final>"
        assert reeval_artifact.parsed_output.final_answer == "42"
        assert reeval_artifact.parsed_output.reasoning == "think"

    def test_rejects_reward_lmdb(self, tmp_path: pathlib.Path) -> None:
        """Passing a reward LMDB (record_type='reward') raises ValueError."""
        lmdb_path = tmp_path / "reward_lmdb"
        writer = pyine.data.utils.lmdb_io.LMDBWriter(
            path=lmdb_path,
            serialization_config=pyine.data.utils.lmdb_io.SerializationConfig(
                method=pyine.data.utils.lmdb_io.SerializationMethod.JSON_ZSTD,
            ),
        )
        writer.write_metadata({"record_type": "reward"})
        writer.close()
        with pytest.raises(ValueError, match="record_type='reward'"):
            asyncio.run(pyine.evals.code_exec.reeval.reevaluate_from_lmdb(lmdb_paths=[lmdb_path]))
