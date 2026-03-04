"""Tests for pyine.evals.persistence -- pickle-based EvalResult persistence."""

from __future__ import annotations

import pickle
import typing
import unittest.mock

import numpy as np
import pydantic
import pytest

import pyine.evals.code_exec.utils
import pyine.evals.common
import pyine.evals.persistence
import pyine.evals.utils
import pyine.utils.code.output_compare

if typing.TYPE_CHECKING:
    import pathlib


class _MockEvalResult(pyine.evals.common.EvalResult):
    """Lightweight mock EvalResult for core API tests."""

    model_config = pydantic.ConfigDict(frozen=True)
    label: str = "mock"


class _OtherMockEvalResult(pyine.evals.common.EvalResult):
    """Another mock subclass for type-mismatch tests."""

    model_config = pydantic.ConfigDict(frozen=True)
    other_field: int = 42


class TestSaveAndLoadEvalResult:
    """Round-trip and error-handling tests for pickle persistence."""

    def test_round_trip(self, tmp_path: pathlib.Path) -> None:
        """save -> load returns identical object."""
        result = _MockEvalResult(metrics={"acc": 0.95}, label="test")
        path = tmp_path / "result.pkl"
        pyine.evals.persistence.save_eval_result(result, path)
        loaded = pyine.evals.persistence.load_eval_result(path, expected_type=_MockEvalResult)
        assert loaded.metrics == result.metrics
        assert loaded.label == result.label

    def test_file_exists_error_without_overwrite(self, tmp_path: pathlib.Path) -> None:
        """Second save raises FileExistsError when overwrite=False."""
        result = _MockEvalResult(metrics={"acc": 0.9})
        path = tmp_path / "result.pkl"
        pyine.evals.persistence.save_eval_result(result, path)
        with pytest.raises(FileExistsError, match="overwrite=True"):
            pyine.evals.persistence.save_eval_result(result, path)

    def test_overwrite_succeeds(self, tmp_path: pathlib.Path) -> None:
        """Second save succeeds when overwrite=True."""
        result1 = _MockEvalResult(metrics={"acc": 0.8}, label="v1")
        result2 = _MockEvalResult(metrics={"acc": 0.9}, label="v2")
        path = tmp_path / "result.pkl"
        pyine.evals.persistence.save_eval_result(result1, path)
        pyine.evals.persistence.save_eval_result(result2, path, overwrite=True)
        loaded = pyine.evals.persistence.load_eval_result(path, expected_type=_MockEvalResult)
        assert loaded.label == "v2"

    def test_type_mismatch_raises_type_error(self, tmp_path: pathlib.Path) -> None:
        """load_eval_result with wrong expected_type raises TypeError."""
        result = _MockEvalResult(metrics={"acc": 0.9})
        path = tmp_path / "result.pkl"
        pyine.evals.persistence.save_eval_result(result, path)
        with pytest.raises(TypeError, match="expected _OtherMockEvalResult"):
            pyine.evals.persistence.load_eval_result(path, expected_type=_OtherMockEvalResult)

    def test_expected_type_none_allows_any(self, tmp_path: pathlib.Path) -> None:
        """load_eval_result with expected_type=None returns any EvalResult subclass."""
        result = _MockEvalResult(metrics={"acc": 0.9}, label="any")
        path = tmp_path / "result.pkl"
        pyine.evals.persistence.save_eval_result(result, path)
        loaded = pyine.evals.persistence.load_eval_result(path)
        assert isinstance(loaded, _MockEvalResult)
        assert loaded.label == "any"

    def test_non_eval_result_rejected(self, tmp_path: pathlib.Path) -> None:
        """load_eval_result rejects pickled objects that aren't EvalResult subclasses."""
        import pickle

        path = tmp_path / "not_eval.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"just": "a dict"}, fh)
        with pytest.raises(TypeError, match="expected an EvalResult subclass"):
            pyine.evals.persistence.load_eval_result(path)

    def test_creates_parent_dirs(self, tmp_path: pathlib.Path) -> None:
        """save_eval_result creates parent directories if they don't exist."""
        result = _MockEvalResult(metrics={"acc": 0.9})
        path = tmp_path / "nested" / "dirs" / "result.pkl"
        pyine.evals.persistence.save_eval_result(result, path)
        assert path.exists()
        loaded = pyine.evals.persistence.load_eval_result(path, expected_type=_MockEvalResult)
        assert loaded.metrics == result.metrics

    def test_atomic_write_cleanup_on_failure(self, tmp_path: pathlib.Path) -> None:
        """Pickle failure leaves no .pkl or .tmp files behind."""
        result = _MockEvalResult(metrics={"acc": 0.9})
        path = tmp_path / "result.pkl"
        with (
            unittest.mock.patch.object(pickle, "dump", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            pyine.evals.persistence.save_eval_result(result, path)
        assert not path.exists()
        remaining = list(tmp_path.glob("*.tmp"))
        assert remaining == []


class TestBuildResultDumpPath:
    """Path construction and sanitization tests."""

    def test_basic_path(self, tmp_path: pathlib.Path) -> None:
        """Produces {dump_dir}/{eval_subset_name}.pkl."""
        path = pyine.evals.persistence.build_result_dump_path(tmp_path, "valid")
        assert path == tmp_path / "valid.pkl"

    def test_with_type_name(self, tmp_path: pathlib.Path) -> None:
        """Produces {dump_dir}/{eval_subset_name}__{type_name}.pkl."""
        path = pyine.evals.persistence.build_result_dump_path(tmp_path, "test", type_name="mean_pool_L8")
        assert path == tmp_path / "test__mean_pool_L8.pkl"

    def test_sanitizes_unsafe_chars(self, tmp_path: pathlib.Path) -> None:
        """Slashes, spaces, etc. replaced with underscores."""
        path = pyine.evals.persistence.build_result_dump_path(tmp_path, "a/b c")
        assert "a_b_c" in path.name
        assert "/" not in path.name.replace(".pkl", "")

    def test_distinct_names_produce_distinct_paths(self, tmp_path: pathlib.Path) -> None:
        """Different (subset, type_name) pairs produce different paths."""
        path1 = pyine.evals.persistence.build_result_dump_path(tmp_path, "valid", type_name="typeA")
        path2 = pyine.evals.persistence.build_result_dump_path(tmp_path, "valid", type_name="typeB")
        path3 = pyine.evals.persistence.build_result_dump_path(tmp_path, "test", type_name="typeA")
        assert path1 != path2
        assert path1 != path3

    def test_whitespace_only_subset_name_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="sanitizes to an empty string"):
            pyine.evals.persistence.build_result_dump_path(tmp_path, "   ")

    def test_whitespace_only_type_name_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="sanitizes to an empty string"):
            pyine.evals.persistence.build_result_dump_path(tmp_path, "valid", type_name="  \t  ")


class TestBuildResultDumpPathCollisions:
    """Sanitization collision edge cases."""

    def test_collision_detected_with_overwrite_false(self, tmp_path: pathlib.Path) -> None:
        """'a/b' and 'a b' produce same sanitized name; second save raises FileExistsError."""
        result = _MockEvalResult(metrics={})
        path1 = pyine.evals.persistence.build_result_dump_path(tmp_path, "a/b")
        path2 = pyine.evals.persistence.build_result_dump_path(tmp_path, "a b")
        assert path1 == path2  # both sanitize to a_b.pkl
        pyine.evals.persistence.save_eval_result(result, path1)
        with pytest.raises(FileExistsError):
            pyine.evals.persistence.save_eval_result(result, path2)

    def test_collision_logs_warning_with_overwrite_true(
        self,
        tmp_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """'a/b' and 'a b' -> same path; overwrite=True succeeds but logs warning."""
        import logging

        logger_name = "pyine.evals.persistence"
        target_logger = logging.getLogger(logger_name)
        target_logger.addHandler(caplog.handler)  # bypass propagate=False on ancestor
        try:
            result = _MockEvalResult(metrics={})
            path1 = pyine.evals.persistence.build_result_dump_path(tmp_path, "a/b")
            path2 = pyine.evals.persistence.build_result_dump_path(tmp_path, "a b")
            pyine.evals.persistence.save_eval_result(result, path1)
            with caplog.at_level(logging.WARNING, logger=logger_name):
                pyine.evals.persistence.save_eval_result(result, path2, overwrite=True)
        finally:
            target_logger.removeHandler(caplog.handler)
        assert "overwriting" in caplog.text.lower()


class TestMaybeDumpEvalResult:
    """Tests for the centralized maybe_dump_eval_result helper."""

    def test_returns_none_when_dump_dir_is_none(self) -> None:
        """No-op when dump_dir is None."""
        result = _MockEvalResult(metrics={})
        path = pyine.evals.persistence.maybe_dump_eval_result(result, None, "valid")
        assert path is None

    def test_raises_when_subset_name_missing(self, tmp_path: pathlib.Path) -> None:
        """ValueError when dump_dir set but eval_subset_name is None/empty/whitespace."""
        result = _MockEvalResult(metrics={})
        with pytest.raises(ValueError, match="eval_subset_name must be set"):
            pyine.evals.persistence.maybe_dump_eval_result(result, tmp_path, None)
        with pytest.raises(ValueError, match="eval_subset_name must be set"):
            pyine.evals.persistence.maybe_dump_eval_result(result, tmp_path, "")
        with pytest.raises(ValueError, match="eval_subset_name must be set"):
            pyine.evals.persistence.maybe_dump_eval_result(result, tmp_path, "   ")

    def test_returns_path_on_success(self, tmp_path: pathlib.Path) -> None:
        """Returns the path written to on success."""
        result = _MockEvalResult(metrics={"x": 1})
        path = pyine.evals.persistence.maybe_dump_eval_result(result, tmp_path, "valid")
        assert path is not None
        assert path.exists()
        assert path.name == "valid.pkl"

    def test_passes_type_name_through(self, tmp_path: pathlib.Path) -> None:
        """type_name is forwarded to build_result_dump_path."""
        result = _MockEvalResult(metrics={})
        path = pyine.evals.persistence.maybe_dump_eval_result(
            result,
            tmp_path,
            "test",
            type_name="pool_L4",
        )
        assert path is not None
        assert "pool_L4" in path.name


class TestRealResultRoundTrip:
    """Round-trip tests with real EvalResult subclasses (not mocks).
    Not marked slow -- these are core safety tests for a new execution path."""

    def test_code_exec_result_round_trip(self, tmp_path: pathlib.Path) -> None:
        """Construct minimal real CodeExecEvalResult and verify pickle round-trip."""
        import pyine.evals.code_exec.utils
        import pyine.evals.utils
        import pyine.organisms.datamodules.samples.common as samples_common

        sample = samples_common.SampleData(
            identifier="s1",
            code="x = 1",
            description="",
            entrypoint="",
            first_line=0,
            last_line=1,
            inputs="",
            expected_output="1",
            predict_type=samples_common.SamplePredictType.program_output,
            code_type="original",
            trace_step_count=1,
            comma_separated_tags="",
            has_code_override=False,
            complexity_metrics={},
            first_line_hit=0,
            last_line_hit=1,
            first_step_idx=0,
            last_step_idx=0,
        )
        eval_result = pyine.evals.code_exec.utils.SampleEval(
            identifier="s1",
            expected="1",
            predicted="1",
            hard_match=True,
            soft_match=pyine.utils.code.output_compare.CompareResult(equal=True),
            _llm_score=None,
            tags=[],
        )
        token_usage = pyine.evals.utils.TokenUsageInfo.get_default()
        artifact = pyine.evals.code_exec.utils.CodeExecEvalArtifact(
            sample=sample,
            token_usage=token_usage,
            eval_result=eval_result,
        )
        result = pyine.evals.code_exec.utils.CodeExecEvalResult(
            metrics={"hard_accuracy": 1.0},
            artifacts=[artifact],
            category_to_identifiers={"code_type/original": ["s1"]},
            eval_metadata={
                "eval_subset_name": "valid",
                "evaluation_backend": "hf",
                "eval_config": {"foo": "bar"},
                "reprod_metadata": {"framework_version": "test"},
            },
        )
        path = tmp_path / "code_exec.pkl"
        pyine.evals.persistence.save_eval_result(result, path)
        loaded = pyine.evals.persistence.load_eval_result(
            path,
            expected_type=pyine.evals.code_exec.utils.CodeExecEvalResult,
        )
        assert loaded.metrics == result.metrics
        assert len(loaded.artifacts) == 1
        assert loaded.artifacts[0].sample.identifier == "s1"
        assert loaded.eval_metadata["eval_subset_name"] == "valid"
        assert loaded.eval_metadata["reprod_metadata"]["framework_version"] == "test"

    def test_correctness_result_round_trip(self, tmp_path: pathlib.Path) -> None:
        """Construct minimal real CorrectnessEvalResult and verify round-trip."""
        import pyine.evals.correctness._impl as correctness_impl
        import pyine.evals.correctness.types as correctness_types
        import pyine.utils.metrics.confidence

        threshold_free = correctness_types.ThresholdFreeMetrics(
            auroc=0.85,
            average_precision=0.80,
            tpr_at_fpr={0.01: 0.7},
            fpr_grid=np.linspace(0, 1, 10),
            tpr_grid=np.linspace(0, 1, 10),
            precision_grid=np.linspace(1, 0, 10),
            recall_grid=np.linspace(0, 1, 10),
        )
        run_result = correctness_types.SingleRunResult(
            guardrail_metadata={"name": "test"},
            attempt_metadata={
                ("s1", 0, 0): {"input_token_count": 12},
            },
            attempt_records=[
                correctness_types.AttemptInspectionRecord(
                    sample_id="s1",
                    problem_id="p1",
                    attempt_index=0,
                    draw_index=0,
                    label=True,
                    score=0.9,
                    verification_cost=None,
                    final_answer="1",
                    code_type="original",
                    difficulty_score=None,
                    attempt_metadata={"input_token_count": 12},
                )
            ],
            threshold_free=threshold_free,
            attempt_metrics={},
            sample_metrics={},
            category_results={},
            bootstrap_cis={},
            difficulty_stats=None,
            verification_cost_stats=None,
        )
        class_balance = correctness_types.ClassBalanceStats(
            overall_positive_rate=0.5,
            per_sample_positive_rates=[0.5],
            num_all_correct_samples=0,
            num_all_incorrect_samples=0,
            code_type_proportions={"original": 1.0},
            predict_type_proportions={"program_output": 1.0},
        )
        aggregated = correctness_types.AggregatedResult(
            split_summary={"test": 10},
            class_balance=class_balance,
            per_run=[run_result],
            attempt_records_by_key={("s1", 0, 0): {"sample_id": "s1"}},
            cross_run_mean={"auroc": 0.85},
            cross_run_std={"auroc": 0.0},
            cross_run_p5={"auroc": 0.85},
            cross_run_num_valid={"auroc": 1},
            hierarchical_cis={},
            difficulty_stats=None,
            verification_cost_stats=None,
        )
        result = correctness_impl.CorrectnessEvalResult(
            metrics={"auroc/mean": 0.85},
            aggregated=aggregated,
            eval_metadata={
                "eval_subset_name": "guardrail_test",
                "eval_config": {"target_fpr_values": [0.01]},
                "reprod_metadata": {"framework_version": "test"},
            },
        )
        path = tmp_path / "correctness.pkl"
        pyine.evals.persistence.save_eval_result(result, path)
        loaded = pyine.evals.persistence.load_eval_result(
            path,
            expected_type=correctness_impl.CorrectnessEvalResult,
        )
        assert loaded.metrics == result.metrics
        assert loaded.aggregated.per_run[0].threshold_free.auroc == 0.85
        assert loaded.aggregated.per_run[0].attempt_metadata is not None
        assert loaded.aggregated.per_run[0].attempt_metadata[("s1", 0, 0)]["input_token_count"] == 12
        assert loaded.aggregated.per_run[0].attempt_records[0].score == pytest.approx(0.9)
        assert loaded.aggregated.per_run[0].attempt_records[0].sample_id == "s1"
        assert loaded.aggregated.per_run[0].attempt_records[0].draw_index == 0
        assert loaded.aggregated.attempt_records_by_key is not None
        assert loaded.aggregated.attempt_records_by_key[("s1", 0, 0)]["sample_id"] == "s1"
        assert loaded.eval_metadata["eval_subset_name"] == "guardrail_test"
        assert loaded.eval_metadata["reprod_metadata"]["framework_version"] == "test"
        # verify numpy arrays survived round-trip
        np.testing.assert_array_equal(
            loaded.aggregated.per_run[0].threshold_free.fpr_grid,
            threshold_free.fpr_grid,
        )


class TestCodeExecAutoDump:
    """Verify finalize_evaluation_results calls persistence when result_dump_dir is set."""

    @pytest.mark.asyncio
    async def test_dump_called_with_dir(self, tmp_path: pathlib.Path) -> None:
        """Patch save_eval_result and verify it's called when result_dump_dir is provided."""
        import pyine.evals.code_exec._impl as impl_module

        # build minimal evaluator + stores
        evaluator = unittest.mock.MagicMock()
        evaluator.results = []
        evaluator.compute_metrics = unittest.mock.AsyncMock(return_value={})
        evaluator.get_attempt_groups = unittest.mock.MagicMock(return_value={})
        token_usage = pyine.evals.utils.TokenUsageInfo.get_default()
        with (
            unittest.mock.patch.object(
                pyine.evals.code_exec.utils,
                "get_metrics",
                new_callable=unittest.mock.AsyncMock,
                return_value={"acc": 0.9},
            ),
            unittest.mock.patch.object(
                pyine.evals.persistence,
                "maybe_dump_eval_result",
                wraps=pyine.evals.persistence.maybe_dump_eval_result,
            ) as mock_dump,
        ):
            await impl_module.finalize_evaluation_results(
                evaluator=evaluator,
                total_token_usage=token_usage,
                attempt_token_usage={},
                sample_data_store={},
                result_dump_dir=tmp_path,
                result_dump_overwrite=False,
                eval_subset_name="valid",
            )
            mock_dump.assert_called_once()
            assert mock_dump.call_args.kwargs["dump_dir"] == tmp_path
            assert mock_dump.call_args.kwargs["eval_subset_name"] == "valid"

    @pytest.mark.asyncio
    async def test_dump_is_noop_without_dir(self, tmp_path: pathlib.Path) -> None:
        """maybe_dump_eval_result is called but receives dump_dir=None, so it no-ops."""
        import pyine.evals.code_exec._impl as impl_module

        evaluator = unittest.mock.MagicMock()
        evaluator.results = []
        token_usage = pyine.evals.utils.TokenUsageInfo.get_default()
        with (
            unittest.mock.patch.object(
                pyine.evals.code_exec.utils,
                "get_metrics",
                new_callable=unittest.mock.AsyncMock,
                return_value={},
            ),
            unittest.mock.patch.object(
                pyine.evals.persistence,
                "maybe_dump_eval_result",
            ) as mock_dump,
        ):
            await impl_module.finalize_evaluation_results(
                evaluator=evaluator,
                total_token_usage=token_usage,
                attempt_token_usage={},
                sample_data_store={},
            )
            mock_dump.assert_called_once()
            assert mock_dump.call_args.kwargs["dump_dir"] is None

    @pytest.mark.asyncio
    async def test_finalize_result_keeps_eval_metadata(self) -> None:
        """finalize_evaluation_results should attach eval_metadata to returned result."""
        import pyine.evals.code_exec._impl as impl_module

        evaluator = unittest.mock.MagicMock()
        evaluator.results = []
        token_usage = pyine.evals.utils.TokenUsageInfo.get_default()
        eval_metadata = {
            "eval_subset_name": "valid",
            "eval_config": {"some_flag": True},
            "reprod_metadata": {"framework_version": "test"},
        }
        with unittest.mock.patch.object(
            pyine.evals.code_exec.utils,
            "get_metrics",
            new_callable=unittest.mock.AsyncMock,
            return_value={},
        ):
            result = await impl_module.finalize_evaluation_results(
                evaluator=evaluator,
                total_token_usage=token_usage,
                attempt_token_usage={},
                sample_data_store={},
                eval_metadata=eval_metadata,
            )
        assert result.eval_metadata == eval_metadata

    @pytest.mark.asyncio
    async def test_raises_without_eval_subset_name(self, tmp_path: pathlib.Path) -> None:
        """ValueError when result_dump_dir set but eval_subset_name is empty."""
        import pyine.evals.code_exec._impl as impl_module

        evaluator = unittest.mock.MagicMock()
        evaluator.results = []
        token_usage = pyine.evals.utils.TokenUsageInfo.get_default()
        with (
            unittest.mock.patch.object(
                pyine.evals.code_exec.utils,
                "get_metrics",
                new_callable=unittest.mock.AsyncMock,
                return_value={},
            ),
            pytest.raises(ValueError, match="eval_subset_name must be set"),
        ):
            await impl_module.finalize_evaluation_results(
                evaluator=evaluator,
                total_token_usage=token_usage,
                attempt_token_usage={},
                sample_data_store={},
                result_dump_dir=tmp_path,
            )


class TestCorrectnessAutoDump:
    """Verify correctness paths call persistence when result_dump_dir is set."""

    @pytest.mark.asyncio
    async def test_evaluate_guardrail_types_dumps_per_type(self, tmp_path: pathlib.Path) -> None:
        """Patch save_eval_result; verify called once per type with type_name in path."""
        import pyine.evals.correctness._impl as correctness_impl

        mock_result = _MockEvalResult(metrics={"auroc/mean": 0.9})
        # pretend it's a CorrectnessEvalResult for the purpose of the test
        with (
            unittest.mock.patch.object(
                correctness_impl,
                "evaluate_guardrail_replicas",
                new_callable=unittest.mock.AsyncMock,
                return_value=mock_result,
            ),
            unittest.mock.patch.object(
                pyine.evals.persistence,
                "maybe_dump_eval_result",
            ) as mock_dump,
        ):
            config = unittest.mock.MagicMock()
            config.result_dump_dir = tmp_path
            config.result_dump_overwrite = False
            datamodule = unittest.mock.MagicMock()
            guardrails_by_type = {"typeA": [unittest.mock.MagicMock()], "typeB": [unittest.mock.MagicMock()]}
            await correctness_impl.evaluate_guardrail_types(
                config=config,
                guardrails_by_type=guardrails_by_type,
                datamodule=datamodule,
                eval_subset_name="test",
                wandb_run=None,
            )
            assert mock_dump.call_count == 2
            type_names_in_calls = [call.kwargs["type_name"] for call in mock_dump.call_args_list]
            assert "typeA" in type_names_in_calls
            assert "typeB" in type_names_in_calls

    @pytest.mark.asyncio
    async def test_evaluate_guardrail_types_logs_type_names_metadata(self) -> None:
        """Verify _guardrail_type_names written to W&B summary after loop."""
        import pyine.evals.correctness._impl as correctness_impl

        mock_result = _MockEvalResult(metrics={"auroc/mean": 0.9})
        mock_wandb_run = unittest.mock.MagicMock()
        mock_wandb_run.summary = {}
        with (
            unittest.mock.patch.object(
                correctness_impl,
                "evaluate_guardrail_replicas",
                new_callable=unittest.mock.AsyncMock,
                return_value=mock_result,
            ),
            unittest.mock.patch.object(
                pyine.evals.persistence,
                "maybe_dump_eval_result",
            ),
        ):
            config = unittest.mock.MagicMock()
            config.result_dump_dir = None
            datamodule = unittest.mock.MagicMock()
            guardrails_by_type = {"typeA": [unittest.mock.MagicMock()], "typeB": [unittest.mock.MagicMock()]}
            await correctness_impl.evaluate_guardrail_types(
                config=config,
                guardrails_by_type=guardrails_by_type,
                datamodule=datamodule,
                eval_subset_name="test",
                wandb_run=mock_wandb_run,
            )
            assert mock_wandb_run.summary["benchmark/test/_guardrail_type_names"] == ["typeA", "typeB"]

    @pytest.mark.asyncio
    async def test_evaluate_wrapped_model_dumps_single_type(self, tmp_path: pathlib.Path) -> None:
        """Patch save_eval_result; verify called once without type_name in path."""
        import pyine.evals.correctness._impl as correctness_impl
        import pyine.evals.correctness.configs as correctness_configs
        import pyine.evals.correctness.datamodule as correctness_datamodule

        mock_result = unittest.mock.MagicMock(spec=correctness_impl.CorrectnessEvalResult)
        mock_result.metrics = {"auroc/mean": 0.9}
        with (
            unittest.mock.patch.object(
                correctness_impl,
                "evaluate_guardrail_replicas",
                new_callable=unittest.mock.AsyncMock,
                return_value=mock_result,
            ),
            unittest.mock.patch.object(
                pyine.evals.persistence,
                "maybe_dump_eval_result",
            ) as mock_dump,
        ):
            config = unittest.mock.MagicMock(spec=correctness_configs.CorrectnessEvalsConfig)
            config.result_dump_dir = tmp_path
            config.result_dump_overwrite = False
            # call the real evaluate_wrapped_model method
            real_method = correctness_configs.CorrectnessEvalsConfig.evaluate_wrapped_model
            mock_datamodule = unittest.mock.MagicMock(spec=correctness_datamodule.CorrectnessDataModule)
            mock_scorer = unittest.mock.MagicMock()
            await real_method(config, mock_scorer, mock_datamodule, "test")
            mock_dump.assert_called_once()
            call_args = mock_dump.call_args
            assert call_args.kwargs["dump_dir"] == tmp_path
            assert call_args.kwargs["eval_subset_name"] == "test"
            # type_name should not be passed (single-type path)
            assert call_args.kwargs.get("type_name") is None or "type_name" not in call_args.kwargs


def _make_mock_lmdb_reader(
    metadata: dict[str, typing.Any] | None = None,
    sample_count: int = 0,
) -> unittest.mock.MagicMock:
    """Create a MagicMock LMDBReader that supports context manager protocol."""
    reader = unittest.mock.MagicMock()
    reader.get_metadata.return_value = metadata or {"record_type": "benchmark"}
    reader.sample_count = sample_count
    reader.__enter__ = unittest.mock.MagicMock(return_value=reader)
    reader.__exit__ = unittest.mock.MagicMock(return_value=False)
    return reader


class TestReevalAutoDump:
    """Verify reevaluate_from_lmdb threads persistence params through."""

    @pytest.mark.asyncio
    async def test_empty_lmdb_paths_raises(self) -> None:
        """ValueError when lmdb_paths is empty."""
        import pyine.evals.code_exec.reeval as reeval_module

        with pytest.raises(ValueError, match="lmdb_paths must be non-empty"):
            await reeval_module.reevaluate_from_lmdb(lmdb_paths=[])

    @pytest.mark.asyncio
    async def test_dump_called_with_dir(self, tmp_path: pathlib.Path) -> None:
        """Patch save_eval_result; verify called when result_dump_dir + eval_subset_name provided."""
        import pyine.evals.code_exec.reeval as reeval_module

        mock_reader = _make_mock_lmdb_reader()
        with (
            unittest.mock.patch(
                "pyine.data.utils.lmdb_io.LMDBReader",
                return_value=mock_reader,
            ),
            unittest.mock.patch.object(
                pyine.evals.code_exec.utils,
                "get_metrics",
                new_callable=unittest.mock.AsyncMock,
                return_value={"acc": 0.9},
            ),
            unittest.mock.patch.object(
                pyine.evals.persistence,
                "maybe_dump_eval_result",
            ) as mock_dump,
        ):
            result = await reeval_module.reevaluate_from_lmdb(
                lmdb_paths=[tmp_path / "fake.lmdb"],
                eval_subset_name="valid",
                result_dump_dir=tmp_path,
            )
            mock_dump.assert_called_once()
            assert mock_dump.call_args.kwargs["dump_dir"] == tmp_path
            assert mock_dump.call_args.kwargs["eval_subset_name"] == "valid"
            assert result.eval_metadata["evaluation_backend"] == "lmdb_reeval"
            assert result.eval_metadata["eval_subset_name"] == "valid"
            assert "reprod_metadata" in result.eval_metadata

    @pytest.mark.asyncio
    async def test_metadata_mismatch_raises(self, tmp_path: pathlib.Path) -> None:
        """ValueError when eval_subset_name disagrees with LMDB metadata."""
        import pyine.evals.code_exec.reeval as reeval_module

        mock_reader = _make_mock_lmdb_reader(
            metadata={"record_type": "benchmark", "eval_subset_name": "test"},
        )
        with (
            unittest.mock.patch(
                "pyine.data.utils.lmdb_io.LMDBReader",
                return_value=mock_reader,
            ),
            pytest.raises(ValueError, match="disagrees with"),
        ):
            await reeval_module.reevaluate_from_lmdb(
                lmdb_paths=[tmp_path / "fake.lmdb"],
                eval_subset_name="valid",
            )

    @pytest.mark.asyncio
    async def test_metadata_missing_raises(self, tmp_path: pathlib.Path) -> None:
        """ValueError when LMDB metadata lacks eval_subset_name and dump is enabled."""
        import pyine.evals.code_exec.reeval as reeval_module

        mock_reader = _make_mock_lmdb_reader()
        with (
            unittest.mock.patch(
                "pyine.data.utils.lmdb_io.LMDBReader",
                return_value=mock_reader,
            ),
            pytest.raises(ValueError, match="could not be determined"),
        ):
            await reeval_module.reevaluate_from_lmdb(
                lmdb_paths=[tmp_path / "fake.lmdb"],
                result_dump_dir=tmp_path,
            )

    @pytest.mark.asyncio
    async def test_conflicting_metadata_across_lmdbs_raises(self, tmp_path: pathlib.Path) -> None:
        """ValueError when multiple LMDBs have different eval_subset_name in metadata."""
        import pyine.evals.code_exec.reeval as reeval_module

        metadata_values = [
            {"record_type": "benchmark", "eval_subset_name": "test"},
            {"record_type": "benchmark", "eval_subset_name": "valid"},
        ]
        call_count = 0

        def make_reader(path: typing.Any) -> unittest.mock.MagicMock:
            nonlocal call_count
            reader = _make_mock_lmdb_reader(metadata=metadata_values[call_count])
            call_count += 1
            return reader

        with (
            unittest.mock.patch(
                "pyine.data.utils.lmdb_io.LMDBReader",
                side_effect=make_reader,
            ),
            pytest.raises(ValueError, match="conflicting"),
        ):
            await reeval_module.reevaluate_from_lmdb(
                lmdb_paths=[tmp_path / "a.lmdb", tmp_path / "b.lmdb"],
            )

    @pytest.mark.asyncio
    async def test_metadata_derived_subset_name_with_dump(self, tmp_path: pathlib.Path) -> None:
        """eval_subset_name derived from metadata when caller omits it and dump is enabled."""
        import pyine.evals.code_exec.reeval as reeval_module

        mock_reader = _make_mock_lmdb_reader(
            metadata={"record_type": "benchmark", "eval_subset_name": "valid"},
        )
        with (
            unittest.mock.patch(
                "pyine.data.utils.lmdb_io.LMDBReader",
                return_value=mock_reader,
            ),
            unittest.mock.patch.object(
                pyine.evals.code_exec.utils,
                "get_metrics",
                new_callable=unittest.mock.AsyncMock,
                return_value={"acc": 0.9},
            ),
            unittest.mock.patch.object(
                pyine.evals.persistence,
                "maybe_dump_eval_result",
            ) as mock_dump,
        ):
            await reeval_module.reevaluate_from_lmdb(
                lmdb_paths=[tmp_path / "fake.lmdb"],
                result_dump_dir=tmp_path,
                # eval_subset_name intentionally omitted -- should be derived from metadata
            )
            mock_dump.assert_called_once()
            assert mock_dump.call_args.kwargs["eval_subset_name"] == "valid"

    @pytest.mark.asyncio
    async def test_partial_metadata_raises_unconditionally(self, tmp_path: pathlib.Path) -> None:
        """ValueError when some LMDBs have eval_subset_name and others don't."""
        import pyine.evals.code_exec.reeval as reeval_module

        metadata_values = [
            {"record_type": "benchmark", "eval_subset_name": "valid"},
            {"record_type": "benchmark"},  # missing eval_subset_name
        ]
        call_count = 0

        def make_reader(path: typing.Any) -> unittest.mock.MagicMock:
            nonlocal call_count
            reader = _make_mock_lmdb_reader(metadata=metadata_values[call_count])
            call_count += 1
            return reader

        with (
            unittest.mock.patch(
                "pyine.data.utils.lmdb_io.LMDBReader",
                side_effect=make_reader,
            ),
            pytest.raises(ValueError, match="partially missing"),
        ):
            await reeval_module.reevaluate_from_lmdb(
                lmdb_paths=[tmp_path / "a.lmdb", tmp_path / "b.lmdb"],
            )
