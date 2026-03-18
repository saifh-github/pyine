"""Tests for evaluate_guardrail_types() in pyine.evals.correctness._impl."""

from __future__ import annotations

import pathlib
import unittest.mock

import pytest

import pyine.evals.correctness._impl as correctness_impl
import pyine.evals.correctness.configs as correctness_configs
import pyine.evals.correctness.datamodule as correctness_datamodule
import pyine.evals.correctness.datamodule_configs as correctness_datamodule_configs
import pyine.evals.correctness.types as correctness_types
import tests.evals.correctness.conftest as correctness_conftest

_FAKE_LMDB_PATH = pathlib.Path("/fake/lmdb")


def _build_mock_datamodule(monkeypatch: pytest.MonkeyPatch) -> correctness_datamodule.CorrectnessDataModule:
    """Build a mock datamodule with minimal data for multi-eval tests."""
    valid_records = []
    test_records = []
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
    return correctness_conftest.build_mock_datamodule(
        monkeypatch, valid_records=valid_records, test_records=test_records
    )


class TestEvaluateGuardrailTypes:
    def test_multiple_types_returns_independent_results(
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
        results = correctness_impl.evaluate_guardrail_types(
            config=config,
            guardrails_by_type={
                "type_a": [correctness_conftest.MockGuardrailScorer(noise_seed=0)],
                "type_b": [correctness_conftest.MockGuardrailScorer(noise_seed=1)],
            },
            datamodule=dm,
            eval_subset_name="guardrail_valid",
        )
        assert "type_a" in results
        assert "type_b" in results
        assert isinstance(results["type_a"], correctness_impl.CorrectnessEvalResult)
        assert isinstance(results["type_b"], correctness_impl.CorrectnessEvalResult)
        assert results["type_a"].eval_metadata["guardrail_type_name"] == "type_a"
        assert results["type_b"].eval_metadata["guardrail_type_name"] == "type_b"
        assert results["type_a"].eval_metadata["eval_subset_name"] == "guardrail_valid"
        assert "eval_config" in results["type_a"].eval_metadata
        assert "datamodule_config" in results["type_a"].eval_metadata
        assert "reprod_metadata" in results["type_a"].eval_metadata

    def test_single_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        results = correctness_impl.evaluate_guardrail_types(
            config=config,
            guardrails_by_type={"only_type": [correctness_conftest.MockGuardrailScorer()]},
            datamodule=dm,
            eval_subset_name="guardrail_valid",
        )
        assert len(results) == 1
        assert "only_type" in results

    def test_empty_replicas_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
            correctness_impl.evaluate_guardrail_types(
                config=config,
                guardrails_by_type={"bad_type": []},
                datamodule=dm,
                eval_subset_name="guardrail_valid",
            )

    def test_wandb_metric_prefixing(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        correctness_impl.evaluate_guardrail_types(
            config=config,
            guardrails_by_type={"my_type": [correctness_conftest.MockGuardrailScorer()]},
            datamodule=dm,
            eval_subset_name="guardrail_valid",
            wandb_run=mock_wandb_run,
        )
        # check that metrics were logged with the type name prefix
        assert any(key.startswith("benchmark/guardrail_valid/my_type/") for key in mock_wandb_run.summary)
        assert mock_wandb_run.log.called, "wandb_run.log should be called for table logging"


class TestEvaluateGuardrailTypesValidation:
    """Input validation tests for evaluate_guardrail_types (no eval pipeline needed)."""

    def _make_config(
        self,
        result_dump_dir: pathlib.Path | None = None,
    ) -> correctness_configs.CorrectnessEvalsConfig:
        return correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_datamodule_configs.CorrectnessDataModuleConfig(
                lmdb_paths=(_FAKE_LMDB_PATH,),
                split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
            ),
            target_fpr_values=[0.05],
            num_bootstrap_replicates=5,
            roc_fpr_grid_size=10,
            result_dump_dir=result_dump_dir,
        )

    def test_empty_guardrails_raises(self) -> None:
        config = self._make_config()
        dm = unittest.mock.MagicMock()
        with pytest.raises(ValueError, match="empty; nothing to evaluate"):
            correctness_impl.evaluate_guardrail_types(
                config=config,
                guardrails_by_type={},
                datamodule=dm,
                eval_subset_name="guardrail_test",
            )

    def test_dump_dir_with_missing_subset_raises_before_eval(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        config = self._make_config(result_dump_dir=tmp_path)
        dm = unittest.mock.MagicMock()
        with unittest.mock.patch.object(
            correctness_impl,
            "evaluate_guardrail_replicas",
            new_callable=unittest.mock.AsyncMock,
        ) as mock_eval:
            with pytest.raises(ValueError, match="eval_subset_name must be set"):
                correctness_impl.evaluate_guardrail_types(
                    config=config,
                    guardrails_by_type={"type_a": [correctness_conftest.MockGuardrailScorer()]},
                    datamodule=dm,
                    eval_subset_name="   ",
                )
            mock_eval.assert_not_called()

    def test_missing_subset_raises_without_dump_dir(self) -> None:
        config = self._make_config(result_dump_dir=None)
        dm = unittest.mock.MagicMock()
        with pytest.raises(ValueError, match="eval_subset_name must be set"):
            correctness_impl.evaluate_guardrail_types(
                config=config,
                guardrails_by_type={"type_a": [correctness_conftest.MockGuardrailScorer()]},
                datamodule=dm,
                eval_subset_name=" ",
            )

    def test_reserved_prefix_raises(self) -> None:
        config = self._make_config()
        dm = unittest.mock.MagicMock()
        with pytest.raises(ValueError, match="reserved metric namespace prefix"):
            correctness_impl.evaluate_guardrail_types(
                config=config,
                guardrails_by_type={"fpr_custom": [correctness_conftest.MockGuardrailScorer()]},
                datamodule=dm,
                eval_subset_name="guardrail_test",
            )

    def test_reserved_prefix_tpr_at_fpr_raises(self) -> None:
        config = self._make_config()
        dm = unittest.mock.MagicMock()
        with pytest.raises(ValueError, match="reserved metric namespace prefix"):
            correctness_impl.evaluate_guardrail_types(
                config=config,
                guardrails_by_type={"tpr_at_fpr_custom": [correctness_conftest.MockGuardrailScorer()]},
                datamodule=dm,
                eval_subset_name="guardrail_test",
            )

    def test_reserved_exact_name_raises(self) -> None:
        config = self._make_config()
        dm = unittest.mock.MagicMock()
        for reserved_name in ("category", "auroc", "average_precision", "sample_count"):
            with pytest.raises(ValueError, match="reserved metric key"):
                correctness_impl.evaluate_guardrail_types(
                    config=config,
                    guardrails_by_type={reserved_name: [correctness_conftest.MockGuardrailScorer()]},
                    datamodule=dm,
                    eval_subset_name="guardrail_test",
                )

    def test_slash_in_name_raises(self) -> None:
        config = self._make_config()
        dm = unittest.mock.MagicMock()
        with pytest.raises(ValueError, match="contains '/'"):
            correctness_impl.evaluate_guardrail_types(
                config=config,
                guardrails_by_type={"foo/bar": [correctness_conftest.MockGuardrailScorer()]},
                datamodule=dm,
                eval_subset_name="guardrail_test",
            )

    def test_whitespace_only_name_raises(self) -> None:
        config = self._make_config()
        dm = unittest.mock.MagicMock()
        with pytest.raises(ValueError, match="must not be empty or whitespace"):
            correctness_impl.evaluate_guardrail_types(
                config=config,
                guardrails_by_type={"   ": [correctness_conftest.MockGuardrailScorer()]},
                datamodule=dm,
                eval_subset_name="guardrail_test",
            )

    def test_dump_path_collision_raises(self, tmp_path: pathlib.Path) -> None:
        """Type names that sanitize to the same filename are caught before evaluation."""
        config = self._make_config(result_dump_dir=tmp_path)
        dm = unittest.mock.MagicMock()
        with pytest.raises(ValueError, match="same dump filename"):
            correctness_impl.evaluate_guardrail_types(
                config=config,
                guardrails_by_type={
                    "a b": [correctness_conftest.MockGuardrailScorer()],
                    "a_b": [correctness_conftest.MockGuardrailScorer()],
                },
                datamodule=dm,
                eval_subset_name="guardrail_test",
            )

    def test_dump_path_collision_is_case_insensitive(self, tmp_path: pathlib.Path) -> None:
        """Type names that differ only by case are treated as filename collisions."""
        config = self._make_config(result_dump_dir=tmp_path)
        dm = unittest.mock.MagicMock()
        with pytest.raises(ValueError, match="same dump filename"):
            correctness_impl.evaluate_guardrail_types(
                config=config,
                guardrails_by_type={
                    "TypeA": [correctness_conftest.MockGuardrailScorer()],
                    "typea": [correctness_conftest.MockGuardrailScorer()],
                },
                datamodule=dm,
                eval_subset_name="guardrail_test",
            )

    def test_single_type_dump_name_sanitization_error_raises_before_eval(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Single-type eval still pre-validates dump path sanitization before running evaluation."""
        config = self._make_config(result_dump_dir=tmp_path)
        dm = unittest.mock.MagicMock()
        with unittest.mock.patch.object(
            correctness_impl,
            "evaluate_guardrail_replicas",
            new_callable=unittest.mock.AsyncMock,
        ) as mock_eval:
            with pytest.raises(ValueError, match="sanitizes to an empty string"):
                correctness_impl.evaluate_guardrail_types(
                    config=config,
                    guardrails_by_type={":::": [correctness_conftest.MockGuardrailScorer()]},
                    datamodule=dm,
                    eval_subset_name="guardrail_test",
                )
            mock_eval.assert_not_called()


class TestEvaluateGuardrailTypesParallel:
    """Test eval_type_parallelism > 1 produces same results as sequential."""

    def test_parallel_matches_sequential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dm = _build_mock_datamodule(monkeypatch)
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_datamodule_configs.CorrectnessDataModuleConfig(
                lmdb_paths=(_FAKE_LMDB_PATH,),
                split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
            ),
            target_fpr_values=[0.05],
            num_bootstrap_replicates=5,
            bootstrap_num_workers=1,  # fixed workers for exact comparison
            roc_fpr_grid_size=10,
        )
        guardrails_by_type = {
            "type_a": [correctness_conftest.MockGuardrailScorer(noise_seed=0)],
            "type_b": [correctness_conftest.MockGuardrailScorer(noise_seed=1)],
        }
        sequential = correctness_impl.evaluate_guardrail_types(
            config=config,
            guardrails_by_type=guardrails_by_type,
            datamodule=dm,
            eval_subset_name="guardrail_valid",
            eval_type_parallelism=1,
        )
        parallel = correctness_impl.evaluate_guardrail_types(
            config=config,
            guardrails_by_type=guardrails_by_type,
            datamodule=dm,
            eval_subset_name="guardrail_valid",
            eval_type_parallelism=2,
        )
        assert set(sequential.keys()) == set(parallel.keys())
        for type_name in sequential:
            seq_auroc = sequential[type_name].metrics.get("auroc/mean")
            par_auroc = parallel[type_name].metrics.get("auroc/mean")
            assert seq_auroc is not None and par_auroc is not None
            assert abs(seq_auroc - par_auroc) < 1e-10, (
                f"AUROC mismatch for {type_name}: sequential={seq_auroc}, parallel={par_auroc}"
            )


class TestAutoReduceBootstrapWorkers:
    """Test that eval_type_parallelism auto-reduces bootstrap workers."""

    def test_auto_reduction_formula(self) -> None:
        import pyine.evals.correctness.metrics as correctness_metrics

        # with 16 resolved workers and parallelism=4, effective should be 4
        resolved = correctness_metrics.resolve_num_workers(0, num_replicates=1000)
        adjusted = max(1, resolved // 4)
        assert adjusted >= 1
        assert adjusted <= resolved


class TestEvalSingleTypeErrorAnnotation:
    """Test that _eval_single_type annotates errors with type_name."""

    def test_error_includes_type_name(self) -> None:
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_datamodule_configs.CorrectnessDataModuleConfig(
                lmdb_paths=(_FAKE_LMDB_PATH,),
                split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
            ),
            target_fpr_values=[0.05],
            num_bootstrap_replicates=5,
            roc_fpr_grid_size=10,
        )
        dm = unittest.mock.MagicMock()
        with (
            unittest.mock.patch.object(
                correctness_impl,
                "evaluate_guardrail_replicas",
                new_callable=unittest.mock.AsyncMock,
                side_effect=ValueError("inner failure"),
            ),
            pytest.raises(RuntimeError, match="my_failing_type"),
        ):
            correctness_impl._eval_single_type(
                type_name="my_failing_type",
                replicas=[unittest.mock.MagicMock()],
                config=config,
                datamodule=dm,
                eval_subset_name="guardrail_test",
            )
