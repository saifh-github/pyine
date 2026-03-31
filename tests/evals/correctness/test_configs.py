"""Tests for pyine.evals.correctness.configs."""

from __future__ import annotations

import pathlib
import unittest.mock

import pydantic
import pytest

import pyine.evals.correctness.configs as correctness_configs
import pyine.evals.correctness.datamodule_configs as correctness_datamodule_configs
import pyine.evals.correctness.types as correctness_types

_FAKE_LMDB_PATH = pathlib.Path("/fake/lmdb")


def _make_datamodule_config() -> correctness_datamodule_configs.CorrectnessDataModuleConfig:
    return correctness_datamodule_configs.CorrectnessDataModuleConfig(
        lmdb_paths=(_FAKE_LMDB_PATH,),
        split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
    )


class TestLabelType:
    def test_hard_match(self) -> None:
        assert correctness_types.LabelType.HARD_MATCH == "hard_match"

    def test_soft_match(self) -> None:
        assert correctness_types.LabelType.SOFT_MATCH == "soft_match"


class TestGuardrailSplitConfig:
    def test_valid(self) -> None:
        config = correctness_types.GuardrailSplitConfig(
            split_source="TACO",
            guardrail_valid_fraction=0.3,
        )
        assert config.guardrail_valid_fraction == 0.3

    def test_fraction_zero_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="must be in"):
            correctness_types.GuardrailSplitConfig(
                split_source="TACO",
                guardrail_valid_fraction=0.0,
            )

    def test_fraction_one_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="must be in"):
            correctness_types.GuardrailSplitConfig(
                split_source="TACO",
                guardrail_valid_fraction=1.0,
            )


class TestRecordCategoryConfig:
    def test_defaults(self) -> None:
        config = correctness_configs.RecordCategoryConfig()
        assert "original" in config.code_type_to_category
        assert config.code_type_to_category["original"] == "regular"
        assert config.report_per_code_type is True


class TestGetEvalsConfigs:
    def test_total_config_count(self) -> None:
        configs = correctness_configs.get_evals_configs("test_group")
        # 4 base configs + 6 presets * 2 groups = 16 total
        assert len(configs) == 16

    def test_split_source_prefills_nested_datamodule_config(self) -> None:
        configs = correctness_configs.get_evals_configs("test_group", split_source="TACO")
        dm_config = next(cfg for cfg in configs if cfg.name == "correctness_dm_base")
        split_config_factory = dm_config.config.__dataclass_fields__["split_config"].default_factory
        assert split_config_factory() == {"split_source": "TACO"}

    def test_custom_base_and_datamodule_names_are_supported(self) -> None:
        configs = correctness_configs.get_evals_configs(
            "test_group",
            split_source="TACO",
            base_name="correctness_taco_base",
            datamodule_name="correctness_taco_dm_base",
        )
        assert any(cfg.name == "correctness_taco_base" and cfg.group == "test_group" for cfg in configs)
        assert any(
            cfg.name == "correctness_taco_dm_base" and cfg.group == "test_group/datamodule_config" for cfg in configs
        )

    def test_preset_names_registered_in_both_groups(self) -> None:
        configs = correctness_configs.get_evals_configs("test_group")
        preset_names = {
            "skewed_weak_bias",
            "balanced_weak_bias",
            "skewed_moderate_bias",
            "balanced_moderate_bias",
            "skewed_strong_bias",
            "balanced_strong_bias",
        }
        cal_names = {cfg.name for cfg in configs if cfg.group == "test_group/calibration_resampling"}
        train_names = {cfg.name for cfg in configs if cfg.group == "test_group/datamodule_config/resampling"}
        assert preset_names.issubset(cal_names)
        assert preset_names.issubset(train_names)


class TestCorrectnessEvalsConfig:
    def test_target_fpr_values_empty_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="must not be empty"):
            correctness_configs.CorrectnessEvalsConfig(
                datamodule_config=_make_datamodule_config(),
                target_fpr_values=[],
            )

    def test_target_fpr_values_unsorted_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="must be sorted"):
            correctness_configs.CorrectnessEvalsConfig(
                datamodule_config=_make_datamodule_config(),
                target_fpr_values=[0.05, 0.01],
            )

    def test_target_fpr_values_duplicates_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="must be unique"):
            correctness_configs.CorrectnessEvalsConfig(
                datamodule_config=_make_datamodule_config(),
                target_fpr_values=[0.01, 0.01],
            )

    def test_target_fpr_values_out_of_range_raises(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="must be in"):
            correctness_configs.CorrectnessEvalsConfig(
                datamodule_config=_make_datamodule_config(),
                target_fpr_values=[0.0, 0.5],
            )

    def test_valid_config(self) -> None:
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=_make_datamodule_config(),
        )
        assert config.target_fpr_values == [0.001, 0.01, 0.05, 0.1]
        assert config.num_bootstrap_replicates == 1000
        assert config.confidence_level == 0.95

    def test_prepare_eval_datamodule_rejects_provided_datamodule(self) -> None:
        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=_make_datamodule_config(),
        )
        mock_dm = unittest.mock.MagicMock()
        with pytest.raises(ValueError, match="should not provide one"):
            config.prepare_eval_datamodule(mock_dm)

    def test_log_metrics_writes_compact_summary_and_tables(self) -> None:
        import numpy as np

        import pyine.evals.correctness._impl as correctness_impl
        import pyine.evals.correctness.types as correctness_types

        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=_make_datamodule_config(),
            target_fpr_values=[0.05],
        )
        mock_wandb_run = unittest.mock.MagicMock()
        mock_wandb_run.summary = {}
        # build a minimal CorrectnessEvalResult
        attempt_metrics = {
            0.05: correctness_types.ThresholdedMetrics(
                target_fpr=0.05,
                threshold=0.5,
                tp=40,
                fp=2,
                tn=198,
                fn=10,
                tpr=0.8,
                fpr=0.01,
                fnr=0.2,
                precision=40 / 42,
                npv=198 / 208,
            ),
        }
        sample_metrics = {
            0.05: correctness_types.SampleLevelMetrics(
                target_fpr=0.05,
                base_pass_rate=0.9,
                guarded_pass_rate=0.84,
                unsafe_slip_rate=0.05,
                total_block_rate=0.02,
                best_of_k_success_rate=0.95,
                cons_pass_rate=0.7,
                cons_unsafe_slip_rate=0.03,
                cons_justified_reject_rate=0.8,
            ),
        }
        run_result = correctness_types.SingleRunResult(
            guardrail_metadata={},
            threshold_free=correctness_types.ThresholdFreeMetrics(
                auroc=0.91,
                average_precision=0.86,
                tpr_at_fpr={0.05: 0.73},
                fpr_grid=np.linspace(0, 1, 10),
                tpr_grid=np.linspace(0, 1, 10),
                precision_grid=np.linspace(1, 0.5, 10),
                recall_grid=np.linspace(0, 1, 10),
            ),
            attempt_metrics=attempt_metrics,
            sample_metrics=sample_metrics,
            category_results={},
            bootstrap_cis={},
            difficulty_stats=None,
            verification_cost_stats=None,
        )
        aggregated = correctness_types.AggregatedResult(
            split_summary={"test_record_count": 500},
            class_balance=correctness_types.ClassBalanceStats(
                overall_positive_rate=0.6,
                per_sample_positive_rates=[0.6] * 100,
                num_all_correct_samples=0,
                num_all_incorrect_samples=0,
                code_type_proportions={"original": 1.0},
                predict_type_proportions={},
            ),
            per_run=[run_result],
            cross_run_mean={
                "auroc": 0.91,
                "average_precision": 0.86,
                "tpr_at_fpr_0_05": 0.73,
                "fpr_0_05/tpr": 0.80,
                "fpr_0_05/guarded_pass_rate": 0.84,
                "fpr_0_05/unsafe_slip_rate": 0.05,
                "fpr_0_05/best_of_k_success_rate": 0.95,
                "fpr_0_05/cons_pass_rate": 0.7,
                "fpr_0_05/cons_unsafe_slip_rate": 0.03,
                "fpr_0_05/cons_justified_reject_rate": 0.8,
            },
            cross_run_std={},
            cross_run_p5={},
            cross_run_num_valid={},
            hierarchical_cis={},
            difficulty_stats=None,
            verification_cost_stats=None,
        )
        subset_result = correctness_impl.CorrectnessEvalResult(
            metrics=aggregated.to_flat_dict(),
            eval_metadata={},
            aggregated=aggregated,
        )
        config.log_metrics(
            wandb_run=mock_wandb_run,
            results_by_subset={"guardrail_test": subset_result},
        )
        # verify compact summary keys are present
        assert "benchmark/guardrail_test/auroc/mean" in mock_wandb_run.summary
        assert "benchmark/guardrail_test/fpr_0_05/tpr/mean" in mock_wandb_run.summary
        # verify category stat keys are NOT in summary (they're in tables now)
        assert not any("category/" in key for key in mock_wandb_run.summary)
        # verify wandb_run.log was called (for tables)
        assert mock_wandb_run.log.called


class TestGetDatamoduleConfigs:
    def test_total_config_count(self) -> None:
        configs = correctness_configs.get_datamodule_configs("test_group")
        # 1 base config + 1 base resampling config + 6 presets = 8 total
        assert len(configs) == 8

    def test_split_source_prefills_base_datamodule_config(self) -> None:
        configs = correctness_configs.get_datamodule_configs("test_group", split_source="TACO")
        base_config = next(cfg for cfg in configs if cfg.name == "correctness_base")
        split_config_factory = base_config.config.__dataclass_fields__["split_config"].default_factory
        assert split_config_factory() == {"split_source": "TACO"}

    def test_custom_base_name_is_supported(self) -> None:
        configs = correctness_configs.get_datamodule_configs(
            "test_group",
            split_source="TACO",
            base_name="correctness_taco_base",
        )
        assert any(cfg.name == "correctness_taco_base" and cfg.group == "test_group" for cfg in configs)
