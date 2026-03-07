"""Tests for pyine.evals.correctness.configs."""

from __future__ import annotations

import pathlib

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
        # 4 base configs + 7 presets * 2 groups = 18 total
        assert len(configs) == 18

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
            "skewed_pos",
            "balanced",
            "original_only",
            "mostly_original",
            "helpful_bias",
            "skewed_pos_helpful_bias",
            "oversample_balanced",
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
        assert config.target_fpr_values == [0.001, 0.01, 0.05]
        assert config.num_bootstrap_replicates == 1000
        assert config.confidence_level == 0.95

    def test_prepare_eval_datamodule_rejects_provided_datamodule(self) -> None:
        import unittest.mock

        config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=_make_datamodule_config(),
        )
        mock_dm = unittest.mock.MagicMock()
        with pytest.raises(ValueError, match="should not provide one"):
            config.prepare_eval_datamodule(mock_dm)


class TestGetDatamoduleConfigs:
    def test_total_config_count(self) -> None:
        configs = correctness_configs.get_datamodule_configs("test_group")
        # 1 base config + 1 base resampling config + 7 presets = 9 total
        assert len(configs) == 9

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
