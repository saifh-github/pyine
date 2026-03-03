"""Tests for LLMClassifierTrainerAppMainConfig and Hydra config registration."""

from __future__ import annotations

import typing
import warnings

if typing.TYPE_CHECKING:
    import pathlib

import pydantic
import pytest
import torch

from pyine.apps.trainers.llm_classifier_trainer_configs import (
    LLMClassifierTrainerAppMainConfig,
)


def _make_minimal_training_args(**overrides: object) -> dict:
    """Minimal valid TrainingArgsConfig kwargs."""
    base: dict[str, object] = {
        "output_dir": "/tmp/test-output",  # noqa: S108
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "report_to": "none",
        "use_cpu": True,
    }
    base.update(overrides)
    return base


def _make_minimal_config(**overrides: object) -> dict:
    """Minimal valid config kwargs for LLMClassifierTrainerAppMainConfig."""
    base: dict[str, object] = {
        "base_model": "some-model",
        "datamodule_config": {
            "lmdb_path": "/tmp/fake-lmdb",  # noqa: S108
        },
        "training_args_config": _make_minimal_training_args(),
    }
    base.update(overrides)
    return base


class TestLLMClassifierTrainerAppMainConfig:
    def test_valid_config_construction(self) -> None:
        cfg = LLMClassifierTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.base_model == "some-model"
        assert cfg.datamodule_config.lmdb_path == "/tmp/fake-lmdb"  # noqa: S108
        assert cfg.num_labels == 2

    def test_datamodule_config_required(self) -> None:
        kwargs = _make_minimal_config()
        del kwargs["datamodule_config"]
        with pytest.raises(pydantic.ValidationError):
            LLMClassifierTrainerAppMainConfig(**kwargs)

    def test_training_args_config_required(self) -> None:
        kwargs = _make_minimal_config()
        del kwargs["training_args_config"]
        with pytest.raises(pydantic.ValidationError):
            LLMClassifierTrainerAppMainConfig(**kwargs)

    def test_target_dtype_from_training_args_bf16(self) -> None:
        cfg = LLMClassifierTrainerAppMainConfig(
            **_make_minimal_config(
                training_args_config=_make_minimal_training_args(bf16=True, use_cpu=False),
            )
        )
        assert cfg.target_dtype == torch.bfloat16

    def test_target_dtype_from_training_args_fp16(self) -> None:
        cfg = LLMClassifierTrainerAppMainConfig(
            **_make_minimal_config(
                training_args_config=_make_minimal_training_args(fp16=True, use_cpu=False),
            )
        )
        assert cfg.target_dtype == torch.float16

    def test_target_dtype_from_training_args_default(self) -> None:
        cfg = LLMClassifierTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.target_dtype == torch.float32

    def test_label_mapping_consistency_id2label(self) -> None:
        with pytest.raises(ValueError, match="id2label"):
            LLMClassifierTrainerAppMainConfig(
                **_make_minimal_config(
                    num_labels=2,
                    id2label={0: "a", 1: "b", 2: "c"},
                )
            )

    def test_label_mapping_consistency_label2id(self) -> None:
        with pytest.raises(ValueError, match="label2id"):
            LLMClassifierTrainerAppMainConfig(
                **_make_minimal_config(
                    num_labels=2,
                    label2id={"a": 0, "b": 1, "c": 2},
                )
            )

    def test_num_labels_default_is_2(self) -> None:
        cfg = LLMClassifierTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.num_labels == 2

    def test_save_model_default_is_true(self) -> None:
        cfg = LLMClassifierTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.save_model is True

    def test_truncation_side_default_not_left(self) -> None:
        cfg = LLMClassifierTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.tokenizer_override_truncation_to_left_side is False

    def test_evals_config_optional(self) -> None:
        cfg = LLMClassifierTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.evals_config is None

    def test_class_weight_with_label_balance_warns(self) -> None:
        from pyine.guardrails.data.datamodule_configs import LabelBalanceConfig

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LLMClassifierTrainerAppMainConfig(
                **_make_minimal_config(
                    class_weight_mode="balanced",
                    datamodule_config={
                        "lmdb_path": "/tmp/fake-lmdb",  # noqa: S108
                        "label_balance": LabelBalanceConfig(target_positive_ratio=0.5),
                    },
                )
            )
            balance_warnings = [x for x in w if "double correction" in str(x.message)]
            assert len(balance_warnings) >= 1

    def test_max_seq_length_default(self) -> None:
        cfg = LLMClassifierTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.max_seq_length == 3000

    def test_log_per_code_type_metrics_default(self) -> None:
        cfg = LLMClassifierTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.log_per_code_type_metrics is True

    @pytest.mark.slow
    def test_get_model_returns_sequence_classification(self) -> None:
        cfg = LLMClassifierTrainerAppMainConfig(**_make_minimal_config(base_model="prajjwal1/bert-tiny"))
        model = cfg.get_model()
        assert hasattr(model, "classifier") or hasattr(model, "score")
        assert model.config.num_labels == 2

    @pytest.mark.slow
    def test_get_model_with_lora(self) -> None:
        import peft

        cfg = LLMClassifierTrainerAppMainConfig(
            **_make_minimal_config(
                base_model="prajjwal1/bert-tiny",
                lora_config=peft.LoraConfig(
                    r=4,
                    lora_alpha=8,
                    target_modules=["query", "value"],
                ),
            )
        )
        model = cfg.get_model()
        assert isinstance(model, peft.PeftModel)


class TestClassifierEvalOnlyConfig:
    def test_checkpoint_path_accepted(self) -> None:
        cfg = LLMClassifierTrainerAppMainConfig(
            **_make_minimal_config(
                classifier_checkpoint_path="/tmp/fake-checkpoint",  # noqa: S108
            )
        )
        assert cfg.classifier_checkpoint_path is not None

    def test_checkpoint_path_default_is_none(self) -> None:
        cfg = LLMClassifierTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.classifier_checkpoint_path is None


class TestHydraConfigRegistration:
    def test_register_hydra_configs_no_errors(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import pyine.apps.trainers.llm_classifier_trainer_configs as llm_cfg
        import pyine.evals.common
        import pyine.utils.filesystem

        monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
        configs = llm_cfg.register_hydra_configs(pyine.evals.common.EvalType.CODE_EXEC)
        assert len(configs) > 0

        entrypoint = next(
            (cfg for cfg in configs if cfg.name == "entrypoint" and cfg.group is None),
            None,
        )
        assert entrypoint is not None

    def test_register_hydra_configs_includes_datamodule(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import pyine.apps.trainers.llm_classifier_trainer_configs as llm_cfg
        import pyine.evals.common
        import pyine.utils.filesystem

        monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
        configs = llm_cfg.register_hydra_configs(pyine.evals.common.EvalType.CODE_EXEC)

        dm_config = next(
            (cfg for cfg in configs if cfg.name == "probe_base" and cfg.group == "config/datamodule_config"),
            None,
        )
        assert dm_config is not None

    def test_register_hydra_configs_includes_training_args(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import pyine.apps.trainers.llm_classifier_trainer_configs as llm_cfg
        import pyine.evals.common
        import pyine.utils.filesystem

        monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
        configs = llm_cfg.register_hydra_configs(pyine.evals.common.EvalType.CODE_EXEC)

        ta_config = next(
            (cfg for cfg in configs if cfg.name == "train_default" and cfg.group == "config/training_args_config"),
            None,
        )
        assert ta_config is not None
