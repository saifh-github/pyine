"""Tests for ProbeTrainerAppMainConfig and Hydra config registration."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import pathlib

import pytest

import pyine.evals.correctness.configs as correctness_configs
import pyine.evals.correctness.datamodule_configs as correctness_datamodule_configs
import pyine.evals.correctness.types as correctness_types
from pyine.apps.trainers.probe_trainer_configs import ProbeTrainerAppMainConfig
from pyine.guardrails.probes.base import ProbeConfig


def _make_minimal_config(**overrides: object) -> dict:
    """Minimal valid config kwargs for ProbeTrainerAppMainConfig."""
    base: dict[str, object] = {
        "base_model": "some-model",
        "datamodule_config": {
            "lmdb_path": "/tmp/fake-lmdb",  # noqa: S108
        },
        "probe_configs": [
            ProbeConfig(name="mean_L0", architecture="mean", layer=0),
        ],
    }
    base.update(overrides)
    return base


class TestProbeTrainerAppMainConfig:
    def test_valid_config_construction(self) -> None:
        cfg = ProbeTrainerAppMainConfig(**_make_minimal_config())
        assert len(cfg.probe_configs) == 1
        assert cfg.datamodule_config.lmdb_path == "/tmp/fake-lmdb"  # noqa: S108

    def test_duplicate_probe_names_raises(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            ProbeTrainerAppMainConfig(
                **_make_minimal_config(
                    probe_configs=[
                        ProbeConfig(name="dup", architecture="mean", layer=0),
                        ProbeConfig(name="dup", architecture="max", layer=0),
                    ]
                )
            )

    def test_evals_config_optional(self) -> None:
        cfg = ProbeTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.evals_config is None

    def test_save_best_probe_checkpoint_default_is_true(self) -> None:
        cfg = ProbeTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.save_best_probe_checkpoint is True

    def test_save_best_probe_checkpoint_requires_save_probes(self) -> None:
        with pytest.raises(ValueError, match="save_best_probe_checkpoint=True requires save_probes=True"):
            ProbeTrainerAppMainConfig(
                **_make_minimal_config(
                    save_probes=False,
                    save_best_probe_checkpoint=True,
                )
            )

    def test_save_best_probe_checkpoint_requires_num_epochs(self) -> None:
        with pytest.raises(ValueError, match="save_best_probe_checkpoint=True requires num_epochs >= 1"):
            ProbeTrainerAppMainConfig(
                **_make_minimal_config(
                    num_epochs=0,
                    save_best_probe_checkpoint=True,
                )
            )

    def test_target_dtype_property(self) -> None:
        import torch

        cfg = ProbeTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.target_dtype in (torch.bfloat16, torch.float16)

    def test_datamodule_config_accessible(self) -> None:
        from pyine.guardrails.data.datamodule_configs import ProbeDataModuleConfig

        cfg = ProbeTrainerAppMainConfig(**_make_minimal_config())
        assert isinstance(cfg.datamodule_config, ProbeDataModuleConfig)
        assert cfg.datamodule_config.selection_strategy == "latest"

    def test_text_field_default(self) -> None:
        cfg = ProbeTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.text_field == "model_output"

    def test_text_field_must_match_evals_config(self) -> None:
        evals_config = correctness_configs.CorrectnessEvalsConfig(
            datamodule_config=correctness_datamodule_configs.CorrectnessDataModuleConfig(
                lmdb_paths=("/tmp/fake-correctness-lmdb",),  # noqa: S108
                split_config=correctness_types.GuardrailSplitConfig(split_source="TACO"),
            ),
            text_field="final_answer",
        )

        with pytest.raises(ValueError, match="same text field"):
            ProbeTrainerAppMainConfig(
                **_make_minimal_config(
                    text_field="model_output",
                    evals_config=evals_config,
                )
            )


class TestHydraConfigRegistration:
    def test_register_hydra_configs_no_errors(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import pyine.apps.trainers.probe_trainer_configs as ptc
        import pyine.evals.common
        import pyine.utils.filesystem

        monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
        configs = ptc.register_hydra_configs(pyine.evals.common.EvalType.CODE_EXEC)
        assert len(configs) > 0

        # Should have an entrypoint config
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
        import pyine.apps.trainers.probe_trainer_configs as ptc
        import pyine.evals.common
        import pyine.utils.filesystem

        monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
        configs = ptc.register_hydra_configs(pyine.evals.common.EvalType.CODE_EXEC)

        # Should have a probe_base datamodule config
        dm_config = next(
            (cfg for cfg in configs if cfg.name == "probe_base" and cfg.group == "config/datamodule_config"),
            None,
        )
        assert dm_config is not None

    def test_register_hydra_configs_includes_correctness_taco_presets(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import pyine.apps.trainers.probe_trainer_configs as ptc
        import pyine.evals.common
        import pyine.utils.filesystem

        monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
        configs = ptc.register_hydra_configs(pyine.evals.common.EvalType.CODE_EXEC)

        assert any(cfg.name == "correctness_taco_base" and cfg.group == "config/datamodule_config" for cfg in configs)
        assert any(cfg.name == "correctness_taco_base" and cfg.group == "config/evals_config" for cfg in configs)


class TestProbeTrainerEvalOnlyConfig:
    def test_checkpoint_dir_accepted(self) -> None:
        cfg = ProbeTrainerAppMainConfig(
            **_make_minimal_config(
                probe_checkpoint_dir="/tmp/fake-probes",  # noqa: S108
            )
        )
        assert cfg.probe_checkpoint_dir is not None

    def test_checkpoint_name_accepted(self) -> None:
        cfg = ProbeTrainerAppMainConfig(
            **_make_minimal_config(
                probe_checkpoint_dir="/tmp/fake-probes",  # noqa: S108
                probe_checkpoint_name="best",
            )
        )
        assert cfg.probe_checkpoint_name == "best"

    def test_empty_probes_with_checkpoint_dir_valid(self) -> None:
        cfg = ProbeTrainerAppMainConfig(
            **_make_minimal_config(
                probe_configs=[],
                probe_checkpoint_dir="/tmp/fake-probes",  # noqa: S108
            )
        )
        assert len(cfg.probe_configs) == 0
        assert cfg.probe_checkpoint_dir is not None

    def test_empty_probes_without_checkpoint_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="probe_configs must be non-empty"):
            ProbeTrainerAppMainConfig(**_make_minimal_config(probe_configs=[]))

    def test_checkpoint_dir_default_is_none(self) -> None:
        cfg = ProbeTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.probe_checkpoint_dir is None


class TestProbeTrainerConfigReplicas:
    def test_num_replicas_default_is_1(self) -> None:
        cfg = ProbeTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.num_replicas == 1

    def test_num_replicas_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            ProbeTrainerAppMainConfig(**_make_minimal_config(num_replicas=0))

    def test_replica_fields_accepted(self) -> None:
        cfg = ProbeTrainerAppMainConfig(
            **_make_minimal_config(
                num_replicas=5,
                replica_base_seed=42,
                log_individual_replicas=True,
            )
        )
        assert cfg.num_replicas == 5
        assert cfg.replica_base_seed == 42
        assert cfg.log_individual_replicas is True

    def test_replica_base_seed_default_is_0(self) -> None:
        cfg = ProbeTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.replica_base_seed == 0

    def test_log_individual_replicas_default_is_false(self) -> None:
        cfg = ProbeTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.log_individual_replicas is False


class TestProbeTrainerConfigCheckpointSaving:
    """Section 6.2: Config validation tests for save_steps and save_total_limit."""

    def test_save_steps_default(self) -> None:
        """Test 14: Default save_steps is -1."""
        cfg = ProbeTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.save_steps == -1

    def test_save_total_limit_default(self) -> None:
        """Test 15: Default save_total_limit is None."""
        cfg = ProbeTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.save_total_limit is None

    def test_save_total_limit_validation_ge_1(self) -> None:
        """Test 16: save_total_limit=0 raises ValidationError."""
        with pytest.raises(ValueError):
            ProbeTrainerAppMainConfig(**_make_minimal_config(save_total_limit=0))

    def test_save_steps_with_save_probes_false_warns(self) -> None:
        """Test 17: Warning when save_steps > 0 and save_probes=False."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ProbeTrainerAppMainConfig(
                **_make_minimal_config(
                    save_steps=100,
                    save_probes=False,
                    save_best_probe_checkpoint=False,
                )
            )
            matching = [x for x in w if "save_steps" in str(x.message)]
            assert len(matching) >= 1, f"Expected warning about save_steps, got: {[str(x.message) for x in w]}"
