"""Tests for ProbeTrainerAppMainConfig and Hydra config registration."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import pathlib

import pytest

from pyine.apps.trainers.probe_trainer_configs import ProbeTrainerAppMainConfig
from pyine.probes.base import ProbeConfig


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

    def test_target_dtype_property(self) -> None:
        import torch

        cfg = ProbeTrainerAppMainConfig(**_make_minimal_config())
        assert cfg.target_dtype in (torch.bfloat16, torch.float16)

    def test_datamodule_config_accessible(self) -> None:
        from pyine.probes.datamodule_configs import ProbeDataModuleConfig

        cfg = ProbeTrainerAppMainConfig(**_make_minimal_config())
        assert isinstance(cfg.datamodule_config, ProbeDataModuleConfig)
        assert cfg.datamodule_config.selection_strategy == "latest"


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
