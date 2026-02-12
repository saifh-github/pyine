"""Tests for ProbeTrainerAppMainConfig and Hydra config registration."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import pathlib

import pytest

from pyine.probes.base import ProbeConfig


class TestProbeTrainerAppMainConfig:
    """Tests for ProbeTrainerAppMainConfig validation."""

    def _make_minimal_config(self, **overrides: object) -> dict:
        """Minimal valid config kwargs."""
        base = {
            "base_model": "some-model",
            "dataset_path": "/tmp/fake-dataset",  # noqa: S108
            "probe_configs": [
                ProbeConfig(name="mean_L0", architecture="mean", layer=0),
            ],
        }
        base.update(overrides)
        return base

    def test_valid_config_construction(self) -> None:
        """Config with all required fields validates successfully."""
        from pyine.apps.trainers.probe_trainer_configs import ProbeTrainerAppMainConfig

        cfg = ProbeTrainerAppMainConfig(**self._make_minimal_config())
        assert len(cfg.probe_configs) == 1
        assert cfg.dataset_path == "/tmp/fake-dataset"  # noqa: S108

    def test_duplicate_probe_names_raises(self) -> None:
        """Config rejects probe_configs with duplicate names."""
        from pyine.apps.trainers.probe_trainer_configs import ProbeTrainerAppMainConfig

        with pytest.raises(ValueError, match="unique"):
            ProbeTrainerAppMainConfig(
                **self._make_minimal_config(
                    probe_configs=[
                        ProbeConfig(name="dup", architecture="mean", layer=0),
                        ProbeConfig(name="dup", architecture="max", layer=0),
                    ]
                )
            )

    def test_evals_config_optional(self) -> None:
        """evals_config defaults to None without error."""
        from pyine.apps.trainers.probe_trainer_configs import ProbeTrainerAppMainConfig

        cfg = ProbeTrainerAppMainConfig(**self._make_minimal_config())
        assert cfg.evals_config is None

    def test_target_dtype_property(self) -> None:
        """target_dtype returns a torch dtype."""
        import torch

        from pyine.apps.trainers.probe_trainer_configs import ProbeTrainerAppMainConfig

        cfg = ProbeTrainerAppMainConfig(**self._make_minimal_config())
        assert cfg.target_dtype in (torch.bfloat16, torch.float16)


class TestHydraConfigRegistration:
    """Tests for Hydra-zen config registration."""

    def test_register_hydra_configs_no_errors(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """register_hydra_configs() completes without exceptions."""
        import pyine.apps.trainers.probe_trainer_configs as ptc
        import pyine.evals.common
        import pyine.utils.filesystem

        monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
        configs = ptc.register_hydra_configs(pyine.evals.common.EvalType.CODE_EXEC)
        assert len(configs) > 0

        # Should have an entrypoint config
        entrypoint = next(
            (c for c in configs if c.name == "entrypoint" and c.group is None),
            None,
        )
        assert entrypoint is not None


class TestProbeTrainerConfigReplicas:
    """Tests for replica-related config fields on ProbeTrainerAppMainConfig."""

    def _make_minimal_config(self, **overrides: object) -> dict:
        """Minimal valid config kwargs."""
        base: dict[str, object] = {
            "base_model": "some-model",
            "dataset_path": "/tmp/fake-dataset",  # noqa: S108
            "probe_configs": [
                ProbeConfig(name="mean_L0", architecture="mean", layer=0),
            ],
        }
        base.update(overrides)
        return base

    def test_num_replicas_default_is_1(self) -> None:
        """Default num_replicas is 1."""
        from pyine.apps.trainers.probe_trainer_configs import ProbeTrainerAppMainConfig

        cfg = ProbeTrainerAppMainConfig(**self._make_minimal_config())
        assert cfg.num_replicas == 1

    def test_num_replicas_must_be_positive(self) -> None:
        """num_replicas < 1 raises validation error (ge=1 constraint)."""
        from pyine.apps.trainers.probe_trainer_configs import ProbeTrainerAppMainConfig

        with pytest.raises(ValueError):
            ProbeTrainerAppMainConfig(**self._make_minimal_config(num_replicas=0))

    def test_replica_fields_accepted(self) -> None:
        """num_replicas, replica_base_seed, log_individual_replicas are accepted without error."""
        from pyine.apps.trainers.probe_trainer_configs import ProbeTrainerAppMainConfig

        cfg = ProbeTrainerAppMainConfig(
            **self._make_minimal_config(
                num_replicas=5,
                replica_base_seed=42,
                log_individual_replicas=True,
            )
        )
        assert cfg.num_replicas == 5
        assert cfg.replica_base_seed == 42
        assert cfg.log_individual_replicas is True

    def test_replica_base_seed_default_is_0(self) -> None:
        """Default replica_base_seed is 0."""
        from pyine.apps.trainers.probe_trainer_configs import ProbeTrainerAppMainConfig

        cfg = ProbeTrainerAppMainConfig(**self._make_minimal_config())
        assert cfg.replica_base_seed == 0

    def test_log_individual_replicas_default_is_false(self) -> None:
        """Default log_individual_replicas is False."""
        from pyine.apps.trainers.probe_trainer_configs import ProbeTrainerAppMainConfig

        cfg = ProbeTrainerAppMainConfig(**self._make_minimal_config())
        assert cfg.log_individual_replicas is False
