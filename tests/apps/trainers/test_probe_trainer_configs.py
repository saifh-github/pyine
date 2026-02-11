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
