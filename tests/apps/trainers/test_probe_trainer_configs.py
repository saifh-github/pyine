"""Tests for ProbeTrainerAppMainConfig and Hydra config registration."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import pathlib

import pytest

from pyine.apps.trainers.probe_trainer_configs import ProbeTrainerAppMainConfig
from pyine.probes.base import ProbeConfig


class TestProbeTrainerAppMainConfig:
    def _make_minimal_config(self, **overrides: object) -> dict:
        """Minimal valid config kwargs."""
        base = {
            "base_model": "some-model",
            "lmdb_path": "/tmp/fake-lmdb",  # noqa: S108
            "probe_configs": [
                ProbeConfig(name="mean_L0", architecture="mean", layer=0),
            ],
        }
        base.update(overrides)
        return base

    def test_valid_config_construction(self) -> None:
        cfg = ProbeTrainerAppMainConfig(**self._make_minimal_config())
        assert len(cfg.probe_configs) == 1
        assert cfg.lmdb_path == "/tmp/fake-lmdb"  # noqa: S108

    def test_duplicate_probe_names_raises(self) -> None:
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
        cfg = ProbeTrainerAppMainConfig(**self._make_minimal_config())
        assert cfg.evals_config is None

    def test_target_dtype_property(self) -> None:
        import torch

        cfg = ProbeTrainerAppMainConfig(**self._make_minimal_config())
        assert cfg.target_dtype in (torch.bfloat16, torch.float16)

    def test_recompute_labels_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="recompute_labels"):
            ProbeTrainerAppMainConfig(
                **self._make_minimal_config(
                    recompute_labels=True,
                    label_metric_key="some/custom_metric",
                )
            )

    def test_recompute_labels_valid_metric_accepted(self) -> None:
        cfg = ProbeTrainerAppMainConfig(
            **self._make_minimal_config(
                recompute_labels=True,
                label_metric_key="reward/metrics/soft_match/is_match",
            )
        )
        assert cfg.recompute_labels is True


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


class TestProbeTrainerConfigReplicas:
    def _make_minimal_config(self, **overrides: object) -> dict:
        """Minimal valid config kwargs."""
        base: dict[str, object] = {
            "base_model": "some-model",
            "lmdb_path": "/tmp/fake-lmdb",  # noqa: S108
            "probe_configs": [
                ProbeConfig(name="mean_L0", architecture="mean", layer=0),
            ],
        }
        base.update(overrides)
        return base

    def test_num_replicas_default_is_1(self) -> None:
        cfg = ProbeTrainerAppMainConfig(**self._make_minimal_config())
        assert cfg.num_replicas == 1

    def test_num_replicas_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            ProbeTrainerAppMainConfig(**self._make_minimal_config(num_replicas=0))

    def test_replica_fields_accepted(self) -> None:
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
        cfg = ProbeTrainerAppMainConfig(**self._make_minimal_config())
        assert cfg.replica_base_seed == 0

    def test_log_individual_replicas_default_is_false(self) -> None:
        cfg = ProbeTrainerAppMainConfig(**self._make_minimal_config())
        assert cfg.log_individual_replicas is False


class TestProbeTrainerConfigEvalOnly:
    def _make_minimal_config(self, **overrides: object) -> dict:
        """Minimal valid config kwargs."""
        base: dict[str, object] = {
            "base_model": "some-model",
            "lmdb_path": "/tmp/fake-lmdb",  # noqa: S108
            "probe_configs": [
                ProbeConfig(name="mean_L0", architecture="mean", layer=0),
            ],
        }
        base.update(overrides)
        return base

    def test_use_eval_only_split_default_false(self) -> None:
        cfg = ProbeTrainerAppMainConfig(**self._make_minimal_config())
        assert cfg.use_eval_only_split is False

    def test_train_split_ratio_bounds(self) -> None:
        with pytest.raises(ValueError):
            ProbeTrainerAppMainConfig(**self._make_minimal_config(train_split_ratio=0.0))
        with pytest.raises(ValueError):
            ProbeTrainerAppMainConfig(**self._make_minimal_config(train_split_ratio=1.0))

    def test_eval_only_fields_accepted(self) -> None:
        cfg = ProbeTrainerAppMainConfig(
            **self._make_minimal_config(
                use_eval_only_split=True,
                eval_only_source_prefix="eval/",
                train_split_ratio=0.7,
                split_by_family=False,
                code_type_filter=["original", "hinted"],
                log_per_code_type_metrics=True,
            )
        )
        assert cfg.use_eval_only_split is True
        assert cfg.eval_only_source_prefix == "eval/"
        assert cfg.train_split_ratio == 0.7
        assert cfg.split_by_family is False
        assert cfg.code_type_filter == ["original", "hinted"]
        assert cfg.log_per_code_type_metrics is True

    def test_code_type_filter_accepts_list(self) -> None:
        cfg = ProbeTrainerAppMainConfig(
            **self._make_minimal_config(code_type_filter=["original", "hinted", "misleading"])
        )
        assert cfg.code_type_filter == ["original", "hinted", "misleading"]

    def test_code_type_filter_default_none(self) -> None:
        cfg = ProbeTrainerAppMainConfig(**self._make_minimal_config())
        assert cfg.code_type_filter is None

    def test_code_type_filter_empty_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty list"):
            ProbeTrainerAppMainConfig(**self._make_minimal_config(code_type_filter=[]))

    def test_eval_only_empty_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="eval_only_source_prefix"):
            ProbeTrainerAppMainConfig(
                **self._make_minimal_config(
                    use_eval_only_split=True,
                    eval_only_source_prefix="",
                )
            )
