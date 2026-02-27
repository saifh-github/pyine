"""Tests for ProbeDataModuleConfig."""

from __future__ import annotations

import pytest

from pyine.guardrails.data.datamodule_configs import LabelBalanceConfig, ProbeDataModuleConfig
from pyine.guardrails.data.reward_keys import HARD_MATCH_KEY, SOFT_MATCH_KEY


class TestProbeDataModuleConfig:
    def _make_minimal(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "lmdb_path": "/tmp/fake-lmdb",  # noqa: S108
        }
        base.update(overrides)
        return base

    def test_minimal_valid_config(self) -> None:
        cfg = ProbeDataModuleConfig(**self._make_minimal())
        assert cfg.lmdb_path == "/tmp/fake-lmdb"  # noqa: S108

    def test_defaults(self) -> None:
        cfg = ProbeDataModuleConfig(**self._make_minimal())
        assert cfg.label_metric_key == SOFT_MATCH_KEY
        assert cfg.selection_strategy == "latest"
        assert cfg.recompute_labels is False
        assert cfg.skip_malformed_records is False
        assert cfg.train_key_prefix == "train/"
        assert cfg.valid_key_prefix == "eval/"
        assert cfg.use_eval_only_split is False
        assert cfg.eval_only_source_prefix == "eval/"
        assert cfg.train_split_ratio == 0.8
        assert cfg.split_by_family is True
        assert cfg.max_samples_per_split is None
        assert cfg.code_type_filter is None
        assert cfg.subset_names == ("train", "valid")

    def test_recompute_labels_soft_match_accepted(self) -> None:
        cfg = ProbeDataModuleConfig(**self._make_minimal(recompute_labels=True, label_metric_key=SOFT_MATCH_KEY))
        assert cfg.recompute_labels is True

    def test_recompute_labels_hard_match_accepted(self) -> None:
        cfg = ProbeDataModuleConfig(**self._make_minimal(recompute_labels=True, label_metric_key=HARD_MATCH_KEY))
        assert cfg.recompute_labels is True

    def test_recompute_labels_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="recompute_labels"):
            ProbeDataModuleConfig(**self._make_minimal(recompute_labels=True, label_metric_key="some/custom_metric"))

    def test_eval_only_empty_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="eval_only_source_prefix"):
            ProbeDataModuleConfig(**self._make_minimal(use_eval_only_split=True, eval_only_source_prefix=""))

    def test_code_type_filter_empty_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty list"):
            ProbeDataModuleConfig(**self._make_minimal(code_type_filter=[]))

    def test_code_type_filter_none_accepted(self) -> None:
        cfg = ProbeDataModuleConfig(**self._make_minimal(code_type_filter=None))
        assert cfg.code_type_filter is None

    def test_code_type_filter_list_accepted(self) -> None:
        cfg = ProbeDataModuleConfig(**self._make_minimal(code_type_filter=["original", "hinted"]))
        assert cfg.code_type_filter == ["original", "hinted"]

    def test_train_split_ratio_bounds(self) -> None:
        with pytest.raises(ValueError):
            ProbeDataModuleConfig(**self._make_minimal(train_split_ratio=0.0))
        with pytest.raises(ValueError):
            ProbeDataModuleConfig(**self._make_minimal(train_split_ratio=1.0))

    def test_eval_only_fields_accepted(self) -> None:
        cfg = ProbeDataModuleConfig(
            **self._make_minimal(
                use_eval_only_split=True,
                eval_only_source_prefix="eval/",
                train_split_ratio=0.7,
                split_by_family=False,
                code_type_filter=["original", "hinted"],
            )
        )
        assert cfg.use_eval_only_split is True
        assert cfg.eval_only_source_prefix == "eval/"
        assert cfg.train_split_ratio == 0.7
        assert cfg.split_by_family is False
        assert cfg.code_type_filter == ["original", "hinted"]

    def test_datamodule_class_path_resolves(self) -> None:
        cfg = ProbeDataModuleConfig(**self._make_minimal())
        assert cfg.datamodule_class_path == "pyine.guardrails.data.datamodule.ProbeDataModule"
        # Verify that instantiate_datamodule works
        from pyine.guardrails.data.datamodule import ProbeDataModule

        dm = cfg.instantiate_datamodule()
        assert isinstance(dm, ProbeDataModule)

    def test_label_balance_none_default(self) -> None:
        cfg = ProbeDataModuleConfig(**self._make_minimal())
        assert cfg.label_balance is None

    def test_probe_data_module_config_with_label_balance(self) -> None:
        lb = LabelBalanceConfig(target_positive_ratio=0.5)
        cfg = ProbeDataModuleConfig(**self._make_minimal(label_balance=lb))
        assert cfg.label_balance is not None
        assert cfg.label_balance.target_positive_ratio == 0.5


class TestLabelBalanceConfig:
    def test_simple_mode_valid(self) -> None:
        cfg = LabelBalanceConfig(target_positive_ratio=0.5)
        assert cfg.target_positive_ratio == 0.5
        assert cfg.group_proportions is None
        assert cfg.strategy == "subsample"
        assert cfg.apply_to == ("train",)

    def test_group_mode_valid(self) -> None:
        cfg = LabelBalanceConfig(group_proportions={"original:1": 0.8, "misleading:0": 0.2})
        assert cfg.target_positive_ratio is None
        assert cfg.group_proportions == {"original:1": 0.8, "misleading:0": 0.2}

    def test_mutual_exclusivity_raises(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            LabelBalanceConfig(target_positive_ratio=0.5, group_proportions={"original:1": 1.0})

    def test_neither_set_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            LabelBalanceConfig()

    def test_invalid_group_key_raises(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            LabelBalanceConfig(group_proportions={"bad_key": 1.0})

    def test_invalid_group_key_label_not_binary_raises(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            LabelBalanceConfig(group_proportions={"original:2": 1.0})

    def test_negative_proportion_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            LabelBalanceConfig(group_proportions={"original:1": -0.5})

    def test_zero_proportion_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            LabelBalanceConfig(group_proportions={"original:1": 0.0})

    def test_apply_to_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            LabelBalanceConfig(target_positive_ratio=0.5, apply_to=())

    def test_apply_to_invalid_split_raises(self) -> None:
        with pytest.raises(ValueError, match="'test'"):
            LabelBalanceConfig(target_positive_ratio=0.5, apply_to=("test",))

    def test_apply_to_both_splits(self) -> None:
        cfg = LabelBalanceConfig(target_positive_ratio=0.5, apply_to=("train", "valid"))
        assert cfg.apply_to == ("train", "valid")

    def test_oversample_strategy(self) -> None:
        cfg = LabelBalanceConfig(target_positive_ratio=0.5, strategy="oversample")
        assert cfg.strategy == "oversample"

    def test_label_balance_config_round_trip(self) -> None:
        """Serialize/deserialize LabelBalanceConfig via model_dump/model_validate."""
        original = LabelBalanceConfig(
            group_proportions={"original:1": 0.8, "misleading:0": 0.2},
            strategy="oversample",
            apply_to=("train", "valid"),
        )
        dumped = original.model_dump()
        restored = LabelBalanceConfig.model_validate(dumped)
        assert restored.group_proportions == original.group_proportions
        assert restored.strategy == original.strategy
        # Explicitly verify tuple survives round-trip (JSON serializes as list)
        assert restored.apply_to == ("train", "valid")
        assert isinstance(restored.apply_to, tuple)
