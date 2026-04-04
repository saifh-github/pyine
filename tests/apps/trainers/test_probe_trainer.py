"""Unit tests for the probe training loop with mocked LLM."""

from __future__ import annotations

import json
import typing
from types import SimpleNamespace
from unittest.mock import MagicMock

if typing.TYPE_CHECKING:
    import pathlib

import accelerate
import datasets
import pytest
import torch

import pyine.apps.trainers.common
import pyine.apps.trainers.probe_trainer
import pyine.guardrails.probes
import pyine.guardrails.probes.base
import pyine.guardrails.probes.collection
import pyine.guardrails.probes.extraction


class MockTransformerBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SmallMockLLM(torch.nn.Module):
    """Tiny model with N transformer blocks, hidden_dim=64.

    Mimics the interface needed by ActivationExtractor and probe_train().
    """

    def __init__(self, n_layers: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            hidden_size=hidden_dim,
        )
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([MockTransformerBlock(hidden_dim) for _ in range(n_layers)])
        self.embed = torch.nn.Embedding(1000, hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        h = self.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return h


@pytest.fixture
def mock_llm() -> SmallMockLLM:
    model = SmallMockLLM(n_layers=2, hidden_dim=64)
    model.eval()
    model.requires_grad_(False)
    return model


@pytest.fixture
def probe_configs() -> list[pyine.guardrails.probes.base.ProbeConfig]:
    return [
        pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0, learning_rate=1e-2),
        pyine.guardrails.probes.base.ProbeConfig(name="max_L1", architecture="max", layer=1, learning_rate=1e-2),
    ]


@pytest.fixture
def probe_collection(
    probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
) -> pyine.guardrails.probes.collection.ProbeCollection:
    return pyine.guardrails.probes.collection.ProbeCollection(probe_configs, hidden_dim=64)


class TestProbeTrainUnit:
    def test_train_step_reduces_loss(
        self,
        mock_llm: SmallMockLLM,
        probe_collection: pyine.guardrails.probes.collection.ProbeCollection,
    ) -> None:
        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(mock_llm, target_layers=[0, 1])
        optimizer = torch.optim.AdamW(probe_collection.get_parameter_groups())
        loss_fn = torch.nn.BCEWithLogitsLoss()

        batch_size, seq_len = 8, 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = torch.randint(0, 2, (batch_size,), dtype=torch.float)

        # Initial loss
        with torch.no_grad():
            mock_llm(input_ids=input_ids, attention_mask=attention_mask)
        activations = extractor.get_activations()
        logits = probe_collection(activations, attention_mask)
        initial_loss = sum(loss_fn(logit.squeeze(-1), labels) for logit in logits.values()).item()

        # Train for 30 steps
        for _ in range(30):
            with torch.no_grad():
                mock_llm(input_ids=input_ids, attention_mask=attention_mask)
            activations = extractor.get_activations()
            logits = probe_collection(activations, attention_mask)
            total_loss = sum(loss_fn(logit.squeeze(-1), labels) for logit in logits.values())
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        final_loss = total_loss.item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"

        extractor.remove_hooks()


class TestBalancedClassWeight:
    """Tests for compute_binary_pos_weight used by probe_train's class_weight_mode='balanced'."""

    def test_balanced_pos_weight(self) -> None:
        n_neg, n_pos = 14, 6  # 70/30 split
        labels = [0] * n_neg + [1] * n_pos
        pw = pyine.apps.trainers.common.compute_binary_pos_weight(labels)
        assert pw == pytest.approx(n_neg / n_pos)

    def test_balanced_equal_classes(self) -> None:
        labels = [0] * 10 + [1] * 10
        pw = pyine.apps.trainers.common.compute_binary_pos_weight(labels)
        assert pw == pytest.approx(1.0)

    def test_balanced_raises_on_all_negative(self) -> None:
        with pytest.raises(ValueError, match="need both classes"):
            pyine.apps.trainers.common.compute_binary_pos_weight([0, 0, 0, 0, 0])

    def test_balanced_raises_on_all_positive(self) -> None:
        with pytest.raises(ValueError, match="need both classes"):
            pyine.apps.trainers.common.compute_binary_pos_weight([1, 1, 1])


class TestGetModuleDevice:
    def test_returns_first_parameter_device(self) -> None:
        module = torch.nn.Linear(4, 2)

        assert pyine.apps.trainers.probe_trainer._get_module_device(module) == module.weight.device

    def test_raises_when_module_has_no_parameters(self) -> None:
        with pytest.raises(ValueError, match="at least one parameter"):
            pyine.apps.trainers.probe_trainer._get_module_device(torch.nn.ReLU())


class TestBestProbeCheckpointPreconditions:
    def test_raises_for_empty_validation_split(self) -> None:
        config = MagicMock(save_best_probe_checkpoint=True)
        valid_dataset = datasets.Dataset.from_dict(
            {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "code_type_id": [],
            }
        )

        with pytest.raises(ValueError, match="non-empty validation split"):
            pyine.apps.trainers.probe_trainer._validate_best_probe_checkpoint_preconditions(
                config,
                valid_dataset,
            )

    def test_allows_non_empty_validation_split(self) -> None:
        config = MagicMock(save_best_probe_checkpoint=True)
        valid_dataset = datasets.Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
                "labels": [1],
                "code_type_id": [0],
            }
        )

        pyine.apps.trainers.probe_trainer._validate_best_probe_checkpoint_preconditions(
            config,
            valid_dataset,
        )

    def test_validation_produces_metrics(
        self,
        probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
    ) -> None:
        acc = accelerate.Accelerator()
        device = acc.device

        # Create fresh model + collection on correct device
        model = SmallMockLLM(n_layers=2, hidden_dim=64).to(device)
        model.eval()
        model.requires_grad_(False)
        coll = pyine.guardrails.probes.collection.ProbeCollection(probe_configs, hidden_dim=64).to(device)

        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(model, target_layers=[0, 1])
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # Build a small dataloader
        batch_size, seq_len = 8, 16
        ds = datasets.Dataset.from_dict(
            {
                "input_ids": torch.randint(0, 100, (20, seq_len)).tolist(),
                "attention_mask": torch.ones(20, seq_len, dtype=torch.long).tolist(),
                "labels": ([0] * 10 + [1] * 10),
            }
        )
        ds.set_format("torch")
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

        coll_prepared = acc.prepare(coll)
        loader = acc.prepare(loader)

        metrics = pyine.apps.trainers.probe_trainer.validate_probes(
            coll_prepared,
            model,
            extractor,
            loader,
            loss_fn,
            global_step=0,
            accelerator=acc,
            runtime=None,
        )

        assert len(metrics) > 0
        for _name, metric_value in metrics.items():
            assert "loss" in metric_value
            assert "auroc" in metric_value
            assert metric_value["loss"] >= 0.0

        extractor.remove_hooks()

    def test_auroc_guard_single_class(
        self,
        probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
    ) -> None:
        import math

        acc = accelerate.Accelerator()
        device = acc.device

        model = SmallMockLLM(n_layers=2, hidden_dim=64).to(device)
        model.eval()
        model.requires_grad_(False)
        coll = pyine.guardrails.probes.collection.ProbeCollection(probe_configs, hidden_dim=64).to(device)

        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(model, target_layers=[0, 1])
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # All labels are 0 -> single-class
        batch_size, seq_len = 4, 16
        ds = datasets.Dataset.from_dict(
            {
                "input_ids": torch.randint(0, 100, (8, seq_len)).tolist(),
                "attention_mask": torch.ones(8, seq_len, dtype=torch.long).tolist(),
                "labels": [0] * 8,
            }
        )
        ds.set_format("torch")
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

        coll_prepared = acc.prepare(coll)
        loader_prepared = acc.prepare(loader)

        metrics = pyine.apps.trainers.probe_trainer.validate_probes(
            coll_prepared,
            model,
            extractor,
            loader_prepared,
            loss_fn,
            global_step=0,
            accelerator=acc,
            runtime=None,
        )

        for name, metric_value in metrics.items():
            assert math.isnan(metric_value["auroc"]), f"Expected NaN AUROC for {name}, got {metric_value['auroc']}"

        extractor.remove_hooks()

    # NOTE: test_save_probe_checkpoints and test_probe_checkpoints_loadable
    # have been moved to TestMidTrainingSave below to cover the new
    # probe-first directory hierarchy with checkpoint_subdir.


def _replica_pc(
    name: str,
    arch: str = "mean",
    layer: int = 0,
    base_name: str = "a",
    replica_idx: int = 0,
    replica_seed: int = 1,
) -> pyine.guardrails.probes.base.ProbeConfig:
    """Shorthand for creating ProbeConfig with replica metadata in tests."""
    return pyine.guardrails.probes.base.ProbeConfig(
        name=name,
        architecture=arch,
        layer=layer,
        base_name=base_name,
        replica_idx=replica_idx,
        replica_seed=replica_seed,
    )


class TestLoadFromCheckpointRoundTrip:
    """Tests for ProbeCollection.load_from_checkpoint via save_probe_checkpoints."""

    def test_save_then_load_preserves_weights(
        self,
        tmp_path: pathlib.Path,
        probe_collection: pyine.guardrails.probes.collection.ProbeCollection,
    ) -> None:
        acc = accelerate.Accelerator()
        probe_collection_prepared = acc.prepare(probe_collection)
        runtime = MagicMock()
        runtime.output_dir = str(tmp_path)
        config = MagicMock()
        output_dir = pyine.apps.trainers.probe_trainer.save_probe_checkpoints(
            probe_collection_prepared,
            config,
            runtime,
            acc,
        )
        assert output_dir is not None
        loaded = pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(
            checkpoint_dir=output_dir,
            hidden_dim=64,
        )
        for name in probe_collection.probes:
            for (orig_name, orig_param), (load_name, load_param) in zip(
                probe_collection.probes[name].named_parameters(),
                loaded.probes[name].named_parameters(),
                strict=True,
            ):
                assert orig_name == load_name
                torch.testing.assert_close(orig_param.cpu(), load_param)


class TestBestProbeCheckpointHelpers:
    def test_is_better_probe_metric_uses_loss_as_auroc_tiebreak(self) -> None:
        best_record = {"loss": 0.5, "auroc": 0.8}
        current_metrics = {"loss": 0.4, "auroc": 0.8}

        assert pyine.apps.trainers.probe_trainer._is_better_probe_metric(
            current_metrics,
            best_record,
            "auroc",
        )

    def test_update_best_probe_checkpoints_saves_best_dirs_and_summary(
        self,
        tmp_path: pathlib.Path,
        probe_collection: pyine.guardrails.probes.collection.ProbeCollection,
    ) -> None:
        acc = accelerate.Accelerator()
        probe_collection_prepared = acc.prepare(probe_collection)
        runtime = MagicMock()
        runtime.output_dir = str(tmp_path)
        config = MagicMock(
            save_best_probe_checkpoint=True,
            best_probe_checkpoint_name="best",
            best_probe_metric="auroc",
        )
        best_probe_records: dict[str, dict[str, typing.Any]] = {}
        metrics = {
            "mean_L0": {"loss": 0.4, "auroc": 0.81},
            "max_L1": {"loss": 0.6, "auroc": 0.72},
        }

        probes_base = pyine.apps.trainers.probe_trainer.update_best_probe_checkpoints(
            probe_collection_prepared,
            config,
            runtime,
            acc,
            metrics,
            epoch=1,
            global_step=10,
            best_probe_records=best_probe_records,
        )

        assert probes_base == tmp_path / "probes"
        assert (probes_base / "mean_L0" / "best" / "probe_state_dict.pt").exists()
        assert (probes_base / "max_L1" / "best" / "probe_state_dict.pt").exists()
        summary = json.loads((probes_base / "best_summary.json").read_text())
        assert summary["checkpoint_name"] == "best"
        assert summary["metric_name"] == "auroc"
        assert summary["probes"]["mean_L0"]["global_step"] == 10
        assert summary["probes"]["max_L1"]["epoch"] == 1

    def test_save_probe_checkpoint_artifacts_cleans_existing_dir(
        self,
        tmp_path: pathlib.Path,
        probe_collection: pyine.guardrails.probes.collection.ProbeCollection,
    ) -> None:
        probe = typing.cast(
            "pyine.guardrails.probes.base.BaseProbe",
            probe_collection.probes["mean_L0"],
        )
        stale_dir = tmp_path / "probes" / "mean_L0" / "best"
        stale_dir.mkdir(parents=True)
        (stale_dir / "stale.txt").write_text("old")

        saved_dir = pyine.apps.trainers.probe_trainer._save_probe_checkpoint_artifacts(
            probe=probe,
            probes_base=tmp_path / "probes",
            probe_name="mean_L0",
            checkpoint_subdir="best",
        )

        assert saved_dir == stale_dir
        assert not (saved_dir / "stale.txt").exists()
        assert (saved_dir / "probe_state_dict.pt").exists()
        assert (saved_dir / "probe_config.json").exists()


class TestSkipTrainingProbeTrainer:
    """Tests for the skip_training code path in probe trainer main."""

    def test_skip_training_requires_checkpoint_dir(self) -> None:
        from pyine.apps.trainers.probe_trainer_configs import ProbeTrainerAppMainConfig

        cfg = ProbeTrainerAppMainConfig(
            base_model="some-model",
            datamodule_config={"lmdb_path": "/tmp/fake-lmdb"},  # noqa: S108
            probe_configs=[],
            probe_checkpoint_dir="/tmp/fake-probes",  # noqa: S108
        )
        # clear the checkpoint dir to trigger the validation
        cfg.probe_checkpoint_dir = None
        with pytest.raises(ValueError, match="skip_training=True requires"):
            pyine.apps.trainers.probe_trainer.main(
                config=cfg,
                runtime=None,
                skip_training=True,
            )


class TestStableReplicaSeed:
    def test_deterministic_across_calls(self) -> None:
        s1 = pyine.apps.trainers.common.stable_replica_seed(0, "mean_L0", 0)
        s2 = pyine.apps.trainers.common.stable_replica_seed(0, "mean_L0", 0)
        assert s1 == s2

    def test_different_replica_idx_different_seeds(self) -> None:
        s0 = pyine.apps.trainers.common.stable_replica_seed(0, "mean_L0", 0)
        s1 = pyine.apps.trainers.common.stable_replica_seed(0, "mean_L0", 1)
        assert s0 != s1

    def test_different_base_seed_different_seeds(self) -> None:
        s0 = pyine.apps.trainers.common.stable_replica_seed(0, "mean_L0", 0)
        s1 = pyine.apps.trainers.common.stable_replica_seed(42, "mean_L0", 0)
        assert s0 != s1

    def test_different_name_different_seeds(self) -> None:
        s0 = pyine.apps.trainers.common.stable_replica_seed(0, "mean_L0", 0)
        s1 = pyine.apps.trainers.common.stable_replica_seed(0, "attn_L8", 0)
        assert s0 != s1

    def test_seed_within_valid_range(self) -> None:
        for i in range(100):
            seed = pyine.apps.trainers.common.stable_replica_seed(i, f"probe_{i}", i % 10)
            assert 0 <= seed < 2**31


class TestExpandProbeConfigsWithReplicas:
    def test_no_expansion_when_num_replicas_1(self) -> None:
        configs = [pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0)]
        result = pyine.apps.trainers.probe_trainer.expand_probe_configs_with_replicas(
            configs,
            num_replicas=1,
            replica_base_seed=0,
        )
        assert result is configs

    def test_expansion_creates_correct_count(self) -> None:
        configs = [
            pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0),
            pyine.guardrails.probes.base.ProbeConfig(name="attn_L8", architecture="attention", layer=8),
        ]
        result = pyine.apps.trainers.probe_trainer.expand_probe_configs_with_replicas(
            configs,
            num_replicas=3,
            replica_base_seed=0,
        )
        assert len(result) == 6

    def test_expanded_names_follow_pattern(self) -> None:
        configs = [pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0)]
        result = pyine.apps.trainers.probe_trainer.expand_probe_configs_with_replicas(
            configs,
            num_replicas=3,
            replica_base_seed=0,
        )
        names = [pc.name for pc in result]
        assert names == ["mean_L0_r0", "mean_L0_r1", "mean_L0_r2"]

    def test_replica_metadata_populated(self) -> None:
        configs = [pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0)]
        result = pyine.apps.trainers.probe_trainer.expand_probe_configs_with_replicas(
            configs,
            num_replicas=2,
            replica_base_seed=0,
        )
        for pc in result:
            assert pc.replica_idx is not None
            assert pc.replica_seed is not None
            assert pc.base_name is not None

    def test_seeds_are_unique(self) -> None:
        configs = [
            pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0),
            pyine.guardrails.probes.base.ProbeConfig(name="attn_L8", architecture="attention", layer=8),
        ]
        result = pyine.apps.trainers.probe_trainer.expand_probe_configs_with_replicas(
            configs,
            num_replicas=5,
            replica_base_seed=0,
        )
        seeds = [pc.replica_seed for pc in result]
        assert len(set(seeds)) == len(seeds), f"Duplicate seeds found: {seeds}"

    def test_seeds_are_deterministic(self) -> None:
        configs = [pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0)]
        r1 = pyine.apps.trainers.probe_trainer.expand_probe_configs_with_replicas(
            configs,
            num_replicas=3,
            replica_base_seed=42,
        )
        r2 = pyine.apps.trainers.probe_trainer.expand_probe_configs_with_replicas(
            configs,
            num_replicas=3,
            replica_base_seed=42,
        )
        for a, b in zip(r1, r2, strict=True):
            assert a.replica_seed == b.replica_seed

    def test_base_name_matches_original(self) -> None:
        configs = [pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0)]
        result = pyine.apps.trainers.probe_trainer.expand_probe_configs_with_replicas(
            configs,
            num_replicas=2,
            replica_base_seed=0,
        )
        for pc in result:
            assert pc.base_name == "mean_L0"

    def test_non_replica_fields_preserved(self) -> None:
        configs = [
            pyine.guardrails.probes.base.ProbeConfig(
                name="attn_L8",
                architecture="attention",
                layer=8,
                learning_rate=5e-4,
                attn_dim=32,
            )
        ]
        result = pyine.apps.trainers.probe_trainer.expand_probe_configs_with_replicas(
            configs,
            num_replicas=2,
            replica_base_seed=0,
        )
        for pc in result:
            assert pc.architecture == "attention"
            assert pc.layer == 8
            assert pc.learning_rate == 5e-4
            assert pc.attn_dim == 32


class TestAggregateReplicaMetrics:
    def test_single_group(self) -> None:
        configs = {
            "mean_L0_r0": _replica_pc("mean_L0_r0", base_name="mean_L0", replica_idx=0),
            "mean_L0_r1": _replica_pc("mean_L0_r1", base_name="mean_L0", replica_idx=1, replica_seed=2),
        }
        result = pyine.apps.trainers.probe_trainer.aggregate_replica_metrics(
            {"mean_L0_r0": 0.5, "mean_L0_r1": 0.7},
            configs,
        )
        assert "mean_L0" in result
        assert len(result) == 1

    def test_multiple_groups(self) -> None:
        configs = {
            "a_r0": _replica_pc("a_r0", replica_idx=0),
            "a_r1": _replica_pc("a_r1", replica_idx=1, replica_seed=2),
            "b_r0": _replica_pc("b_r0", arch="max", base_name="b", replica_idx=0, replica_seed=3),
        }
        result = pyine.apps.trainers.probe_trainer.aggregate_replica_metrics(
            {"a_r0": 0.5, "a_r1": 0.7, "b_r0": 0.3},
            configs,
        )
        assert set(result.keys()) == {"a", "b"}

    def test_single_value_std_is_zero(self) -> None:
        configs = {
            "b_r0": _replica_pc("b_r0", arch="max", base_name="b"),
        }
        result = pyine.apps.trainers.probe_trainer.aggregate_replica_metrics({"b_r0": 0.42}, configs)
        assert result["b"]["std"] == 0.0

    def test_non_replicated_probes(self) -> None:
        configs = {
            "mean_L0": pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0),
        }
        result = pyine.apps.trainers.probe_trainer.aggregate_replica_metrics({"mean_L0": 0.5}, configs)
        assert "mean_L0" in result
        assert result["mean_L0"]["mean"] == 0.5

    def test_mean_std_correctness(self) -> None:
        import statistics

        values = [0.3, 0.5, 0.7]
        configs = {f"a_r{i}": _replica_pc(f"a_r{i}", replica_idx=i, replica_seed=i) for i in range(3)}
        per_probe = {f"a_r{i}": v for i, v in enumerate(values)}
        result = pyine.apps.trainers.probe_trainer.aggregate_replica_metrics(per_probe, configs)
        assert abs(result["a"]["mean"] - statistics.mean(values)) < 1e-10
        assert abs(result["a"]["std"] - statistics.stdev(values)) < 1e-10

    def test_nan_excluded_from_aggregation(self) -> None:
        import math

        configs = {
            "a_r0": _replica_pc("a_r0", replica_idx=0),
            "a_r1": _replica_pc("a_r1", replica_idx=1, replica_seed=2),
            "a_r2": _replica_pc("a_r2", replica_idx=2, replica_seed=3),
        }
        result = pyine.apps.trainers.probe_trainer.aggregate_replica_metrics(
            {"a_r0": 0.5, "a_r1": float("nan"), "a_r2": 0.7},
            configs,
        )
        assert not math.isnan(result["a"]["mean"])
        assert abs(result["a"]["mean"] - 0.6) < 1e-10

    def test_all_nan_returns_nan(self) -> None:
        import math

        configs = {
            "a_r0": _replica_pc("a_r0", replica_idx=0),
            "a_r1": _replica_pc("a_r1", replica_idx=1, replica_seed=2),
        }
        result = pyine.apps.trainers.probe_trainer.aggregate_replica_metrics(
            {"a_r0": float("nan"), "a_r1": float("nan")},
            configs,
        )
        assert math.isnan(result["a"]["mean"])
        assert math.isnan(result["a"]["std"])
        assert math.isnan(result["a"]["min"])
        assert math.isnan(result["a"]["max"])


class TestReplicaTables:
    def test_train_table_has_correct_columns(self) -> None:
        configs = {"a_r0": _replica_pc("a_r0")}
        table = pyine.apps.trainers.probe_trainer.build_train_replica_table(
            {"a_r0": 0.5},
            configs,
            global_step=10,
            epoch=0,
        )
        expected_cols = [
            "probe_name",
            "base_name",
            "architecture",
            "layer",
            "replica_idx",
            "step",
            "epoch",
            "train_loss",
        ]
        assert table.columns == expected_cols

    def test_train_table_row_count_matches_probes(self) -> None:
        configs = {
            "a_r0": _replica_pc("a_r0", replica_idx=0),
            "a_r1": _replica_pc("a_r1", replica_idx=1, replica_seed=2),
            "b_r0": _replica_pc("b_r0", arch="max", layer=4, base_name="b", replica_seed=3),
        }
        table = pyine.apps.trainers.probe_trainer.build_train_replica_table(
            {"a_r0": 0.5, "a_r1": 0.6, "b_r0": 0.7},
            configs,
            global_step=10,
            epoch=0,
        )
        assert len(table.data) == 3

    def test_train_table_base_name_populated(self) -> None:
        configs = {"a_r0": _replica_pc("a_r0")}
        table = pyine.apps.trainers.probe_trainer.build_train_replica_table(
            {"a_r0": 0.5},
            configs,
            global_step=10,
            epoch=0,
        )
        base_name_col_idx = table.columns.index("base_name")
        assert table.data[0][base_name_col_idx] == "a"

    def test_valid_table_has_correct_columns(self) -> None:
        configs = {"a_r0": _replica_pc("a_r0")}
        table = pyine.apps.trainers.probe_trainer.build_valid_replica_table(
            {"a_r0": {"loss": 0.5, "auroc": 0.8}},
            configs,
            global_step=10,
        )
        expected_cols = [
            "probe_name",
            "base_name",
            "architecture",
            "layer",
            "replica_idx",
            "step",
            "valid_loss",
            "auroc",
        ]
        assert table.columns == expected_cols

    def test_valid_table_includes_loss_and_auroc(self) -> None:
        configs = {"a_r0": _replica_pc("a_r0")}
        table = pyine.apps.trainers.probe_trainer.build_valid_replica_table(
            {"a_r0": {"loss": 0.5, "auroc": 0.8}},
            configs,
            global_step=10,
        )
        loss_idx = table.columns.index("valid_loss")
        auroc_idx = table.columns.index("auroc")
        assert table.data[0][loss_idx] == 0.5
        assert table.data[0][auroc_idx] == 0.8

    def test_valid_table_row_count_matches_probes(self) -> None:
        configs = {f"a_r{i}": _replica_pc(f"a_r{i}", replica_idx=i, replica_seed=i) for i in range(4)}
        metrics = {f"a_r{i}": {"loss": 0.5 + i * 0.1, "auroc": 0.7 + i * 0.05} for i in range(4)}
        table = pyine.apps.trainers.probe_trainer.build_valid_replica_table(metrics, configs, global_step=10)
        assert len(table.data) == 4

    def test_tables_with_non_replicated_probes(self) -> None:
        configs = {
            "mean_L0": pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0),
        }
        table = pyine.apps.trainers.probe_trainer.build_train_replica_table(
            {"mean_L0": 0.5},
            configs,
            global_step=10,
            epoch=0,
        )
        replica_idx_col = table.columns.index("replica_idx")
        assert table.data[0][replica_idx_col] == 0


class TestValidateProbesWithReplicas:
    def test_validation_returns_per_replica_metrics(self) -> None:
        acc = accelerate.Accelerator()
        device = acc.device

        base_configs = [
            pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0, learning_rate=1e-2),
            pyine.guardrails.probes.base.ProbeConfig(name="max_L1", architecture="max", layer=1, learning_rate=1e-2),
        ]
        expanded = pyine.apps.trainers.probe_trainer.expand_probe_configs_with_replicas(
            base_configs, num_replicas=2, replica_base_seed=0
        )
        expanded_by_name = {pc.name: pc for pc in expanded}

        model = SmallMockLLM(n_layers=2, hidden_dim=64).to(device)
        model.eval()
        model.requires_grad_(False)
        coll = pyine.guardrails.probes.collection.ProbeCollection(expanded, hidden_dim=64).to(device)

        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(model, target_layers=[0, 1])
        loss_fn = torch.nn.BCEWithLogitsLoss()

        seq_len = 16
        ds = datasets.Dataset.from_dict(
            {
                "input_ids": torch.randint(0, 100, (20, seq_len)).tolist(),
                "attention_mask": torch.ones(20, seq_len, dtype=torch.long).tolist(),
                "labels": ([0] * 10 + [1] * 10),
            }
        )
        ds.set_format("torch")
        loader = torch.utils.data.DataLoader(ds, batch_size=8)

        coll_prepared = acc.prepare(coll)
        loader = acc.prepare(loader)

        metrics = pyine.apps.trainers.probe_trainer.validate_probes(
            coll_prepared,
            model,
            extractor,
            loader,
            loss_fn,
            global_step=0,
            accelerator=acc,
            runtime=None,
            expanded_configs_by_name=expanded_by_name,
        )

        assert len(metrics) == 4  # 2 base x 2 replicas
        assert "mean_L0_r0" in metrics
        assert "mean_L0_r1" in metrics
        assert "max_L1_r0" in metrics
        assert "max_L1_r1" in metrics
        extractor.remove_hooks()

    def test_aggregation_of_validation_metrics(self) -> None:
        acc = accelerate.Accelerator()
        device = acc.device

        base_configs = [
            pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0, learning_rate=1e-2),
        ]
        expanded = pyine.apps.trainers.probe_trainer.expand_probe_configs_with_replicas(
            base_configs, num_replicas=3, replica_base_seed=0
        )
        expanded_by_name = {pc.name: pc for pc in expanded}

        model = SmallMockLLM(n_layers=2, hidden_dim=64).to(device)
        model.eval()
        model.requires_grad_(False)
        coll = pyine.guardrails.probes.collection.ProbeCollection(expanded, hidden_dim=64).to(device)

        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(model, target_layers=[0, 1])
        loss_fn = torch.nn.BCEWithLogitsLoss()

        seq_len = 16
        ds = datasets.Dataset.from_dict(
            {
                "input_ids": torch.randint(0, 100, (20, seq_len)).tolist(),
                "attention_mask": torch.ones(20, seq_len, dtype=torch.long).tolist(),
                "labels": ([0] * 10 + [1] * 10),
            }
        )
        ds.set_format("torch")
        loader = torch.utils.data.DataLoader(ds, batch_size=8)

        coll_prepared = acc.prepare(coll)
        loader = acc.prepare(loader)

        metrics = pyine.apps.trainers.probe_trainer.validate_probes(
            coll_prepared,
            model,
            extractor,
            loader,
            loss_fn,
            global_step=0,
            accelerator=acc,
            runtime=None,
            expanded_configs_by_name=expanded_by_name,
        )

        # Aggregate loss
        loss_agg = pyine.apps.trainers.probe_trainer.aggregate_replica_metrics(
            {name: metric_value["loss"] for name, metric_value in metrics.items()},
            expanded_by_name,
        )
        assert "mean_L0" in loss_agg
        assert "mean" in loss_agg["mean_L0"]
        assert "std" in loss_agg["mean_L0"]
        assert "min" in loss_agg["mean_L0"]
        assert "max" in loss_agg["mean_L0"]
        extractor.remove_hooks()

    def test_auroc_nan_handling_in_aggregation(self) -> None:
        import math

        acc = accelerate.Accelerator()
        device = acc.device

        base_configs = [
            pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0, learning_rate=1e-2),
        ]
        expanded = pyine.apps.trainers.probe_trainer.expand_probe_configs_with_replicas(
            base_configs, num_replicas=2, replica_base_seed=0
        )
        expanded_by_name = {pc.name: pc for pc in expanded}

        model = SmallMockLLM(n_layers=2, hidden_dim=64).to(device)
        model.eval()
        model.requires_grad_(False)
        coll = pyine.guardrails.probes.collection.ProbeCollection(expanded, hidden_dim=64).to(device)

        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(model, target_layers=[0, 1])
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # All labels 0 -> single class -> NaN AUROC
        seq_len = 16
        ds = datasets.Dataset.from_dict(
            {
                "input_ids": torch.randint(0, 100, (8, seq_len)).tolist(),
                "attention_mask": torch.ones(8, seq_len, dtype=torch.long).tolist(),
                "labels": [0] * 8,
            }
        )
        ds.set_format("torch")
        loader = torch.utils.data.DataLoader(ds, batch_size=4)

        coll_prepared = acc.prepare(coll)
        loader = acc.prepare(loader)

        metrics = pyine.apps.trainers.probe_trainer.validate_probes(
            coll_prepared,
            model,
            extractor,
            loader,
            loss_fn,
            global_step=0,
            accelerator=acc,
            runtime=None,
            expanded_configs_by_name=expanded_by_name,
        )

        auroc_agg = pyine.apps.trainers.probe_trainer.aggregate_replica_metrics(
            {name: metric_value["auroc"] for name, metric_value in metrics.items()},
            expanded_by_name,
        )
        # All replicas had NaN AUROC, so aggregation should be NaN
        assert math.isnan(auroc_agg["mean_L0"]["mean"])
        extractor.remove_hooks()


class TestTrainStepWithReplicas:
    def test_train_step_reduces_loss_with_replicas(self) -> None:
        base_configs = [
            pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0, learning_rate=1e-2),
            pyine.guardrails.probes.base.ProbeConfig(name="max_L1", architecture="max", layer=1, learning_rate=1e-2),
        ]
        expanded = pyine.apps.trainers.probe_trainer.expand_probe_configs_with_replicas(
            base_configs, num_replicas=2, replica_base_seed=42
        )
        assert len(expanded) == 4

        model = SmallMockLLM(n_layers=2, hidden_dim=64)
        model.eval()
        model.requires_grad_(False)

        coll = pyine.guardrails.probes.collection.ProbeCollection(expanded, hidden_dim=64)
        extractor = pyine.guardrails.probes.extraction.ActivationExtractor(model, target_layers=[0, 1])
        optimizer = torch.optim.AdamW(coll.get_parameter_groups())
        loss_fn = torch.nn.BCEWithLogitsLoss()

        batch_size, seq_len = 8, 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = torch.randint(0, 2, (batch_size,), dtype=torch.float)

        # Initial loss
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        activations = extractor.get_activations()
        logits = coll(activations, attention_mask)
        initial_loss = sum(loss_fn(logit.squeeze(-1), labels) for logit in logits.values()).item()

        # Train for 30 steps
        for _ in range(30):
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            activations = extractor.get_activations()
            logits = coll(activations, attention_mask)
            total_loss = sum(loss_fn(logit.squeeze(-1), labels) for logit in logits.values())
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        final_loss = total_loss.item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
        extractor.remove_hooks()


# ---------------------------------------------------------------------------
# Section 6.1: TestMidTrainingSave (tests 1-5)
# ---------------------------------------------------------------------------


class TestMidTrainingSave:
    """Tests for save_probe_checkpoints() with the new checkpoint_subdir parameter."""

    def test_save_probe_checkpoints_with_step_subdir(
        self,
        tmp_path: pathlib.Path,
        probe_collection: pyine.guardrails.probes.collection.ProbeCollection,
    ) -> None:
        """Test 1: checkpoint_subdir='step-0010' creates files at <probes_base>/<probe>/step-0010/."""
        acc = accelerate.Accelerator()
        probe_collection_prepared = acc.prepare(probe_collection)

        runtime = MagicMock()
        runtime.output_dir = str(tmp_path)
        config = MagicMock()

        probes_base = pyine.apps.trainers.probe_trainer.save_probe_checkpoints(
            probe_collection_prepared,
            config,
            runtime,
            acc,
            checkpoint_subdir="step-0010",
        )

        assert probes_base is not None
        for name in probe_collection.probes:
            probe_dir = probes_base / name / "step-0010"
            assert (probe_dir / "probe_state_dict.pt").exists()
            assert (probe_dir / "probe_config.json").exists()

    def test_save_probe_checkpoints_final_subdir(
        self,
        tmp_path: pathlib.Path,
        probe_collection: pyine.guardrails.probes.collection.ProbeCollection,
    ) -> None:
        """Test 2: Default checkpoint_subdir='final' creates files at <probes_base>/<probe>/final/."""
        acc = accelerate.Accelerator()
        probe_collection_prepared = acc.prepare(probe_collection)

        runtime = MagicMock()
        runtime.output_dir = str(tmp_path)
        config = MagicMock()

        probes_base = pyine.apps.trainers.probe_trainer.save_probe_checkpoints(
            probe_collection_prepared,
            config,
            runtime,
            acc,
        )

        assert probes_base is not None
        for name in probe_collection.probes:
            probe_dir = probes_base / name / "final"
            assert (probe_dir / "probe_state_dict.pt").exists()
            assert (probe_dir / "probe_config.json").exists()

    def test_multiple_checkpoints_same_probe(
        self,
        tmp_path: pathlib.Path,
        probe_collection: pyine.guardrails.probes.collection.ProbeCollection,
    ) -> None:
        """Test 3: Multiple saves create separate subdirs under each probe folder."""
        acc = accelerate.Accelerator()
        probe_collection_prepared = acc.prepare(probe_collection)

        runtime = MagicMock()
        runtime.output_dir = str(tmp_path)
        config = MagicMock()

        for subdir in ("step-0010", "step-0020", "final"):
            pyine.apps.trainers.probe_trainer.save_probe_checkpoints(
                probe_collection_prepared,
                config,
                runtime,
                acc,
                checkpoint_subdir=subdir,
            )

        probes_base = tmp_path / "probes"
        for name in probe_collection.probes:
            for subdir in ("step-0010", "step-0020", "final"):
                probe_dir = probes_base / name / subdir
                assert (probe_dir / "probe_state_dict.pt").exists()
                assert (probe_dir / "probe_config.json").exists()

    def test_checkpoints_loadable_from_step_subdir(
        self,
        tmp_path: pathlib.Path,
        probe_collection: pyine.guardrails.probes.collection.ProbeCollection,
    ) -> None:
        """Test 4: Save to step subdir, load via ProbeCollection.load_from_checkpoint with checkpoint_name."""
        acc = accelerate.Accelerator()
        probe_collection_prepared = acc.prepare(probe_collection)

        runtime = MagicMock()
        runtime.output_dir = str(tmp_path)
        config = MagicMock()

        probes_base = pyine.apps.trainers.probe_trainer.save_probe_checkpoints(
            probe_collection_prepared,
            config,
            runtime,
            acc,
            checkpoint_subdir="step-0010",
        )
        assert probes_base is not None

        loaded = pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(
            probes_base, hidden_dim=64, checkpoint_name="step-0010"
        )
        assert set(loaded.probes.keys()) == set(probe_collection.probes.keys())

    def test_checkpoints_loadable_from_final(
        self,
        tmp_path: pathlib.Path,
        probe_collection: pyine.guardrails.probes.collection.ProbeCollection,
    ) -> None:
        """Test 5: Save to 'final' subdir, load via load_from_checkpoint (default checkpoint_name)."""
        acc = accelerate.Accelerator()
        probe_collection_prepared = acc.prepare(probe_collection)

        runtime = MagicMock()
        runtime.output_dir = str(tmp_path)
        config = MagicMock()

        probes_base = pyine.apps.trainers.probe_trainer.save_probe_checkpoints(
            probe_collection_prepared,
            config,
            runtime,
            acc,
            checkpoint_subdir="final",
        )
        assert probes_base is not None

        loaded = pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(probes_base, hidden_dim=64)
        assert set(loaded.probes.keys()) == set(probe_collection.probes.keys())

        # Verify loaded weights match originals
        for name in probe_collection.probes:
            original_params = dict(probe_collection.probes[name].named_parameters())
            loaded_params = dict(loaded.probes[name].named_parameters())
            assert set(original_params.keys()) == set(loaded_params.keys())
            for pname in original_params:
                torch.testing.assert_close(original_params[pname].cpu(), loaded_params[pname])


# ---------------------------------------------------------------------------
# Section 6.1: TestEnforceCheckpointLimit (tests 6-13)
# ---------------------------------------------------------------------------


class TestEnforceCheckpointLimit:
    """Tests for enforce_checkpoint_limit() utility."""

    @staticmethod
    def _make_step_dirs(probe_dir: pathlib.Path, steps: list[int]) -> None:
        """Create step-NNNN subdirectories with a dummy file inside each."""
        for step in steps:
            d = probe_dir / f"step-{step:04d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "probe_state_dict.pt").write_text("dummy")

    def test_no_deletion_when_under_limit(self, tmp_path: pathlib.Path) -> None:
        """Test 6: All dirs remain when count is under the limit."""
        probe_dir = tmp_path / "mean_L16"
        self._make_step_dirs(probe_dir, [10, 20])

        pyine.apps.trainers.probe_trainer.enforce_checkpoint_limit(tmp_path, save_total_limit=5)

        assert (probe_dir / "step-0010").exists()
        assert (probe_dir / "step-0020").exists()

    def test_deletes_oldest_when_over_limit(self, tmp_path: pathlib.Path) -> None:
        """Test 7: Oldest step dir is deleted when count exceeds limit."""
        probe_dir = tmp_path / "mean_L16"
        self._make_step_dirs(probe_dir, [10, 20, 30])

        pyine.apps.trainers.probe_trainer.enforce_checkpoint_limit(tmp_path, save_total_limit=2)

        assert not (probe_dir / "step-0010").exists()
        assert (probe_dir / "step-0020").exists()
        assert (probe_dir / "step-0030").exists()

    def test_enforces_per_probe_independently(self, tmp_path: pathlib.Path) -> None:
        """Test 8: Each probe folder independently retains only limit dirs."""
        for probe_name in ("mean_L16", "max_L16"):
            probe_dir = tmp_path / probe_name
            self._make_step_dirs(probe_dir, [10, 20, 30])

        pyine.apps.trainers.probe_trainer.enforce_checkpoint_limit(tmp_path, save_total_limit=2)

        for probe_name in ("mean_L16", "max_L16"):
            probe_dir = tmp_path / probe_name
            assert not (probe_dir / "step-0010").exists()
            assert (probe_dir / "step-0020").exists()
            assert (probe_dir / "step-0030").exists()

    def test_non_contiguous_step_numbers(self, tmp_path: pathlib.Path) -> None:
        """Test 9: Integer-based sorting removes the numerically smallest step."""
        probe_dir = tmp_path / "mean_L16"
        self._make_step_dirs(probe_dir, [10, 50, 100])

        pyine.apps.trainers.probe_trainer.enforce_checkpoint_limit(tmp_path, save_total_limit=2)

        assert not (probe_dir / "step-0010").exists()
        assert (probe_dir / "step-0050").exists()
        assert (probe_dir / "step-0100").exists()

    def test_final_dir_never_deleted(self, tmp_path: pathlib.Path) -> None:
        """Test 10: The 'final/' subdir is never removed by the retention policy."""
        probe_dir = tmp_path / "mean_L16"
        self._make_step_dirs(probe_dir, [10, 20, 30])
        final_dir = probe_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "probe_state_dict.pt").write_text("dummy")

        pyine.apps.trainers.probe_trainer.enforce_checkpoint_limit(tmp_path, save_total_limit=1)

        assert final_dir.exists()
        # Only step-0030 should remain (newest); step-0010 and step-0020 deleted
        assert not (probe_dir / "step-0010").exists()
        assert not (probe_dir / "step-0020").exists()
        assert (probe_dir / "step-0030").exists()

    def test_extraneous_entries_ignored(self, tmp_path: pathlib.Path) -> None:
        """Test 11: Non-step dirs/files inside probe folders and at root are preserved."""
        probe_dir = tmp_path / "mean_L16"
        self._make_step_dirs(probe_dir, [10, 20, 30])
        # Add non-step entries
        (probe_dir / "notes.txt").write_text("keep me")
        (tmp_path / "replica_summary.json").write_text("{}")

        pyine.apps.trainers.probe_trainer.enforce_checkpoint_limit(tmp_path, save_total_limit=2)

        # Extraneous entries are preserved
        assert (probe_dir / "notes.txt").exists()
        assert (tmp_path / "replica_summary.json").exists()
        # Only oldest step dir deleted
        assert not (probe_dir / "step-0010").exists()
        assert (probe_dir / "step-0020").exists()
        assert (probe_dir / "step-0030").exists()

    def test_no_op_when_limit_is_none(self, tmp_path: pathlib.Path) -> None:
        """Test 12: All dirs remain when save_total_limit is None."""
        probe_dir = tmp_path / "mean_L16"
        self._make_step_dirs(probe_dir, [10, 20, 30, 40, 50])

        pyine.apps.trainers.probe_trainer.enforce_checkpoint_limit(tmp_path, save_total_limit=None)

        for step in (10, 20, 30, 40, 50):
            assert (probe_dir / f"step-{step:04d}").exists()

    def test_handles_empty_probes_dir(self, tmp_path: pathlib.Path) -> None:
        """Test 13: No error when called on an empty directory."""
        empty_dir = tmp_path / "empty_probes"
        empty_dir.mkdir()

        # Should not raise
        pyine.apps.trainers.probe_trainer.enforce_checkpoint_limit(empty_dir, save_total_limit=1)
