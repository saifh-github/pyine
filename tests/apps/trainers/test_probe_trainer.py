"""Unit tests for the probe training loop with mocked LLM."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import datasets
import pytest
import torch

from pyine.probes import build_probe
from pyine.probes.base import ProbeConfig
from pyine.probes.collection import ProbeCollection
from pyine.probes.extraction import ActivationExtractor

# ---------------------------------------------------------------------------
# Small mock LLM for testing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> SmallMockLLM:
    model = SmallMockLLM(n_layers=2, hidden_dim=64)
    model.eval()
    model.requires_grad_(False)
    return model


@pytest.fixture
def probe_configs() -> list[ProbeConfig]:
    return [
        ProbeConfig(name="mean_L0", architecture="mean", layer=0, learning_rate=1e-2),
        ProbeConfig(name="max_L1", architecture="max", layer=1, learning_rate=1e-2),
    ]


@pytest.fixture
def probe_collection(probe_configs: list[ProbeConfig]) -> ProbeCollection:
    return ProbeCollection(probe_configs, hidden_dim=64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProbeTrainUnit:
    """Unit tests for the probe training components with mocked LLM."""

    def test_train_step_reduces_loss(
        self,
        mock_llm: SmallMockLLM,
        probe_collection: ProbeCollection,
    ) -> None:
        """After a few steps, train loss is lower than initial loss."""
        extractor = ActivationExtractor(mock_llm, target_layers=[0, 1])
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

    def test_validation_produces_metrics(
        self,
        probe_configs: list[ProbeConfig],
    ) -> None:
        """validate_probes() returns loss and AUROC per probe."""
        from accelerate import Accelerator

        from pyine.apps.trainers.probe_trainer import validate_probes

        accelerator = Accelerator()
        device = accelerator.device

        # Create fresh model + collection on correct device
        model = SmallMockLLM(n_layers=2, hidden_dim=64).to(device)
        model.eval()
        model.requires_grad_(False)
        coll = ProbeCollection(probe_configs, hidden_dim=64).to(device)

        extractor = ActivationExtractor(model, target_layers=[0, 1])
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

        coll_prepared = accelerator.prepare(coll)
        loader = accelerator.prepare(loader)

        metrics = validate_probes(
            coll_prepared,
            model,
            extractor,
            loader,
            loss_fn,
            global_step=0,
            accelerator=accelerator,
            runtime=None,
        )

        assert len(metrics) > 0
        for _name, m in metrics.items():
            assert "loss" in m
            assert "auroc" in m
            assert m["loss"] >= 0.0

        extractor.remove_hooks()

    def test_auroc_guard_single_class(
        self,
        probe_configs: list[ProbeConfig],
    ) -> None:
        """validate_probes() returns NaN AUROC for single-class valid set."""
        import math

        from accelerate import Accelerator

        from pyine.apps.trainers.probe_trainer import validate_probes

        accelerator = Accelerator()
        device = accelerator.device

        model = SmallMockLLM(n_layers=2, hidden_dim=64).to(device)
        model.eval()
        model.requires_grad_(False)
        coll = ProbeCollection(probe_configs, hidden_dim=64).to(device)

        extractor = ActivationExtractor(model, target_layers=[0, 1])
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

        coll_prepared = accelerator.prepare(coll)
        loader_prepared = accelerator.prepare(loader)

        metrics = validate_probes(
            coll_prepared,
            model,
            extractor,
            loader_prepared,
            loss_fn,
            global_step=0,
            accelerator=accelerator,
            runtime=None,
        )

        for name, m in metrics.items():
            assert math.isnan(m["auroc"]), f"Expected NaN AUROC for {name}, got {m['auroc']}"

        extractor.remove_hooks()

    def test_save_probe_checkpoints(
        self,
        tmp_path: Path,
        probe_collection: ProbeCollection,
    ) -> None:
        """save_probe_checkpoints() writes state_dict + config JSON per probe."""
        from accelerate import Accelerator

        from pyine.apps.trainers.probe_trainer import save_probe_checkpoints

        accelerator = Accelerator()
        probe_collection_prepared = accelerator.prepare(probe_collection)

        runtime = MagicMock()
        runtime.output_dir = str(tmp_path)

        config = MagicMock()

        output_dir = save_probe_checkpoints(
            probe_collection_prepared,
            config,
            runtime,
            accelerator,
        )

        assert output_dir is not None
        for name in probe_collection.probes:
            probe_dir = output_dir / name
            assert (probe_dir / "probe_state_dict.pt").exists()
            assert (probe_dir / "probe_config.json").exists()

    def test_probe_checkpoints_loadable(
        self,
        tmp_path: Path,
        probe_collection: ProbeCollection,
        probe_configs: list[ProbeConfig],
    ) -> None:
        """Saved probe can be reconstructed from config JSON + state_dict."""
        import json

        from accelerate import Accelerator

        from pyine.apps.trainers.probe_trainer import save_probe_checkpoints

        accelerator = Accelerator()
        probe_collection_prepared = accelerator.prepare(probe_collection)

        runtime = MagicMock()
        runtime.output_dir = str(tmp_path)
        config = MagicMock()

        output_dir = save_probe_checkpoints(
            probe_collection_prepared,
            config,
            runtime,
            accelerator,
        )

        # Reload each probe
        for name in probe_collection.probes:
            probe_dir = output_dir / name
            config_json = json.loads((probe_dir / "probe_config.json").read_text())
            loaded_config = ProbeConfig(**config_json)
            loaded_probe = build_probe(loaded_config)
            state_dict = torch.load(probe_dir / "probe_state_dict.pt", weights_only=True)
            loaded_probe.load_state_dict(state_dict)

            # Check shapes match
            original_params = dict(probe_collection.probes[name].named_parameters())
            loaded_params = dict(loaded_probe.named_parameters())
            assert set(original_params.keys()) == set(loaded_params.keys())


# ---------------------------------------------------------------------------
# Replica feature tests
# ---------------------------------------------------------------------------


def _replica_pc(
    name: str,
    arch: str = "mean",
    layer: int = 0,
    base_name: str = "a",
    replica_idx: int = 0,
    replica_seed: int = 1,
) -> ProbeConfig:
    """Shorthand for creating ProbeConfig with replica metadata in tests."""
    return ProbeConfig(
        name=name,
        architecture=arch,
        layer=layer,
        base_name=base_name,
        replica_idx=replica_idx,
        replica_seed=replica_seed,
    )


class TestStableReplicaSeed:
    """Tests for _stable_replica_seed determinism."""

    def test_deterministic_across_calls(self) -> None:
        """Same (base_seed, name, replica_idx) produces same seed on every call."""
        from pyine.apps.trainers.probe_trainer import _stable_replica_seed

        s1 = _stable_replica_seed(0, "mean_L0", 0)
        s2 = _stable_replica_seed(0, "mean_L0", 0)
        assert s1 == s2

    def test_different_replica_idx_different_seeds(self) -> None:
        """Different replica_idx values produce different seeds."""
        from pyine.apps.trainers.probe_trainer import _stable_replica_seed

        s0 = _stable_replica_seed(0, "mean_L0", 0)
        s1 = _stable_replica_seed(0, "mean_L0", 1)
        assert s0 != s1

    def test_different_base_seed_different_seeds(self) -> None:
        """Different base_seed values produce different seeds."""
        from pyine.apps.trainers.probe_trainer import _stable_replica_seed

        s0 = _stable_replica_seed(0, "mean_L0", 0)
        s1 = _stable_replica_seed(42, "mean_L0", 0)
        assert s0 != s1

    def test_different_name_different_seeds(self) -> None:
        """Different probe names produce different seeds."""
        from pyine.apps.trainers.probe_trainer import _stable_replica_seed

        s0 = _stable_replica_seed(0, "mean_L0", 0)
        s1 = _stable_replica_seed(0, "attn_L8", 0)
        assert s0 != s1

    def test_seed_within_valid_range(self) -> None:
        """Returned seed is in [0, 2^31)."""
        from pyine.apps.trainers.probe_trainer import _stable_replica_seed

        for i in range(100):
            seed = _stable_replica_seed(i, f"probe_{i}", i % 10)
            assert 0 <= seed < 2**31


class TestExpandProbeConfigsWithReplicas:
    """Tests for the config expansion function."""

    def test_no_expansion_when_num_replicas_1(self) -> None:
        """num_replicas=1 returns original list unchanged (same object)."""
        from pyine.apps.trainers.probe_trainer import expand_probe_configs_with_replicas

        configs = [ProbeConfig(name="mean_L0", architecture="mean", layer=0)]
        result = expand_probe_configs_with_replicas(configs, num_replicas=1, replica_base_seed=0)
        assert result is configs

    def test_expansion_creates_correct_count(self) -> None:
        """2 configs x 3 replicas = 6 expanded configs."""
        from pyine.apps.trainers.probe_trainer import expand_probe_configs_with_replicas

        configs = [
            ProbeConfig(name="mean_L0", architecture="mean", layer=0),
            ProbeConfig(name="attn_L8", architecture="attention", layer=8),
        ]
        result = expand_probe_configs_with_replicas(configs, num_replicas=3, replica_base_seed=0)
        assert len(result) == 6

    def test_expanded_names_follow_pattern(self) -> None:
        """Expanded name is '{original}_r{idx}'."""
        from pyine.apps.trainers.probe_trainer import expand_probe_configs_with_replicas

        configs = [ProbeConfig(name="mean_L0", architecture="mean", layer=0)]
        result = expand_probe_configs_with_replicas(configs, num_replicas=3, replica_base_seed=0)
        names = [pc.name for pc in result]
        assert names == ["mean_L0_r0", "mean_L0_r1", "mean_L0_r2"]

    def test_replica_metadata_populated(self) -> None:
        """Each expanded config has replica_idx, replica_seed, base_name set."""
        from pyine.apps.trainers.probe_trainer import expand_probe_configs_with_replicas

        configs = [ProbeConfig(name="mean_L0", architecture="mean", layer=0)]
        result = expand_probe_configs_with_replicas(configs, num_replicas=2, replica_base_seed=0)
        for pc in result:
            assert pc.replica_idx is not None
            assert pc.replica_seed is not None
            assert pc.base_name is not None

    def test_seeds_are_unique(self) -> None:
        """All replica seeds are distinct across all expanded configs."""
        from pyine.apps.trainers.probe_trainer import expand_probe_configs_with_replicas

        configs = [
            ProbeConfig(name="mean_L0", architecture="mean", layer=0),
            ProbeConfig(name="attn_L8", architecture="attention", layer=8),
        ]
        result = expand_probe_configs_with_replicas(configs, num_replicas=5, replica_base_seed=0)
        seeds = [pc.replica_seed for pc in result]
        assert len(set(seeds)) == len(seeds), f"Duplicate seeds found: {seeds}"

    def test_seeds_are_deterministic(self) -> None:
        """Same inputs produce same seeds on repeated calls."""
        from pyine.apps.trainers.probe_trainer import expand_probe_configs_with_replicas

        configs = [ProbeConfig(name="mean_L0", architecture="mean", layer=0)]
        r1 = expand_probe_configs_with_replicas(configs, num_replicas=3, replica_base_seed=42)
        r2 = expand_probe_configs_with_replicas(configs, num_replicas=3, replica_base_seed=42)
        for a, b in zip(r1, r2, strict=True):
            assert a.replica_seed == b.replica_seed

    def test_base_name_matches_original(self) -> None:
        """base_name equals the original probe config name."""
        from pyine.apps.trainers.probe_trainer import expand_probe_configs_with_replicas

        configs = [ProbeConfig(name="mean_L0", architecture="mean", layer=0)]
        result = expand_probe_configs_with_replicas(configs, num_replicas=2, replica_base_seed=0)
        for pc in result:
            assert pc.base_name == "mean_L0"

    def test_non_replica_fields_preserved(self) -> None:
        """architecture, layer, learning_rate, etc. are unchanged in replicas."""
        from pyine.apps.trainers.probe_trainer import expand_probe_configs_with_replicas

        configs = [ProbeConfig(name="attn_L8", architecture="attention", layer=8, learning_rate=5e-4, attn_dim=32)]
        result = expand_probe_configs_with_replicas(configs, num_replicas=2, replica_base_seed=0)
        for pc in result:
            assert pc.architecture == "attention"
            assert pc.layer == 8
            assert pc.learning_rate == 5e-4
            assert pc.attn_dim == 32


class TestAggregateReplicaMetrics:
    """Tests for the metric aggregation helper."""

    def test_single_group(self) -> None:
        """All probes share base_name -> single aggregated entry."""
        from pyine.apps.trainers.probe_trainer import aggregate_replica_metrics

        configs = {
            "mean_L0_r0": _replica_pc("mean_L0_r0", base_name="mean_L0", replica_idx=0),
            "mean_L0_r1": _replica_pc("mean_L0_r1", base_name="mean_L0", replica_idx=1, replica_seed=2),
        }
        result = aggregate_replica_metrics({"mean_L0_r0": 0.5, "mean_L0_r1": 0.7}, configs)
        assert "mean_L0" in result
        assert len(result) == 1

    def test_multiple_groups(self) -> None:
        """Different base_names produce separate aggregated entries."""
        from pyine.apps.trainers.probe_trainer import aggregate_replica_metrics

        configs = {
            "a_r0": _replica_pc("a_r0", replica_idx=0),
            "a_r1": _replica_pc("a_r1", replica_idx=1, replica_seed=2),
            "b_r0": _replica_pc("b_r0", arch="max", base_name="b", replica_idx=0, replica_seed=3),
        }
        result = aggregate_replica_metrics({"a_r0": 0.5, "a_r1": 0.7, "b_r0": 0.3}, configs)
        assert set(result.keys()) == {"a", "b"}

    def test_single_value_std_is_zero(self) -> None:
        """A group with 1 value has std=0.0."""
        from pyine.apps.trainers.probe_trainer import aggregate_replica_metrics

        configs = {
            "b_r0": _replica_pc("b_r0", arch="max", base_name="b"),
        }
        result = aggregate_replica_metrics({"b_r0": 0.42}, configs)
        assert result["b"]["std"] == 0.0

    def test_non_replicated_probes(self) -> None:
        """Probes with base_name=None use name as base."""
        from pyine.apps.trainers.probe_trainer import aggregate_replica_metrics

        configs = {
            "mean_L0": ProbeConfig(name="mean_L0", architecture="mean", layer=0),
        }
        result = aggregate_replica_metrics({"mean_L0": 0.5}, configs)
        assert "mean_L0" in result
        assert result["mean_L0"]["mean"] == 0.5

    def test_mean_std_correctness(self) -> None:
        """Mean and std match statistics.mean() and statistics.stdev()."""
        import statistics

        from pyine.apps.trainers.probe_trainer import aggregate_replica_metrics

        values = [0.3, 0.5, 0.7]
        configs = {f"a_r{i}": _replica_pc(f"a_r{i}", replica_idx=i, replica_seed=i) for i in range(3)}
        per_probe = {f"a_r{i}": v for i, v in enumerate(values)}
        result = aggregate_replica_metrics(per_probe, configs)
        assert abs(result["a"]["mean"] - statistics.mean(values)) < 1e-10
        assert abs(result["a"]["std"] - statistics.stdev(values)) < 1e-10

    def test_nan_excluded_from_aggregation(self) -> None:
        """NaN values are excluded; non-NaN values still produce correct stats."""
        import math

        from pyine.apps.trainers.probe_trainer import aggregate_replica_metrics

        configs = {
            "a_r0": _replica_pc("a_r0", replica_idx=0),
            "a_r1": _replica_pc("a_r1", replica_idx=1, replica_seed=2),
            "a_r2": _replica_pc("a_r2", replica_idx=2, replica_seed=3),
        }
        result = aggregate_replica_metrics({"a_r0": 0.5, "a_r1": float("nan"), "a_r2": 0.7}, configs)
        assert not math.isnan(result["a"]["mean"])
        assert abs(result["a"]["mean"] - 0.6) < 1e-10

    def test_all_nan_returns_nan(self) -> None:
        """When all values for a base_name are NaN, all stats are NaN."""
        import math

        from pyine.apps.trainers.probe_trainer import aggregate_replica_metrics

        configs = {
            "a_r0": _replica_pc("a_r0", replica_idx=0),
            "a_r1": _replica_pc("a_r1", replica_idx=1, replica_seed=2),
        }
        result = aggregate_replica_metrics({"a_r0": float("nan"), "a_r1": float("nan")}, configs)
        assert math.isnan(result["a"]["mean"])
        assert math.isnan(result["a"]["std"])
        assert math.isnan(result["a"]["min"])
        assert math.isnan(result["a"]["max"])


class TestReplicaTables:
    """Tests for W&B Table construction helpers."""

    def test_train_table_has_correct_columns(self) -> None:
        """build_train_replica_table returns table with expected column names."""
        from pyine.apps.trainers.probe_trainer import build_train_replica_table

        configs = {"a_r0": _replica_pc("a_r0")}
        table = build_train_replica_table({"a_r0": 0.5}, configs, global_step=10, epoch=0)
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
        """Table has one row per probe in per_probe_losses dict."""
        from pyine.apps.trainers.probe_trainer import build_train_replica_table

        configs = {
            "a_r0": _replica_pc("a_r0", replica_idx=0),
            "a_r1": _replica_pc("a_r1", replica_idx=1, replica_seed=2),
            "b_r0": _replica_pc("b_r0", arch="max", layer=4, base_name="b", replica_seed=3),
        }
        table = build_train_replica_table(
            {"a_r0": 0.5, "a_r1": 0.6, "b_r0": 0.7},
            configs,
            global_step=10,
            epoch=0,
        )
        assert len(table.data) == 3

    def test_train_table_base_name_populated(self) -> None:
        """base_name column reflects the original probe name, not the replica name."""
        from pyine.apps.trainers.probe_trainer import build_train_replica_table

        configs = {"a_r0": _replica_pc("a_r0")}
        table = build_train_replica_table({"a_r0": 0.5}, configs, global_step=10, epoch=0)
        base_name_col_idx = table.columns.index("base_name")
        assert table.data[0][base_name_col_idx] == "a"

    def test_valid_table_has_correct_columns(self) -> None:
        """build_valid_replica_table returns table with expected column names."""
        from pyine.apps.trainers.probe_trainer import build_valid_replica_table

        configs = {"a_r0": _replica_pc("a_r0")}
        table = build_valid_replica_table(
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
        """Each row has valid_loss and auroc values from the per-probe metrics."""
        from pyine.apps.trainers.probe_trainer import build_valid_replica_table

        configs = {"a_r0": _replica_pc("a_r0")}
        table = build_valid_replica_table(
            {"a_r0": {"loss": 0.5, "auroc": 0.8}},
            configs,
            global_step=10,
        )
        loss_idx = table.columns.index("valid_loss")
        auroc_idx = table.columns.index("auroc")
        assert table.data[0][loss_idx] == 0.5
        assert table.data[0][auroc_idx] == 0.8

    def test_valid_table_row_count_matches_probes(self) -> None:
        """Table has one row per probe in per_probe_metrics dict."""
        from pyine.apps.trainers.probe_trainer import build_valid_replica_table

        configs = {f"a_r{i}": _replica_pc(f"a_r{i}", replica_idx=i, replica_seed=i) for i in range(4)}
        metrics = {f"a_r{i}": {"loss": 0.5 + i * 0.1, "auroc": 0.7 + i * 0.05} for i in range(4)}
        table = build_valid_replica_table(metrics, configs, global_step=10)
        assert len(table.data) == 4

    def test_tables_with_non_replicated_probes(self) -> None:
        """Tables work when replica_idx is None (non-replica ProbeConfigs)."""
        from pyine.apps.trainers.probe_trainer import build_train_replica_table

        configs = {
            "mean_L0": ProbeConfig(name="mean_L0", architecture="mean", layer=0),
        }
        table = build_train_replica_table(
            {"mean_L0": 0.5},
            configs,
            global_step=10,
            epoch=0,
        )
        replica_idx_col = table.columns.index("replica_idx")
        assert table.data[0][replica_idx_col] == 0


class TestValidateProbesWithReplicas:
    """Tests for validate_probes with replica-expanded probe collections."""

    def test_validation_returns_per_replica_metrics(self) -> None:
        """validate_probes returns metrics keyed by replica name (e.g., 'mean_L0_r0')."""
        from accelerate import Accelerator

        from pyine.apps.trainers.probe_trainer import expand_probe_configs_with_replicas, validate_probes

        accelerator = Accelerator()
        device = accelerator.device

        base_configs = [
            ProbeConfig(name="mean_L0", architecture="mean", layer=0, learning_rate=1e-2),
            ProbeConfig(name="max_L1", architecture="max", layer=1, learning_rate=1e-2),
        ]
        expanded = expand_probe_configs_with_replicas(base_configs, num_replicas=2, replica_base_seed=0)
        expanded_by_name = {pc.name: pc for pc in expanded}

        model = SmallMockLLM(n_layers=2, hidden_dim=64).to(device)
        model.eval()
        model.requires_grad_(False)
        coll = ProbeCollection(expanded, hidden_dim=64).to(device)

        extractor = ActivationExtractor(model, target_layers=[0, 1])
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

        coll_prepared = accelerator.prepare(coll)
        loader = accelerator.prepare(loader)

        metrics = validate_probes(
            coll_prepared,
            model,
            extractor,
            loader,
            loss_fn,
            global_step=0,
            accelerator=accelerator,
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
        """aggregate_replica_metrics correctly groups validate_probes output by base_name."""
        from accelerate import Accelerator

        from pyine.apps.trainers.probe_trainer import (
            aggregate_replica_metrics,
            expand_probe_configs_with_replicas,
            validate_probes,
        )

        accelerator = Accelerator()
        device = accelerator.device

        base_configs = [
            ProbeConfig(name="mean_L0", architecture="mean", layer=0, learning_rate=1e-2),
        ]
        expanded = expand_probe_configs_with_replicas(base_configs, num_replicas=3, replica_base_seed=0)
        expanded_by_name = {pc.name: pc for pc in expanded}

        model = SmallMockLLM(n_layers=2, hidden_dim=64).to(device)
        model.eval()
        model.requires_grad_(False)
        coll = ProbeCollection(expanded, hidden_dim=64).to(device)

        extractor = ActivationExtractor(model, target_layers=[0, 1])
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

        coll_prepared = accelerator.prepare(coll)
        loader = accelerator.prepare(loader)

        metrics = validate_probes(
            coll_prepared,
            model,
            extractor,
            loader,
            loss_fn,
            global_step=0,
            accelerator=accelerator,
            runtime=None,
            expanded_configs_by_name=expanded_by_name,
        )

        # Aggregate loss
        loss_agg = aggregate_replica_metrics(
            {name: m["loss"] for name, m in metrics.items()},
            expanded_by_name,
        )
        assert "mean_L0" in loss_agg
        assert "mean" in loss_agg["mean_L0"]
        assert "std" in loss_agg["mean_L0"]
        assert "min" in loss_agg["mean_L0"]
        assert "max" in loss_agg["mean_L0"]
        extractor.remove_hooks()

    def test_auroc_nan_handling_in_aggregation(self) -> None:
        """NaN AUROC values (single-class valid set) are excluded from aggregation."""
        import math

        from accelerate import Accelerator

        from pyine.apps.trainers.probe_trainer import (
            aggregate_replica_metrics,
            expand_probe_configs_with_replicas,
            validate_probes,
        )

        accelerator = Accelerator()
        device = accelerator.device

        base_configs = [
            ProbeConfig(name="mean_L0", architecture="mean", layer=0, learning_rate=1e-2),
        ]
        expanded = expand_probe_configs_with_replicas(base_configs, num_replicas=2, replica_base_seed=0)
        expanded_by_name = {pc.name: pc for pc in expanded}

        model = SmallMockLLM(n_layers=2, hidden_dim=64).to(device)
        model.eval()
        model.requires_grad_(False)
        coll = ProbeCollection(expanded, hidden_dim=64).to(device)

        extractor = ActivationExtractor(model, target_layers=[0, 1])
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # All labels 0 → single class → NaN AUROC
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

        coll_prepared = accelerator.prepare(coll)
        loader = accelerator.prepare(loader)

        metrics = validate_probes(
            coll_prepared,
            model,
            extractor,
            loader,
            loss_fn,
            global_step=0,
            accelerator=accelerator,
            runtime=None,
            expanded_configs_by_name=expanded_by_name,
        )

        auroc_agg = aggregate_replica_metrics(
            {name: m["auroc"] for name, m in metrics.items()},
            expanded_by_name,
        )
        # All replicas had NaN AUROC, so aggregation should be NaN
        assert math.isnan(auroc_agg["mean_L0"]["mean"])
        extractor.remove_hooks()


class TestTrainStepWithReplicas:
    """End-to-end test for training with replica-expanded probes."""

    def test_train_step_reduces_loss_with_replicas(self) -> None:
        """Train loop with replica-expanded probes reduces loss."""
        from pyine.apps.trainers.probe_trainer import expand_probe_configs_with_replicas

        base_configs = [
            ProbeConfig(name="mean_L0", architecture="mean", layer=0, learning_rate=1e-2),
            ProbeConfig(name="max_L1", architecture="max", layer=1, learning_rate=1e-2),
        ]
        expanded = expand_probe_configs_with_replicas(base_configs, num_replicas=2, replica_base_seed=42)
        assert len(expanded) == 4

        model = SmallMockLLM(n_layers=2, hidden_dim=64)
        model.eval()
        model.requires_grad_(False)

        coll = ProbeCollection(expanded, hidden_dim=64)
        extractor = ActivationExtractor(model, target_layers=[0, 1])
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
