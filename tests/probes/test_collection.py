"""Tests for ProbeCollection."""

from __future__ import annotations

import typing

import pytest
import torch

import pyine.probes.base
import pyine.probes.collection
import tests.probes.conftest

if typing.TYPE_CHECKING:
    import pathlib


class TestProbeCollection:
    def test_forward_returns_all_probe_logits(
        self,
        sample_probe_configs: list[pyine.probes.base.ProbeConfig],
        random_activations: dict[int, torch.Tensor],
        random_attention_mask: torch.Tensor,
    ) -> None:
        coll = pyine.probes.collection.ProbeCollection(
            sample_probe_configs, hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM
        )
        results = coll(random_activations, random_attention_mask)

        assert set(results.keys()) == {probe_config.name for probe_config in sample_probe_configs}
        for name, logits in results.items():
            assert logits.shape == (tests.probes.conftest.PROBE_BATCH_SIZE, 1), f"{name}: {logits.shape}"

    def test_parameter_groups_match_configs(
        self,
        sample_probe_configs: list[pyine.probes.base.ProbeConfig],
    ) -> None:
        coll = pyine.probes.collection.ProbeCollection(
            sample_probe_configs, hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM
        )
        groups = coll.get_parameter_groups()
        assert len(groups) == len(sample_probe_configs)

        for group, probe_config in zip(groups, sample_probe_configs, strict=True):
            assert group["lr"] == probe_config.learning_rate
            assert group["weight_decay"] == probe_config.weight_decay

    def test_parameter_groups_nonempty(
        self,
        sample_probe_configs: list[pyine.probes.base.ProbeConfig],
    ) -> None:
        coll = pyine.probes.collection.ProbeCollection(
            sample_probe_configs, hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM
        )
        groups = coll.get_parameter_groups()
        for i, group in enumerate(groups):
            assert len(group["params"]) > 0, f"Group {i} has no parameters"

    def test_hidden_dim_auto_populated(
        self,
        sample_probe_configs: list[pyine.probes.base.ProbeConfig],
    ) -> None:
        """Probes receive hidden_dim from ProbeCollection constructor, not from config."""
        # Configs start with hidden_dim=None
        for probe_config in sample_probe_configs:
            assert probe_config.hidden_dim is None

        coll = pyine.probes.collection.ProbeCollection(
            sample_probe_configs, hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM
        )

        # All probes in the collection should have hidden_dim set
        for name, probe in coll.probes.items():
            assert probe.config.hidden_dim == tests.probes.conftest.PROBE_HIDDEN_DIM, (
                f"Probe {name} has hidden_dim={probe.config.hidden_dim}"
            )

    def test_summed_loss_gradient_independence(
        self,
        sample_probe_configs: list[pyine.probes.base.ProbeConfig],
        random_activations: dict[int, torch.Tensor],
        random_attention_mask: torch.Tensor,
    ) -> None:
        """Gradient of sum(losses) w.r.t. probe_i params equals gradient of loss_i alone."""
        coll = pyine.probes.collection.ProbeCollection(
            sample_probe_configs, hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()
        labels = torch.randint(0, 2, (tests.probes.conftest.PROBE_BATCH_SIZE,), dtype=torch.float)

        # 1) Compute gradient of sum(losses) w.r.t. first probe's params
        logits = coll(random_activations, random_attention_mask)
        total_loss = sum(loss_fn(logit.squeeze(-1), labels) for logit in logits.values())
        total_loss.backward()

        first_probe_name = sample_probe_configs[0].name
        grads_from_sum = {
            param_name: param.grad.clone() for param_name, param in coll.probes[first_probe_name].named_parameters()
        }
        coll.zero_grad()

        # 2) Compute gradient of only the first probe's loss
        logits2 = coll(random_activations, random_attention_mask)
        single_loss = loss_fn(logits2[first_probe_name].squeeze(-1), labels)
        single_loss.backward()

        grads_from_single = {
            param_name: param.grad.clone() for param_name, param in coll.probes[first_probe_name].named_parameters()
        }

        # They should be identical
        for param_name in grads_from_sum:
            torch.testing.assert_close(
                grads_from_sum[param_name],
                grads_from_single[param_name],
                msg=f"Gradient mismatch for {first_probe_name}.{param_name}",
            )

    def test_all_submodules_visible_to_ddp(
        self,
        sample_probe_configs: list[pyine.probes.base.ProbeConfig],
    ) -> None:
        """All probe parameters are in collection.parameters() (needed for DDP)."""
        coll = pyine.probes.collection.ProbeCollection(
            sample_probe_configs, hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM
        )

        # Collect all params from individual probes
        probe_param_ids = set()
        for probe in coll.probes.values():
            for param in probe.parameters():
                probe_param_ids.add(id(param))

        # All should be in collection.parameters()
        collection_param_ids = {id(param) for param in coll.parameters()}
        assert probe_param_ids == collection_param_ids


class TestReplicaSeeding:
    def test_different_replicas_have_different_weights(
        self,
        replica_probe_configs: list[pyine.probes.base.ProbeConfig],
    ) -> None:
        coll = pyine.probes.collection.ProbeCollection(
            replica_probe_configs, hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM
        )
        # mean_L0_r0 and mean_L0_r1 should have different weight tensors
        params_r0 = dict(coll.probes["mean_L0_r0"].named_parameters())
        params_r1 = dict(coll.probes["mean_L0_r1"].named_parameters())
        any_different = False
        for param_name in params_r0:
            if not torch.equal(params_r0[param_name], params_r1[param_name]):
                any_different = True
                break
        assert any_different, "Replica r0 and r1 should have different weights"

    def test_same_seed_produces_same_weights(
        self,
        replica_probe_configs: list[pyine.probes.base.ProbeConfig],
    ) -> None:
        coll1 = pyine.probes.collection.ProbeCollection(
            replica_probe_configs, hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM
        )
        coll2 = pyine.probes.collection.ProbeCollection(
            replica_probe_configs, hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM
        )
        for name in coll1.probes:
            for (param_name_1, param_1), (param_name_2, param_2) in zip(
                coll1.probes[name].named_parameters(),
                coll2.probes[name].named_parameters(),
                strict=True,
            ):
                assert param_name_1 == param_name_2
                assert torch.equal(param_1, param_2), f"Weight mismatch for {name}.{param_name_1}"

    def test_rng_state_restored_after_construction(
        self,
        replica_probe_configs: list[pyine.probes.base.ProbeConfig],
    ) -> None:
        """Global RNG state is restored after ProbeCollection construction with replicas."""
        torch.manual_seed(12345)
        before_val = torch.randn(1).item()

        # Re-seed and construct collection (should save/restore RNG state)
        torch.manual_seed(12345)
        pyine.probes.collection.ProbeCollection(
            replica_probe_configs, hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM
        )
        after_val = torch.randn(1).item()

        assert before_val == after_val, "RNG state was not restored after replica construction"

    def test_non_seeded_probes_unaffected(
        self,
        sample_probe_configs: list[pyine.probes.base.ProbeConfig],
    ) -> None:
        # Should construct normally without any issues
        coll = pyine.probes.collection.ProbeCollection(
            sample_probe_configs, hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM
        )
        assert len(coll.probes) == len(sample_probe_configs)


def _save_collection_to_dir(
    collection: pyine.probes.collection.ProbeCollection,
    output_dir: pathlib.Path,
) -> None:
    """Save a ProbeCollection to disk in the same format as save_probe_checkpoints."""
    for name, module in collection.probes.items():
        probe = typing.cast("pyine.probes.base.BaseProbe", module)
        probe_dir = output_dir / name
        probe_dir.mkdir(parents=True, exist_ok=True)
        torch.save(probe.state_dict(), probe_dir / "probe_state_dict.pt")
        (probe_dir / "probe_config.json").write_text(probe.config.model_dump_json(indent=2))


class TestLoadFromCheckpoint:
    @pytest.fixture
    def saved_collection(
        self,
        sample_probe_configs: list[pyine.probes.base.ProbeConfig],
        tmp_path: pathlib.Path,
    ) -> tuple[pyine.probes.collection.ProbeCollection, pathlib.Path]:
        coll = pyine.probes.collection.ProbeCollection(
            sample_probe_configs, hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM
        )
        checkpoint_dir = tmp_path / "probes"
        _save_collection_to_dir(coll, checkpoint_dir)
        return coll, checkpoint_dir

    def test_round_trip_preserves_weights(
        self,
        saved_collection: tuple[pyine.probes.collection.ProbeCollection, pathlib.Path],
    ) -> None:
        original, checkpoint_dir = saved_collection
        loaded = pyine.probes.collection.ProbeCollection.load_from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM,
        )
        assert set(loaded.probes.keys()) == set(original.probes.keys())
        for name in original.probes:
            for (orig_name, orig_param), (load_name, load_param) in zip(
                original.probes[name].named_parameters(),
                loaded.probes[name].named_parameters(),
                strict=True,
            ):
                assert orig_name == load_name
                torch.testing.assert_close(orig_param, load_param)

    def test_round_trip_preserves_probe_configs(
        self,
        saved_collection: tuple[pyine.probes.collection.ProbeCollection, pathlib.Path],
    ) -> None:
        original, checkpoint_dir = saved_collection
        loaded = pyine.probes.collection.ProbeCollection.load_from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM,
        )
        for name in original.probes:
            orig_cfg = original._probe_configs[name]
            load_cfg = loaded._probe_configs[name]
            assert orig_cfg.name == load_cfg.name
            assert orig_cfg.architecture == load_cfg.architecture
            assert orig_cfg.layer == load_cfg.layer

    def test_loaded_collection_produces_output(
        self,
        saved_collection: tuple[pyine.probes.collection.ProbeCollection, pathlib.Path],
        random_activations: dict[int, torch.Tensor],
        random_attention_mask: torch.Tensor,
    ) -> None:
        _, checkpoint_dir = saved_collection
        loaded = pyine.probes.collection.ProbeCollection.load_from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM,
        )
        results = loaded(random_activations, random_attention_mask)
        assert len(results) > 0
        for logits in results.values():
            assert logits.shape == (tests.probes.conftest.PROBE_BATCH_SIZE, 1)

    def test_empty_dir_raises(self, tmp_path: pathlib.Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No probe configs found"):
            pyine.probes.collection.ProbeCollection.load_from_checkpoint(
                checkpoint_dir=empty_dir,
                hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM,
            )

    def test_loads_to_cpu_by_default(
        self,
        saved_collection: tuple[pyine.probes.collection.ProbeCollection, pathlib.Path],
    ) -> None:
        """Weights are loaded to CPU regardless of where they were saved."""
        _, checkpoint_dir = saved_collection
        loaded = pyine.probes.collection.ProbeCollection.load_from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            hidden_dim=tests.probes.conftest.PROBE_HIDDEN_DIM,
        )
        for module in loaded.probes.values():
            for param in module.parameters():
                assert param.device == torch.device("cpu")
