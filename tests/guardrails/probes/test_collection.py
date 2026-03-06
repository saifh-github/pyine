"""Tests for ProbeCollection."""

from __future__ import annotations

import typing

import pytest
import torch

import pyine.guardrails.probes.base
import pyine.guardrails.probes.collection
import tests.guardrails.probes.conftest

if typing.TYPE_CHECKING:
    import pathlib


class TestProbeCollection:
    def test_forward_returns_all_probe_logits(
        self,
        sample_probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
        random_activations: dict[int, torch.Tensor],
        random_attention_mask: torch.Tensor,
    ) -> None:
        coll = pyine.guardrails.probes.collection.ProbeCollection(
            sample_probe_configs, hidden_dim=tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
        )
        results = coll(random_activations, random_attention_mask)

        assert set(results.keys()) == {probe_config.name for probe_config in sample_probe_configs}
        for name, logits in results.items():
            assert logits.shape == (tests.guardrails.probes.conftest.PROBE_BATCH_SIZE, 1), f"{name}: {logits.shape}"

    def test_parameter_groups_match_configs(
        self,
        sample_probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
    ) -> None:
        coll = pyine.guardrails.probes.collection.ProbeCollection(
            sample_probe_configs, hidden_dim=tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
        )
        groups = coll.get_parameter_groups()
        assert len(groups) == len(sample_probe_configs)

        for group, probe_config in zip(groups, sample_probe_configs, strict=True):
            assert group["lr"] == probe_config.learning_rate
            assert group["weight_decay"] == probe_config.weight_decay

    def test_parameter_groups_nonempty(
        self,
        sample_probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
    ) -> None:
        coll = pyine.guardrails.probes.collection.ProbeCollection(
            sample_probe_configs, hidden_dim=tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
        )
        groups = coll.get_parameter_groups()
        for i, group in enumerate(groups):
            assert len(group["params"]) > 0, f"Group {i} has no parameters"

    def test_hidden_dim_auto_populated(
        self,
        sample_probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
    ) -> None:
        """Probes receive hidden_dim from ProbeCollection constructor, not from config."""
        # Configs start with hidden_dim=None
        for probe_config in sample_probe_configs:
            assert probe_config.hidden_dim is None

        coll = pyine.guardrails.probes.collection.ProbeCollection(
            sample_probe_configs, hidden_dim=tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
        )

        # All probes in the collection should have hidden_dim set
        for name, probe in coll.probes.items():
            assert probe.config.hidden_dim == tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM, (
                f"Probe {name} has hidden_dim={probe.config.hidden_dim}"
            )

    def test_summed_loss_gradient_independence(
        self,
        sample_probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
        random_activations: dict[int, torch.Tensor],
        random_attention_mask: torch.Tensor,
    ) -> None:
        """Gradient of sum(losses) w.r.t. probe_i params equals gradient of loss_i alone."""
        coll = pyine.guardrails.probes.collection.ProbeCollection(
            sample_probe_configs, hidden_dim=tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()
        labels = torch.randint(0, 2, (tests.guardrails.probes.conftest.PROBE_BATCH_SIZE,), dtype=torch.float)

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
        sample_probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
    ) -> None:
        """All probe parameters are in collection.parameters() (needed for DDP)."""
        coll = pyine.guardrails.probes.collection.ProbeCollection(
            sample_probe_configs, hidden_dim=tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
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
        replica_probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
    ) -> None:
        coll = pyine.guardrails.probes.collection.ProbeCollection(
            replica_probe_configs, hidden_dim=tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
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
        replica_probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
    ) -> None:
        coll1 = pyine.guardrails.probes.collection.ProbeCollection(
            replica_probe_configs, hidden_dim=tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
        )
        coll2 = pyine.guardrails.probes.collection.ProbeCollection(
            replica_probe_configs, hidden_dim=tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
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
        replica_probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
    ) -> None:
        """Global RNG state is restored after ProbeCollection construction with replicas."""
        torch.manual_seed(12345)
        before_val = torch.randn(1).item()

        # Re-seed and construct collection (should save/restore RNG state)
        torch.manual_seed(12345)
        pyine.guardrails.probes.collection.ProbeCollection(
            replica_probe_configs, hidden_dim=tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
        )
        after_val = torch.randn(1).item()

        assert before_val == after_val, "RNG state was not restored after replica construction"

    def test_non_seeded_probes_unaffected(
        self,
        sample_probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
    ) -> None:
        # Should construct normally without any issues
        coll = pyine.guardrails.probes.collection.ProbeCollection(
            sample_probe_configs, hidden_dim=tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
        )
        assert len(coll.probes) == len(sample_probe_configs)


# ---------------------------------------------------------------------------
# Section 6.3: ProbeCollection.load_from_checkpoint tests (tests 18-20)
# ---------------------------------------------------------------------------


def _save_single_probe_to_disk(
    collection: pyine.guardrails.probes.collection.ProbeCollection,
    probe_dir: pathlib.Path,
    probe_name: str,
    checkpoint_name: str = "final",
) -> None:
    """Helper: save a single probe from *collection* into the flat checkpoint layout.

    Creates ``probe_dir/<checkpoint_name>/probe_config.json`` and
    ``probe_dir/<checkpoint_name>/probe_state_dict.pt``.
    """
    import typing as _typing

    import pyine.guardrails.probes.base as probe_base

    probe = _typing.cast("probe_base.BaseProbe", collection.probes[probe_name])
    ckpt_dir = probe_dir / probe_name / checkpoint_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), ckpt_dir / "probe_state_dict.pt")
    (ckpt_dir / "probe_config.json").write_text(probe.config.model_dump_json(indent=2))


class TestLoadFromCheckpoint:
    """Tests for ProbeCollection.load_from_checkpoint."""

    def test_load_auto_detects_final(
        self,
        sample_probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
        tmp_path: pathlib.Path,
    ) -> None:
        """Default (checkpoint_name=None) auto-detects 'final'."""
        hidden_dim = tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
        probe_name = sample_probe_configs[0].name
        original = pyine.guardrails.probes.collection.ProbeCollection(sample_probe_configs, hidden_dim=hidden_dim)

        probe_dir = tmp_path / "probe"
        _save_single_probe_to_disk(original, probe_dir, probe_name, checkpoint_name="final")

        loaded = pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(
            probe_dir, hidden_dim=hidden_dim
        )

        assert set(loaded.probes.keys()) == {probe_name}
        for (pn1, p1), (pn2, p2) in zip(
            original.probes[probe_name].named_parameters(),
            loaded.probes[probe_name].named_parameters(),
            strict=True,
        ):
            assert pn1 == pn2
            torch.testing.assert_close(p1, p2)

    def test_load_auto_detects_highest_step(
        self,
        sample_probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
        tmp_path: pathlib.Path,
    ) -> None:
        """When no 'final/' exists, auto-detect picks the highest step."""
        hidden_dim = tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
        probe_name = sample_probe_configs[0].name
        original = pyine.guardrails.probes.collection.ProbeCollection(sample_probe_configs, hidden_dim=hidden_dim)

        probe_dir = tmp_path / "probe"
        _save_single_probe_to_disk(original, probe_dir, probe_name, checkpoint_name="step-0050")
        _save_single_probe_to_disk(original, probe_dir, probe_name, checkpoint_name="step-0200")

        loaded = pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(
            probe_dir, hidden_dim=hidden_dim
        )

        # Should have loaded - auto-detected step-0200
        assert set(loaded.probes.keys()) == {probe_name}

    def test_load_final_preferred_over_steps(
        self,
        sample_probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
        tmp_path: pathlib.Path,
    ) -> None:
        """When both 'final/' and 'step-*/' exist, 'final' is preferred."""
        hidden_dim = tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
        probe_name = sample_probe_configs[0].name
        original = pyine.guardrails.probes.collection.ProbeCollection(sample_probe_configs, hidden_dim=hidden_dim)

        probe_dir = tmp_path / "probe"
        _save_single_probe_to_disk(original, probe_dir, probe_name, checkpoint_name="final")
        _save_single_probe_to_disk(original, probe_dir, probe_name, checkpoint_name="step-9999")

        loaded = pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(
            probe_dir, hidden_dim=hidden_dim
        )

        assert set(loaded.probes.keys()) == {probe_name}

    def test_load_explicit_step(
        self,
        sample_probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
        tmp_path: pathlib.Path,
    ) -> None:
        """Explicit checkpoint_name bypasses auto-detection."""
        hidden_dim = tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
        probe_name = sample_probe_configs[0].name
        original = pyine.guardrails.probes.collection.ProbeCollection(sample_probe_configs, hidden_dim=hidden_dim)

        probe_dir = tmp_path / "probe"
        _save_single_probe_to_disk(original, probe_dir, probe_name, checkpoint_name="step-0050")

        loaded = pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(
            probe_dir, hidden_dim=hidden_dim, checkpoint_name="step-0050"
        )

        assert set(loaded.probes.keys()) == {probe_name}

    def test_load_missing_checkpoint_name(
        self,
        sample_probe_configs: list[pyine.guardrails.probes.base.ProbeConfig],
        tmp_path: pathlib.Path,
    ) -> None:
        """FileNotFoundError when explicit checkpoint_name doesn't exist."""
        hidden_dim = tests.guardrails.probes.conftest.PROBE_HIDDEN_DIM
        probe_name = sample_probe_configs[0].name
        original = pyine.guardrails.probes.collection.ProbeCollection(sample_probe_configs, hidden_dim=hidden_dim)

        probe_dir = tmp_path / "probe"
        _save_single_probe_to_disk(original, probe_dir, probe_name, checkpoint_name="final")

        with pytest.raises(FileNotFoundError):
            pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(
                probe_dir, hidden_dim=hidden_dim, checkpoint_name="step-9999"
            )

    def test_load_empty_dir_raises(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """FileNotFoundError when auto-detect finds no valid checkpoints."""
        probe_dir = tmp_path / "empty_probe"
        probe_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(probe_dir, hidden_dim=64)
