"""Tests for ProbeCollection."""

from __future__ import annotations

import pytest
import torch

from pyine.probes import ProbeConfig
from pyine.probes.collection import ProbeCollection
from tests.probes.conftest import PROBE_BATCH_SIZE, PROBE_HIDDEN_DIM


class TestProbeCollection:
    """Tests for the ProbeCollection wrapper module."""

    def test_forward_returns_all_probe_logits(
        self,
        sample_probe_configs: list[ProbeConfig],
        random_activations: dict[int, torch.Tensor],
        random_attention_mask: torch.Tensor,
    ) -> None:
        """Forward returns dict with one (batch, 1) tensor per probe."""
        coll = ProbeCollection(sample_probe_configs, hidden_dim=PROBE_HIDDEN_DIM)
        results = coll(random_activations, random_attention_mask)

        assert set(results.keys()) == {pc.name for pc in sample_probe_configs}
        for name, logits in results.items():
            assert logits.shape == (PROBE_BATCH_SIZE, 1), f"{name}: {logits.shape}"

    def test_parameter_groups_match_configs(
        self,
        sample_probe_configs: list[ProbeConfig],
    ) -> None:
        """get_parameter_groups() returns one group per probe with correct lr/wd."""
        coll = ProbeCollection(sample_probe_configs, hidden_dim=PROBE_HIDDEN_DIM)
        groups = coll.get_parameter_groups()
        assert len(groups) == len(sample_probe_configs)

        for group, pc in zip(groups, sample_probe_configs):
            assert group["lr"] == pc.learning_rate
            assert group["weight_decay"] == pc.weight_decay

    def test_parameter_groups_nonempty(
        self,
        sample_probe_configs: list[ProbeConfig],
    ) -> None:
        """Each parameter group has at least one parameter."""
        coll = ProbeCollection(sample_probe_configs, hidden_dim=PROBE_HIDDEN_DIM)
        groups = coll.get_parameter_groups()
        for i, group in enumerate(groups):
            assert len(group["params"]) > 0, f"Group {i} has no parameters"

    def test_hidden_dim_auto_populated(
        self,
        sample_probe_configs: list[ProbeConfig],
    ) -> None:
        """Probes receive hidden_dim from ProbeCollection constructor, not from config."""
        # Configs start with hidden_dim=None
        for pc in sample_probe_configs:
            assert pc.hidden_dim is None

        coll = ProbeCollection(sample_probe_configs, hidden_dim=PROBE_HIDDEN_DIM)

        # All probes in the collection should have hidden_dim set
        for name, probe in coll.probes.items():
            assert probe.config.hidden_dim == PROBE_HIDDEN_DIM, (
                f"Probe {name} has hidden_dim={probe.config.hidden_dim}"
            )

    def test_summed_loss_gradient_independence(
        self,
        sample_probe_configs: list[ProbeConfig],
        random_activations: dict[int, torch.Tensor],
        random_attention_mask: torch.Tensor,
    ) -> None:
        """Gradient of sum(losses) w.r.t. probe_i params equals gradient of loss_i alone."""
        coll = ProbeCollection(sample_probe_configs, hidden_dim=PROBE_HIDDEN_DIM)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        labels = torch.randint(0, 2, (PROBE_BATCH_SIZE,), dtype=torch.float)

        # 1) Compute gradient of sum(losses) w.r.t. first probe's params
        logits = coll(random_activations, random_attention_mask)
        total_loss = sum(
            loss_fn(l.squeeze(-1), labels) for l in logits.values()
        )
        total_loss.backward()

        first_probe_name = sample_probe_configs[0].name
        grads_from_sum = {
            n: p.grad.clone()
            for n, p in coll.probes[first_probe_name].named_parameters()
        }
        coll.zero_grad()

        # 2) Compute gradient of only the first probe's loss
        logits2 = coll(random_activations, random_attention_mask)
        single_loss = loss_fn(logits2[first_probe_name].squeeze(-1), labels)
        single_loss.backward()

        grads_from_single = {
            n: p.grad.clone()
            for n, p in coll.probes[first_probe_name].named_parameters()
        }

        # They should be identical
        for n in grads_from_sum:
            torch.testing.assert_close(
                grads_from_sum[n], grads_from_single[n],
                msg=f"Gradient mismatch for {first_probe_name}.{n}",
            )

    def test_all_submodules_visible_to_ddp(
        self,
        sample_probe_configs: list[ProbeConfig],
    ) -> None:
        """All probe parameters are in collection.parameters() (needed for DDP)."""
        coll = ProbeCollection(sample_probe_configs, hidden_dim=PROBE_HIDDEN_DIM)

        # Collect all params from individual probes
        probe_param_ids = set()
        for probe in coll.probes.values():
            for p in probe.parameters():
                probe_param_ids.add(id(p))

        # All should be in collection.parameters()
        collection_param_ids = {id(p) for p in coll.parameters()}
        assert probe_param_ids == collection_param_ids
