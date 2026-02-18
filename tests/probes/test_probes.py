"""Unit tests for all 6 probe architectures and the build_probe factory."""

from __future__ import annotations

import pytest
import torch

import pyine.probes
import pyine.probes.base
import tests.probes.conftest

ALL_ARCHITECTURES = [
    pytest.param("mean", {}, id="mean"),
    pytest.param("max", {}, id="max"),
    pytest.param("last_token", {}, id="last_token"),
    pytest.param("rolling_mean", {"window_size": 4}, id="rolling_mean"),
    pytest.param("softmax", {"temperature": 0.5}, id="softmax"),
    pytest.param("attention", {"attn_dim": 16}, id="attention"),
]


def _make_probe(
    architecture: str, extra_kwargs: dict, hidden_dim: int = tests.probes.conftest.PROBE_HIDDEN_DIM
) -> pyine.probes.base.BaseProbe:
    config = pyine.probes.base.ProbeConfig(
        name=f"test_{architecture}",
        architecture=architecture,
        layer=0,
        hidden_dim=hidden_dim,
        **extra_kwargs,
    )
    return pyine.probes.build_probe(config)


@pytest.mark.parametrize(("architecture", "extra_kwargs"), ALL_ARCHITECTURES)
class TestProbeArchitectures:
    def test_output_shape(
        self,
        architecture: str,
        extra_kwargs: dict,
        random_activations: dict[int, torch.Tensor],
        random_attention_mask: torch.Tensor,
    ) -> None:
        probe = _make_probe(architecture, extra_kwargs)
        hidden_states = random_activations[0]
        out = probe(hidden_states, random_attention_mask)
        assert out.shape == (tests.probes.conftest.PROBE_BATCH_SIZE, 1)

    def test_gradient_flow(
        self,
        architecture: str,
        extra_kwargs: dict,
        random_activations: dict[int, torch.Tensor],
        random_attention_mask: torch.Tensor,
    ) -> None:
        probe = _make_probe(architecture, extra_kwargs)
        hidden_states = random_activations[0]
        out = probe(hidden_states, random_attention_mask)
        loss = out.sum()
        loss.backward()
        for name, param in probe.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_masked_positions_ignored(
        self,
        architecture: str,
        extra_kwargs: dict,
        probe_hidden_dim: int,
    ) -> None:
        """Changing masked positions does not affect output."""
        probe = _make_probe(architecture, extra_kwargs, hidden_dim=probe_hidden_dim)
        probe.eval()

        batch, seq_len = 2, 16
        hidden_states = torch.randn(batch, seq_len, probe_hidden_dim)
        mask = torch.ones(batch, seq_len, dtype=torch.long)
        mask[:, -4:] = 0  # last 4 positions are padding

        with torch.no_grad():
            out1 = probe(hidden_states, mask)

        # Modify masked positions
        hidden_states_modified = hidden_states.clone()
        hidden_states_modified[:, -4:, :] = torch.randn(batch, 4, probe_hidden_dim) * 100
        with torch.no_grad():
            out2 = probe(hidden_states_modified, mask)

        torch.testing.assert_close(out1, out2, rtol=1e-4, atol=1e-5)

    def test_parameter_count(
        self,
        architecture: str,
        extra_kwargs: dict,
        probe_hidden_dim: int,
    ) -> None:
        probe = _make_probe(architecture, extra_kwargs, hidden_dim=probe_hidden_dim)
        n_params = sum(param.numel() for param in probe.parameters())
        assert n_params > 0, "Probe should have at least one parameter"
        # All probes should be lightweight -- at most a few thousand params
        assert n_params < 100_000, f"Probe has {n_params} params -- too many for a lightweight probe"


class TestBuildProbe:
    def test_build_probe_all_architectures(self) -> None:
        for arch_name, expected_cls in pyine.probes.PROBE_REGISTRY.items():
            config = pyine.probes.base.ProbeConfig(
                name=f"test_{arch_name}", architecture=arch_name, layer=0, hidden_dim=32
            )
            probe = pyine.probes.build_probe(config)
            assert isinstance(probe, expected_cls)

    def test_build_probe_unknown_architecture_raises(self) -> None:
        config = pyine.probes.base.ProbeConfig(name="bad", architecture="nonexistent", layer=0, hidden_dim=32)
        with pytest.raises(KeyError, match="nonexistent"):
            pyine.probes.build_probe(config)

    def test_build_probe_missing_hidden_dim_raises(self) -> None:
        config = pyine.probes.base.ProbeConfig(name="bad", architecture="mean", layer=0)
        assert config.hidden_dim is None
        with pytest.raises(ValueError, match="hidden_dim"):
            pyine.probes.build_probe(config)
