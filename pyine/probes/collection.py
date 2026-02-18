"""ProbeCollection -- DDP-friendly wrapper for multiple probes."""

from __future__ import annotations

import typing

import torch

if typing.TYPE_CHECKING:
    import pyine.probes.base

# avoid circular import at module level; build_probe is imported lazily.


class ProbeCollection(torch.nn.Module):
    """Holds N probes as submodules via ``nn.ModuleDict``.

    This is the unit wrapped by DDP so that all probe parameters participate
    in gradient synchronisation with a single all-reduce call.
    """

    def __init__(
        self,
        probe_configs: list[pyine.probes.base.ProbeConfig],
        hidden_dim: int,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # nn.Module stub
        import pyine.probes

        self._probe_configs: dict[str, pyine.probes.base.ProbeConfig] = {}
        probes: dict[str, torch.nn.Module] = {}

        # save RNG state so replica seeding doesn't affect downstream randomness
        any_seeded = any(probe_config.replica_seed is not None for probe_config in probe_configs)
        saved_rng_state = torch.random.get_rng_state() if any_seeded else None

        for probe_config in probe_configs:
            probe_config = probe_config.model_copy(update={"hidden_dim": hidden_dim})
            if probe_config.replica_seed is not None:
                torch.manual_seed(probe_config.replica_seed)  # pyright: ignore[reportUnknownMemberType]  # torch stubs
            probes[probe_config.name] = pyine.probes.build_probe(probe_config)
            self._probe_configs[probe_config.name] = probe_config

        # restore RNG state after seeded construction
        if saved_rng_state is not None:
            torch.random.set_rng_state(saved_rng_state)

        self.probes = torch.nn.ModuleDict(probes)

    def forward(
        self,
        activations: dict[int, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run all probes on their respective layer activations.

        Args:
            activations: ``{layer_idx: (batch, seq_len, hidden_dim)}``
            attention_mask: ``(batch, seq_len)``

        Returns:
            ``{probe_name: (batch, 1) logits}``
        """
        results: dict[str, torch.Tensor] = {}
        for name, module in self.probes.items():
            probe = typing.cast("pyine.probes.base.BaseProbe", module)
            hidden_states = activations[probe.config.layer]
            results[name] = probe(hidden_states, attention_mask)
        return results

    def get_parameter_groups(self) -> list[dict[str, list[torch.nn.Parameter] | float]]:
        """Per-probe parameter groups with individual learning rates."""
        groups: list[dict[str, list[torch.nn.Parameter] | float]] = []
        for name, probe in self.probes.items():
            probe_config = self._probe_configs[name]
            params = list(probe.parameters())
            assert len(params) > 0, f"Probe '{name}' has no parameters"
            groups.append(
                {
                    "params": params,
                    "lr": probe_config.learning_rate,
                    "weight_decay": probe_config.weight_decay,
                }
            )
        return groups
