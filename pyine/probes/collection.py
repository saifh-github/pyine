"""ProbeCollection — DDP-friendly wrapper for multiple probes."""

from __future__ import annotations

import typing

import torch

if typing.TYPE_CHECKING:
    from pyine.probes.base import BaseProbe, ProbeConfig

# Avoid circular import at module level; build_probe is imported lazily.


class ProbeCollection(torch.nn.Module):
    """Holds N probes as submodules via ``nn.ModuleDict``.

    This is the unit wrapped by DDP so that all probe parameters participate
    in gradient synchronisation with a single all-reduce call.
    """

    def __init__(self, probe_configs: list[ProbeConfig], hidden_dim: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # nn.Module stub
        from pyine.probes import build_probe

        self._probe_configs: dict[str, ProbeConfig] = {}
        probes: dict[str, torch.nn.Module] = {}
        for pc in probe_configs:
            pc = pc.model_copy(update={"hidden_dim": hidden_dim})
            probes[pc.name] = build_probe(pc)
            self._probe_configs[pc.name] = pc
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
            probe = typing.cast("BaseProbe", module)
            h = activations[probe.config.layer]
            results[name] = probe(h, attention_mask)
        return results

    def get_parameter_groups(self) -> list[dict[str, list[torch.nn.Parameter] | float]]:
        """Per-probe parameter groups with individual learning rates."""
        groups: list[dict[str, list[torch.nn.Parameter] | float]] = []
        for name, probe in self.probes.items():
            pc = self._probe_configs[name]
            params = list(probe.parameters())
            assert len(params) > 0, f"Probe '{name}' has no parameters"
            groups.append(
                {
                    "params": params,
                    "lr": pc.learning_rate,
                    "weight_decay": pc.weight_decay,
                }
            )
        return groups
