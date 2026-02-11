"""Last-token probe."""

from __future__ import annotations

import torch

from pyine.probes.base import BaseProbe, ProbeConfig


class LastTokenProbe(BaseProbe):
    """Extracts the hidden state at the last non-padding position and applies a linear head."""

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__(config)
        self.head = torch.nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # last non-padding position per sample
        # attention_mask.sum(dim=1) gives length; subtract 1 for 0-based index
        last_positions = attention_mask.sum(dim=1).long() - 1  # (batch,)
        last_positions = last_positions.clamp(min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_idx, last_positions, :]  # (batch, hidden_dim)
        return self.head(pooled)  # (batch, 1)
