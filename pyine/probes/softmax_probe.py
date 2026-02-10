"""Softmax-weighted probe."""

from __future__ import annotations

import torch

from pyine.probes.base import BaseProbe, ProbeConfig


class SoftmaxProbe(BaseProbe):
    """Temperature-scaled softmax over per-token scores, then weighted sum."""

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__(config)
        self.head = torch.nn.Linear(config.hidden_dim, 1)
        self.temperature = config.temperature

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Per-token scores: (batch, seq_len, 1)
        scores = self.head(hidden_states)

        # Masked softmax with temperature
        mask = attention_mask.unsqueeze(-1).bool()  # (batch, seq_len, 1)
        scaled = scores / self.temperature
        scaled = scaled.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scaled, dim=1)  # (batch, seq_len, 1)

        # Weighted sum
        pooled = (weights * scores).sum(dim=1)  # (batch, 1)
        return pooled
