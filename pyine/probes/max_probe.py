"""Max-pooling probe."""

from __future__ import annotations

import torch

from pyine.probes.base import BaseProbe, ProbeConfig


class MaxProbe(BaseProbe):
    """Takes the maximum per-token score over non-masked positions."""

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__(config)
        self.head = torch.nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Per-token scores: (batch, seq_len, 1)
        scores = self.head(hidden_states)
        # Set masked positions to -inf so they don't affect max
        mask = attention_mask.unsqueeze(-1).bool()  # (batch, seq_len, 1)
        scores = scores.masked_fill(~mask, float("-inf"))
        # Max over sequence: (batch, 1)
        pooled, _ = scores.max(dim=1)
        return pooled
