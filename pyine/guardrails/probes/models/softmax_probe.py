"""Softmax-weighted probe."""

from __future__ import annotations

import torch

from pyine.guardrails.probes.base import BaseProbe, ProbeConfig  # noqa: TID252 -- avoids circular import via __init__


class SoftmaxProbe(BaseProbe):
    """Temperature-scaled softmax over per-token scores, then weighted sum."""

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__(config)
        self.head = torch.nn.Linear(self.hidden_dim, 1)
        self.temperature = config.temperature

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # per-token scores: (batch, seq_len, 1)
        scores = self.head(hidden_states)

        # masked softmax with temperature
        mask = attention_mask.unsqueeze(-1).bool()  # (batch, seq_len, 1)
        scaled = scores / self.temperature
        scaled = scaled.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scaled, dim=1)  # (batch, seq_len, 1)

        # weighted sum
        return (weights * scores).sum(dim=1)  # (batch, 1)
