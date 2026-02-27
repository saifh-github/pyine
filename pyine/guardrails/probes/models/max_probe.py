"""Max-pooling probe."""

from __future__ import annotations

import torch

from pyine.guardrails.probes.base import BaseProbe, ProbeConfig  # noqa: TID252 -- avoids circular import via __init__


class MaxProbe(BaseProbe):
    """Takes the maximum per-token score over non-masked positions."""

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__(config)
        self.head = torch.nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # per-token scores: (batch, seq_len, 1)
        scores = self.head(hidden_states)
        # set masked positions to -inf so they don't affect max
        mask = attention_mask.unsqueeze(-1).bool()  # (batch, seq_len, 1)
        scores = scores.masked_fill(~mask, float("-inf"))
        # max over sequence: (batch, 1)
        pooled, _ = scores.max(dim=1)
        return pooled
