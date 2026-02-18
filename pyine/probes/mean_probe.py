"""Mean-pooling probe."""

from __future__ import annotations

import torch

from pyine.probes.base import BaseProbe, ProbeConfig  # noqa: TID252 -- avoids circular import via __init__


class MeanProbe(BaseProbe):
    """Computes the mean activation over non-masked positions, then applies a linear head."""

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__(config)
        self.head = torch.nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # mask: (batch, seq_len, 1)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        # masked sum / count
        summed = (hidden_states * mask).sum(dim=1)  # (batch, hidden_dim)
        lengths = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
        pooled = summed / lengths  # (batch, hidden_dim)
        return self.head(pooled)  # (batch, 1)
