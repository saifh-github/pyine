"""Max-of-rolling-means probe."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from pyine.probes.base import BaseProbe, ProbeConfig  # noqa: TID252 -- avoids circular import via __init__


class RollingMeanProbe(BaseProbe):
    """Applies a 1D rolling-mean window to per-token scores, then takes the max."""

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__(config)
        self.head = torch.nn.Linear(self.hidden_dim, 1)
        self.window_size = config.window_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # per-token scores: (batch, seq_len, 1)
        scores = self.head(hidden_states)

        # zero out masked positions before pooling
        mask_3d = attention_mask.unsqueeze(-1).to(scores.dtype)  # (batch, seq_len, 1)
        scores = scores * mask_3d

        # reshape for avg_pool1d: (batch, 1, seq_len)
        scores_1d = scores.permute(0, 2, 1)

        # also pool the mask to know valid counts per window
        mask_1d = attention_mask.unsqueeze(1).to(scores.dtype)  # (batch, 1, seq_len)

        # pad so we don't lose positions (same-size output)
        pad = self.window_size - 1
        scores_padded = functional.pad(scores_1d, (pad, 0), value=0.0)
        mask_padded = functional.pad(mask_1d, (pad, 0), value=0.0)

        # sum-pool
        score_sums = functional.avg_pool1d(scores_padded, self.window_size, stride=1) * self.window_size
        mask_sums = functional.avg_pool1d(mask_padded, self.window_size, stride=1) * self.window_size

        # rolling mean (avoid division by zero)
        rolling = score_sums / mask_sums.clamp(min=1)  # (batch, 1, seq_len)

        # mask out fully-padded windows
        valid_mask = mask_sums > 0  # (batch, 1, seq_len)
        rolling = rolling.masked_fill(~valid_mask, float("-inf"))

        # max over sequence: (batch, 1)
        pooled, _ = rolling.max(dim=2)
        return pooled
