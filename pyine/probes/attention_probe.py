"""Attention-based probe."""

from __future__ import annotations

import math

import torch

from pyine.probes.base import BaseProbe, ProbeConfig


class AttentionProbe(BaseProbe):
    """Learned query/value projections with attention-weighted pooling.

    Parameters:
        W_q: ``(hidden_dim, attn_dim)`` — key projection
        Q_global: ``(attn_dim,)`` — learned query vector
        W_v: ``(hidden_dim, 1)`` — value projection
    """

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__(config)
        attn_dim = config.attn_dim
        hidden_dim = config.hidden_dim

        self.W_q = torch.nn.Linear(hidden_dim, attn_dim, bias=False)
        self.Q_global = torch.nn.Parameter(torch.randn(attn_dim))
        self.W_v = torch.nn.Linear(hidden_dim, 1, bias=False)
        self._scale = math.sqrt(attn_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Key projection: (batch, seq_len, attn_dim)
        K = self.W_q(hidden_states)

        # Attention scores: (batch, seq_len, 1)
        attn_scores = (K @ self.Q_global.unsqueeze(-1)) / self._scale

        # Masked softmax
        mask = attention_mask.unsqueeze(-1).bool()  # (batch, seq_len, 1)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

        # Value projection: (batch, seq_len, 1)
        V = self.W_v(hidden_states)

        # Weighted sum: (batch, 1)
        pooled = (attn_weights * V).sum(dim=1)
        return pooled
