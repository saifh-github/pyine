"""Base probe class and configuration."""

from __future__ import annotations

import abc

import pydantic
import torch


class ProbeConfig(pydantic.BaseModel):
    """Configuration for a single probe instance."""

    model_config = pydantic.ConfigDict(frozen=False)

    name: str
    architecture: str
    layer: int
    hidden_dim: int | None = None
    # Architecture-specific hyperparams
    window_size: int = 16
    temperature: float = 1.0
    attn_dim: int = 64
    # Training hyperparams (per-probe)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


class BaseProbe(torch.nn.Module, abc.ABC):
    """Abstract base class for all probes.

    All probes receive hidden states ``(batch, seq_len, hidden_dim)``
    and an attention mask ``(batch, seq_len)`` and produce a scalar
    logit per sample ``(batch, 1)``.
    """

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__()
        if config.hidden_dim is None:
            raise ValueError("hidden_dim must be set before constructing a probe")
        self.config = config

    @abc.abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: ``(batch, seq_len, hidden_dim)`` – detached
                activations from the frozen LLM.
            attention_mask: ``(batch, seq_len)`` – 1 for real tokens,
                0 for padding.

        Returns:
            Logits of shape ``(batch, 1)``.
        """
        ...
