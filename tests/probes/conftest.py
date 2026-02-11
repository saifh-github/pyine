"""Shared fixtures for probe tests."""

from __future__ import annotations

import pytest
import torch

from pyine.probes import ProbeConfig

PROBE_HIDDEN_DIM = 64
PROBE_SEQ_LEN = 32
PROBE_BATCH_SIZE = 4


@pytest.fixture
def probe_hidden_dim() -> int:
    return PROBE_HIDDEN_DIM


@pytest.fixture
def random_activations() -> dict[int, torch.Tensor]:
    """Random activations for layers 0, 4, 8 with small dimensions."""
    return {layer: torch.randn(PROBE_BATCH_SIZE, PROBE_SEQ_LEN, PROBE_HIDDEN_DIM) for layer in [0, 4, 8]}


@pytest.fixture
def random_attention_mask() -> torch.Tensor:
    """Attention mask with some padding (last few positions masked out)."""
    mask = torch.ones(PROBE_BATCH_SIZE, PROBE_SEQ_LEN, dtype=torch.long)
    # Mask out last 4 positions for first 2 samples
    mask[:2, -4:] = 0
    return mask


@pytest.fixture
def sample_probe_configs() -> list[ProbeConfig]:
    """Minimal probe configs covering all 6 architectures."""
    return [
        ProbeConfig(name="mean_L0", architecture="mean", layer=0),
        ProbeConfig(name="max_L0", architecture="max", layer=0),
        ProbeConfig(name="last_L4", architecture="last_token", layer=4),
        ProbeConfig(name="rolling_L4", architecture="rolling_mean", layer=4, window_size=4),
        ProbeConfig(name="softmax_L8", architecture="softmax", layer=8, temperature=0.5),
        ProbeConfig(name="attn_L8", architecture="attention", layer=8, attn_dim=16),
    ]
