"""Shared fixtures for probe tests."""

from __future__ import annotations

import typing

import pytest
import torch

import pyine.apps.trainers.probe_trainer
import pyine.probes.base
import pyine.probes.collection
import pyine.probes.data.debug_dataset

if typing.TYPE_CHECKING:
    import pathlib

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
def sample_probe_configs() -> list[pyine.probes.base.ProbeConfig]:
    """Minimal probe configs covering all 6 architectures."""
    return [
        pyine.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0),
        pyine.probes.base.ProbeConfig(name="max_L0", architecture="max", layer=0),
        pyine.probes.base.ProbeConfig(name="last_L4", architecture="last_token", layer=4),
        pyine.probes.base.ProbeConfig(name="rolling_L4", architecture="rolling_mean", layer=4, window_size=4),
        pyine.probes.base.ProbeConfig(name="softmax_L8", architecture="softmax", layer=8, temperature=0.5),
        pyine.probes.base.ProbeConfig(name="attn_L8", architecture="attention", layer=8, attn_dim=16),
    ]


@pytest.fixture
def replica_probe_configs() -> list[pyine.probes.base.ProbeConfig]:
    """Two base configs expanded to 3 replicas each (6 total), with replica metadata."""
    base_configs = [
        pyine.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0),
        pyine.probes.base.ProbeConfig(name="attn_L8", architecture="attention", layer=8, attn_dim=16),
    ]
    return pyine.apps.trainers.probe_trainer.expand_probe_configs_with_replicas(
        base_configs, num_replicas=3, replica_base_seed=42
    )


@pytest.fixture
def replica_probe_collection(
    replica_probe_configs: list[pyine.probes.base.ProbeConfig],
) -> pyine.probes.collection.ProbeCollection:
    return pyine.probes.collection.ProbeCollection(replica_probe_configs, hidden_dim=PROBE_HIDDEN_DIM)


@pytest.fixture
def debug_lmdb_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a debug LMDB at tmp_path and return its path."""
    lmdb_path = tmp_path / "debug.lmdb"
    pyine.probes.data.debug_dataset.create_debug_probe_lmdb(lmdb_path, n_train=50, n_valid=20, seed=42)
    return lmdb_path
