"""Probe module library -- lightweight classifiers on frozen LLM activations."""

from __future__ import annotations

import typing

from pyine.probes.attention_probe import AttentionProbe
from pyine.probes.last_token_probe import LastTokenProbe
from pyine.probes.lmdb_dataset import load_probe_dataset_from_lmdb as load_probe_dataset_from_lmdb
from pyine.probes.max_probe import MaxProbe
from pyine.probes.mean_probe import MeanProbe
from pyine.probes.rolling_mean_probe import RollingMeanProbe
from pyine.probes.softmax_probe import SoftmaxProbe

if typing.TYPE_CHECKING:
    import pyine.probes.base

PROBE_REGISTRY: dict[str, type[pyine.probes.base.BaseProbe]] = {
    "mean": MeanProbe,
    "max": MaxProbe,
    "last_token": LastTokenProbe,
    "rolling_mean": RollingMeanProbe,
    "softmax": SoftmaxProbe,
    "attention": AttentionProbe,
}


def build_probe(config: pyine.probes.base.ProbeConfig) -> pyine.probes.base.BaseProbe:
    """Instantiate a probe from its config."""
    if config.hidden_dim is None:
        raise ValueError(f"hidden_dim must be set before building probe '{config.name}'")
    if config.architecture not in PROBE_REGISTRY:
        raise KeyError(
            f"Unknown probe architecture '{config.architecture}'. Available: {sorted(PROBE_REGISTRY.keys())}"
        )
    cls = PROBE_REGISTRY[config.architecture]
    return cls(config)
