"""Tests for ProbeMetricsConnector."""

from __future__ import annotations

import typing
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

if typing.TYPE_CHECKING:
    import pathlib

from pyine.guardrails.probes.metrics_connector import (
    ProbeMetricsConnector,
    _generate_synthetic_activations,
    _generate_synthetic_batch,
)

# ---------------------------------------------------------------------------
# Synthetic data helper tests
# ---------------------------------------------------------------------------


class TestGenerateSyntheticBatch:
    def test_shapes(self) -> None:
        batch = _generate_synthetic_batch(
            batch_size=4,
            seq_length=128,
            vocab_size=32000,
            device=torch.device("cpu"),
        )
        assert batch["input_ids"].shape == (4, 128)
        assert batch["attention_mask"].shape == (4, 128)
        assert batch["input_ids"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long

    def test_vocab_range(self) -> None:
        batch = _generate_synthetic_batch(
            batch_size=2,
            seq_length=64,
            vocab_size=100,
            device=torch.device("cpu"),
        )
        assert batch["input_ids"].min() >= 0
        assert batch["input_ids"].max() < 100

    def test_attention_all_ones(self) -> None:
        batch = _generate_synthetic_batch(
            batch_size=2,
            seq_length=64,
            vocab_size=100,
            device=torch.device("cpu"),
        )
        assert (batch["attention_mask"] == 1).all()


class TestGenerateSyntheticActivations:
    def test_shapes(self) -> None:
        layers = [4, 8, 16]
        activations, mask = _generate_synthetic_activations(
            batch_size=2,
            seq_length=64,
            hidden_dim=128,
            target_layers=layers,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert set(activations.keys()) == set(layers)
        for layer in layers:
            assert activations[layer].shape == (2, 64, 128)
            assert activations[layer].dtype == torch.float32
        assert mask.shape == (2, 64)

    def test_dtype(self) -> None:
        activations, _ = _generate_synthetic_activations(
            batch_size=1,
            seq_length=32,
            hidden_dim=64,
            target_layers=[0],
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        assert activations[0].dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# flatten_for_csv tests
# ---------------------------------------------------------------------------


class TestProbeConnectorFlattenForCSV:
    @staticmethod
    def _make_connector() -> ProbeMetricsConnector:
        """Create connector with a minimal mock config (load() not called)."""
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        return ProbeMetricsConnector(mock_config)

    def test_oom_entry(self) -> None:
        connector = self._make_connector()
        static_info: dict[str, typing.Any] = {"probes": {}}
        benchmarks = [{"batch_size": 32, "seq_length": 4096, "status": "OOM"}]

        rows = connector.flatten_for_csv(static_info, benchmarks)
        assert len(rows) == 1
        assert rows[0]["status"] == "OOM"
        assert rows[0]["classifier_type"] == "probe"

    def test_per_probe_and_e2e_rows(self) -> None:
        connector = self._make_connector()
        static_info: dict[str, typing.Any] = {
            "probes": {
                "mean_L4": {"architecture": "mean", "layer": 4, "param_count": 512},
                "max_L8": {"architecture": "max", "layer": 8, "param_count": 256},
            },
        }
        benchmarks = [
            {
                "batch_size": 1,
                "seq_length": 512,
                "status": "ok",
                "probe_flops": {"mean_L4": 1024, "max_L8": 768},
                "llm_forward_with_hooks_flops": 5000,
                "end_to_end_latency": {"mean_ms": 10.0, "p99_ms": 12.0},
                "probe_only_latency": {"mean_ms": 0.5, "p99_ms": 0.7},
                "probe_memory": {"peak_activation_memory_bytes": 2048},
                "throughput_samples_per_sec": {"end_to_end": 100.0, "probe_only": 2000.0},
            },
        ]

        rows = connector.flatten_for_csv(static_info, benchmarks)
        # 2 per-probe rows + 1 end-to-end row
        assert len(rows) == 3

        # Per-probe rows
        probe_rows = [r for r in rows if r["probe_name"] != "-"]
        assert len(probe_rows) == 2
        names = {r["probe_name"] for r in probe_rows}
        assert names == {"mean_L4", "max_L8"}

        mean_row = next(r for r in probe_rows if r["probe_name"] == "mean_L4")
        assert mean_row["architecture"] == "mean"
        assert mean_row["layer"] == 4
        assert mean_row["flops"] == 1024
        assert mean_row["param_count"] == 512

        # End-to-end row
        e2e_rows = [r for r in rows if r["model_name"] == "end_to_end"]
        assert len(e2e_rows) == 1
        assert e2e_rows[0]["flops"] == 5000
        assert e2e_rows[0]["mean_latency_ms"] == 10.0

    def test_empty_benchmarks(self) -> None:
        connector = self._make_connector()
        rows = connector.flatten_for_csv({"probes": {}}, [])
        assert rows == []


# ---------------------------------------------------------------------------
# WandB helper tests
# ---------------------------------------------------------------------------


class TestProbeConnectorWandBSummary:
    @staticmethod
    def _make_connector() -> ProbeMetricsConnector:
        from unittest.mock import MagicMock

        return ProbeMetricsConnector(MagicMock())

    def test_summary_keys(self) -> None:
        connector = self._make_connector()
        static_info = {
            "frozen_llm": {
                "model_name": "Qwen/Qwen2.5-3B",
                "param_count": 3_000_000_000,
                "param_memory_bytes": 6_000_000_000,
            },
            "probes": {
                "mean_L16": {"architecture": "mean", "layer": 16, "param_count": 1024},
            },
        }

        summary = connector.get_wandb_summary(static_info)
        assert summary["probe_llm_model"] == "Qwen/Qwen2.5-3B"
        assert summary["probe_llm_param_count"] == 3_000_000_000
        assert summary["probe/mean_L16/architecture"] == "mean"
        assert summary["probe/mean_L16/layer"] == 16


class TestProbeConnectorWandBLogEntries:
    @staticmethod
    def _make_connector() -> ProbeMetricsConnector:
        from unittest.mock import MagicMock

        return ProbeMetricsConnector(MagicMock())

    def test_ok_entry(self) -> None:
        connector = self._make_connector()
        benchmarks = [
            {
                "batch_size": 1,
                "seq_length": 512,
                "status": "ok",
                "llm_forward_flops": 100,
                "probe_flops": {"p1": 50},
                "end_to_end_latency": {"mean_ms": 10.0},
            },
        ]

        entries = connector.get_wandb_log_entries(benchmarks)
        assert len(entries) == 1
        log = entries[0]
        assert log["probe/batch_size"] == 1
        assert log["probe/llm_forward_flops"] == 100
        assert log["probe/flops/p1"] == 50

    def test_oom_entry(self) -> None:
        connector = self._make_connector()
        benchmarks = [{"batch_size": 32, "seq_length": 4096, "status": "OOM"}]

        entries = connector.get_wandb_log_entries(benchmarks)
        assert len(entries) == 1
        assert entries[0]["probe/status"] == "OOM"

    def test_empty_benchmarks(self) -> None:
        connector = self._make_connector()
        assert connector.get_wandb_log_entries([]) == []


class TestProbeConnectorLoad:
    def test_load_passes_checkpoint_name(
        self,
        monkeypatch: typing.Any,
    ) -> None:
        config = SimpleNamespace(
            probe_checkpoint_dir="/tmp/probes",  # noqa: S108
            probe_checkpoint_name="best",
            probe_llm_model="Qwen/Qwen2.5-3B",
            probe_llm_checkpoint_path=None,
            probe_auto_model_config={"use_cache": False},
            use_torch_compile=False,
        )
        connector = ProbeMetricsConnector(config)

        class DummyLLM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(hidden_size=64, vocab_size=128)

        dummy_llm = DummyLLM()
        loaded_collection = MagicMock()
        loaded_collection.to.return_value = loaded_collection
        loaded_collection._probe_configs = {}
        captured: dict[str, typing.Any] = {}

        def _mock_load_from_checkpoint(
            checkpoint_dir: pathlib.Path,
            hidden_dim: int,
            *,
            checkpoint_name: str | None = None,
        ) -> MagicMock:
            captured["checkpoint_dir"] = checkpoint_dir
            captured["hidden_dim"] = hidden_dim
            captured["checkpoint_name"] = checkpoint_name
            return loaded_collection

        monkeypatch.setattr(
            "transformers.AutoModelForCausalLM.from_pretrained",
            lambda *args, **kwargs: dummy_llm,
        )
        monkeypatch.setattr(
            "pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint",
            _mock_load_from_checkpoint,
        )

        connector.load(torch.device("cpu"))

        assert captured["checkpoint_dir"].name == "probes"
        assert captured["hidden_dim"] == 64
        assert captured["checkpoint_name"] == "best"
