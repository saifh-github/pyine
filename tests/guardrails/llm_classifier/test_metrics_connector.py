"""Tests for ClassifierMetricsConnector."""

from __future__ import annotations

import typing

from pyine.guardrails.llm_classifier.metrics_connector import ClassifierMetricsConnector

# ---------------------------------------------------------------------------
# flatten_for_csv tests
# ---------------------------------------------------------------------------


class TestClassifierConnectorFlattenForCSV:
    @staticmethod
    def _make_connector() -> ClassifierMetricsConnector:
        from unittest.mock import MagicMock

        return ClassifierMetricsConnector(MagicMock())

    def test_oom_entry(self) -> None:
        connector = self._make_connector()
        static_info = {"model_name": "test-clf", "param_count": 1000}
        benchmarks = [{"batch_size": 32, "seq_length": 4096, "status": "OOM"}]

        rows = connector.flatten_for_csv(static_info, benchmarks)
        assert len(rows) == 1
        assert rows[0]["status"] == "OOM"
        assert rows[0]["classifier_type"] == "classifier"
        assert rows[0]["model_name"] == "test-clf"

    def test_ok_entry(self) -> None:
        connector = self._make_connector()
        static_info: dict[str, typing.Any] = {"model_name": "test-clf", "param_count": 1000}
        benchmarks = [
            {
                "batch_size": 4,
                "seq_length": 256,
                "status": "ok",
                "flops": 2000,
                "latency": {"mean_ms": 5.0, "p99_ms": 7.0},
                "throughput_samples_per_sec": 800.0,
                "memory": {"peak_activation_memory_bytes": 4096},
            },
        ]

        rows = connector.flatten_for_csv(static_info, benchmarks)
        assert len(rows) == 1
        row = rows[0]
        assert row["classifier_type"] == "classifier"
        assert row["model_name"] == "test-clf"
        assert row["batch_size"] == 4
        assert row["seq_length"] == 256
        assert row["status"] == "ok"
        assert row["param_count"] == 1000
        assert row["flops"] == 2000
        assert row["mean_latency_ms"] == 5.0
        assert row["p99_latency_ms"] == 7.0
        assert row["throughput_samples_per_sec"] == 800.0
        assert row["peak_memory_bytes"] == 4096

    def test_empty_benchmarks(self) -> None:
        connector = self._make_connector()
        rows = connector.flatten_for_csv({"model_name": "x"}, [])
        assert rows == []

    def test_multiple_configs(self) -> None:
        connector = self._make_connector()
        static_info: dict[str, typing.Any] = {"model_name": "clf", "param_count": 500}
        benchmarks = [
            {"batch_size": 1, "seq_length": 128, "status": "ok"},
            {"batch_size": 4, "seq_length": 256, "status": "ok"},
            {"batch_size": 8, "seq_length": 512, "status": "OOM"},
        ]

        rows = connector.flatten_for_csv(static_info, benchmarks)
        assert len(rows) == 3
        assert rows[2]["status"] == "OOM"


# ---------------------------------------------------------------------------
# WandB helper tests
# ---------------------------------------------------------------------------


class TestClassifierConnectorWandBSummary:
    @staticmethod
    def _make_connector() -> ClassifierMetricsConnector:
        from unittest.mock import MagicMock

        return ClassifierMetricsConnector(MagicMock())

    def test_summary_keys(self) -> None:
        connector = self._make_connector()
        static_info = {
            "model_name": "modernbert",
            "param_count": 150_000_000,
            "param_memory_bytes": 300_000_000,
        }

        summary = connector.get_wandb_summary(static_info)
        assert summary["classifier_model_name"] == "modernbert"
        assert summary["classifier_param_count"] == 150_000_000
        assert summary["classifier_param_memory_bytes"] == 300_000_000


class TestClassifierConnectorWandBLogEntries:
    @staticmethod
    def _make_connector() -> ClassifierMetricsConnector:
        from unittest.mock import MagicMock

        return ClassifierMetricsConnector(MagicMock())

    def test_ok_entry(self) -> None:
        connector = self._make_connector()
        benchmarks = [
            {
                "batch_size": 4,
                "seq_length": 256,
                "status": "ok",
                "flops": 2000,
                "latency": {"mean_ms": 5.0, "p99_ms": 7.0},
                "throughput_samples_per_sec": 800.0,
            },
        ]

        entries = connector.get_wandb_log_entries(benchmarks)
        assert len(entries) == 1
        log = entries[0]
        assert log["classifier/batch_size"] == 4
        assert log["classifier/flops"] == 2000
        assert log["classifier/latency/mean_ms"] == 5.0
        assert log["classifier/throughput_samples_per_sec"] == 800.0

    def test_oom_entry(self) -> None:
        connector = self._make_connector()
        benchmarks = [{"batch_size": 32, "seq_length": 4096, "status": "OOM"}]

        entries = connector.get_wandb_log_entries(benchmarks)
        assert len(entries) == 1
        assert entries[0]["classifier/status"] == "OOM"

    def test_memory_logged(self) -> None:
        connector = self._make_connector()
        benchmarks = [
            {
                "batch_size": 1,
                "seq_length": 128,
                "status": "ok",
                "memory": {"peak_activation_memory_bytes": 4096, "param_memory_bytes": 2048},
            },
        ]

        entries = connector.get_wandb_log_entries(benchmarks)
        log = entries[0]
        assert log["classifier/memory/peak_activation_memory_bytes"] == 4096
        assert log["classifier/memory/param_memory_bytes"] == 2048

    def test_empty_benchmarks(self) -> None:
        connector = self._make_connector()
        assert connector.get_wandb_log_entries([]) == []
