"""Tests for guardrail_metrics main app logic (generic runner)."""

from __future__ import annotations

import json
import pathlib
import typing
from unittest.mock import MagicMock

import pytest
import torch

import pyine.guardrails.probes.base
import pyine.guardrails.probes.collection
from pyine.apps.guardrail_metrics.guardrail_metrics import (
    _log_to_wandb,
    benchmark_connector,
    compute_comparison,
    resolve_device,
    run_benchmark_sweep,
    save_results,
)
from pyine.apps.guardrail_metrics.guardrail_metrics_configs import (
    GuardrailMetricsAppMainConfig,
)

# ---------------------------------------------------------------------------
# Mock connector for testing the generic runner
# ---------------------------------------------------------------------------


class _MockConnector:
    """Minimal connector stub implementing the GuardrailMetricsConnector protocol."""

    def __init__(self, guardrail_type: str = "mock") -> None:
        self._guardrail_type = guardrail_type
        self.loaded = False
        self.cleaned_up = False

    @property
    def guardrail_type(self) -> str:
        return self._guardrail_type

    def load(self, device: torch.device) -> None:
        self.loaded = True

    def get_static_info(self) -> dict[str, typing.Any]:
        return {"model_name": "mock-model", "param_count": 100}

    def benchmark_single_config(
        self,
        *,
        batch_size: int,
        seq_length: int,
        config: GuardrailMetricsAppMainConfig,
        device: torch.device,
    ) -> dict[str, typing.Any]:
        return {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "status": "ok",
            "flops": 1000,
        }

    def flatten_for_csv(
        self,
        static_info: dict[str, typing.Any],
        benchmarks: list[dict[str, typing.Any]],
    ) -> list[dict[str, typing.Any]]:
        rows: list[dict[str, typing.Any]] = []
        for bm in benchmarks:
            rows.append(
                {
                    "classifier_type": self._guardrail_type,
                    "model_name": static_info.get("model_name", "-"),
                    "probe_name": "-",
                    "architecture": "-",
                    "layer": "-",
                    "batch_size": bm["batch_size"],
                    "seq_length": bm["seq_length"],
                    "status": bm.get("status", "ok"),
                    "flops": bm.get("flops"),
                }
            )
        return rows

    def get_wandb_summary(
        self,
        static_info: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        return {f"{self._guardrail_type}_model_name": static_info.get("model_name")}

    def get_wandb_log_entries(
        self,
        benchmarks: list[dict[str, typing.Any]],
    ) -> list[dict[str, typing.Any]]:
        entries: list[dict[str, typing.Any]] = []
        for bm in benchmarks:
            entries.append(
                {
                    f"{self._guardrail_type}/batch_size": bm["batch_size"],
                    f"{self._guardrail_type}/status": bm.get("status"),
                }
            )
        return entries

    def cleanup(self) -> None:
        self.cleaned_up = True


# ---------------------------------------------------------------------------
# Checkpoint loading tests (probe-specific, unchanged)
# ---------------------------------------------------------------------------


class TestProbeCheckpointLoading:
    def test_load_from_checkpoint_roundtrip(self, tmp_path: pathlib.Path) -> None:
        """Save probes and reload them via load_from_checkpoint."""
        hidden_dim = 32
        configs = [
            pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0),
            pyine.guardrails.probes.base.ProbeConfig(name="max_L4", architecture="max", layer=4),
        ]
        original = pyine.guardrails.probes.collection.ProbeCollection(configs, hidden_dim)

        # Save
        for name, module in original.probes.items():
            probe = typing.cast("pyine.guardrails.probes.base.BaseProbe", module)
            probe_dir = tmp_path / name / "final"
            probe_dir.mkdir(parents=True)
            torch.save(probe.state_dict(), probe_dir / "probe_state_dict.pt")
            (probe_dir / "probe_config.json").write_text(probe.config.model_dump_json(indent=2))

        # Load
        loaded = pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(
            tmp_path,
            hidden_dim,
        )

        # Verify
        assert set(loaded.probes.keys()) == set(original.probes.keys())
        for name in original.probes:
            for (pn1, p1), (pn2, p2) in zip(
                original.probes[name].named_parameters(),
                loaded.probes[name].named_parameters(),
                strict=True,
            ):
                assert pn1 == pn2
                torch.testing.assert_close(p1, p2)

    def test_load_from_checkpoint_eval_mode(self, tmp_path: pathlib.Path) -> None:
        """Loaded collection should be in eval mode."""
        hidden_dim = 32
        configs = [
            pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0),
        ]
        coll = pyine.guardrails.probes.collection.ProbeCollection(configs, hidden_dim)

        probe_dir = tmp_path / "mean_L0" / "final"
        probe_dir.mkdir(parents=True)
        probe = typing.cast("pyine.guardrails.probes.base.BaseProbe", coll.probes["mean_L0"])
        torch.save(probe.state_dict(), probe_dir / "probe_state_dict.pt")
        (probe_dir / "probe_config.json").write_text(probe.config.model_dump_json(indent=2))

        loaded = pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(
            tmp_path,
            hidden_dim,
        )
        assert not loaded.training

    def test_load_from_checkpoint_missing_dir(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(
                pathlib.Path("/nonexistent/dir"),
                64,
            )

    def test_load_from_checkpoint_empty_dir(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError, match="No valid probes found"):
            pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(
                tmp_path,
                64,
            )

    def test_load_from_checkpoint_skips_incomplete(self, tmp_path: pathlib.Path) -> None:
        """Directories missing either config or weights should be skipped."""
        hidden_dim = 32
        configs = [
            pyine.guardrails.probes.base.ProbeConfig(name="mean_L0", architecture="mean", layer=0),
        ]
        coll = pyine.guardrails.probes.collection.ProbeCollection(configs, hidden_dim)

        # Save a valid probe
        probe_dir = tmp_path / "mean_L0" / "final"
        probe_dir.mkdir(parents=True)
        probe = typing.cast("pyine.guardrails.probes.base.BaseProbe", coll.probes["mean_L0"])
        torch.save(probe.state_dict(), probe_dir / "probe_state_dict.pt")
        (probe_dir / "probe_config.json").write_text(probe.config.model_dump_json(indent=2))

        # Create incomplete directory (missing weights in final/ subdir)
        incomplete_dir = tmp_path / "incomplete" / "final"
        incomplete_dir.mkdir(parents=True)
        (incomplete_dir / "probe_config.json").write_text("{}")

        loaded = pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(
            tmp_path,
            hidden_dim,
        )
        assert set(loaded.probes.keys()) == {"mean_L0"}


# ---------------------------------------------------------------------------
# Device resolution tests
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_explicit_cpu(self) -> None:
        config = GuardrailMetricsAppMainConfig(
            classifier_checkpoint_dir="/tmp/clf",  # noqa: S108
            device="cpu",
        )
        assert resolve_device(config) == torch.device("cpu")

    def test_auto_fallback_to_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On a machine without CUDA or MPS, auto should resolve to cpu."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        config = GuardrailMetricsAppMainConfig(
            classifier_checkpoint_dir="/tmp/clf",  # noqa: S108
            device="auto",
        )
        assert resolve_device(config) == torch.device("cpu")


# ---------------------------------------------------------------------------
# Benchmark sweep tests
# ---------------------------------------------------------------------------


class TestRunBenchmarkSweep:
    def test_sweeps_all_configs(self) -> None:
        call_log: list[tuple[int, int]] = []

        def mock_fn(batch_size: int, seq_length: int) -> dict[str, typing.Any]:
            call_log.append((batch_size, seq_length))
            return {"batch_size": batch_size, "seq_length": seq_length, "status": "ok"}

        results = run_benchmark_sweep(mock_fn, batch_sizes=[1, 2], seq_lengths=[64, 128])
        assert len(results) == 4
        assert len(call_log) == 4

    def test_sorted_ascending_order(self) -> None:
        call_order: list[tuple[int, int]] = []

        def mock_fn(batch_size: int, seq_length: int) -> dict[str, typing.Any]:
            call_order.append((seq_length, batch_size))
            return {"batch_size": batch_size, "seq_length": seq_length, "status": "ok"}

        run_benchmark_sweep(mock_fn, batch_sizes=[32, 1], seq_lengths=[1024, 128])
        # Should be sorted (seq_length asc, batch_size asc)
        assert call_order == sorted(call_order)


# ---------------------------------------------------------------------------
# benchmark_connector tests (generic runner)
# ---------------------------------------------------------------------------


class TestBenchmarkConnector:
    @staticmethod
    def _make_config(**kwargs: typing.Any) -> GuardrailMetricsAppMainConfig:
        defaults: dict[str, typing.Any] = {
            "classifier_checkpoint_dir": "/tmp/clf",  # noqa: S108
        }
        defaults.update(kwargs)
        return GuardrailMetricsAppMainConfig(**defaults)

    def test_calls_load_and_cleanup(self) -> None:
        connector = _MockConnector()
        config = self._make_config()
        device = torch.device("cpu")

        benchmark_connector(connector, config, device)  # type: ignore[arg-type]

        assert connector.loaded
        assert connector.cleaned_up

    def test_returns_guardrail_type_and_benchmarks(self) -> None:
        connector = _MockConnector(guardrail_type="test")
        config = self._make_config(batch_sizes=[1], synthetic_seq_lengths=[64])
        device = torch.device("cpu")

        result = benchmark_connector(connector, config, device)  # type: ignore[arg-type]

        assert result["guardrail_type"] == "test"
        assert "static_info" in result
        assert "benchmarks" in result
        assert len(result["benchmarks"]) >= 1

    def test_static_info_included(self) -> None:
        connector = _MockConnector()
        config = self._make_config(batch_sizes=[1], synthetic_seq_lengths=[128])
        device = torch.device("cpu")

        result = benchmark_connector(connector, config, device)  # type: ignore[arg-type]

        assert result["static_info"]["model_name"] == "mock-model"
        assert result["static_info"]["param_count"] == 100


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------


class TestComputeComparison:
    def test_matching_configs(self) -> None:
        probe_results = {
            "benchmarks": [
                {
                    "batch_size": 1,
                    "seq_length": 512,
                    "status": "ok",
                    "probe_flops": {"mean_L16": 8192},
                    "hook_overhead_flops": 100,
                    "end_to_end_latency": {"mean_ms": 12.5},
                    "probe_only_latency": {"mean_ms": 0.05},
                },
            ],
        }
        classifier_results = {
            "benchmarks": [
                {
                    "batch_size": 1,
                    "seq_length": 512,
                    "status": "ok",
                    "flops": 5_000_000_000,
                    "latency": {"mean_ms": 8.5},
                },
            ],
        }

        comparison = compute_comparison(probe_results, classifier_results)
        assert len(comparison) == 1
        entry = comparison[0]
        assert entry["batch_size"] == 1
        assert entry["seq_length"] == 512
        assert entry["probe_total_overhead_flops"] == 8192 + 100
        assert entry["classifier_flops"] == 5_000_000_000
        assert entry["probe_vs_classifier_flops_ratio"] < 1.0

    def test_oom_entries_skipped(self) -> None:
        probe_results = {
            "benchmarks": [
                {"batch_size": 32, "seq_length": 4096, "status": "OOM"},
            ],
        }
        classifier_results = {
            "benchmarks": [
                {"batch_size": 32, "seq_length": 4096, "status": "ok", "flops": 100},
            ],
        }

        comparison = compute_comparison(probe_results, classifier_results)
        assert len(comparison) == 0

    def test_unmatched_configs_skipped(self) -> None:
        probe_results = {
            "benchmarks": [
                {
                    "batch_size": 1,
                    "seq_length": 512,
                    "status": "ok",
                    "probe_flops": {"p1": 100},
                    "hook_overhead_flops": 0,
                },
            ],
        }
        classifier_results = {
            "benchmarks": [
                {"batch_size": 4, "seq_length": 1024, "status": "ok", "flops": 100},
            ],
        }

        comparison = compute_comparison(probe_results, classifier_results)
        assert len(comparison) == 0


# ---------------------------------------------------------------------------
# Output saving tests
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_save_json(self, tmp_path: pathlib.Path) -> None:
        results: dict[str, typing.Any] = {
            "metadata": {"test": True},
            "guardrail_results": {},
        }

        class MockRuntime:
            output_dir_path = tmp_path

        connectors = [_MockConnector()]
        save_results(results, connectors, MockRuntime(), "json")  # type: ignore[arg-type]
        json_path = tmp_path / "guardrail_metrics.json"
        assert json_path.exists()
        loaded = json.loads(json_path.read_text())
        assert loaded["metadata"]["test"] is True

    def test_save_csv(self, tmp_path: pathlib.Path) -> None:
        connector = _MockConnector(guardrail_type="classifier")
        results: dict[str, typing.Any] = {
            "guardrail_results": {
                "classifier": {
                    "guardrail_type": "classifier",
                    "static_info": {"model_name": "test", "param_count": 1000},
                    "benchmarks": [
                        {
                            "batch_size": 1,
                            "seq_length": 128,
                            "status": "ok",
                            "flops": 500,
                        },
                    ],
                },
            },
        }

        class MockRuntime:
            output_dir_path = tmp_path

        save_results(results, [connector], MockRuntime(), "csv")  # type: ignore[arg-type]
        csv_path = tmp_path / "guardrail_metrics.csv"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "classifier" in content
        assert "test" in content


# ---------------------------------------------------------------------------
# WandB logging tests
# ---------------------------------------------------------------------------


class TestLogToWandB:
    @staticmethod
    def _make_config(**kwargs: typing.Any) -> GuardrailMetricsAppMainConfig:
        defaults: dict[str, typing.Any] = {
            "classifier_checkpoint_dir": "/tmp/clf",  # noqa: S108
        }
        defaults.update(kwargs)
        return GuardrailMetricsAppMainConfig(**defaults)

    def test_noop_when_no_runtime(self) -> None:
        """Should silently return when runtime is None."""
        _log_to_wandb({}, [], self._make_config(), None)

    def test_noop_when_no_wandb_run(self) -> None:
        """Should silently return when wandb_run is None."""
        runtime = MagicMock()
        runtime.wandb_run = None
        _log_to_wandb({}, [], self._make_config(), runtime)

    def test_summary_updated_with_metadata(self) -> None:
        runtime = MagicMock()
        wandb_run = MagicMock()
        runtime.wandb_run = wandb_run

        results: dict[str, typing.Any] = {
            "metadata": {
                "device": "cuda",
                "torch_version": "2.3.0",
                "cuda_device": "A100",
                "use_torch_compile": False,
                "num_warmup_iterations": 10,
                "num_benchmark_iterations": 100,
            },
            "guardrail_results": {},
        }
        _log_to_wandb(results, [], self._make_config(), runtime)

        wandb_run.summary.update.assert_called_once()
        summary_dict = wandb_run.summary.update.call_args[0][0]
        assert summary_dict["device"] == "cuda"
        assert summary_dict["torch_version"] == "2.3.0"
        assert summary_dict["cuda_device"] == "A100"

    def test_connector_benchmarks_logged(self) -> None:
        runtime = MagicMock()
        wandb_run = MagicMock()
        runtime.wandb_run = wandb_run

        connector = _MockConnector(guardrail_type="probe")
        results: dict[str, typing.Any] = {
            "metadata": {},
            "guardrail_results": {
                "probe": {
                    "static_info": {"model_name": "test-llm"},
                    "benchmarks": [
                        {"batch_size": 1, "seq_length": 512, "status": "ok"},
                    ],
                },
            },
        }
        _log_to_wandb(results, [connector], self._make_config(), runtime)  # type: ignore[arg-type]

        assert wandb_run.log.call_count >= 1
        first_log = wandb_run.log.call_args_list[0]
        log_dict = first_log[0][0]
        assert log_dict["probe/batch_size"] == 1

    def test_summary_includes_connector_info(self) -> None:
        runtime = MagicMock()
        wandb_run = MagicMock()
        runtime.wandb_run = wandb_run

        probe_connector = _MockConnector(guardrail_type="probe")
        clf_connector = _MockConnector(guardrail_type="classifier")
        results: dict[str, typing.Any] = {
            "metadata": {},
            "guardrail_results": {
                "probe": {
                    "static_info": {"model_name": "Qwen/Qwen2.5-3B"},
                    "benchmarks": [],
                },
                "classifier": {
                    "static_info": {"model_name": "modernbert"},
                    "benchmarks": [],
                },
            },
        }
        _log_to_wandb(results, [probe_connector, clf_connector], self._make_config(), runtime)  # type: ignore[arg-type]

        summary_dict = wandb_run.summary.update.call_args[0][0]
        assert summary_dict["probe_model_name"] == "Qwen/Qwen2.5-3B"
        assert summary_dict["classifier_model_name"] == "modernbert"

    def test_comparison_table_logged(self) -> None:
        runtime = MagicMock()
        wandb_run = MagicMock()
        runtime.wandb_run = wandb_run

        results: dict[str, typing.Any] = {
            "metadata": {},
            "guardrail_results": {},
            "comparison": [
                {
                    "batch_size": 1,
                    "seq_length": 512,
                    "probe_total_overhead_flops": 8292,
                    "classifier_flops": 5_000_000_000,
                    "probe_vs_classifier_flops_ratio": 0.0000016584,
                    "probe_end_to_end_latency_ms": 12.5,
                    "classifier_latency_ms": 8.5,
                    "probe_marginal_latency_ms": 0.05,
                },
            ],
        }
        _log_to_wandb(results, [], self._make_config(), runtime)

        # Find the call that logs the comparison table
        table_logged = False
        for call in wandb_run.log.call_args_list:
            log_dict = call[0][0]
            if "comparison_table" in log_dict:
                table_logged = True
                break
        assert table_logged

    def test_oom_entries_logged(self) -> None:
        runtime = MagicMock()
        wandb_run = MagicMock()
        runtime.wandb_run = wandb_run

        connector = _MockConnector(guardrail_type="probe")
        results: dict[str, typing.Any] = {
            "metadata": {},
            "guardrail_results": {
                "probe": {
                    "static_info": {},
                    "benchmarks": [
                        {"batch_size": 32, "seq_length": 4096, "status": "OOM"},
                    ],
                },
            },
        }
        _log_to_wandb(results, [connector], self._make_config(), runtime)  # type: ignore[arg-type]

        assert wandb_run.log.call_count >= 1
        log_dict = wandb_run.log.call_args_list[0][0][0]
        assert log_dict["probe/status"] == "OOM"
