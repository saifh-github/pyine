"""Guardrail inference metrics: generic runner for benchmarking guardrail types.

Architecture-agnostic runner that delegates per-config measurement to
``GuardrailMetricsConnector`` implementations (probes, LLM classifiers, etc.).
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import pathlib
import typing
from datetime import datetime

import torch

import pyine.configs.schemas
import pyine.evals.common
import pyine.utils.reprod
from pyine.apps.guardrail_metrics.guardrail_metrics_configs import (
    GuardrailMetricsAppMainConfig,
    register_hydra_configs,
)

if typing.TYPE_CHECKING:
    from pyine.guardrails.connector import GuardrailMetricsConnector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connector factory
# ---------------------------------------------------------------------------


def _build_connectors(config: GuardrailMetricsAppMainConfig) -> list[GuardrailMetricsConnector]:
    """Build connectors based on which config fields are set.

    This is the **only** place with architecture-aware imports. The runner
    itself only depends on the ``GuardrailMetricsConnector`` protocol.
    """
    connectors: list[GuardrailMetricsConnector] = []

    if config.probe_checkpoint_dir is not None:
        from pyine.guardrails.probes.metrics_connector import ProbeMetricsConnector

        connectors.append(ProbeMetricsConnector(config))

    if config.classifier_checkpoint_dir is not None:
        from pyine.guardrails.llm_classifier.metrics_connector import ClassifierMetricsConnector

        connectors.append(ClassifierMetricsConnector(config))

    if not connectors:
        raise ValueError("No guardrail connectors configured. Set probe_checkpoint_dir or classifier_checkpoint_dir.")

    return connectors


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_device(config: GuardrailMetricsAppMainConfig) -> torch.device:
    """Resolve device from config, using auto-detection when ``'auto'``."""
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(config.device)


# ---------------------------------------------------------------------------
# Benchmark sweep runner (OOM-resilient)
# ---------------------------------------------------------------------------


def run_benchmark_sweep(
    benchmark_fn: typing.Callable[..., dict[str, typing.Any]],
    batch_sizes: list[int],
    seq_lengths: list[int],
    **kwargs: typing.Any,
) -> list[dict[str, typing.Any]]:
    """Run ``benchmark_fn`` across all ``(batch_size, seq_length)`` combos with OOM resilience.

    Configs are sorted ``(seq_length asc, batch_size asc)`` to minimize OOM probability.
    On OOM: logs warning, records ``"OOM"``, calls ``empty_cache()``, continues.
    """
    results: list[dict[str, typing.Any]] = []
    configs = sorted(
        [(sl, bs) for bs in batch_sizes for sl in seq_lengths],
        key=lambda x: (x[0], x[1]),
    )
    for seq_length, batch_size in configs:
        try:
            result = benchmark_fn(
                batch_size=batch_size,
                seq_length=seq_length,
                **kwargs,
            )
            results.append(result)
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "OOM at batch_size=%d, seq_length=%d — skipping",
                batch_size,
                seq_length,
            )
            torch.cuda.empty_cache()
            results.append(
                {
                    "batch_size": batch_size,
                    "seq_length": seq_length,
                    "status": "OOM",
                }
            )
    return results


# ---------------------------------------------------------------------------
# Generic benchmark runner
# ---------------------------------------------------------------------------


def benchmark_connector(
    connector: GuardrailMetricsConnector,
    config: GuardrailMetricsAppMainConfig,
    device: torch.device,
) -> dict[str, typing.Any]:
    """Generic benchmark runner for any guardrail connector.

    Delegates model loading to ``connector.load()``, per-config measurement to
    ``connector.benchmark_single_config()``, and cleanup to ``connector.cleanup()``.
    The ``run_benchmark_sweep()`` wrapper provides OOM resilience.
    """
    connector.load(device)
    static_info = connector.get_static_info()

    seq_lengths = sorted(config.synthetic_seq_lengths) if config.use_synthetic_data else [config.max_seq_length]
    batch_sizes = sorted(config.batch_sizes)

    benchmarks = run_benchmark_sweep(
        connector.benchmark_single_config,
        batch_sizes=batch_sizes,
        seq_lengths=seq_lengths,
        config=config,
        device=device,
    )

    connector.cleanup()

    return {
        "guardrail_type": connector.guardrail_type,
        "static_info": static_info,
        "benchmarks": benchmarks,
    }


# ---------------------------------------------------------------------------
# Comparison (cross-connector)
# ---------------------------------------------------------------------------


def compute_comparison(
    probe_results: dict[str, typing.Any],
    classifier_results: dict[str, typing.Any],
) -> list[dict[str, typing.Any]]:
    """Compute per-(batch_size, seq_length) comparison between probes and classifier."""
    comparison: list[dict[str, typing.Any]] = []

    probe_benchmarks = probe_results.get("benchmarks", [])
    classifier_benchmarks = classifier_results.get("benchmarks", [])

    # Index classifier benchmarks by (batch_size, seq_length)
    classifier_by_config: dict[tuple[int, int], dict[str, typing.Any]] = {}
    for cb in classifier_benchmarks:
        key = (cb["batch_size"], cb["seq_length"])
        classifier_by_config[key] = cb

    for pb in probe_benchmarks:
        if pb.get("status") == "OOM":
            continue
        key = (pb["batch_size"], pb["seq_length"])
        cb = classifier_by_config.get(key)
        if cb is None or cb.get("status") == "OOM":
            continue

        entry: dict[str, typing.Any] = {
            "batch_size": pb["batch_size"],
            "seq_length": pb["seq_length"],
        }

        # FLOPs comparison
        probe_flops = pb.get("probe_flops", {})
        hook_overhead = pb.get("hook_overhead_flops", 0)
        if probe_flops:
            total_probe_flops = sum(probe_flops.values()) + hook_overhead
            entry["probe_total_overhead_flops"] = total_probe_flops
        classifier_flops = cb.get("flops")
        if classifier_flops is not None:
            entry["classifier_flops"] = classifier_flops
            if "probe_total_overhead_flops" in entry and classifier_flops > 0:
                entry["probe_vs_classifier_flops_ratio"] = entry["probe_total_overhead_flops"] / classifier_flops

        # Latency comparison
        e2e_lat = pb.get("end_to_end_latency", {}).get("mean_ms")
        clf_lat = cb.get("latency", {}).get("mean_ms")
        if e2e_lat is not None:
            entry["probe_end_to_end_latency_ms"] = e2e_lat
        if clf_lat is not None:
            entry["classifier_latency_ms"] = clf_lat
        probe_only_lat = pb.get("probe_only_latency", {}).get("mean_ms")
        if probe_only_lat is not None:
            entry["probe_marginal_latency_ms"] = probe_only_lat

        comparison.append(entry)

    return comparison


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results(
    results: dict[str, typing.Any],
    connectors: list[GuardrailMetricsConnector],
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    output_format: str,
) -> None:
    """Save benchmark results to disk in the requested format(s)."""
    if runtime is not None:
        output_dir = runtime.output_dir_path
    else:
        output_dir = pathlib.Path("guardrail_metrics_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_format in ("json", "both"):
        json_path = output_dir / "guardrail_metrics.json"
        json_path.write_text(json.dumps(results, indent=2, default=str))
        logger.info("saved JSON results to %s", json_path)

    if output_format in ("csv", "both"):
        csv_path = output_dir / "guardrail_metrics.csv"
        _write_csv(results, connectors, csv_path)
        logger.info("saved CSV results to %s", csv_path)


def _write_csv(
    results: dict[str, typing.Any],
    connectors: list[GuardrailMetricsConnector],
    csv_path: pathlib.Path,
) -> None:
    """Write flattened benchmark results to CSV.

    Delegates row generation to each connector's ``flatten_for_csv()`` method.
    """
    fieldnames = [
        "classifier_type",
        "model_name",
        "probe_name",
        "architecture",
        "layer",
        "batch_size",
        "seq_length",
        "status",
        "param_count",
        "flops",
        "mean_latency_ms",
        "p99_latency_ms",
        "throughput_samples_per_sec",
        "peak_memory_bytes",
    ]

    rows: list[dict[str, typing.Any]] = []
    for connector in connectors:
        cr = results.get("guardrail_results", {}).get(connector.guardrail_type, {})
        rows.extend(connector.flatten_for_csv(cr.get("static_info", {}), cr.get("benchmarks", [])))

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    csv_path.write_text(buf.getvalue())


# ---------------------------------------------------------------------------
# WandB logging
# ---------------------------------------------------------------------------


def _log_to_wandb(
    results: dict[str, typing.Any],
    connectors: list[GuardrailMetricsConnector],
    config: GuardrailMetricsAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> None:
    """Push benchmark results to WandB (summary, per-config logs, and comparison table).

    Delegates summary and log entry generation to each connector's
    ``get_wandb_summary()`` and ``get_wandb_log_entries()`` methods.
    """
    if runtime is None or runtime.wandb_run is None:
        return

    wandb_run = runtime.wandb_run

    # --- 1. Summary: metadata + all connectors' static info ---
    metadata = results.get("metadata", {})
    summary: dict[str, typing.Any] = {
        "device": metadata.get("device"),
        "torch_version": metadata.get("torch_version"),
        "cuda_device": metadata.get("cuda_device"),
        "use_torch_compile": metadata.get("use_torch_compile"),
        "num_warmup_iterations": metadata.get("num_warmup_iterations"),
        "num_benchmark_iterations": metadata.get("num_benchmark_iterations"),
    }

    for connector in connectors:
        cr = results.get("guardrail_results", {}).get(connector.guardrail_type, {})
        summary.update(connector.get_wandb_summary(cr.get("static_info", {})))

    wandb_run.summary.update(summary)  # pyright: ignore[reportUnknownMemberType]  # wandb stubs

    # --- 2. Per-config sweep logs ---
    step = 0
    for connector in connectors:
        cr = results.get("guardrail_results", {}).get(connector.guardrail_type, {})
        for log_dict in connector.get_wandb_log_entries(cr.get("benchmarks", [])):
            wandb_run.log(log_dict, step=step)
            step += 1

    # --- 3. Comparison table ---
    comparison = results.get("comparison", [])
    if comparison:
        import wandb

        columns = [
            "batch_size",
            "seq_length",
            "probe_total_overhead_flops",
            "classifier_flops",
            "probe_vs_classifier_flops_ratio",
            "probe_end_to_end_latency_ms",
            "classifier_latency_ms",
            "probe_marginal_latency_ms",
        ]
        table = wandb.Table(columns=columns)
        for entry in comparison:
            table.add_data(*[entry.get(col) for col in columns])  # pyright: ignore[reportUnknownMemberType]
        wandb_run.log({"comparison_table": table})

    logger.info("logged benchmark results to WandB")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


async def main(
    config: GuardrailMetricsAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Main entrypoint for guardrail inference metrics measurement."""
    pyine.utils.reprod.entrypoint_setup(
        runtime_config=runtime,
        use_wandb_logging=config.use_wandb_logging,
        main_config=config,
    )

    if runtime is not None and runtime.dry_run:
        logger.info("dry run mode -- skipping")
        return

    device = resolve_device(config)
    logger.info("using device: %s", device)

    connectors = _build_connectors(config)

    results: dict[str, typing.Any] = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "use_torch_compile": config.use_torch_compile,
            "num_warmup_iterations": config.num_warmup_iterations,
            "num_benchmark_iterations": config.num_benchmark_iterations,
        },
        "guardrail_results": {},
    }

    # --- Benchmark each connector ---
    for connector in connectors:
        key = connector.guardrail_type
        logger.info("starting %s benchmarking...", key)
        results["guardrail_results"][key] = benchmark_connector(connector, config, device)

    # --- Per-config comparative analysis ---
    guardrail_results = results["guardrail_results"]
    if "probe" in guardrail_results and "classifier" in guardrail_results:
        results["comparison"] = compute_comparison(
            guardrail_results["probe"],
            guardrail_results["classifier"],
        )

    # --- Save output ---
    save_results(results, connectors, runtime, config.output_format)

    # --- Log to WandB ---
    _log_to_wandb(results, connectors, config, runtime)

    logger.info("guardrail metrics measurement complete")

    if runtime is not None:
        runtime.finalize()


def async_guardrail_metrics_main_wrapper(
    config: GuardrailMetricsAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Synchronous wrapper around the async main."""
    asyncio.run(main(config=config, runtime=runtime))


# ---------------------------------------------------------------------------
# __main__ block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pyine.apps.trainers.common

    pyine.apps.trainers.common.hydra_main(
        eval_type=pyine.evals.common.EvalType.CORRECTNESS,
        hydra_config_registration_fn=register_hydra_configs,
        async_main_wrapper=async_guardrail_metrics_main_wrapper,
    )
