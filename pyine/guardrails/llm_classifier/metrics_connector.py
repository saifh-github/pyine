"""ClassifierMetricsConnector -- GuardrailMetricsConnector for LLM classifiers.

Simpler than ProbeMetricsConnector -- classifier measurements are independent
(no hooks, no derived metrics).
"""

from __future__ import annotations

import logging
import pathlib
import typing

import torch
import transformers

import pyine.apps.trainers.common as common
from pyine.apps.guardrail_metrics.metrics_collector import (
    measure_flops,
    measure_latency,
    measure_memory,
)

if typing.TYPE_CHECKING:
    from pyine.apps.guardrail_metrics.guardrail_metrics_configs import GuardrailMetricsAppMainConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forward function helper
# ---------------------------------------------------------------------------


def _llm_forward_fn(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor] | tuple[torch.Tensor, ...],
) -> typing.Any:
    """Standard LLM forward pass."""
    if isinstance(inputs, dict):
        return model(**inputs)
    return model(*inputs)


# ---------------------------------------------------------------------------
# ClassifierMetricsConnector
# ---------------------------------------------------------------------------


class ClassifierMetricsConnector:
    """Implements GuardrailMetricsConnector for LLM classifiers.

    Simpler than ProbeMetricsConnector -- classifier measurements are
    independent (no hooks, no derived metrics). Each measurement
    (FLOPs, memory, latency, throughput) is self-contained.
    """

    def __init__(self, config: GuardrailMetricsAppMainConfig) -> None:
        self._config = config
        self._model: torch.nn.Module | None = None
        self._tokenizer: transformers.PreTrainedTokenizerBase | None = None
        self._vocab_size: int = 0

    @property
    def guardrail_type(self) -> str:
        return "classifier"

    def load(self, device: torch.device) -> None:
        """Load classifier + tokenizer from HF checkpoint dir."""
        assert self._config.classifier_checkpoint_dir is not None

        checkpoint_dir = pathlib.Path(self._config.classifier_checkpoint_dir)

        logger.info("loading classifier from: %s", checkpoint_dir)
        resolved_config = common.resolve_attn_implementation(self._config.classifier_auto_model_config)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # transformers stubs
            checkpoint_dir,
            **resolved_config,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # transformers stubs
        model.eval()  # pyright: ignore[reportUnknownMemberType]  # transformers stubs
        model.to(device)  # pyright: ignore[reportUnknownMemberType]  # transformers stubs

        if self._config.use_torch_compile:
            logger.info("applying torch.compile() to classifier")
            model = torch.compile(model)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

        self._model = model  # pyright: ignore[reportAttributeAccessIssue]
        self._tokenizer = tokenizer  # type: ignore[assignment]  # pyright: ignore[reportUnknownVariableType]
        self._vocab_size = tokenizer.vocab_size  # pyright: ignore[reportUnknownMemberType, reportAssignmentType, reportUnknownVariableType]  # transformers stubs

    def get_static_info(self) -> dict[str, typing.Any]:
        """Return classifier model info."""
        assert self._model is not None
        assert self._config.classifier_checkpoint_dir is not None

        return {
            "model_name": pathlib.Path(self._config.classifier_checkpoint_dir).name,
            "param_count": sum(p.numel() for p in self._model.parameters()),  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # transformers stubs
            "param_memory_bytes": sum(p.numel() * p.element_size() for p in self._model.parameters()),  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # transformers stubs
        }

    def benchmark_single_config(
        self,
        *,
        batch_size: int,
        seq_length: int,
        config: GuardrailMetricsAppMainConfig,
        device: torch.device,
    ) -> dict[str, typing.Any]:
        """Run all classifier measurements for one ``(batch_size, seq_length)`` config.

        Each measurement is independent (no hooks, no derived metrics).
        """
        assert self._model is not None

        result: dict[str, typing.Any] = {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "status": "ok",
        }

        input_ids = torch.randint(0, self._vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=device)
        batch: dict[str, torch.Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}

        if config.measure_flops:
            result["flops"] = measure_flops(self._model, batch, _llm_forward_fn)

        if config.measure_memory:
            result["memory"] = measure_memory(self._model, batch, _llm_forward_fn, device)

        if config.measure_latency:
            lat = measure_latency(
                self._model,
                batch,
                _llm_forward_fn,
                config.num_warmup_iterations,
                config.num_benchmark_iterations,
                device,
            )
            result["latency"] = lat

        if config.measure_throughput and config.measure_latency:
            mean_ms = result.get("latency", {}).get("mean_ms")
            if mean_ms and mean_ms > 0:
                result["throughput_samples_per_sec"] = batch_size / (mean_ms / 1000.0)

        return result

    def flatten_for_csv(
        self,
        static_info: dict[str, typing.Any],
        benchmarks: list[dict[str, typing.Any]],
    ) -> list[dict[str, typing.Any]]:
        """Flatten classifier benchmarks into CSV rows.

        One row per ``(batch_size, seq_length)`` config.
        """
        rows: list[dict[str, typing.Any]] = []
        for bm in benchmarks:
            if bm.get("status") == "OOM":
                rows.append(
                    {
                        "classifier_type": "classifier",
                        "model_name": static_info.get("model_name", "-"),
                        "probe_name": "-",
                        "architecture": "-",
                        "layer": "-",
                        "batch_size": bm["batch_size"],
                        "seq_length": bm["seq_length"],
                        "status": "OOM",
                    }
                )
                continue

            rows.append(
                {
                    "classifier_type": "classifier",
                    "model_name": static_info.get("model_name", "-"),
                    "probe_name": "-",
                    "architecture": "-",
                    "layer": "-",
                    "batch_size": bm["batch_size"],
                    "seq_length": bm["seq_length"],
                    "status": "ok",
                    "param_count": static_info.get("param_count"),
                    "flops": bm.get("flops"),
                    "mean_latency_ms": bm.get("latency", {}).get("mean_ms"),
                    "p99_latency_ms": bm.get("latency", {}).get("p99_ms"),
                    "throughput_samples_per_sec": bm.get("throughput_samples_per_sec"),
                    "peak_memory_bytes": bm.get("memory", {}).get("peak_activation_memory_bytes"),
                }
            )

        return rows

    def get_wandb_summary(
        self,
        static_info: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Return WandB summary entries for classifier static info."""
        return {
            "classifier_model_name": static_info.get("model_name"),
            "classifier_param_count": static_info.get("param_count"),
            "classifier_param_memory_bytes": static_info.get("param_memory_bytes"),
        }

    def get_wandb_log_entries(
        self,
        benchmarks: list[dict[str, typing.Any]],
    ) -> list[dict[str, typing.Any]]:
        """Convert classifier benchmarks to WandB log dicts (``classifier/`` prefix)."""
        entries: list[dict[str, typing.Any]] = []
        for bm in benchmarks:
            log_dict: dict[str, typing.Any] = {
                "classifier/batch_size": bm["batch_size"],
                "classifier/seq_length": bm["seq_length"],
                "classifier/status": bm.get("status"),
            }
            if bm.get("status") == "OOM":
                entries.append(log_dict)
                continue

            if "flops" in bm:
                log_dict["classifier/flops"] = bm["flops"]

            lat = bm.get("latency", {})
            if lat:
                for stat_name, stat_val in lat.items():
                    log_dict[f"classifier/latency/{stat_name}"] = stat_val

            mem = bm.get("memory", {})
            if mem:
                for mem_key, mem_val in mem.items():
                    log_dict[f"classifier/memory/{mem_key}"] = mem_val

            throughput = bm.get("throughput_samples_per_sec")
            if throughput is not None:
                log_dict["classifier/throughput_samples_per_sec"] = throughput

            entries.append(log_dict)

        return entries

    def cleanup(self) -> None:
        """Release resources."""
        pass
