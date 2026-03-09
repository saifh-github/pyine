"""ProbeMetricsConnector -- GuardrailMetricsConnector for activation probes.

Owns the full probe measurement lifecycle: LLM baseline FLOPs, hooked FLOPs,
hook overhead derivation, per-probe FLOPs on captured activations, memory,
latency (baseline/e2e/probe-only), and throughput.
"""

from __future__ import annotations

import logging
import pathlib
import typing

import torch
import transformers

import pyine.apps.trainers.common as common
import pyine.guardrails.probes.collection
import pyine.guardrails.probes.extraction
from pyine.apps.guardrail_metrics.metrics_collector import (
    measure_flops,
    measure_latency,
    measure_memory,
)

if typing.TYPE_CHECKING:
    import pyine.guardrails.probes.base  # noqa: TC004 - used in typing.cast() string form
    from pyine.apps.guardrail_metrics.guardrail_metrics_configs import GuardrailMetricsAppMainConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forward function helpers
# ---------------------------------------------------------------------------


def _llm_forward_fn(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor] | tuple[torch.Tensor, ...],
) -> typing.Any:
    """Standard LLM forward pass."""
    if isinstance(inputs, dict):
        return model(**inputs)
    return model(*inputs)


def _probe_forward_fn(
    probe_collection: torch.nn.Module,
    inputs: dict[str, torch.Tensor] | tuple[torch.Tensor, ...],
) -> typing.Any:
    """Probe collection forward pass on (activations_dict, attention_mask)."""
    assert isinstance(inputs, tuple) and len(inputs) == 2
    activations, attention_mask = inputs
    return probe_collection(activations, attention_mask)


def _single_probe_forward(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor] | tuple[torch.Tensor, ...],
) -> typing.Any:
    """Single probe forward on (hidden_states, attention_mask)."""
    assert isinstance(inputs, tuple) and len(inputs) == 2
    hidden_states, attn_mask = inputs
    return model(hidden_states, attn_mask)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _generate_synthetic_batch(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Generate a synthetic batch of ``input_ids`` and ``attention_mask``."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _generate_synthetic_activations(
    batch_size: int,
    seq_length: int,
    hidden_dim: int,
    target_layers: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Generate synthetic activations mimicking ``ActivationExtractor`` output."""
    activations = {
        layer: torch.randn(batch_size, seq_length, hidden_dim, device=device, dtype=dtype) for layer in target_layers
    }
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=device)
    return activations, attention_mask


# ---------------------------------------------------------------------------
# ProbeMetricsConnector
# ---------------------------------------------------------------------------


class ProbeMetricsConnector:
    """Implements GuardrailMetricsConnector for activation probes.

    Owns the full probe measurement lifecycle: LLM baseline FLOPs,
    hooked FLOPs, hook overhead derivation, per-probe FLOPs on
    captured activations, memory, latency (baseline/e2e/probe-only),
    and throughput.
    """

    def __init__(self, config: GuardrailMetricsAppMainConfig) -> None:
        self._config = config
        self._llm_model: torch.nn.Module | None = None
        self._probe_collection: pyine.guardrails.probes.collection.ProbeCollection | None = None
        self._device: torch.device | None = None
        self._hidden_dim: int = 0
        self._vocab_size: int = 0
        self._dtype: torch.dtype = torch.float32
        self._target_layers: list[int] = []

    @property
    def guardrail_type(self) -> str:
        return "probe"

    def load(self, device: torch.device) -> None:
        """Load frozen LLM + probe collection from checkpoints."""
        self._device = device
        assert self._config.probe_checkpoint_dir is not None
        assert self._config.probe_llm_model is not None

        checkpoint_dir = pathlib.Path(self._config.probe_checkpoint_dir)

        # Load frozen LLM
        logger.info("loading frozen LLM: %s", self._config.probe_llm_model)
        resolved_config = common.resolve_attn_implementation(self._config.probe_auto_model_config)
        llm_model = transformers.AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # transformers stubs
            self._config.probe_llm_checkpoint_path or self._config.probe_llm_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map={"": device},
            **resolved_config,
        )
        llm_model.eval()  # pyright: ignore[reportUnknownMemberType]  # transformers stubs
        llm_model.requires_grad_(False)  # pyright: ignore[reportUnknownMemberType]  # transformers stubs
        self._llm_model = llm_model

        model_config = llm_model.config  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # transformers stubs
        self._hidden_dim = model_config.hidden_size  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]  # transformers stubs
        self._vocab_size = model_config.vocab_size  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]  # transformers stubs
        self._dtype = next(llm_model.parameters()).dtype  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # transformers stubs

        # Load probe collection
        logger.info(
            f"loading probe collection from: {checkpoint_dir}; "
            f"checkpoint_name={self._config.probe_checkpoint_name or 'auto'}"
        )
        probe_collection = pyine.guardrails.probes.collection.ProbeCollection.load_from_checkpoint(
            checkpoint_dir,
            self._hidden_dim,
            checkpoint_name=self._config.probe_checkpoint_name,
        )
        probe_collection.to(device=device, dtype=self._dtype)

        if self._config.use_torch_compile:
            logger.info("applying torch.compile() to probe collection")
            probe_collection = torch.compile(probe_collection)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

        self._probe_collection = probe_collection  # pyright: ignore[reportAttributeAccessIssue]
        self._target_layers = sorted({pc.layer for pc in probe_collection._probe_configs.values()})  # pyright: ignore[reportPrivateUsage, reportFunctionMemberAccess]

    def get_static_info(self) -> dict[str, typing.Any]:
        """Return frozen LLM info + per-probe info."""
        assert self._llm_model is not None
        assert self._probe_collection is not None

        probes_info: dict[str, dict[str, typing.Any]] = {}
        for name, module in self._probe_collection.probes.items():  # pyright: ignore[reportFunctionMemberAccess]
            probe = typing.cast("pyine.guardrails.probes.base.BaseProbe", module)
            probes_info[name] = {
                "architecture": probe.config.architecture,
                "layer": probe.config.layer,
                "param_count": sum(p.numel() for p in probe.parameters()),
                "param_memory_bytes": sum(p.numel() * p.element_size() for p in probe.parameters()),
            }

        frozen_llm_info = {
            "model_name": self._config.probe_llm_model,
            "param_count": sum(p.numel() for p in self._llm_model.parameters()),  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # transformers stubs
            "param_memory_bytes": sum(p.numel() * p.element_size() for p in self._llm_model.parameters()),  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # transformers stubs
        }

        return {
            "frozen_llm": frozen_llm_info,
            "probes": probes_info,
        }

    def benchmark_single_config(
        self,
        *,
        batch_size: int,
        seq_length: int,
        config: GuardrailMetricsAppMainConfig,
        device: torch.device,
    ) -> dict[str, typing.Any]:
        """Run all probe measurements for one ``(batch_size, seq_length)`` config.

        Measurements are interdependent within a single config:

        1. **FLOPs**: baseline (no hooks) → hooked → hook_overhead (derived) →
           per-probe FLOPs on captured activations
        2. **Memory**: probe collection forward on synthetic activations
        3. **Latency**: baseline (no hooks) → e2e (with hooks + probes) →
           probe-only (synthetic activations)
        4. **Throughput**: derived from latency
        """
        assert self._llm_model is not None
        assert self._probe_collection is not None

        result: dict[str, typing.Any] = {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "status": "ok",
        }

        batch = _generate_synthetic_batch(batch_size, seq_length, self._vocab_size, device)

        # --- FLOPs (interdependent) ---
        if config.measure_flops:
            # LLM forward WITHOUT hooks (baseline)
            baseline_flops = measure_flops(self._llm_model, batch, _llm_forward_fn)
            result["llm_forward_flops"] = baseline_flops

            # LLM forward WITH hooks
            extractor = pyine.guardrails.probes.extraction.ActivationExtractor(
                self._llm_model,
                self._target_layers,
                activation_dtype=self._dtype,
            )
            hooked_flops = measure_flops(self._llm_model, batch, _llm_forward_fn)
            result["llm_forward_with_hooks_flops"] = hooked_flops
            result["hook_overhead_flops"] = hooked_flops - baseline_flops

            # Capture activations for per-probe measurement
            with torch.inference_mode():
                self._llm_model(**batch)
            activations = extractor.get_activations()
            attention_mask = batch["attention_mask"]
            extractor.remove_hooks()

            # Per-probe FLOPs on captured activations
            probe_flops: dict[str, int] = {}
            for probe_name, probe_module in self._probe_collection.probes.items():
                probe = typing.cast("pyine.guardrails.probes.base.BaseProbe", probe_module)
                layer_act = activations[probe.config.layer]
                probe_flops[probe_name] = measure_flops(
                    probe,
                    (layer_act, attention_mask),
                    _single_probe_forward,
                )
            result["probe_flops"] = probe_flops

        # --- Memory ---
        if config.measure_memory:
            activations_synth, attn_mask_synth = _generate_synthetic_activations(
                batch_size,
                seq_length,
                self._hidden_dim,
                self._target_layers,
                device,
                self._dtype,
            )
            mem = measure_memory(
                self._probe_collection,
                (activations_synth, attn_mask_synth),  # pyright: ignore[reportArgumentType]
                _probe_forward_fn,
                device,
            )
            result["probe_memory"] = mem

        # --- Latency (hook lifecycle coordination) ---
        if config.measure_latency:
            # LLM forward baseline latency (without hooks)
            llm_baseline_lat = measure_latency(
                self._llm_model,
                batch,
                _llm_forward_fn,
                config.num_warmup_iterations,
                config.num_benchmark_iterations,
                device,
            )
            result["llm_baseline_latency"] = llm_baseline_lat

            # End-to-end latency (LLM forward with hooks + probe forward)
            extractor = pyine.guardrails.probes.extraction.ActivationExtractor(
                self._llm_model,
                self._target_layers,
                activation_dtype=self._dtype,
            )

            probe_collection_ref = self._probe_collection

            def _e2e_forward(
                model: torch.nn.Module,
                inputs: dict[str, torch.Tensor] | tuple[torch.Tensor, ...],
            ) -> None:
                assert isinstance(inputs, dict)
                model(**inputs)
                acts = extractor.get_activations()
                attn = inputs["attention_mask"]
                probe_collection_ref(acts, attn)

            e2e_lat = measure_latency(
                self._llm_model,
                batch,
                _e2e_forward,
                config.num_warmup_iterations,
                config.num_benchmark_iterations,
                device,
            )
            result["end_to_end_latency"] = e2e_lat
            extractor.remove_hooks()

            # Probe-only latency
            activations_synth, attn_mask_synth = _generate_synthetic_activations(
                batch_size,
                seq_length,
                self._hidden_dim,
                self._target_layers,
                device,
                self._dtype,
            )
            probe_only_lat = measure_latency(
                self._probe_collection,
                (activations_synth, attn_mask_synth),  # pyright: ignore[reportArgumentType]
                _probe_forward_fn,
                config.num_warmup_iterations,
                config.num_benchmark_iterations,
                device,
            )
            result["probe_only_latency"] = probe_only_lat

        # --- Throughput (derived from latency) ---
        if config.measure_throughput and config.measure_latency:
            e2e_mean_ms = result.get("end_to_end_latency", {}).get("mean_ms")
            if e2e_mean_ms and e2e_mean_ms > 0:
                result["throughput_samples_per_sec"] = {
                    "end_to_end": batch_size / (e2e_mean_ms / 1000.0),
                }
                probe_only_mean_ms = result.get("probe_only_latency", {}).get("mean_ms")
                if probe_only_mean_ms and probe_only_mean_ms > 0:
                    result["throughput_samples_per_sec"]["probe_only"] = batch_size / (probe_only_mean_ms / 1000.0)

        return result

    def flatten_for_csv(
        self,
        static_info: dict[str, typing.Any],
        benchmarks: list[dict[str, typing.Any]],
    ) -> list[dict[str, typing.Any]]:
        """Flatten probe benchmarks into CSV rows.

        Produces per-probe rows (with per-probe FLOPs, architecture, layer)
        and an end-to-end row (with LLM+hooks FLOPs and e2e latency) for
        each ``(batch_size, seq_length)`` config.
        """
        rows: list[dict[str, typing.Any]] = []
        probes_info = static_info.get("probes", {})

        for bm in benchmarks:
            if bm.get("status") == "OOM":
                rows.append(
                    {
                        "classifier_type": "probe",
                        "model_name": "end_to_end",
                        "probe_name": "-",
                        "architecture": "-",
                        "layer": "-",
                        "batch_size": bm["batch_size"],
                        "seq_length": bm["seq_length"],
                        "status": "OOM",
                    }
                )
                continue

            # Per-probe rows
            probe_flops = bm.get("probe_flops", {})
            for probe_name, flops_val in probe_flops.items():
                info = probes_info.get(probe_name, {})
                rows.append(
                    {
                        "classifier_type": "probe",
                        "model_name": probe_name,
                        "probe_name": probe_name,
                        "architecture": info.get("architecture", "-"),
                        "layer": info.get("layer", "-"),
                        "batch_size": bm["batch_size"],
                        "seq_length": bm["seq_length"],
                        "status": "ok",
                        "param_count": info.get("param_count"),
                        "flops": flops_val,
                        "mean_latency_ms": bm.get("probe_only_latency", {}).get("mean_ms"),
                        "p99_latency_ms": bm.get("probe_only_latency", {}).get("p99_ms"),
                        "throughput_samples_per_sec": bm.get("throughput_samples_per_sec", {}).get("probe_only"),
                        "peak_memory_bytes": bm.get("probe_memory", {}).get("peak_activation_memory_bytes"),
                    }
                )

            # End-to-end row
            rows.append(
                {
                    "classifier_type": "probe",
                    "model_name": "end_to_end",
                    "probe_name": "-",
                    "architecture": "-",
                    "layer": "-",
                    "batch_size": bm["batch_size"],
                    "seq_length": bm["seq_length"],
                    "status": "ok",
                    "param_count": "-",
                    "flops": bm.get("llm_forward_with_hooks_flops"),
                    "mean_latency_ms": bm.get("end_to_end_latency", {}).get("mean_ms"),
                    "p99_latency_ms": bm.get("end_to_end_latency", {}).get("p99_ms"),
                    "throughput_samples_per_sec": bm.get("throughput_samples_per_sec", {}).get("end_to_end"),
                }
            )

        return rows

    def get_wandb_summary(
        self,
        static_info: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Return WandB summary entries for probe static info."""
        summary: dict[str, typing.Any] = {}
        frozen_llm = static_info.get("frozen_llm", {})
        summary["probe_llm_model"] = frozen_llm.get("model_name")
        summary["probe_llm_param_count"] = frozen_llm.get("param_count")
        summary["probe_llm_param_memory_bytes"] = frozen_llm.get("param_memory_bytes")
        for probe_name, info in static_info.get("probes", {}).items():
            summary[f"probe/{probe_name}/architecture"] = info.get("architecture")
            summary[f"probe/{probe_name}/layer"] = info.get("layer")
            summary[f"probe/{probe_name}/param_count"] = info.get("param_count")
        return summary

    def get_wandb_log_entries(
        self,
        benchmarks: list[dict[str, typing.Any]],
    ) -> list[dict[str, typing.Any]]:
        """Convert probe benchmark results to WandB log dicts.

        Each entry is one WandB step with keys prefixed ``probe/``.
        """
        entries: list[dict[str, typing.Any]] = []
        for bm in benchmarks:
            log_dict: dict[str, typing.Any] = {
                "probe/batch_size": bm["batch_size"],
                "probe/seq_length": bm["seq_length"],
                "probe/status": bm.get("status"),
            }
            if bm.get("status") == "OOM":
                entries.append(log_dict)
                continue

            if "llm_forward_flops" in bm:
                log_dict["probe/llm_forward_flops"] = bm["llm_forward_flops"]
            if "llm_forward_with_hooks_flops" in bm:
                log_dict["probe/llm_forward_with_hooks_flops"] = bm["llm_forward_with_hooks_flops"]
            if "hook_overhead_flops" in bm:
                log_dict["probe/hook_overhead_flops"] = bm["hook_overhead_flops"]
            for probe_name, flops_val in bm.get("probe_flops", {}).items():
                log_dict[f"probe/flops/{probe_name}"] = flops_val

            for key in ("llm_baseline_latency", "end_to_end_latency", "probe_only_latency"):
                lat = bm.get(key, {})
                if lat:
                    for stat_name, stat_val in lat.items():
                        log_dict[f"probe/{key}/{stat_name}"] = stat_val

            mem = bm.get("probe_memory", {})
            if mem:
                for mem_key, mem_val in mem.items():
                    log_dict[f"probe/memory/{mem_key}"] = mem_val

            throughput = bm.get("throughput_samples_per_sec", {})
            if isinstance(throughput, dict):
                for tp_key, tp_val in throughput.items():  # pyright: ignore[reportUnknownVariableType]
                    log_dict[f"probe/throughput_samples_per_sec/{tp_key}"] = tp_val

            entries.append(log_dict)

        return entries

    def cleanup(self) -> None:
        """Release resources. No persistent hooks after benchmark_single_config returns."""
        pass
