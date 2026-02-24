"""GuardrailMetricsConnector Protocol — the core abstraction for pluggable guardrail benchmarking.

Each guardrail architecture (activation probes, LLM classifiers, future types)
implements this protocol.  The metrics runner calls these methods without knowing
the architecture details, enabling open/closed extensibility.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import torch

    from pyine.apps.guardrail_metrics.guardrail_metrics_configs import GuardrailMetricsAppMainConfig


class GuardrailMetricsConnector(typing.Protocol):
    """Protocol for guardrail types to plug into the metrics runner.

    Each guardrail architecture (probes, LLM classifier, future types)
    implements this protocol.  The metrics runner calls these methods
    without knowing the architecture details.

    Key design: each connector owns the measurement lifecycle for a single
    ``(batch_size, seq_length)`` config via ``benchmark_single_config()``.
    This allows architecture-specific coordination (e.g., probes need to
    install/remove hooks, compute derived metrics like hook overhead, and
    share captured activations across sub-measurements).  The generic runner
    only handles the sweep over configs, OOM resilience, and output —
    never architecture-specific measurement orchestration.
    """

    @property
    def guardrail_type(self) -> str:
        """Short identifier for the guardrail type (e.g., ``'probe'``, ``'classifier'``)."""
        ...

    def load(self, device: torch.device) -> None:
        """Load model(s) and checkpoint(s) onto the given device.

        Called once before benchmarking starts.  Implementations should:

        - Load the model(s) from checkpoint paths in the config
        - Move model(s) to device
        - Set model(s) to eval mode
        - Optionally apply ``torch.compile()`` if requested
        """
        ...

    def get_static_info(self) -> dict[str, typing.Any]:
        """Return static info about the loaded model(s).

        E.g., model name, total parameter count, parameter memory bytes.
        This is called once after ``load()`` and included in the output metadata.
        """
        ...

    def benchmark_single_config(
        self,
        *,
        batch_size: int,
        seq_length: int,
        config: GuardrailMetricsAppMainConfig,
        device: torch.device,
    ) -> dict[str, typing.Any]:
        """Run all measurements for one ``(batch_size, seq_length)`` config.

        This method owns the full measurement lifecycle, allowing
        architecture-specific coordination.  For example, probes need to:

        - Measure LLM FLOPs without hooks (baseline), then with hooks
        - Compute ``hook_overhead = hooked - baseline`` (derived metric)
        - Capture activations from the hooked pass for per-probe FLOPs
        - Install/remove hooks between latency measurements

        Implementations should call ``measure_flops()``,
        ``measure_memory()``, and ``measure_latency()`` from
        ``metrics_collector`` as needed.  The returned dict has
        architecture-specific keys (e.g., ``probe_flops``,
        ``hook_overhead_flops`` for probes; ``flops``, ``latency``
        for classifiers).

        Returns a dict that always includes:

        - ``batch_size``: int
        - ``seq_length``: int
        - ``status``: ``"ok"`` or ``"OOM"``

        Plus architecture-specific measurement results.
        """
        ...

    def flatten_for_csv(
        self,
        static_info: dict[str, typing.Any],
        benchmarks: list[dict[str, typing.Any]],
    ) -> list[dict[str, typing.Any]]:
        """Flatten benchmark results into rows for CSV output.

        Each row should use the common CSV fieldnames:
        ``classifier_type``, ``model_name``, ``probe_name``,
        ``architecture``, ``layer``, ``batch_size``, ``seq_length``,
        ``status``, ``param_count``, ``flops``, ``mean_latency_ms``,
        ``p99_latency_ms``, ``throughput_samples_per_sec``,
        ``peak_memory_bytes``.

        This moves the architecture-specific result-to-row mapping
        (currently hardcoded in ``_write_csv()``) into each connector.
        """
        ...

    def get_wandb_summary(
        self,
        static_info: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Return WandB summary entries for static model info.

        Keys should be prefixed with the guardrail type (e.g.,
        ``probe/mean_L16/architecture``, ``classifier_model_name``).
        """
        ...

    def get_wandb_log_entries(
        self,
        benchmarks: list[dict[str, typing.Any]],
    ) -> list[dict[str, typing.Any]]:
        """Convert per-config benchmark results to WandB log dicts.

        Each entry in the returned list corresponds to one
        ``(batch_size, seq_length)`` config and will be logged as one
        WandB step.  Keys should be prefixed with the guardrail type.
        """
        ...

    def cleanup(self) -> None:
        """Release resources (hooks, model references, etc.)."""
        ...
