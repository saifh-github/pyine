"""Hydra-zen config builder for the guardrail inference metrics app."""

from __future__ import annotations

import logging
import typing

import pydantic

import pyine.apps.trainers.common as common
import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.configs.utils
import pyine.evals.common
import pyine.guardrails.data.datamodule_configs  # noqa: TC001
import pyine.utils.reprod

logger = logging.getLogger(__name__)


class GuardrailMetricsAppMainConfig(common.AppMainConfig):
    """Configuration for guardrail inference metrics measurement.

    Unlike the training configs, this app does NOT inherit from
    ``ModelTokenizerConfigBase`` because it may load two different models
    (a frozen causal LLM for probes and an encoder for classification).
    Each model's settings are configured via separate field groups below,
    and loading reuses ``common.instantiate_model()`` /
    ``common.instantiate_tokenizer()`` directly with per-model arguments.
    """

    # --- Disable unused base fields ---
    evals_config: pyine.evals.common.BaseEvalsConfig | None = None  # type: ignore[assignment]

    # --- Data (optional, only for real-data mode) ---
    datamodule_config: (  # pyright: ignore[reportIncompatibleVariableOverride]
        pydantic.SerializeAsAny[  # type: ignore[assignment]
            pyine.guardrails.data.datamodule_configs.ProbeDataModuleConfig
        ]
        | None
    ) = None
    """Probe data configuration. Required only when ``use_synthetic_data=False``.
    When None and ``use_synthetic_data=True`` (default), synthetic tensors are used."""

    # --- Probe classifier settings ---
    probe_checkpoint_dir: str | None = None
    """Path to directory containing probe checkpoints (from probe_trainer).
    Each subdirectory should have ``probe_state_dict.pt`` and ``probe_config.json``.
    If None, probe benchmarking is skipped."""
    probe_checkpoint_name: str | None = None
    """Optional probe checkpoint subdirectory name to load for every probe (e.g. ``"best"``)."""

    probe_llm_model: str | None = None
    """HuggingFace model ID or local path for the frozen LLM used with probes.
    Required when ``probe_checkpoint_dir`` is set."""

    probe_llm_checkpoint_path: str | None = None
    """Optional checkpoint path for the frozen LLM (overrides ``probe_llm_model``)."""

    probe_auto_model_config: dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: {"use_cache": False, "attn_implementation": "flash_attention_2"},
    )
    """Model config for the frozen LLM. ``use_cache`` MUST be False for benchmarking
    (KV cache allocation distorts memory/latency measurements)."""

    # --- LLM classifier settings ---
    classifier_checkpoint_dir: str | None = None
    """Path to a saved HuggingFace model directory (from llm_classifier_trainer).
    If None, classifier benchmarking is skipped."""

    classifier_auto_model_config: dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: {"attn_implementation": "flash_attention_2"},
    )
    """Model config for the encoder classifier."""

    # --- Benchmarking settings ---
    num_warmup_iterations: int = 10
    """Number of warmup forward passes (not measured)."""

    num_benchmark_iterations: int = 100
    """Number of timed forward passes for latency/throughput statistics."""

    batch_sizes: list[int] = pydantic.Field(default_factory=lambda: [1, 4, 8, 16, 32])
    """Batch sizes to benchmark. Results reported per batch size.
    Sorted ascending at runtime to minimize OOM probability (smaller first)."""

    max_seq_length: int = 2048
    """Maximum sequence length for tokenization (real-data mode)."""

    use_synthetic_data: bool = True
    """If True, generate synthetic input tensors instead of loading real data.
    Faster and avoids LMDB dependency for pure compute benchmarks."""

    synthetic_seq_lengths: list[int] = pydantic.Field(default_factory=lambda: [512, 1024, 2048, 4096])
    """Sequence lengths to benchmark when ``use_synthetic_data=True``.
    Sorted ascending at runtime to minimize OOM probability (smaller first)."""

    use_torch_compile: bool = False
    """If True, apply ``torch.compile()`` to models before benchmarking.
    Useful for measuring compiled-model performance, but adds compilation time."""

    # --- Hardware settings ---
    device: str = "auto"
    """Device for benchmarking. ``'auto'`` uses CUDA > MPS > CPU auto-detection.
    Can override to ``'cuda'``, ``'cpu'``, or ``'mps'``."""

    # --- Output ---
    output_format: typing.Literal["json", "csv", "both"] = "both"
    """Format for the metrics report file."""

    measure_flops: bool = True
    measure_memory: bool = True
    measure_latency: bool = True
    measure_throughput: bool = True

    # --- Validators ---

    @pydantic.model_validator(mode="after")
    def _validate_probe_settings(self) -> GuardrailMetricsAppMainConfig:
        """Ensure ``probe_llm_model`` is set when probe benchmarking is requested."""
        if self.probe_checkpoint_dir is not None and self.probe_llm_model is None:
            raise ValueError(
                "probe_llm_model is required when probe_checkpoint_dir is set. "
                "Provide the HuggingFace model ID or local path for the frozen LLM."
            )
        return self

    @pydantic.model_validator(mode="after")
    def _validate_data_mode(self) -> GuardrailMetricsAppMainConfig:
        """Ensure ``datamodule_config`` is provided when real data is requested."""
        if not self.use_synthetic_data and self.datamodule_config is None:
            raise ValueError(
                "datamodule_config is required when use_synthetic_data=False. Provide LMDB path and data configuration."
            )
        return self

    @pydantic.model_validator(mode="after")
    def _validate_probe_checkpoint_name(self) -> GuardrailMetricsAppMainConfig:
        if self.probe_checkpoint_name == "":
            raise ValueError("probe_checkpoint_name must be non-empty when provided")
        return self

    @pydantic.model_validator(mode="after")
    def _validate_at_least_one_model(self) -> GuardrailMetricsAppMainConfig:
        """Ensure at least one classifier type is configured."""
        if self.probe_checkpoint_dir is None and self.classifier_checkpoint_dir is None:
            raise ValueError(
                "At least one of probe_checkpoint_dir or classifier_checkpoint_dir must be set. Nothing to benchmark."
            )
        return self

    @pydantic.model_validator(mode="after")
    def _validate_use_cache_false(self) -> GuardrailMetricsAppMainConfig:
        """Enforce ``use_cache=False`` for probe LLM (KV cache distorts benchmarks)."""
        if self.probe_checkpoint_dir is not None:
            if self.probe_auto_model_config.get("use_cache") is not False:
                raise ValueError(
                    "probe_auto_model_config.use_cache must be False for benchmarking. "
                    "KV cache allocation significantly affects memory and latency measurements."
                )
        return self


def _get_app_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns guardrail metrics application configs for hydra zen storage."""
    app_main_config = pyine.configs.utils.make_config_description(
        GuardrailMetricsAppMainConfig,
        name="base",
        group=group,
        description="Base settings for the guardrail metrics app.",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    return [app_main_config]


def register_hydra_configs(
    eval_type: pyine.evals.common.EvalType,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Registers guardrail-metrics-specific configs in hydra and returns config descriptions."""
    from pyine.apps.guardrail_metrics.guardrail_metrics import async_guardrail_metrics_main_wrapper

    pyine.utils.reprod.load_dotenv()

    entrypoint_config = pyine.configs.utils.make_config_description(
        async_guardrail_metrics_main_wrapper,
        name="entrypoint",
        group=None,
        description="Entrypoint settings for the guardrail metrics app.",
        config={
            "populate_full_signature": True,
            "hydra_defaults": [
                "_self_",
                {"config": "base"},
                {"runtime": "default"},
                *pyine.configs.base.get_base_hydra_default_overrides(),
            ],
        },
    )

    store, base_configs = pyine.configs.base.get_base_store_and_configs("guardrail_metrics")
    app_configs = _get_app_configs(group="config")
    configs_to_register = [entrypoint_config, *app_configs]

    # pick up external experiment YAMLs
    external_configs = pyine.configs.searchpath.SearchPathPlugin.get_external_configs(
        app_name="guardrail_metrics",
        eval_type=eval_type,
        entrypoint_config=entrypoint_config,
        app_configs=[*base_configs, *configs_to_register],
    )
    configs_to_register.extend(external_configs)

    for config in configs_to_register:
        assert config.name is not None
        store(
            typing.cast("typing.Any", config.config),
            name=config.name,
            group=config.group,
            package=config.package,
        )
    store.add_to_hydra_store(overwrite_ok=True)
    return [*base_configs, *configs_to_register]


if __name__ == "__main__":
    import sys

    pyine.configs.base.register_searchpath_plugin()
    pyine.configs.utils.print_experiment_configs(
        config_descriptions=register_hydra_configs(eval_type=pyine.evals.common.EvalType.CORRECTNESS),
        app_name="guardrail_metrics",
        cli_args=sys.argv[1:],
    )
