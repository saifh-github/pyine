"""Hydra-zen config builder for probe_trainer.py app."""

from __future__ import annotations

import logging
import typing

import pydantic
import torch

import pyine.apps.trainers.common as common
import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.configs.utils
import pyine.evals.common
import pyine.evals.configs
import pyine.probes.base  # noqa: TC001
import pyine.probes.data.datamodule_configs  # noqa: TC001
import pyine.utils.reprod

logger = logging.getLogger(__name__)


class ProbeTrainerAppMainConfig(common.AppMainConfig, common.ModelTokenizerConfigBase):
    """Configuration for probe training on frozen LLM activations."""

    # --- Override: optional correctness benchmarking after probe training ---
    evals_config: pyine.evals.common.BaseEvalsConfig | None = None  # type: ignore[assignment]
    """Optional correctness eval config for post-training benchmarking.

    When set (e.g. via ``+evals_config=correctness_base`` on the hydra command line), the trained
    probes are evaluated as guardrail scorers on the correctness pipeline after training completes.
    """

    # --- Override: use ProbeDataModuleConfig instead of generic BaseDataModuleConfig ---
    datamodule_config: pydantic.SerializeAsAny[pyine.probes.data.datamodule_configs.ProbeDataModuleConfig] = ...  # type: ignore[assignment]
    """Probe data configuration (LMDB source, splitting, filtering)."""

    # --- LLM checkpoint ---
    llm_checkpoint_path: str | None = None
    """Path to model checkpoint. If None, uses base_model directly."""

    # --- Probe configurations ---
    probe_configs: list[pyine.probes.base.ProbeConfig]
    """List of probe configs, each specifying architecture, layer, and hyperparams."""

    # --- Training/logging options (not data-related, stay here) ---
    log_per_code_type_metrics: bool = True
    """Log per-code-type validation metrics (loss, AUROC) to W&B.
    Requires code_type column in the dataset (always present)."""

    # --- Training loop ---
    num_epochs: int = 10
    train_batch_size: int = 4
    eval_batch_size: int = 8
    max_seq_length: int = 20000
    """Maximum sequence length for tokenization."""
    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps before optimizer update."""
    logging_steps: int = 10
    eval_steps: int = -1
    """Run validation every N optimizer steps. -1 = only at epoch end."""
    dataloader_num_workers: int = 4

    # --- Output ---
    save_probes: bool = True
    """Whether to save trained probe weights at the end of training."""

    # --- Replica settings ---
    num_replicas: int = pydantic.Field(default=1, ge=1)
    """Number of replicas per probe config. Each replica is initialized with a
    different random seed. Metrics are aggregated (mean/std) across replicas.
    Default 1 = no replication."""
    replica_base_seed: int = 0
    """Base seed for deterministic replica initialization."""
    log_individual_replicas: bool = False
    """When true, also log per-replica metrics to W&B (in addition to
    aggregated mean/std). Useful for debugging but adds many metrics."""

    @property
    @typing.override
    def target_dtype(self) -> torch.dtype:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    @pydantic.model_validator(mode="after")
    def _validate_probe_names_unique(self) -> ProbeTrainerAppMainConfig:
        names = [probe_config.name for probe_config in self.probe_configs]
        if len(names) != len(set(names)):
            raise ValueError(f"Probe names must be unique. Got duplicates in: {names}")
        return self


def _get_app_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns probe trainer application configs for hydra zen storage."""
    # --- Probe datamodule config ---
    datamodule_config = pyine.configs.utils.make_config_description(
        pyine.probes.data.datamodule_configs.ProbeDataModuleConfig,
        name="probe_base",
        group=f"{group}/datamodule_config",
        description="Base probe datamodule settings (LMDB source, splitting, filtering).",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )

    # --- Correctness eval configs (registered so user can opt-in via hydra) ---
    evals_configs = pyine.evals.configs.get_evals_configs(
        eval_type=pyine.evals.common.EvalType.CORRECTNESS,
        group=f"{group}/evals_config",
    )

    # --- Main app config ---
    app_main_config = pyine.configs.utils.make_config_description(
        ProbeTrainerAppMainConfig,
        name="base",
        group=group,
        description="Base settings for the probe trainer app.",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
            "hydra_defaults": [
                "_self_",
                {"datamodule_config": "probe_base"},
            ],
        },
    )
    return [app_main_config, datamodule_config, *evals_configs]


def register_hydra_configs(
    eval_type: pyine.evals.common.EvalType,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Registers probe-trainer-specific configs in hydra and returns config descriptions."""
    # import here to avoid circular imports at module load time
    from pyine.apps.trainers.probe_trainer import async_probe_trainer_main_wrapper

    pyine.utils.reprod.load_dotenv()

    entrypoint_config = pyine.configs.utils.make_config_description(
        async_probe_trainer_main_wrapper,
        name="entrypoint",
        group=None,
        description="Entrypoint settings for the probe trainer app.",
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

    store, base_configs = pyine.configs.base.get_base_store_and_configs("probe_trainer")
    app_configs = _get_app_configs(group="config")
    configs_to_register = [entrypoint_config, *app_configs]

    # pick up external experiment YAMLs
    external_configs = pyine.configs.searchpath.SearchPathPlugin.get_external_configs(
        app_name="probe_trainer",
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
