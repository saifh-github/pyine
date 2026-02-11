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
import pyine.utils.reprod
from pyine.probes.base import ProbeConfig  # noqa: TC001

logger = logging.getLogger(__name__)


class ProbeTrainerAppMainConfig(common.AppMainConfig, common.ModelTokenizerConfigBase):
    """Configuration for probe training on frozen LLM activations."""

    # --- Override: evals and datamodule not needed for probe training ---
    evals_config: pyine.evals.common.BaseEvalsConfig | None = pydantic.Field(  # type: ignore[assignment]
        default=None,
        description="Not used for probe training. Kept for AppMainConfig compatibility.",
    )
    datamodule_config: typing.Any = pydantic.Field(  # type: ignore[assignment]
        default=None,
        description="Not used for probe training. Kept for AppMainConfig compatibility.",
    )

    # --- LLM checkpoint ---
    llm_checkpoint_path: str | None = pydantic.Field(
        default=None,
        description="Path to model checkpoint. If None, uses base_model directly.",
    )

    # --- Probe configurations ---
    probe_configs: list[ProbeConfig] = pydantic.Field(
        ...,
        description="List of probe configs, each specifying architecture, layer, and hyperparams.",
    )

    # --- Dataset ---
    dataset_path: str = pydantic.Field(
        ...,
        description="HuggingFace dataset path (local dir or Hub name) with train/valid splits.",
    )
    text_field: str = pydantic.Field(
        default="messages",
        description="Column name for text input. 'messages' for chat format, or a plain text column.",
    )
    label_field: str = pydantic.Field(
        default="label",
        description="Column name for binary labels (0/1).",
    )

    # --- Training loop ---
    num_epochs: int = pydantic.Field(default=10)
    train_batch_size: int = pydantic.Field(default=4)
    eval_batch_size: int = pydantic.Field(default=8)
    max_seq_length: int = pydantic.Field(
        default=3000,
        description="Maximum sequence length for tokenization.",
    )
    gradient_accumulation_steps: int = pydantic.Field(
        default=1,
        description="Number of gradient accumulation steps before optimizer update.",
    )
    logging_steps: int = pydantic.Field(default=10)
    eval_steps: int = pydantic.Field(
        default=-1,
        description="Run validation every N optimizer steps. -1 = only at epoch end.",
    )
    dataloader_num_workers: int = pydantic.Field(default=4)

    # --- Output ---
    save_probes: bool = pydantic.Field(
        default=True,
        description="Whether to save trained probe weights at the end of training.",
    )

    @property
    @typing.override
    def target_dtype(self) -> torch.dtype:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    @pydantic.model_validator(mode="after")
    def _validate_probe_names_unique(self) -> ProbeTrainerAppMainConfig:
        names = [pc.name for pc in self.probe_configs]
        if len(names) != len(set(names)):
            raise ValueError(f"Probe names must be unique. Got duplicates in: {names}")
        return self


def _get_app_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns probe trainer application configs for hydra zen storage."""
    app_main_config = pyine.configs.utils.make_config_description(
        ProbeTrainerAppMainConfig,
        name="base",
        group=group,
        description="Base settings for the probe trainer app.",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    return [app_main_config]


def register_hydra_configs(
    eval_type: pyine.evals.common.EvalType,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Registers probe-trainer-specific configs in hydra and returns config descriptions."""
    # Import here to avoid circular imports at module load time
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

    # Pick up external experiment YAMLs
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
