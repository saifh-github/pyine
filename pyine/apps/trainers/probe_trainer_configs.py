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
import pyine.probes.base  # noqa: TC001
import pyine.probes.reward_keys
import pyine.utils.reprod

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
    probe_configs: list[pyine.probes.base.ProbeConfig] = pydantic.Field(
        ...,
        description="List of probe configs, each specifying architecture, layer, and hyperparams.",
    )

    # --- LMDB data source ---
    lmdb_path: str = pydantic.Field(
        ...,
        description="Path to LMDB database exported by DiskRewardLogger.",
    )
    label_metric_key: str = pydantic.Field(
        default=pyine.probes.reward_keys.SOFT_MATCH_KEY,
        description=(
            "Key in reward_metrics dict for binary label derivation. "
            f"Common values: '{pyine.probes.reward_keys.SOFT_MATCH_KEY}', '{pyine.probes.reward_keys.HARD_MATCH_KEY}'."
        ),
    )
    train_key_prefix: str = pydantic.Field(
        default="train/",
        description="LMDB key prefix for training records.",
    )
    valid_key_prefix: str = pydantic.Field(
        default="eval/",
        description="LMDB key prefix for validation records.",
    )
    selection_strategy: typing.Literal["latest", "best_reward"] = pydantic.Field(
        default="latest",
        description=(
            "Strategy for deduplicating multiple generations per sample. "
            "'latest' uses highest generation_count, 'best_reward' uses highest reward_total."
        ),
    )
    recompute_labels: bool = pydantic.Field(
        default=False,
        description=(
            "If True, re-compute labels instead of using stored reward_metrics. "
            f"Only valid when label_metric_key is '{pyine.probes.reward_keys.SOFT_MATCH_KEY}' or "
            f"'{pyine.probes.reward_keys.HARD_MATCH_KEY}' -- validated at config construction time."
        ),
    )
    max_samples_per_split: int | None = pydantic.Field(
        default=None,
        description="Cap samples per split. Useful for debugging or fast iteration.",
    )
    skip_malformed_records: bool = pydantic.Field(
        default=False,
        description=(
            "If True, skip records missing required fields instead of raising. "
            "Skipped records are counted and logged at WARNING level. "
            "If False (default), raise ValueError on any malformed record."
        ),
    )

    # --- Eval-only split mode ---
    use_eval_only_split: bool = pydantic.Field(
        default=False,
        description=(
            "When True, read data from a single LMDB prefix (eval_only_source_prefix) "
            "and split internally into train/valid. When False (default), use separate "
            "train_key_prefix and valid_key_prefix as before."
        ),
    )
    eval_only_source_prefix: str = pydantic.Field(
        default="eval/",
        description="LMDB key prefix to read from when use_eval_only_split=True.",
    )
    train_split_ratio: float = pydantic.Field(
        default=0.8,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of data used for training when use_eval_only_split=True. Remainder is used for validation."
        ),
    )
    split_by_family: bool = pydantic.Field(
        default=True,
        description=(
            "When True (default), split by family (problem) ID so that all code-type "
            "variants of the same problem go to the same split. Prevents data leakage "
            "from shared problem structure. When False, split randomly at the record level."
        ),
    )

    # --- Code type filtering and metrics ---
    code_type_filter: list[str] | None = pydantic.Field(
        default=None,
        description=(
            "If set, only include records with code_type matching one of the listed values. "
            "Example: ['original', 'hinted', 'misleading']. "
            "None (default) includes all records."
        ),
    )
    log_per_code_type_metrics: bool = pydantic.Field(
        default=True,
        description=(
            "Log per-code-type validation metrics (loss, AUROC) to W&B. "
            "Requires code_type column in the dataset (always present)."
        ),
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

    # --- Replica settings ---
    num_replicas: int = pydantic.Field(
        default=1,
        ge=1,
        description=(
            "Number of replicas per probe config. Each replica is initialized with a "
            "different random seed. Metrics are aggregated (mean/std) across replicas. "
            "Default 1 = no replication."
        ),
    )
    replica_base_seed: int = pydantic.Field(
        default=0,
        description="Base seed for deterministic replica initialization.",
    )
    log_individual_replicas: bool = pydantic.Field(
        default=False,
        description=(
            "When true, also log per-replica metrics to W&B (in addition to "
            "aggregated mean/std). Useful for debugging but adds many metrics."
        ),
    )

    @property
    @typing.override
    def target_dtype(self) -> torch.dtype:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    @pydantic.model_validator(mode="after")
    def _validate_recompute_label_metric(self) -> ProbeTrainerAppMainConfig:
        recomputable = {pyine.probes.reward_keys.SOFT_MATCH_KEY, pyine.probes.reward_keys.HARD_MATCH_KEY}
        if self.recompute_labels and self.label_metric_key not in recomputable:
            raise ValueError(
                f"recompute_labels=True is only supported for label_metric_key in "
                f"{recomputable}, got '{self.label_metric_key}'"
            )
        return self

    @pydantic.model_validator(mode="after")
    def _validate_eval_only_split_config(self) -> ProbeTrainerAppMainConfig:
        if self.use_eval_only_split:
            if not self.eval_only_source_prefix:
                raise ValueError("eval_only_source_prefix must be non-empty when use_eval_only_split=True")
        if self.code_type_filter is not None and len(self.code_type_filter) == 0:
            raise ValueError("code_type_filter must be None (include all) or a non-empty list; got an empty list")
        return self

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
