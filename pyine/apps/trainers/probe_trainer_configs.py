"""Hydra-zen config builder for probe_trainer.py app."""

from __future__ import annotations

import logging
import pathlib  # noqa: TC003
import typing
import warnings

import pydantic
import torch

import pyine.apps.trainers.common as common
import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.configs.utils
import pyine.evals.common
import pyine.evals.correctness.datamodule_configs  # noqa: TC001
import pyine.guardrails.data.datamodule_configs  # noqa: TC001
import pyine.guardrails.probes.base  # noqa: TC001
import pyine.utils.reprod

logger = logging.getLogger(__name__)


class ProbeTrainerAppMainConfig(common.AppMainConfig, common.ModelTokenizerConfigBase):
    """Configuration for probe training on frozen LLM activations."""

    # --- Override: optional correctness benchmarking after probe training ---
    evals_config: pyine.evals.common.BaseEvalsConfig | None = None  # type: ignore[assignment]
    """Optional correctness eval config for post-training benchmarking.

    When set, the trained probes are evaluated as guardrail scorers on the correctness pipeline
    after training completes.
    """

    # --- Override: accept ProbeDataModuleConfig or CorrectnessDataModuleConfig ---
    datamodule_config: pydantic.SerializeAsAny[  # pyright: ignore[reportIncompatibleVariableOverride]
        pyine.guardrails.data.datamodule_configs.ProbeDataModuleConfig
        | pyine.evals.correctness.datamodule_configs.CorrectnessDataModuleConfig
    ] = ...  # type: ignore[assignment]
    """Data configuration (LMDB source, splitting, filtering).

    Accepts ProbeDataModuleConfig or CorrectnessDataModuleConfig.
    """

    # --- LLM checkpoint ---
    llm_checkpoint_path: str | None = None
    """Path to model checkpoint. If None, uses base_model directly."""

    # --- Probe configurations ---
    probe_configs: list[pyine.guardrails.probes.base.ProbeConfig] = pydantic.Field(default_factory=lambda: [])
    """List of probe configs, each specifying architecture, layer, and hyperparams.

    May be empty when ``probe_checkpoint_dir`` is set (eval-only mode).
    """

    # --- Checkpoint loading (eval-only mode) ---
    probe_checkpoint_dir: pathlib.Path | None = None
    """Path to a directory of saved probe checkpoints (as written by ``save_probe_checkpoints``).

    Used in eval-only mode (``skip_training=True``) to load pretrained probes without re-training.
    """
    probe_checkpoint_name: str | None = None
    """Optional checkpoint subdirectory name to load for every probe (e.g. ``"best"`` or ``"final"``).

    When ``None``, each probe auto-resolves its checkpoint using the collection defaults.
    """

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
    truncation_side: typing.Literal["right", "left"] = "left"
    """Which side to truncate when sequences exceed max_seq_length.

    'left' preserves the assistant completion (end of sequence) which is typically
    the content probes need to classify. 'right' preserves the prompt prefix instead.
    """
    text_field: str = "model_output"
    """Which ``EvalRecord`` field to use as the assistant output when building model inputs.

    This is only used with ``CorrectnessDataModuleConfig``. Probe LMDB datasets already contain
    structured messages and ignore this setting.
    """
    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps before optimizer update."""
    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping. Set to 0 to disable. Matches default used in other trainer apps."""
    logging_steps: int = 10
    eval_steps: int = -1
    """Run validation every N optimizer steps. -1 = only at epoch end."""
    dataloader_num_workers: int = 4

    # --- Class imbalance ---
    class_weight_mode: typing.Literal["none", "balanced"] = "none"
    """How to handle class imbalance in the loss function.

    'none' = standard BCEWithLogitsLoss. 'balanced' = compute pos_weight inversely proportional to
    positive class frequency and pass it to BCEWithLogitsLoss. Label distribution is always logged
    regardless of this setting. Note: this is separate from label balancing via datamodule_config
    label_balance (which resamples the data). Using both simultaneously is usually undesirable.
    """

    # --- Output ---
    save_probes: bool = True
    """Whether to save trained probe weights at the end of training."""
    save_best_probe_checkpoint: bool = True
    """Whether to maintain a per-probe best checkpoint export during training."""
    best_probe_checkpoint_name: str = "best"
    """Checkpoint subdirectory name used for the per-probe best export."""
    best_probe_metric: typing.Literal["auroc", "loss"] = "auroc"
    """Validation metric used to decide whether a probe improved."""

    # --- Checkpoint saving ---
    save_steps: int = -1
    """Save probe checkpoints every N optimizer steps during training.
    -1 = only at the end of training (current behavior).
    Must be > 0 to enable mid-training saves. Follows the same convention
    as eval_steps."""

    save_total_limit: int | None = pydantic.Field(default=None, ge=1)
    """Maximum number of mid-training checkpoint directories to keep for probes.
    When exceeded, the oldest checkpoint (by step number) is deleted.
    None = keep all checkpoints (no limit). Does not count the final
    checkpoint saved at end-of-training."""

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
        if not self.probe_configs:
            return self  # empty is valid when using probe_checkpoint_dir
        names = [probe_config.name for probe_config in self.probe_configs]
        if len(names) != len(set(names)):
            raise ValueError(f"Probe names must be unique. Got duplicates in: {names}")
        return self

    @pydantic.model_validator(mode="after")
    def _validate_has_probes_or_checkpoint(self) -> ProbeTrainerAppMainConfig:
        if not self.probe_configs and self.probe_checkpoint_dir is None:
            raise ValueError(
                "Either probe_configs must be non-empty (training mode) or "
                "probe_checkpoint_dir must be set (eval-only mode)."
            )
        return self

    @pydantic.model_validator(mode="after")
    def _validate_save_steps(self) -> ProbeTrainerAppMainConfig:
        if self.save_steps > 0 and not self.save_probes:
            warnings.warn(
                "save_steps > 0 has no effect when save_probes=False",
                stacklevel=2,
            )
        return self

    @pydantic.model_validator(mode="after")
    def _validate_class_weight_with_label_balance(self) -> ProbeTrainerAppMainConfig:
        # note: only checks label_balance (ProbeDataModule / LMDB). The CorrectnessDataModule
        # uses 'resampling' for distribution shaping, which is not a label-balance mechanism
        # and does not cause double correction with class_weight_mode.
        label_balance = getattr(self.datamodule_config, "label_balance", None)
        if self.class_weight_mode == "balanced" and label_balance is not None:
            warnings.warn(
                "Both class_weight_mode='balanced' and datamodule_config.label_balance are active. "
                "This applies double correction for class imbalance (resampling + weighted loss). "
                "This is probably undesirable; consider using only one.",
                stacklevel=2,
            )
        return self

    @pydantic.model_validator(mode="after")
    def _validate_calibration_resampling_matches_training(self) -> ProbeTrainerAppMainConfig:
        if self.evals_config is not None:
            common.warn_on_calibration_resampling_mismatch(self.datamodule_config, self.evals_config)
        return self

    @pydantic.model_validator(mode="after")
    def _validate_text_field_matches_evals_config(self) -> ProbeTrainerAppMainConfig:
        evals_text_field = getattr(self.evals_config, "text_field", None)
        if evals_text_field is not None and evals_text_field != self.text_field:
            raise ValueError(
                f"config.text_field={self.text_field!r} does not match "
                f"config.evals_config.text_field={evals_text_field!r}; "
                "use the same text field for training and correctness evaluation"
            )
        return self

    @pydantic.model_validator(mode="after")
    def _validate_evals_require_best_checkpoint(self) -> ProbeTrainerAppMainConfig:
        if self.evals_config is not None and not self.save_best_probe_checkpoint:
            raise ValueError(
                "evals_config is set but save_best_probe_checkpoint=False; "
                "post-training evaluation requires best probe checkpoints to be saved "
                "so the best (not last) probes are used for benchmarking. "
                "Set save_best_probe_checkpoint=True or remove evals_config."
            )
        return self

    @pydantic.model_validator(mode="after")
    def _validate_best_checkpoint_settings(self) -> ProbeTrainerAppMainConfig:
        if self.save_best_probe_checkpoint and not self.save_probes:
            raise ValueError("save_best_probe_checkpoint=True requires save_probes=True")
        if self.save_best_probe_checkpoint and self.num_epochs < 1:
            raise ValueError("save_best_probe_checkpoint=True requires num_epochs >= 1")
        if not self.best_probe_checkpoint_name:
            raise ValueError("best_probe_checkpoint_name must be non-empty")
        if self.probe_checkpoint_name == "":
            raise ValueError("probe_checkpoint_name must be non-empty when provided")
        return self


def _get_app_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns probe trainer application configs for hydra zen storage."""
    support_configs = common.get_probe_and_correctness_support_configs(group=group)
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
    return [app_main_config, *support_configs]


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
            "skip_training": False,
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


if __name__ == "__main__":
    import sys

    pyine.configs.base.register_searchpath_plugin()
    pyine.configs.utils.print_experiment_configs(
        config_descriptions=register_hydra_configs(eval_type=pyine.evals.common.EvalType.CORRECTNESS),
        app_name="probe_trainer",
        cli_args=sys.argv[1:],
    )
