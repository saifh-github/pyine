"""Hydra-zen config builder for llm_classifier_trainer.py app."""

from __future__ import annotations

import logging
import typing

import pydantic
import torch
import transformers

if typing.TYPE_CHECKING:
    import pathlib

import pyine.apps.trainers.common as common
import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.configs.utils
import pyine.evals.common
import pyine.evals.configs
import pyine.probes.data.datamodule_configs  # noqa: TC001
import pyine.utils.reprod
import pyine.utils.transformers

logger = logging.getLogger(__name__)


class LLMClassifierTrainerAppMainConfig(common.AppMainConfig, common.ModelTokenizerConfigBase):
    """Configuration for fine-tuning an encoder LLM as a binary classifier.

    Inherits from ModelTokenizerConfigBase for model/tokenizer fields (base_model,
    auto_model_config, lora_config, etc.) and overrides get_model() to use
    AutoModelForSequenceClassification instead of AutoModelForCausalLM.

    Embeds ProbeDataModuleConfig for LMDB data loading — same config used by the
    probe trainer. All LMDB fields (lmdb_path, label_metric_key, split config,
    code_type_filter, label_balance, etc.) live in datamodule_config.
    """

    # --- Override: optional correctness benchmarking after classifier training ---
    evals_config: pyine.evals.common.BaseEvalsConfig | None = None  # type: ignore[assignment]
    """Optional correctness eval config for post-training benchmarking.

    When set (e.g. via ``+evals_config=correctness_base`` on the hydra command line), the trained
    classifier is evaluated as a guardrail scorer on the correctness pipeline after training completes.
    """

    # --- Override: use ProbeDataModuleConfig (same as probe trainer) ---
    datamodule_config: pydantic.SerializeAsAny[pyine.probes.data.datamodule_configs.ProbeDataModuleConfig] = ...  # type: ignore[assignment]
    """Probe data configuration (LMDB source, splitting, filtering, label balancing)."""

    # --- HuggingFace Training Arguments ---
    training_args_config: pydantic.SerializeAsAny[pyine.utils.transformers.TrainingArgsConfig] = ...  # type: ignore[assignment]
    """HuggingFace TrainingArguments configuration."""

    # --- Encoder-appropriate tokenizer defaults ---
    # Override causal-LM defaults from ModelTokenizerConfigBase:
    # Encoder models typically truncate on the right (drop end tokens, preserving
    # [CLS] at the start) and pad on the right. The causal-LM defaults (left
    # truncation, right padding) are inappropriate for encoder classification.
    tokenizer_override_truncation_to_left_side: bool = False
    """Disabled for encoder models. Encoder tokenizers should truncate right (default)."""

    # --- Code type metrics ---
    log_per_code_type_metrics: bool = True

    # --- Tokenization ---
    max_seq_length: int = 3000
    """Maximum sequence length for tokenization. Should not exceed model's max position embeddings."""
    truncation_side: typing.Literal["right", "left"] | None = None
    """Which side to truncate when sequences exceed max_seq_length.

    None = use tokenizer default (usually 'right'). 'left' preserves the completion
    (at the end) at the cost of dropping prompt tokens. This is applied after
    get_tokenizer() and overrides its settings.
    """

    # --- Output ---
    save_model: bool = True
    """Whether to save the trained model at the end of training."""

    # --- Class imbalance ---
    class_weight_mode: typing.Literal["none", "balanced"] = "none"
    """How to handle class imbalance in the loss function.

    'none' = standard CrossEntropyLoss. 'balanced' = compute class weights inversely
    proportional to class frequency and pass them to CrossEntropyLoss via a custom
    Trainer. Label distribution is always logged regardless of this setting.
    Note: this is separate from label balancing via datamodule_config.label_balance
    (which resamples the data). Using both simultaneously is usually undesirable.
    """

    # --- Classification model ---
    num_labels: int = 2
    """Number of classification labels. Default 2 for binary classification."""
    id2label: dict[int, str] = pydantic.Field(default_factory=lambda: {0: "incorrect", 1: "correct"})
    """Mapping from label ID to human-readable name."""
    label2id: dict[str, int] = pydantic.Field(default_factory=lambda: {"incorrect": 0, "correct": 1})
    """Mapping from human-readable name to label ID."""

    @property
    @typing.override
    def target_dtype(self) -> torch.dtype:
        """Derive dtype from training_args_config (matching SFT trainer pattern)."""
        if self.training_args_config.bf16:
            return torch.bfloat16
        if self.training_args_config.fp16:
            return torch.float16
        return torch.float32

    @typing.override
    def get_model(
        self,
        checkpoint_path: pathlib.Path | None = None,
    ) -> transformers.PreTrainedModel:
        """Load an AutoModelForSequenceClassification (not AutoModelForCausalLM).

        Overrides the base class to use the correct AutoModel class for encoder
        classification. Handles attention implementation fallback and LoRA.
        """
        resolved_config = common.resolve_attn_implementation(self.auto_model_config)
        model_kwargs: dict[str, typing.Any] = {
            "torch_dtype": self.target_dtype,
            "device_map": self.device_map,
            "num_labels": self.num_labels,
            "id2label": self.id2label,
            "label2id": self.label2id,
            **resolved_config,
        }
        model_path = checkpoint_path if checkpoint_path is not None else self.base_model
        model: transformers.PreTrainedModel = transformers.AutoModelForSequenceClassification.from_pretrained(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # transformers stubs
            model_path,
            **model_kwargs,
        )
        if self.lora_config is not None and checkpoint_path is None:
            import peft

            if isinstance(self.lora_config, pyine.utils.transformers.LoraConfig):
                lora_peft_config = self.lora_config.to_peft_config()
            else:
                lora_peft_config = self.lora_config
            model = peft.get_peft_model(model, lora_peft_config)  # type: ignore[assignment]  # pyright: ignore[reportUnknownVariableType]
        return model  # pyright: ignore[reportUnknownVariableType]  # peft stubs

    # --- Validators ---
    @pydantic.model_validator(mode="after")
    def _validate_label_mappings(self) -> LLMClassifierTrainerAppMainConfig:
        if len(self.id2label) != self.num_labels:
            raise ValueError(f"id2label has {len(self.id2label)} entries but num_labels={self.num_labels}")
        if len(self.label2id) != self.num_labels:
            raise ValueError(f"label2id has {len(self.label2id)} entries but num_labels={self.num_labels}")
        return self

    @pydantic.model_validator(mode="after")
    def _validate_class_weight_with_label_balance(self) -> LLMClassifierTrainerAppMainConfig:
        if self.class_weight_mode == "balanced" and self.datamodule_config.label_balance is not None:
            import warnings

            warnings.warn(
                "Both class_weight_mode='balanced' and datamodule_config.label_balance "
                "are active. This applies double correction for class imbalance "
                "(resampling + weighted loss). This is usually undesirable — consider "
                "using only one.",
                stacklevel=2,
            )
        return self


def _get_training_args_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generate training args configs with classifier-appropriate defaults."""
    hw_flags = common.get_hardware_training_flags()
    base = pyine.configs.utils.make_config_description(
        pyine.utils.transformers.TrainingArgsConfig,
        name="base",
        group=group,
        description=(
            "Base training arguments for classifier trainer configs; auto-determines some "
            "arguments based on available hardware."
        ),
        config={
            **hw_flags,
            "run_name": "${runtime.exp_name}-${runtime.run_name}",
            "output_dir": "${hydra:runtime.output_dir}",
            "logging_dir": "${hydra:runtime.output_dir}/logs",
            "seed": "${runtime.seed}",
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )

    train_default = pyine.configs.utils.make_config_description(
        pyine.utils.transformers.TrainingArgsConfig,
        name="train_default",
        group=group,
        description=(
            "Default training arguments for classifier training; applies on top of the `base` "
            "arguments config with reasonable defaults for encoder classification."
        ),
        config={
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 32,
            "num_train_epochs": 3,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "logging_steps": 10,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "auroc",
            "greater_is_better": True,
            "report_to": "none",
            "builds_bases": (base.config,),
        },
    )
    return [base, train_default]


def _get_app_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns classifier trainer app configs for hydra zen storage."""
    # --- Probe datamodule config (reused from probe trainer) ---
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
        LLMClassifierTrainerAppMainConfig,
        name="base",
        group=group,
        description="Base settings for the LLM classifier trainer app.",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
            "hydra_defaults": [
                "_self_",
                {"datamodule_config": "probe_base"},
                {"training_args_config": "base"},
            ],
        },
    )
    return [app_main_config, datamodule_config, *evals_configs]


def register_hydra_configs(
    eval_type: pyine.evals.common.EvalType,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Registers classifier-trainer-specific configs in hydra."""
    from pyine.apps.trainers.llm_classifier_trainer import (
        async_classifier_trainer_main_wrapper,
    )

    pyine.utils.reprod.load_dotenv()

    entrypoint_config = pyine.configs.utils.make_config_description(
        async_classifier_trainer_main_wrapper,
        name="entrypoint",
        group=None,
        description="Entrypoint settings for the LLM classifier trainer app.",
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

    store, base_configs = pyine.configs.base.get_base_store_and_configs("llm_classifier_trainer")
    app_configs = _get_app_configs(group="config")
    training_args_configs = _get_training_args_configs(group="config/training_args_config")
    configs_to_register = [entrypoint_config, *app_configs, *training_args_configs]

    external_configs = pyine.configs.searchpath.SearchPathPlugin.get_external_configs(
        app_name="llm_classifier_trainer",
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
