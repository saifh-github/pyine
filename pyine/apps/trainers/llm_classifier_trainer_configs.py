"""Hydra-zen config builder for llm_classifier_trainer.py app."""

from __future__ import annotations

import logging
import pathlib  # noqa: TC003
import typing

import peft
import pydantic
import torch
import transformers

import pyine.apps.trainers.common as common
import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.configs.utils
import pyine.evals.common
import pyine.evals.correctness.datamodule_configs  # noqa: TC001
import pyine.guardrails.data.datamodule_configs  # noqa: TC001
import pyine.utils.reprod
import pyine.utils.transformers

logger = logging.getLogger(__name__)


class LLMClassifierTrainerAppMainConfig(common.AppMainConfig, common.ModelTokenizerConfigBase):
    """Configuration for fine-tuning an encoder LLM as a binary classifier.

    Inherits from ModelTokenizerConfigBase for model/tokenizer fields (base_model,
    auto_model_config, lora_config, etc.) and overrides get_model() to use
    AutoModelForSequenceClassification instead of AutoModelForCausalLM.

    Embeds ProbeDataModuleConfig (or CorrectnessDataModuleConfig) for LMDB data loading. Data is
    loaded as structured message lists; the trainer auto-detects chat template support on the
    tokenizer. If a ``chat_template`` is defined (chat-tuned models), it is applied; otherwise
    (encoder models like ModernBERT, DeBERTa), messages are concatenated into role-tagged plain text.
    """

    # --- Override: optional correctness benchmarking after classifier training ---
    evals_config: pyine.evals.common.BaseEvalsConfig | None = None  # type: ignore[assignment]
    """Optional correctness eval config for post-training benchmarking.

    When set, the trained classifier is evaluated as a guardrail scorer on the correctness pipeline
    after training completes.
    """

    # --- Override: accept ProbeDataModuleConfig or CorrectnessDataModuleConfig ---
    datamodule_config: pydantic.SerializeAsAny[  # pyright: ignore[reportIncompatibleVariableOverride]
        pyine.guardrails.data.datamodule_configs.ProbeDataModuleConfig
        | pyine.evals.correctness.datamodule_configs.CorrectnessDataModuleConfig
    ] = ...  # type: ignore[assignment]
    """Data configuration (LMDB source, splitting, filtering, label balancing).

    Accepts ProbeDataModuleConfig or CorrectnessDataModuleConfig.
    """

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
    max_seq_length: pydantic.PositiveInt = pyine.evals.common.PASS_AT_K_DEFAULTS.max_new_tokens
    """Maximum sequence length for tokenization. Should not exceed model's max position embeddings."""
    text_field: str = "model_output"
    """Which ``EvalRecord`` field to use as the assistant output when building model inputs.

    This is only used with ``CorrectnessDataModuleConfig``. Probe LMDB datasets already contain
    structured messages and ignore this setting.
    """
    truncation_side: typing.Literal["right", "left"] | None = None
    """Which side to truncate when sequences exceed max_seq_length.

    None = use tokenizer default (usually 'right'). 'left' preserves the completion
    (at the end) at the cost of dropping prompt tokens. This is applied after
    get_tokenizer() and overrides its settings.
    """

    # --- Checkpoint loading (eval-only mode) ---
    classifier_checkpoint_path: pathlib.Path | None = None
    """Path to a saved classifier checkpoint directory (as written by ``Trainer.save_model()``).

    Used in eval-only mode (``skip_training=True``) to load a pretrained model without re-training.
    """

    # --- Output ---
    save_model: bool = True
    """Whether to save the trained model at the end of training."""
    save_best_model_export: bool = True
    """Whether to also export the best loaded model to an explicit directory under the run output dir."""
    best_model_output_suffix: str = "/checkpoint-best"
    """Relative export path under ``training_args_config.output_dir`` for the best-model export.

    A leading path separator is ignored for convenience, so ``/checkpoint-best`` resolves to
    ``<output_dir>/checkpoint-best``.
    """

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
        if checkpoint_path is not None:
            adapter_config_path = checkpoint_path / "adapter_config.json"
            if adapter_config_path.exists():
                logger.debug(f"loading PEFT adapter classifier checkpoint from: {checkpoint_path}")
                model = typing.cast(
                    "transformers.PreTrainedModel",
                    peft.AutoPeftModelForSequenceClassification.from_pretrained(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                        checkpoint_path,
                        **model_kwargs,
                    ),
                )
            else:
                logger.debug(f"loading full classifier checkpoint from: {checkpoint_path}")
                model = transformers.AutoModelForSequenceClassification.from_pretrained(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # transformers stubs
                    checkpoint_path,
                    **model_kwargs,
                )
        else:
            logger.debug(f"loading base sequence-classification model from: {self.base_model}")
            model = transformers.AutoModelForSequenceClassification.from_pretrained(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # transformers stubs
                self.base_model,
                **model_kwargs,
            )
        model_config = typing.cast("typing.Any", model.config)  # pyright: ignore[reportUnknownMemberType]
        if (
            getattr(model_config, "pad_token_id", None) is None
            and self.tokenizer_set_padding_to_eos_if_needed
            and getattr(model_config, "eos_token_id", None) is not None
        ):
            # keep standalone get_model() reasonably safe for decoder-only classifiers: when the
            # tokenizer logic will reuse EOS as PAD, mirror that on the model config too. The
            # trainer path still performs an explicit tokenizer->model sync after loading both.
            model.config.pad_token_id = model.config.eos_token_id  # type: ignore[reportUnknownMemberType]
        if self.lora_config is not None and checkpoint_path is None:
            if isinstance(self.lora_config, pyine.utils.transformers.LoraConfig):
                lora_peft_config = self.lora_config.to_peft_config()
            else:
                lora_peft_config = self.lora_config
            logger.debug(f"applying LoRA adapters to classifier model loaded from: {self.base_model}")
            model = peft.get_peft_model(model, lora_peft_config)  # type: ignore[assignment]  # pyright: ignore[reportUnknownVariableType]
        return typing.cast("transformers.PreTrainedModel", model)  # pyright: ignore[reportUnknownVariableType]

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
        label_balance = getattr(self.datamodule_config, "label_balance", None)
        if self.class_weight_mode == "balanced" and label_balance is not None:
            import warnings

            warnings.warn(
                "Both class_weight_mode='balanced' and datamodule_config.label_balance "
                "are active. This applies double correction for class imbalance "
                "(resampling + weighted loss). This is usually undesirable - consider "
                "using only one.",
                stacklevel=2,
            )
        return self

    @pydantic.model_validator(mode="after")
    def _validate_text_field_matches_evals_config(self) -> LLMClassifierTrainerAppMainConfig:
        evals_text_field = getattr(self.evals_config, "text_field", None)
        if evals_text_field is not None and evals_text_field != self.text_field:
            raise ValueError(
                f"config.text_field={self.text_field!r} does not match "
                f"config.evals_config.text_field={evals_text_field!r}; "
                "use the same text field for training and correctness evaluation"
            )
        return self

    @pydantic.model_validator(mode="after")
    def _validate_best_model_export_settings(self) -> LLMClassifierTrainerAppMainConfig:
        if self.save_best_model_export and not self.save_model:
            raise ValueError("save_best_model_export=True requires save_model=True")
        if not self.best_model_output_suffix:
            raise ValueError("best_model_output_suffix must be non-empty")
        normalized_output_suffix = self.best_model_output_suffix.lstrip("/\\")
        if not normalized_output_suffix:
            raise ValueError("best_model_output_suffix must not be empty or root-only")
        if any(part == ".." for part in pathlib.Path(normalized_output_suffix).parts):
            raise ValueError("best_model_output_suffix must stay within output_dir")
        if self.save_best_model_export:

            def _normalize_interval_strategy(
                value: typing.Any,
            ) -> str:
                return str(getattr(value, "value", value)).lower()

            eval_strategy = _normalize_interval_strategy(self.training_args_config.eval_strategy)
            save_strategy = _normalize_interval_strategy(self.training_args_config.save_strategy)
            if eval_strategy == "no":
                raise ValueError(
                    "save_best_model_export=True requires training_args_config.eval_strategy != 'no' "
                    "so a best checkpoint can be selected"
                )
            if save_strategy == "no":
                raise ValueError(
                    "save_best_model_export=True requires training_args_config.save_strategy != 'no' "
                    "so a best checkpoint can be materialized"
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
    support_configs = common.get_probe_and_correctness_support_configs(group=group)
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
    return [app_main_config, *support_configs]


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
            "skip_training": False,
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


if __name__ == "__main__":
    import sys

    pyine.configs.base.register_searchpath_plugin()
    pyine.configs.utils.print_experiment_configs(
        config_descriptions=register_hydra_configs(eval_type=pyine.evals.common.EvalType.CORRECTNESS),
        app_name="llm_classifier_trainer",
        cli_args=sys.argv[1:],
    )
