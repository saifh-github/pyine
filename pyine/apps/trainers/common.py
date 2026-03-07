"""Common trainer utilities and helper shared by all training apps."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import json
import logging
import os
import pathlib
import shutil
import sys
import time
import typing

import datasets as hf_datasets  # noqa: TC002
import deepdiff
import hydra_zen
import peft
import pydantic
import torch
import torch.distributed.elastic.multiprocessing.errors
import transformers

import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.utils
import pyine.data.datamodule
import pyine.evals.common
import pyine.evals.configs
import pyine.evals.correctness.configs as correctness_configs
import pyine.evals.utils
import pyine.guardrails.data.datamodule_configs
import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.logging as reward_logging
import pyine.organisms.models.rewards.core.manager as reward_manager_mod
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.organisms.models.rewards.trl as reward_trl
import pyine.utils.distrib
import pyine.utils.filesystem
import pyine.utils.interrupts
import pyine.utils.langchain
import pyine.utils.llm_providers
import pyine.utils.portability
import pyine.utils.reprod
import pyine.utils.timers
import pyine.utils.tokenizers
import pyine.utils.transformers
import pyine.utils.transformers.data

logger = logging.getLogger(__name__)


def _batch_format_messages_to_text(
    batch: dict[str, typing.Any],
    messages_key: str,
    output_key: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> dict[str, typing.Any]:
    """Batch transform: convert each sample's messages to text via ``format_messages_to_text``.

    Handles both chat-template and no-template tokenizers. Each sample is formatted independently
    through ``format_messages_to_text``, which is the single source of truth for message-to-text
    conversion semantics. All existing columns (except ``messages_key``, which is handled by
    ``remove_columns`` in the caller) are preserved.
    """
    all_messages: list[list[dict[str, str]]] = batch[messages_key]
    texts: list[str] = [
        pyine.utils.transformers.data.format_messages_to_text(messages, tokenizer) for messages in all_messages
    ]
    return {output_key: texts}


def apply_messages_formatting(
    dataset_dict: hf_datasets.DatasetDict,
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages_key: str = "messages",
    output_key: str = "text",
) -> hf_datasets.DatasetDict:
    """Convert ``messages`` to ``text`` for all splits, auto-detecting chat template support.

    This is a batch wrapper around ``format_messages_to_text`` from ``pyine.utils.transformers.data``.
    Each sample is formatted independently through that function, which is the single source of
    truth for message -> text conversion semantics.

    If the tokenizer has a ``chat_template``, ``format_messages_to_text`` delegates to
    ``tokenizer.apply_chat_template``. Otherwise, it falls back to role-tagged plain text
    concatenation (suitable for encoder models like BERT, ModernBERT, DeBERTa).

    Args:
        dataset_dict: HF DatasetDict with a ``messages`` column in each split.
        tokenizer: Tokenizer to use for chat template formatting.
        messages_key: Column name containing the messages lists.
        output_key: Column name for the resulting text.

    Returns:
        New DatasetDict with ``output_key`` column added and ``messages_key`` removed.
    """
    has_template = pyine.utils.transformers.data.tokenizer_has_chat_template(tokenizer)
    if has_template:
        logger.info("tokenizer has chat_template; applying chat template to format messages")
    else:
        logger.info(
            "tokenizer does not have chat_template (typical for encoder models); "
            "falling back to role-tagged plain text concatenation"
        )
    result = hf_datasets.DatasetDict()
    for split_name in dataset_dict:
        result[split_name] = dataset_dict[split_name].map(  # pyright: ignore[reportUnknownMemberType]
            _batch_format_messages_to_text,
            batched=True,
            fn_kwargs={"messages_key": messages_key, "output_key": output_key, "tokenizer": tokenizer},
            remove_columns=[messages_key],
            desc="formatting messages to text",
        )
    return result


def resolve_attn_implementation(
    auto_model_config: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    """Resolve attention implementation with fallback to SDPA if flash-attn unavailable."""
    requested_impl = auto_model_config.get("attn_implementation")
    if requested_impl == "flash_attention_2":
        if not transformers.utils.is_flash_attn_2_available():  # pyright: ignore[reportPrivateImportUsage]
            logger.warning(
                "flash-attn not available, falling back to sdpa attention implementation; "
                "install flash-attn for better performance: uv sync --extra flash_attn"
            )
            return {**auto_model_config, "attn_implementation": "sdpa"}
    return auto_model_config


def validate_wandb_sweeper_requirements(config: AppMainConfig) -> None:
    """Validates that wandb logging is enabled when using the wandb sweeper.

    This function should be called early in the main entrypoint of any trainer app that supports
    Hydra multirun sweeps. It checks if the app is running in multirun mode with the wandb sweeper,
    and raises an error if wandb logging is not enabled (which is required for sweep tracking).

    Args:
        config: The application configuration to validate.

    Raises:
        ValueError: If using wandb sweeper without wandb logging enabled.
    """
    from hydra.core.hydra_config import HydraConfig
    from hydra.types import RunMode

    if not HydraConfig.initialized():
        return  # hydra not running, nothing to check
    hydra_cfg = HydraConfig.get()
    hydra_mode = getattr(hydra_cfg, "mode", None)
    if hydra_mode is None or hydra_mode != RunMode.MULTIRUN:
        return  # hydra not running in multirun (sweep) mode
    hydra_sweeper = getattr(hydra_cfg, "sweeper", None)
    if hydra_sweeper is None:
        return
    sweeper_target = getattr(hydra_sweeper, "_target_", None)
    sweeper_config = getattr(hydra_sweeper, "wandb_sweep_config", None)
    if (sweeper_target is not None and "wandb" in str(sweeper_target).lower()) or sweeper_config is not None:
        if not config.use_wandb_logging:
            raise ValueError(
                "When using the hydra-wandb-sweeper (multirun with wandb sweeper), "
                "config.use_wandb_logging must be set to True. "
                "Please set config.use_wandb_logging=true or remove the wandb sweeper configuration."
            )


def get_wandb_run_for_callback(
    config: typing.Any,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    callback_config: typing.Any,
) -> typing.Any | None:
    """Get wandb_run for callback, with validation for only_main_process setting.

    This helper resolves the appropriate wandb_run for callbacks that log directly to wandb
    (like ThroughputLoggingCallback and GPUStatsLoggingCallback). It handles:
    - Returning None if runtime or wandb_run is unavailable
    - Returning None on non-main ranks when callback_config.only_main_process=True
    - Warning if only_main_process=False but wandb_init_on_all_ranks=False

    Args:
        config: The app config object (must have wandb_init_on_all_ranks attribute).
        runtime: The runtime config object (may be None).
        callback_config: The callback's config object (must have only_main_process attribute).

    Returns:
        wandb_run if available and appropriate for this process, None otherwise.
    """
    if not runtime or not runtime.wandb_run:
        return None
    if callback_config.only_main_process:
        if not pyine.utils.distrib.is_main_process():
            return None
        return runtime.wandb_run
    # log from all ranks - warn if wandb not available on all ranks
    if not config.wandb_init_on_all_ranks:
        logger.warning(
            f"{callback_config.__class__.__name__}.only_main_process=False requires "
            "wandb_init_on_all_ranks=True to log from all ranks. Metrics from non-main "
            "ranks will be silently dropped."
        )
    return runtime.wandb_run


def validate_training_prediction_vllm_compatibility(config: AppMainConfig) -> None:
    """Validates that training, prediction, and vLLM server settings are compatible.

    This function should be called early in the main entrypoint of any trainer app that supports
    vLLM-based evaluation. It checks if the app is configured to run both training and prediction
    with vLLM provider enabled, which is invalid because a manual step (starting the vLLM server
    with the trained checkpoint) must occur between training and prediction.

    Args:
        config: The application configuration to validate.
    """
    do_train, _do_eval, do_predict = get_training_flags(config)
    if not isinstance(config.evals_config, pyine.evals.common.GenerationEvalsConfig):
        return  # not relying on vllm if we don't have any generation to do...
    has_vllm_evals_config = config.evals_config.vllm_provider_config is not None
    if do_train and do_predict and has_vllm_evals_config:
        training_type = "RL" if is_rl_config(config) else "SFT"
        raise ValueError(
            f"Invalid configuration: cannot run {training_type} training and vLLM-based prediction in the same run. "
            "When using vLLM provider for evaluation, you must manually start the vLLM server with the "
            "trained checkpoint between training and prediction. Please choose one of these options:\n"
            "  Option A: Train only (do_train=True, do_predict=False), then manually start vLLM "
            "server with the checkpoint, then run prediction only (do_train=False, do_predict=True, "
            "vllm_provider_config=<config>)\n"
            "  Option B: Train and predict in one run without vLLM (do_train=True, do_predict=True, "
            "vllm_provider_config=null)",
        )


def resolve_save_on_each_node(
    training_args_dict: dict[str, typing.Any],
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> None:
    """Resolve ``save_on_each_node='auto'`` based on multi-node and filesystem detection.

    When the value is ``'auto'`` (the default in ``TrainingArgsConfig``), this function checks
    whether the run is multi-node and the output directory is on node-local storage. If so,
    ``save_on_each_node`` is set to True so that each node saves checkpoints locally; otherwise
    it defaults to False (standard HF behavior).

    Explicit ``True`` or ``False`` values are left unchanged.

    Args:
        training_args_dict: Mutable dict of training arguments (modified in-place).
        runtime: Runtime config for the current run (used for output dir detection).
    """
    if training_args_dict.get("save_on_each_node") != pyine.utils.transformers.SAVE_ON_EACH_NODE_AUTO:
        return
    should_save = False
    num_nodes = pyine.utils.distrib.get_num_nodes(default=1)
    if num_nodes is not None and num_nodes > 1 and runtime is not None:
        output_dir = training_args_dict.get("output_dir") or str(runtime.output_dir_path)
        should_save = not pyine.utils.filesystem.is_path_on_shared_filesystem(output_dir)
    training_args_dict["save_on_each_node"] = should_save
    if should_save:
        logger.info("auto-enabled save_on_each_node (multi-node + local FS detected)")
    else:
        logger.debug(f"resolved save_on_each_node to False (num_nodes={num_nodes})")


class ModelTokenizerConfigBase(pydantic.BaseModel):
    """Base configuration providing model and tokenizer fields shared across trainers.

    Trainer configs that need to instantiate HuggingFace models and tokenizers should inherit
    from this class. Subclasses must implement the ``target_dtype`` property since it depends
    on trainer-specific config fields.

    The default ``get_model()`` loads a causal LM via ``AutoModelForCausalLM``; subclasses can
    override it for other architectures (e.g., ``AutoModelForSequenceClassification``).
    """

    # --------------- model settings ---------------

    base_model: str
    """HuggingFace model identifier or local path for the base pretrained model."""
    auto_model_config: dict[str, typing.Any] = pydantic.Field(default_factory=lambda: {})
    """Extra kwargs passed to the ``AutoModel*.from_pretrained`` call."""
    quantization_mode: typing.Literal["qlora", "none"] = "none"
    """Quantization mode; ``qlora`` loads the model in 4-bit, ``none`` disables quantization."""
    lora_config: peft.LoraConfig | pyine.utils.transformers.LoraConfig | None = None
    """LoRA adapter configuration; if None, LoRA is not applied."""

    # --------------- tokenizer settings ---------------

    auto_tokenizer_config: dict[str, typing.Any] = pydantic.Field(default_factory=lambda: {"use_fast": True})
    """Extra kwargs passed to ``AutoTokenizer.from_pretrained``."""
    tokenizer_set_padding_to_eos_if_needed: bool = True
    """If True and tokenizer has no PAD token, reuse EOS token as PAD for batching."""
    tokenizer_override_padding_to_right_side: bool = True
    """Override the tokenizer's default padding side to ``right``."""
    tokenizer_override_truncation_to_left_side: bool = True
    """Override the tokenizer's default truncation side to ``left``."""

    # --------------- helpers/getters ---------------

    @property
    def target_dtype(self) -> torch.dtype:
        """Returns the target dtype to use with models.

        Subclasses must override this property based on their training args config
        (e.g., training_args_config.bf16 for SFT, grpo_config.bf16 for RL/GRPO).
        """
        raise NotImplementedError("Subclasses must implement target_dtype property")

    @property
    def device_map(self) -> dict[str, torch.device | str] | str | None:
        """Returns the device map to use with models."""
        return get_device_map()

    def get_tokenizer(
        self,
        checkpoint_path: pathlib.Path | None = None,
    ) -> transformers.PreTrainedTokenizer:
        """Returns the tokenizer to use for the targeted model.

        Args:
            checkpoint_path: Optional path to a checkpoint to load from. If provided, loads the
                tokenizer from the checkpoint. If None, loads the tokenizer for the base model.

        Returns:
            The instantiated tokenizer.
        """
        return instantiate_tokenizer(
            base_model=self.base_model,
            checkpoint_path=checkpoint_path,
            auto_tokenizer_config=self.auto_tokenizer_config,
            set_padding_to_eos_if_needed=self.tokenizer_set_padding_to_eos_if_needed,
            override_padding_to_right_side=self.tokenizer_override_padding_to_right_side,
            override_truncation_to_left_side=self.tokenizer_override_truncation_to_left_side,
        )

    def get_model(
        self,
        checkpoint_path: pathlib.Path | None = None,
    ) -> transformers.PreTrainedModel:
        """Returns a pretrained model instance.

        The default implementation loads a causal LM via ``AutoModelForCausalLM``. Subclasses
        can override this for other architectures (e.g., sequence classification).

        Args:
            checkpoint_path: Optional checkpoint to load from (with LoRA adapters if present).
                If None, loads the base pretrained model.

        Returns:
            The instantiated model.
        """
        return instantiate_model(
            base_model=self.base_model,
            checkpoint_path=checkpoint_path,
            target_dtype=self.target_dtype,
            device_map=self.device_map,
            auto_model_config=self.auto_model_config,
            quantization_mode=self.quantization_mode,
            lora_config=self.lora_config,
        )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _coerce_lora_config(
        cls,
        data: typing.Any,
    ) -> typing.Any:
        """Ensures the lora field is a LoraConfig instance when provided as a dict."""
        if isinstance(data, dict):
            typed_data = typing.cast("dict[str, typing.Any]", data)
            lora_config = typed_data.get("lora_config")
            if isinstance(lora_config, pyine.utils.transformers.LoraConfig):
                return typed_data
            if isinstance(lora_config, peft.LoraConfig):
                # use dataclasses.asdict to avoid private attributes like _custom_modules
                typed_data["lora_config"] = pyine.utils.transformers.LoraConfig.model_validate(
                    dataclasses.asdict(lora_config)
                )
                return typed_data
            if isinstance(lora_config, dict):
                typed_lora_config = typing.cast("dict[str, typing.Any]", lora_config)
                typed_data["lora_config"] = pyine.utils.transformers.LoraConfig.model_validate(typed_lora_config)
                return typed_data
        return typing.cast("typing.Any", data)


class AppMainConfig(pydantic.BaseModel):
    """Trainer application main entrypoint configuration settings.

    Should apply to all trainers that intend to train/evaluate models.
    """

    model_config = pydantic.ConfigDict(extra="ignore")
    """Pydantic model configuration (ignore extra fields without strict validation)."""

    datamodule_config: pydantic.SerializeAsAny[pyine.data.datamodule.BaseDataModuleConfig]
    """Configuration for the datamodule to use."""
    evals_config: pydantic.SerializeAsAny[pyine.evals.common.BaseEvalsConfig]
    """Configuration for the task evaluation strategy to use."""
    use_wandb_logging: bool = False
    """Whether to use W&B logging."""
    wandb_init_on_all_ranks: bool = False
    """Whether to initialize W&B on all distributed ranks (shared mode).

    When True, all ranks initialize W&B in shared mode, creating a wandb.Run object on each rank.
    This enables distributed logging coordination where all ranks can access the run object, though
    only the primary rank (rank 0) actually uploads data to W&B servers.

    When False (default), only the main rank (rank 0) initializes W&B. This is more efficient for
    typical training scenarios where only rank 0 needs to log metrics. Components like RewardManager
    automatically handle the absence of a logger on non-main ranks when configured appropriately.

    Note: When using distributed RL training with reward logging enabled, this should generally be
    kept False (the default). The RewardManager and other logging components are designed to work
    correctly in this setup through conditional logger creation and appropriate validation checks.
    """
    gpu_stats_logging: pyine.utils.transformers.GPUStatsLoggingConfig | None = None
    """GPU stats logging configuration. If None, GPU stats logging is disabled."""
    throughput_logging: pyine.utils.transformers.ThroughputLoggingConfig | None = None
    """Throughput logging configuration. If None, throughput logging is disabled."""

    # --------------- resume settings ---------------

    resume_from_run_dir: pathlib.Path | None = pydantic.Field(
        default=None,
        description=(
            "Path to the Hydra output directory of a previous run to resume from. "
            "When provided, the run directory must contain the original logged configs and checkpoints."
        ),
    )
    resume_checkpoint_name: str | None = pydantic.Field(
        default=None,
        description=(
            "Optional checkpoint directory name to resume from inside the run directory. "
            "If not provided, the most recent checkpoint in the run directory will be used."
        ),
    )
    resume_wandb_behavior: typing.Literal["must", "allow", "never"] = pydantic.Field(
        default="allow",  # this is a good default for resuming interrupted/preempted runs
        description=(
            "Whether to resume from a previous W&B run if one is found in the previous run dir. "
            "If 'must', the previous run directory must contain a W&B run. If 'allow', a W&B run "
            "will be resumed if one is found. If 'never', a new run will be started if wandb is "
            "requested."
        ),
    )
    resume_incompatibility_policy: typing.Literal["error", "warn", "ignore"] = pydantic.Field(
        default="error",
        description=(
            "Policy for handling incompatibilities between the previous run metadata and the current launch. "
            "'error' raises when mismatches such as git revision or seed are detected, 'warn' logs a warning, "
            "and 'ignore' skips the checks."
        ),
    )
    auto_resume_if_possible: bool = pydantic.Field(
        default=True,
        description=(
            "Whether to attempt resuming automatically when the current Hydra output directory "
            "already contains checkpoints from a previous run."
        ),
    )

    # --------------- public / utility / helper functions ---------------

    def is_resuming(self) -> bool:
        """Returns whether the application configuration indicates that a run is being resumed."""
        return self.resume_from_run_dir is not None

    def normalize_for_resume_overlap_check(
        self,
        config: AppMainConfig | dict[str, typing.Any] | None = None,
    ) -> dict[str, typing.Any]:
        """Normalizes the config by removing fields that might change without effects on experiments."""
        # derives classes might want to override this and add new pops for other non-important attributes
        if config is None:
            data = pyine.utils.portability.make_json_serializable(self.model_dump(mode="python"))
        elif isinstance(config, AppMainConfig):
            data = pyine.utils.portability.make_json_serializable(config.model_dump(mode="python"))
        else:
            assert isinstance(config, dict)
            data = config.copy()
        data.pop("use_wandb_logging", None)
        data.pop("wandb_init_on_all_ranks", None)
        data.pop("gpu_stats_logging", None)
        data.pop("throughput_logging", None)
        data.pop("resume_from_run_dir", None)
        data.pop("resume_checkpoint_name", None)
        data.pop("resume_wandb_behavior", None)
        data.pop("resume_incompatibility_policy", None)
        data.pop("auto_resume_if_possible", None)
        return data

    @pydantic.model_validator(mode="after")
    def _validate_resume_settings(
        self,
    ) -> AppMainConfig:
        """Ensures resume settings are consistent."""
        if self.resume_from_run_dir is not None:
            run_dir = pathlib.Path(self.resume_from_run_dir).expanduser().resolve()
            if not run_dir.is_dir():
                raise FileNotFoundError(f"invalid resume directory (not found): {run_dir}")
            if self.resume_checkpoint_name is not None:
                checkpoint_path = run_dir / self.resume_checkpoint_name
                if not checkpoint_path.is_dir():
                    raise FileNotFoundError(f"specified resume checkpoint not found: {checkpoint_path}")
        elif self.resume_checkpoint_name is not None:
            raise ValueError("resume_checkpoint_name requires resume_from_run_dir to be set")
        return self


class ResumeArtifacts(pydantic.BaseModel):
    """Container with the information required to resume a training run."""

    run_dir: pathlib.Path
    """Path to the Hydra output directory of the previous run being resumed."""
    checkpoint_path: pathlib.Path
    """Path to the specific checkpoint directory to resume training from."""
    checkpoint_metadata_dict: dict[str, typing.Any]
    """Metadata dictionary stored alongside the checkpoint, if present."""
    checkpoint_metadata_path: pathlib.Path | None
    """Path to the checkpoint metadata file, if found."""
    previous_config_dict: dict[str, typing.Any]
    """Configuration object dump from the previous run, loaded from the logged config file."""
    previous_config_path: pathlib.Path | None
    """Path to the logged config file from the previous run, if found."""
    previous_runtime_dict: dict[str, typing.Any]
    """Runtime configuration dictionary from the previous run, loaded from the logged runtime file."""
    previous_runtime_path: pathlib.Path | None
    """Path to the logged runtime file from the previous run, if found."""
    previous_metadata_dict: dict[str, typing.Any]
    """Reproducibility metadata dictionary from the previous run, loaded from the logged metadata file."""
    previous_metadata_path: pathlib.Path | None
    """Path to the logged reproducibility metadata file from the previous run, if found."""
    wandb_resume_kwargs: dict[str, typing.Any]
    """Dictionary of kwargs for resuming a Weights & Biases run (to be passed to `wandb.init`)."""
    compatibility_issues: list[str] = pydantic.Field(default_factory=list)
    """List of detected compatibility issues between the previous run and the current launch."""

    @classmethod
    def create(
        cls,
        config: AppMainConfig,
        runtime: pyine.configs.schemas.RuntimeConfig | None = None,
    ) -> ResumeArtifacts:
        """Creates a ResumeArtifacts object from a config that should contain resume-related args.

        Args:
            config: The application configuration, which should contain the resume settings. If it
                does not, an error will be raised.
            runtime: The runtime config object to fetch settings from (if available).

        Returns:
            A ResumeArtifacts object containing the information required to resume a training run.
        """
        if config.resume_from_run_dir is None:
            raise ValueError("resume_from_run_dir must be provided for resume functionality")
        assert config.is_resuming()
        run_dir = pathlib.Path(config.resume_from_run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"invalid resume directory (not found): {run_dir}")
        config_path = cls._find_first_matching_path(run_dir, "config.*.rank*.json")
        if config_path is None:
            raise FileNotFoundError(f"could not locate logged config file under resume directory: {run_dir}")
        runtime_path = cls._find_first_matching_path(run_dir, "runtime.*.rank*.json")
        metadata_path = cls._find_first_matching_path(run_dir, "reprod_metadata.*.rank*.json")
        config_payload = cls._load_json_if_exists(config_path)
        previous_config = typing.cast("dict[str, typing.Any] | None", config_payload.get("main_config"))
        if previous_config is None:
            raise ValueError("resume directory is missing the logged main_config payload")
        cls._ensure_resume_config_matches(current_config=config, previous_config=previous_config)
        runtime_payload = cls._load_json_if_exists(runtime_path)
        metadata_payload = cls._load_json_if_exists(metadata_path)
        wandb_resume_kwargs = cls._extract_wandb_resume_kwargs(runtime_payload, config)
        if config.resume_checkpoint_name is not None:
            checkpoint_path = run_dir / config.resume_checkpoint_name
            if not checkpoint_path.is_dir():
                raise FileNotFoundError(f"specified resume checkpoint directory not found: {checkpoint_path}")
        else:
            last_checkpoint = typing.cast(
                "str | None",
                transformers.trainer_utils.get_last_checkpoint(  # pyright: ignore[reportUnknownMemberType]
                    str(run_dir),
                ),
            )
            if last_checkpoint is None:
                raise FileNotFoundError(f"no checkpoint found under resume directory: {run_dir}")
            checkpoint_path = pathlib.Path(last_checkpoint).resolve()
        trainer_state_path = checkpoint_path / "trainer_state.json"
        if not trainer_state_path.is_file():
            raise FileNotFoundError(
                f"invalid checkpoint directory (missing trainer_state.json): {checkpoint_path}",
            )
        missing_artifacts: list[str] = []
        for artifact_name in ("optimizer.pt", "scheduler.pt"):
            artifact_path = checkpoint_path / artifact_name
            if not artifact_path.is_file():
                missing_artifacts.append(artifact_name)
        if missing_artifacts:
            missing_display = ", ".join(sorted(missing_artifacts))
            logger.warning(
                f"resume checkpoint is missing optimizer-related artifacts ({missing_display}); "
                "automatic resumption may skip optimizer or scheduler state restore"
            )
        compatibility_issues: list[str] = []
        policy = getattr(config, "resume_incompatibility_policy", "error")
        if policy != "ignore":
            previous_git = typing.cast("str | None", metadata_payload.get("git_revision_hash"))
            current_git = pyine.utils.reprod.get_git_revision_hash()
            if previous_git and current_git and previous_git != current_git:
                compatibility_issues.append(
                    f"git_revision_hash mismatch (previous: {previous_git}, current: {current_git})",
                )
            previous_seed_value = runtime_payload.get("seed", metadata_payload.get("seed"))
            previous_seed = cls._coerce_optional_int(previous_seed_value)
            current_seed = cls._coerce_optional_int(getattr(runtime, "seed", None)) if runtime is not None else None
            if previous_seed is not None and current_seed is not None and previous_seed != current_seed:
                compatibility_issues.append(
                    f"seed mismatch (previous: {previous_seed}, current: {current_seed})",
                )
        if compatibility_issues:
            issues_str = "; ".join(compatibility_issues)
            if policy == "error":
                raise ValueError(
                    f"resume metadata compatibility check failed: {issues_str}; "
                    "set resume_incompatibility_policy to 'warn' or 'ignore' to override",
                )
            if policy == "warn":
                logger.warning(f"resume metadata compatibility warnings: {issues_str}")
        checkpoint_metadata_dict: dict[str, typing.Any] = {}
        checkpoint_metadata_path = checkpoint_path / "run_meta.json"
        if checkpoint_metadata_path.is_file():
            checkpoint_metadata_dict = cls._load_json_if_exists(checkpoint_metadata_path)
            if config.use_wandb_logging and not wandb_resume_kwargs and config.resume_wandb_behavior != "never":
                checkpoint_runtime_payload = typing.cast(
                    "dict[str, typing.Any]",
                    checkpoint_metadata_dict.get("runtime", {}),
                )
                wandb_run_id_meta = typing.cast("str | None", checkpoint_runtime_payload.get("wandb_run_id"))
                wandb_project_meta = typing.cast("str | None", checkpoint_runtime_payload.get("wandb_run_project"))
                wandb_entity_meta = typing.cast("str | None", checkpoint_runtime_payload.get("wandb_run_entity"))
                if wandb_run_id_meta and wandb_project_meta and wandb_entity_meta:
                    wandb_resume_kwargs = {
                        "project": wandb_project_meta,
                        "entity": wandb_entity_meta,
                        "id": wandb_run_id_meta,
                        "resume": config.resume_wandb_behavior,
                    }
                    logger.info(
                        "resuming wandb run from checkpoint metadata: "
                        f"id={wandb_run_id_meta} project={wandb_project_meta}"
                    )
        return ResumeArtifacts(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata_dict=checkpoint_metadata_dict,
            checkpoint_metadata_path=checkpoint_metadata_path if checkpoint_metadata_path.is_file() else None,
            previous_config_dict=previous_config,
            previous_config_path=config_path,
            previous_runtime_dict=runtime_payload,
            previous_runtime_path=runtime_path,
            previous_metadata_dict=metadata_payload,
            previous_metadata_path=metadata_path,
            wandb_resume_kwargs=wandb_resume_kwargs,
            compatibility_issues=compatibility_issues,
        )

    # ------------- private utility functions -------------

    @staticmethod
    def _find_first_matching_path(
        root_dir: pathlib.Path,
        pattern: str,
    ) -> pathlib.Path | None:
        matches = sorted(root_dir.glob(pattern))
        if not matches:
            return None
        return matches[0]

    @staticmethod
    def _load_json_if_exists(path: pathlib.Path | None) -> dict[str, typing.Any]:
        if path is None:
            return {}
        return typing.cast("dict[str, typing.Any]", json.loads(path.read_text()))

    @staticmethod
    def _coerce_optional_int(value: typing.Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _ensure_resume_config_matches(
        current_config: AppMainConfig,
        previous_config: dict[str, typing.Any] | AppMainConfig,
    ) -> None:
        current_payload = current_config.normalize_for_resume_overlap_check()
        previous_payload = current_config.normalize_for_resume_overlap_check(previous_config)
        # use deepdiff for comparison (handles order-independent list comparison and type coercion)
        diff = deepdiff.DeepDiff(previous_payload, current_payload, verbose_level=2, ignore_order=True)
        if not diff:
            return  # configs match
        raise ValueError(
            f"resume configuration mismatch detected:\n"
            f"{diff.pretty()}\n"  # type: ignore[reportUnknownMemberType]
            "Ensure the resumed launch matches the original configuration (except for resume settings).",
        )

    @staticmethod
    def _extract_wandb_resume_kwargs(
        runtime_dict: dict[str, typing.Any],
        config: AppMainConfig,
    ) -> dict[str, typing.Any]:
        if not config.use_wandb_logging or config.resume_wandb_behavior == "never":
            return {}
        wandb_run_id = typing.cast("str | None", runtime_dict.get("wandb_run_id"))
        if wandb_run_id is None:
            if config.resume_wandb_behavior == "must":
                raise RuntimeError("original run did NOT have wandb logging enabled; cannot resume")
            return {}
        assert config.resume_wandb_behavior in ["allow", "must"]
        assert "wandb_run_project" in runtime_dict and "wandb_run_entity" in runtime_dict
        wandb_run_project = typing.cast("str", runtime_dict["wandb_run_project"])
        wandb_run_entity = typing.cast("str", runtime_dict["wandb_run_entity"])
        return {
            "project": wandb_run_project,
            "entity": wandb_run_entity,
            "id": wandb_run_id,
            "resume": config.resume_wandb_behavior,
        }


def prepare_resume_artifacts(
    config: AppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    *,
    persist_to_runtime: bool = True,
) -> ResumeArtifacts | None:
    """Prepares resume artifacts (if resuming is requested) and returns them.

    Will also update the runtime config with the resume artifacts if resuming is requested.
    """
    auto_resume_activated = False
    has_resume_dir_attr = hasattr(config, "resume_from_run_dir")
    has_resume_checkpoint_attr = hasattr(config, "resume_checkpoint_name")
    original_resume_dir = getattr(config, "resume_from_run_dir", None)
    original_resume_checkpoint = getattr(config, "resume_checkpoint_name", None)
    auto_resume_enabled = bool(getattr(config, "auto_resume_if_possible", False))
    if not config.is_resuming() and auto_resume_enabled:
        inferred_run_dir = _infer_resume_run_dir_from_runtime(runtime)
        if inferred_run_dir is not None:
            config.resume_from_run_dir = inferred_run_dir
            config.resume_checkpoint_name = None
            auto_resume_activated = True
            logger.info(f"auto-resume enabled using run dir: {inferred_run_dir}")
    if not config.is_resuming():
        return None
    try:
        resume_artifacts = ResumeArtifacts.create(config, runtime)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        if auto_resume_activated:
            logger.error(f"auto-resume aborted because prerequisites were not met: {exc}")
            if has_resume_dir_attr:
                config.resume_from_run_dir = original_resume_dir
            elif hasattr(config, "resume_from_run_dir"):
                delattr(config, "resume_from_run_dir")
            if has_resume_checkpoint_attr:
                config.resume_checkpoint_name = original_resume_checkpoint
            elif hasattr(config, "resume_checkpoint_name"):
                delattr(config, "resume_checkpoint_name")
        raise
    logger.info(f"prepared resume artifacts from run dir: {resume_artifacts.run_dir}")
    logger.debug(f"will resume training from checkpoint: {resume_artifacts.checkpoint_path}")
    if resume_artifacts.wandb_resume_kwargs and "id" in resume_artifacts.wandb_resume_kwargs:
        logger.debug(f"will resume wandb run id: {resume_artifacts.wandb_resume_kwargs['id']}")
    elif config.use_wandb_logging:
        logger.debug("will create a new wandb run for resumed training")
    if runtime is not None and pyine.utils.distrib.is_local_main_process():
        runtime.metadata["resumed_from_run_dir"] = str(resume_artifacts.run_dir)
        runtime.metadata["resume_checkpoint_path"] = str(resume_artifacts.checkpoint_path)
    if runtime is not None and persist_to_runtime:
        assert runtime.output_dir_path.is_dir(), "invalid runtime config output dir"
        file_mappings: list[tuple[pathlib.Path | None, str]] = [
            (resume_artifacts.previous_config_path, "previous_config.json"),
            (resume_artifacts.previous_runtime_path, "previous_runtime.json"),
            (resume_artifacts.previous_metadata_path, "previous_reprod_metadata.json"),
            (resume_artifacts.checkpoint_metadata_path, "resume_checkpoint_run_meta.json"),
        ]
        for source_path, target_name in file_mappings:
            if source_path is None or not source_path.is_file():
                continue
            target_path = runtime.output_dir_path / target_name
            shutil.copy2(source_path, target_path)
    return resume_artifacts


def _infer_resume_run_dir_from_runtime(
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> pathlib.Path | None:
    """Return the runtime output directory when it already contains a valid HF Trainer checkpoint."""
    if runtime is None or not runtime.output_dir_path.exists():
        return None
    last_checkpoint = typing.cast(
        "str | None",
        transformers.trainer_utils.get_last_checkpoint(str(runtime.output_dir_path)),  # type: ignore[reportUnknownMemberType]
    )
    if last_checkpoint is None:
        return None
    checkpoint_path = pathlib.Path(last_checkpoint)
    trainer_state_path = checkpoint_path / "trainer_state.json"
    if not trainer_state_path.is_file():
        logger.warning(
            f"auto-resume detected checkpoint without trainer_state.json; ignoring checkpoint at {checkpoint_path}",
        )
        return None
    return runtime.output_dir_path


def prepare_datamodule(
    config: AppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    stage: str | None = None,
) -> pyine.data.datamodule.BaseDataModule[typing.Any]:
    """Prepares the configured datamodule with per-node or global-rank-0 preparation.

    When per-node prep is enabled (i.e. when caching on local FS):
    - Local rank 0 on each node runs prepare_data();
    - Intra-node barrier synchronizes local ranks;
    - Cross-node fingerprint validation ensures data integrity.

    Otherwise, when single-node or using a shared filesystem:
    - Only global rank 0 runs prepare_data();
    - Standard barrier synchronizes all ranks.

    Args:
        config: The application configuration, which should contain the datamodule config.
        runtime: The runtime configuration, which may contain W&B run information.
        stage: Optional Lightning stage string (``"fit"``, ``"validate"``, ``"test"``,
            ``"predict"``) forwarded to ``dm.setup(stage=...)``. When ``None``, all subsets
            are set up.

    Returns:
        The instantiated, prepared, and set-up datamodule that is ready to provide data loaders.
    """
    logger.info("preparing datamodule and setting up parsers/loaders...")
    dm = config.datamodule_config.instantiate_datamodule(verbose=True)
    # use synchronized decision-making to prevent cross-node divergence and ensure DDP is
    # initialized early (before any rank-divergent work like fingerprint computation)
    use_per_node_prep = pyine.utils.distrib.determine_per_node_prep_mode()
    if use_per_node_prep:
        pyine.utils.distrib.validate_node_configuration(use_per_node_prep)
        if pyine.utils.distrib.is_local_main_process():
            logger.info(f"node {pyine.utils.distrib.get_node_rank()} preparing data...")
            dm.prepare_data()
        pyine.utils.distrib.local_barrier()
        fingerprint_payload: pyine.utils.distrib.FingerprintPayload | None = None
        if pyine.utils.distrib.is_local_main_process():
            try:
                fingerprint = _compute_datamodule_fingerprint(dm, config)
                fingerprint_payload = pyine.utils.distrib.FingerprintPayload(
                    ok=True,
                    fingerprint=fingerprint,
                    error=None,
                )
            except Exception as exc:
                fingerprint_payload = pyine.utils.distrib.FingerprintPayload(
                    ok=False,
                    fingerprint=None,
                    error=str(exc),
                )
        pyine.utils.distrib.validate_cross_node_fingerprints(fingerprint_payload, "datamodule")
    else:
        if pyine.utils.distrib.is_main_process():
            dm.prepare_data()
    pyine.utils.distrib.barrier()
    dm.setup(stage=stage)
    if (
        config.use_wandb_logging
        and runtime is not None
        and runtime.wandb_run is not None
        and pyine.utils.distrib.is_main_process()
    ):
        assert runtime is not None and runtime.wandb_run is not None
        active = set(dm.active_subset_names)
        target_subsets: list[str] = [
            name
            for name in {
                *config.datamodule_config.train_subset_names,
                *config.datamodule_config.resolved_valid_subset_names,
                *config.datamodule_config.resolved_eval_subset_names,
            }
            if name in active
        ]
        dm_stats = dm.get_stats(target_subsets)
        summary_stats = {f"dataset_stats/{k}": v for k, v in dm_stats.items()}
        runtime.wandb_run.summary.update(summary_stats)  # type: ignore[reportUnknownMemberType]
    return dm


def _compute_datamodule_fingerprint(
    dm: pyine.data.datamodule.BaseDataModule[typing.Any],
    config: AppMainConfig,
) -> str:
    """Compute fingerprint of prepared datamodule for cross-node validation.

    Uses the datamodule's get_fingerprint_inputs() interface to ensure
    each datamodule explicitly provides its deterministic artifacts.

    Args:
        dm: The prepared datamodule.
        config: The application configuration.

    Returns:
        SHA256 hexdigest fingerprint string.
    """
    config_hash = pyine.utils.reprod.get_versioned_cache_hash(config.datamodule_config.model_dump())
    fingerprint_inputs = dm.get_fingerprint_inputs()
    return pyine.utils.reprod.compute_data_fingerprint(
        inputs=fingerprint_inputs,
        config_hash=config_hash,
    )


def _is_deepspeed_enabled() -> bool:
    """Detects if DeepSpeed is being used for training.

    Checks multiple indicators:
    1. Accelerate's PartialState if available
    2. DeepSpeed initialization status
    3. Environment variables set by Accelerate

    Returns:
        True if DeepSpeed is detected, False otherwise.
    """
    # check environment variables set by accelerate
    if os.environ.get("ACCELERATE_USE_DEEPSPEED"):
        return True
    # try to check accelerate's state
    try:
        from accelerate import PartialState

        state = PartialState()
        if hasattr(state, "distributed_type"):
            from accelerate.utils import DistributedType

            if state.distributed_type == DistributedType.DEEPSPEED:
                return True
    except (ImportError, RuntimeError):
        # accelerate not available or not initialized yet
        pass
    return False


def get_device_map() -> dict[str, torch.device | str] | str | None:
    """Returns the device map to use with models for distributed/single-GPU training.

    This function determines the appropriate device placement strategy based on whether
    we're running in distributed mode or not:
    - DeepSpeed mode: Return None (DeepSpeed handles device placement)
    - Standard DDP mode: Place model on specific GPU per process (allows Accelerate to handle distribution)
    - Single GPU/CPU mode: Use 'auto' for automatic device placement

    Returns:
        Device map specification compatible with transformers.from_pretrained()
    """
    # DeepSpeed handles its own device placement, so we must return None
    if _is_deepspeed_enabled():
        logger.debug("DeepSpeed detected; setting device_map=None")
        return None
    if pyine.utils.distrib.is_distributed():
        if torch.cuda.is_available():
            local_rank = pyine.utils.distrib.get_local_rank(default=0)
            if local_rank is None:
                return None
            return {"": f"cuda:{local_rank}"}
        return None
    return {"": "mps"} if torch.backends.mps.is_available() else "auto"


def instantiate_tokenizer(
    base_model: str,
    checkpoint_path: pathlib.Path | None = None,
    auto_tokenizer_config: dict[str, typing.Any] | None = None,
    set_padding_to_eos_if_needed: bool = True,
    override_padding_to_right_side: bool = True,
    override_truncation_to_left_side: bool = True,
) -> transformers.PreTrainedTokenizer:
    """Shared tokenizer instantiation for SFT and RL trainers.

    Args:
        base_model: HuggingFace model identifier or local path for the base model.
        checkpoint_path: Optional path to a checkpoint to load from. If provided, loads the tokenizer
            from the checkpoint. If None, loads the tokenizer for the base model specified in base_model.
        auto_tokenizer_config: Tokenizer configuration args passed to `transformers.AutoTokenizer.from_pretrained`.
        set_padding_to_eos_if_needed: If True and tokenizer has no PAD token, reuse EOS token as PAD for batching.
        override_padding_to_right_side: Override whichever the tokenizer's default padding side is to 'right'.
        override_truncation_to_left_side: Override whichever the tokenizer's default truncation side is to 'left'.

    Returns:
        The instantiated tokenizer.
    """
    model_path = checkpoint_path if checkpoint_path is not None else base_model
    logger.info(f"setting up tokenizer for: {model_path}")
    if auto_tokenizer_config is None:
        auto_tokenizer_config = {"use_fast": True}
    logger.debug(f"auto tokenizer config: {auto_tokenizer_config}")
    tokenizer = pyine.utils.tokenizers.get_hf_tokenizer(
        pretrained_model_name_or_path=str(model_path),
        set_padding_to_eos_if_needed=set_padding_to_eos_if_needed,
        override_padding_to_right_side=override_padding_to_right_side,
        override_truncation_to_left_side=override_truncation_to_left_side,
        **auto_tokenizer_config,
    )
    logger.info(f"tokenizer successfully created ({type(tokenizer).__name__})")
    logger.debug(f"tokenizer is_fast: {getattr(tokenizer, 'is_fast', False)}")
    logger.debug(f"tokenizer vocab size: {len(tokenizer)}")
    return tokenizer


def instantiate_model(
    base_model: str,
    checkpoint_path: pathlib.Path | None = None,
    target_dtype: torch.dtype = torch.float16,
    device_map: dict[str, torch.device | str] | str | None = None,
    auto_model_config: dict[str, typing.Any] | None = None,
    quantization_mode: typing.Literal["qlora", "none"] = "none",
    lora_config: peft.LoraConfig | pyine.utils.transformers.LoraConfig | None = None,
) -> transformers.PreTrainedModel:
    """Shared model instantiation for SFT and RL trainers.

    Handles:
    - Loading from HF hub or checkpoint
    - Quantization (4-bit/8-bit for QLoRA)
    - PEFT/LoRA adapter application
    - Device placement
    - LoRA checkpoint loading (via AutoPeftModelForCausalLM)

    Args:
        base_model: HuggingFace model identifier or local path for the base model to fine-tune.
        checkpoint_path: Optional path to a checkpoint to load from. If provided, loads the model
            from the checkpoint (with LoRA adapters if present). If None, loads the base pretrained
            model specified in base_model.
        target_dtype: The target dtype to use with the model (e.g., torch.bfloat16, torch.float16).
        device_map: Device map for distributed/single-GPU training.
        auto_model_config: Model configuration args passed to `transformers.AutoModelForCausalLM.from_pretrained`.
        quantization_mode: Quantization mode. "qlora" loads the model in 4-bit for QLoRA; "none" disables quantization.
        lora_config: LoRA adapter configuration. If None, does not apply LoRA.

    Returns:
        The instantiated model.
    """
    if auto_model_config is None:
        auto_model_config = {}

    resolved_auto_config = resolve_attn_implementation(auto_model_config)

    if checkpoint_path is not None:
        # Load model from checkpoint
        logger.info(f"setting up model from checkpoint: {checkpoint_path}")
        dtype, device_map_to_use = target_dtype, device_map

        # Check if checkpoint contains LoRA adapters
        adapter_config_path = checkpoint_path / "adapter_config.json"
        if adapter_config_path.exists():
            # Load PEFT model with LoRA adapters
            logger.info("  (loading model with LoRA adapters from checkpoint)")
            model: transformers.PreTrainedModel = typing.cast(
                "transformers.PreTrainedModel",
                peft.AutoPeftModelForCausalLM.from_pretrained(  # type: ignore[reportUnknownMemberType]
                    checkpoint_path,
                    torch_dtype=dtype,
                    device_map=device_map_to_use,
                    **resolved_auto_config,
                ),
            )
        else:
            # Load regular model without adapters
            logger.info("  (loading model without adapters from checkpoint)")
            model = typing.cast(
                "transformers.PreTrainedModel",
                transformers.AutoModelForCausalLM.from_pretrained(  # type: ignore[reportUnknownMemberType]
                    checkpoint_path,
                    torch_dtype=dtype,
                    device_map=device_map_to_use,
                    **resolved_auto_config,
                ),
            )
    else:
        # Load base pretrained model
        logger.info(f"setting up model: {base_model}")
        dtype, device_map_to_use = target_dtype, device_map
        model_kwargs: dict[str, typing.Any] = {
            "torch_dtype": dtype,
            "device_map": device_map_to_use,
            **resolved_auto_config,
        }
        if quantization_mode == "qlora":
            logger.info("  (setting up model using QLoRA 4-bit quantization)")
            quant_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quant_config
        elif quantization_mode == "none":
            logger.info("  (setting up model using no quantization)")
        else:
            raise ValueError(f"unsupported quantization_mode: {quantization_mode}")
        logger.debug(f"auto model config: {model_kwargs}")
        base_model_instance = typing.cast(
            "transformers.PreTrainedModel",
            transformers.AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                base_model,
                **model_kwargs,
            ),
        )
        model = base_model_instance
        if lora_config is not None:
            logger.info("  (setting up LoRA adapters)")
            logger.debug(f"lora_config: {lora_config}")
            if isinstance(lora_config, pyine.utils.transformers.LoraConfig):
                lora_peft_config = lora_config.to_peft_config()
            else:
                assert isinstance(lora_config, peft.LoraConfig)
                lora_peft_config = lora_config
            model = typing.cast("transformers.PreTrainedModel", peft.get_peft_model(model, lora_peft_config))

    logger.info(f"model successfully created:\n{model}")
    model_config = getattr(model, "config", None)
    if hasattr(model_config, "to_json_string") and callable(model_config.to_json_string):
        logger.debug(f"model config: {model_config.to_json_string()}")
    if hasattr(model, "peft_config"):
        logger.debug(f"model peft_config: {model.peft_config}")
    get_trainable_params = getattr(model, "get_nb_trainable_parameters", None)
    if callable(get_trainable_params):
        trainable_param_count, total_param_count = typing.cast(
            "tuple[int, int]",
            get_trainable_params(),
        )
        logger.info(f"trainable param count: {trainable_param_count:,d}")
        logger.info(f"total param count: {total_param_count:,d}")
        if total_param_count:
            trainable_ratio = (100 * trainable_param_count) / total_param_count
            logger.info(f"trainable param %: {trainable_ratio:.3f}")
    return model


async def evaluate_model(
    model: pyine.evals.utils.InvocableModelChain | transformers.PreTrainedModel | typing.Any | None,
    tokenizer: transformers.PreTrainedTokenizer | None,
    datamodule: pyine.data.datamodule.BaseDataModule[typing.Any] | None,
    config: AppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> dict[str, pyine.evals.common.EvalResult]:
    """Evaluates the given model following the strategy defined in the app's evals config.

    Args:
        model: The model to evaluate, in langchain runnable, HF-transformers, or invocable/wrapped
            format. Can be None when using vLLM provider (model is served remotely via vLLM server).
            If we cannot deduce the model type and it is not None, we'll let the downstream eval
            pipeline determine if it is compatible with its 'evaluate_wrapped_model' function.
        tokenizer: The tokenizer to use for evaluation, if applicable and relevant (likely only
            useful for e.g. HF model evals, or evals involving vLLM and requiring token stats).
        datamodule: The datamodule from which to load the evaluation data (if not configured as
            part of the evals config itself). If one is provided and not expected by the configured
            eval strategy, an error is raised, and vice versa.
        config: The application configuration, which should contain the evals config, but that will
            also likely be used to log evaluation metadata.
        runtime: The runtime configuration, which may contain W&B run information and relevant info
            for logging.

    Returns:
        The evaluation results as a dictionary indexed by evaluated data subset name.
    """
    evaluation_results: dict[str, pyine.evals.common.EvalResult] = {}
    if config.evals_config is None or config.evals_config.eval_type is None:  # type: ignore[reportUnnecessaryComparison]
        return evaluation_results
    eval_dm = config.evals_config.prepare_eval_datamodule(datamodule)
    eval_dm_subset_names = eval_dm.config.eval_subset_names
    logger.info(f"will evaluate using {len(eval_dm_subset_names)} subset(s): {eval_dm_subset_names}")
    if config.use_wandb_logging and runtime is not None and runtime.wandb_run is not None:
        config.evals_config.define_metrics_for_wandb(
            wandb_run=runtime.wandb_run,
            eval_subset_names=eval_dm_subset_names,
        )
    start_time = time.time()
    # resolve which eval method to call based on model type; each branch validates preconditions
    # and binds all arguments except eval_subset_name into a partial
    vllm_provider_config = getattr(config.evals_config, "vllm_provider_config", None)
    if vllm_provider_config is not None:
        # vLLM provider mode: use standard runnable chain evaluation
        assert isinstance(vllm_provider_config, pyine.utils.llm_providers.LLMProviderConfig)
        if model is not None:
            raise ValueError("model should be None when using vLLM provider mode")
        vllm_model = vllm_provider_config.get_model()
        if not isinstance(eval_dm, pyine.data.datamodule.ConversationDataModule):
            raise ValueError(
                "datamodule should be provided as a conversation DM when using vLLM provider mode "
                "(it specifies the prompt config, otherwise we cannot access it)"
            )
        chain = eval_dm.config.get_prompt_chain(vllm_model)
        run_subset_eval = functools.partial(
            config.evals_config.evaluate_runnable_model,
            chain=chain,
            datamodule=eval_dm,
            verbose=True,
        )
        eval_label = "vLLM chain"
    elif pyine.utils.transformers.is_hf_model(model):
        # local HF model mode: assume we must use model.generate()
        if not isinstance(config.evals_config, pyine.evals.common.GenerationEvalsConfig):
            raise ValueError(f"invalid evals config type for HF model eval: {type(config.evals_config).__name__}")
        if tokenizer is None or not pyine.utils.transformers.is_hf_tokenizer(tokenizer):
            raise ValueError("invalid tokenizer (need to provide one to evaluate hf model")
        if not isinstance(eval_dm, pyine.data.datamodule.ConversationDataModule):
            raise ValueError("invalid datamodule type for HF model eval: must be ConversationDataModule")
        if config.evals_config.eval_padding_side != tokenizer.padding_side:
            new_padding_side = config.evals_config.eval_padding_side
            logger.debug(f"overriding tokenizer padding side to '{new_padding_side}' for evals")
            tokenizer.padding_side = new_padding_side
        hf_model = typing.cast("transformers.PreTrainedModel", model)
        run_subset_eval = functools.partial(
            config.evals_config.evaluate_hf_model,
            model=hf_model,
            tokenizer=tokenizer,
            datamodule=eval_dm,
            verbose=True,
        )
        eval_label = "HF model"
    else:
        # not using vLLM and not an HF model: must be a langchain runnable, or a wrapped model
        if model is None:
            raise ValueError("model cannot be None when not using vLLM provider mode")
        if tokenizer is not None:
            raise NotImplementedError("tokenizer support in runnable chain / wrapped model eval is not implemented")
        if pyine.utils.langchain.is_invocable_chain(model):
            if not isinstance(eval_dm, pyine.data.datamodule.ConversationDataModule):
                raise ValueError("invalid datamodule type for runnable chain eval: must be ConversationDataModule")
            chain_model = typing.cast("pyine.evals.utils.InvocableModelChain", model)
            run_subset_eval = functools.partial(
                config.evals_config.evaluate_runnable_model,
                chain=chain_model,
                datamodule=eval_dm,
                verbose=True,
            )
            eval_label = "invocable chain"
        else:
            # let the downstream evaluation pipeline handle whether this model is compatible
            run_subset_eval = functools.partial(
                config.evals_config.evaluate_wrapped_model,
                wrapped_model=model,
                datamodule=eval_dm,
                verbose=True,
            )
            eval_label = "wrapped model"
    for eval_subset_name in eval_dm_subset_names:
        logger.info(f"running {eval_label} evaluation on the {eval_subset_name} subset...")
        evaluation_result = await run_subset_eval(eval_subset_name=eval_subset_name)
        assert isinstance(evaluation_result, pyine.evals.common.EvalResult)
        pyine.evals.utils.print_metrics(evaluation_result.metrics, eval_subset_name, logger.info)
        evaluation_results[eval_subset_name] = evaluation_result
    if config.use_wandb_logging and evaluation_results:
        if runtime is None:
            raise RuntimeError("runtime configuration with wandb run must be provided when logging to wandb")
        if runtime.wandb_run is None:
            raise RuntimeError("wandb run must be initialized before logging to wandb")
        logger.info(f"logging evaluation results to wandb run id: {runtime.wandb_run_id}...")
        config.evals_config.log_metrics(
            wandb_run=runtime.wandb_run,
            results_by_subset=evaluation_results,
        )
        for subset_name, subset_result in evaluation_results.items():
            logger.info(f"logging tables for {subset_name} subset to wandb run id: {runtime.wandb_run_id}...")
            config.evals_config.log_sample_metrics(
                wandb_run=runtime.wandb_run,
                subset_name=subset_name,
                subset_results=subset_result,
            )
            config.evals_config.log_predictions(
                wandb_run=runtime.wandb_run,
                subset_name=subset_name,
                subset_results=subset_result,
            )
    end_time = time.time()
    time_delta_seconds = end_time - start_time
    time_delta_str = pyine.utils.timers.get_human_readable_time(time_delta_seconds)
    logger.info(f"evaluations finished in {time_delta_str}")
    return evaluation_results


def get_training_flags(
    config: AppMainConfig,
) -> tuple[bool, bool, bool]:
    """Extracts training flags (do_train, do_eval, do_predict) from config.

    Handles both SFT configs (with training_args_config) and HF-TRL configs (with grpo_config).

    Args:
        config: The trainer app config.

    Returns:
        Tuple of (do_train, do_eval, do_predict) booleans.
    """
    # check for TRL config (grpo_config)
    grpo_config = getattr(config, "grpo_config", None)
    if grpo_config is not None:
        return (
            getattr(grpo_config, "do_train", False),
            getattr(grpo_config, "do_eval", False),
            getattr(grpo_config, "do_predict", False),
        )
    # check for SFT config (training_args_config)
    training_args_config = getattr(config, "training_args_config", None)
    if training_args_config is not None:
        return (
            getattr(training_args_config, "do_train", False),
            getattr(training_args_config, "do_eval", False),
            getattr(training_args_config, "do_predict", False),
        )
    raise NotImplementedError("cannot deduce training flag from given trainer app config")


def is_rl_config(config: AppMainConfig) -> bool:
    """Determines if the config is for RL training.

    Args:
        config: The trainer app config.

    Returns:
        True if this is an RL config, False otherwise.
    """
    import pyine.apps.trainers.hf_rl_trainer_configs  # import here to avoid circular imports

    if isinstance(config, pyine.apps.trainers.hf_rl_trainer_configs.RLTrainerAppMainConfig):
        return True
    # fallback for test mocks: check for grpo_config without training_args_config
    return hasattr(config, "grpo_config") and not hasattr(config, "training_args_config")


def load_model_and_tokenizer_for_prediction(
    config: ModelTokenizerConfigBase,
    resume_artifacts: ResumeArtifacts | None,
    evals_config: pydantic.SerializeAsAny[pyine.evals.common.BaseEvalsConfig] | None = None,
) -> tuple[transformers.PreTrainedModel | None, transformers.PreTrainedTokenizer | None]:
    """Load model and tokenizer for predict-only mode (no training).

    Handles three cases: vLLM provider (skip loading), resume from checkpoint,
    and fresh base model loading.

    Args:
        config: The model/tokenizer config (provides ``get_model``/``get_tokenizer``).
        resume_artifacts: Optional resume artifacts with checkpoint path.
        evals_config: Optional evals config; when its ``vllm_provider_config`` is set,
            model/tokenizer loading is skipped entirely.

    Returns:
        Tuple of (model, tokenizer), both None when using a vLLM provider.
    """
    if evals_config is not None and getattr(evals_config, "vllm_provider_config", None) is not None:
        logger.info("vLLM provider enabled - skipping local model and tokenizer loading")
        return None, None
    if resume_artifacts is not None:
        return (
            config.get_model(checkpoint_path=resume_artifacts.checkpoint_path),
            config.get_tokenizer(checkpoint_path=resume_artifacts.checkpoint_path),
        )
    return config.get_model(), config.get_tokenizer()


def get_vllm_provider_model_name(
    config: AppMainConfig,
) -> str | None:
    """Extracts vLLM provider model name from evals config if present.

    Args:
        config: The trainer app config.

    Returns:
        The vLLM provider model name or None.
    """
    vllm_provider_config = getattr(config.evals_config, "vllm_provider_config", None)
    if vllm_provider_config is None:
        return None
    return vllm_provider_config.model_kwargs.get("model", "default")


def prepare_resume_train_kwargs(
    resume_artifacts: ResumeArtifacts | None,
) -> dict[str, typing.Any]:
    """Prepares train_kwargs with resume checkpoint if available.

    Args:
        resume_artifacts: Resume artifacts from a previous run.

    Returns:
        Dictionary with resume_from_checkpoint if applicable.
    """
    train_kwargs: dict[str, typing.Any] = {}
    if resume_artifacts is not None:
        assert resume_artifacts.checkpoint_path.is_dir(), f"invalid checkpoint path: {resume_artifacts.checkpoint_path}"
        logger.info(f"resuming from checkpoint: {resume_artifacts.checkpoint_path}")
        train_kwargs["resume_from_checkpoint"] = str(resume_artifacts.checkpoint_path)
    return train_kwargs


def create_shutdown_callback(
    config: AppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    shutdown_manager: pyine.utils.interrupts.GracefulShutdownManager | None,
) -> pyine.utils.interrupts.GracefulShutdownCallback | None:
    """Creates a shutdown callback for graceful training interruption.

    Args:
        config: The trainer app config.
        runtime: Runtime configuration.
        shutdown_manager: The shutdown manager instance.

    Returns:
        A GracefulShutdownCallback instance or None if shutdown_manager is None.
    """
    if shutdown_manager is None:
        return None

    def _metadata_writer(checkpoint_dir: pathlib.Path, state: transformers.TrainerState) -> None:
        pyine.utils.transformers.write_checkpoint_metadata(
            checkpoint_dir,
            config=config,
            runtime=runtime,
            state=state,
            shutdown_manager=shutdown_manager,
        )

    return pyine.utils.interrupts.GracefulShutdownCallback(
        shutdown_manager=shutdown_manager,
        metadata_writer=_metadata_writer,
    )


def add_callback_to_trainer(
    trainer: typing.Any,
    callback: typing.Any,
) -> None:
    """Adds a callback to a trainer, handling different trainer types.

    Args:
        trainer: The trainer instance (transformers.Trainer or trl.GRPOTrainer).
        callback: The callback to add.
    """
    if hasattr(trainer, "add_callback"):
        trainer.add_callback(callback)  # type: ignore[reportUnknownMemberType]
    else:
        trainer.callbacks.append(callback)  # type: ignore[reportUnknownMemberType]


def run_training_with_timing(
    trainer: typing.Any,
    train_kwargs: dict[str, typing.Any],
    training_type: str = "training",
) -> None:
    """Runs trainer.train() with timing and logging.

    Args:
        trainer: The trainer instance.
        train_kwargs: Keyword arguments to pass to trainer.train().
        training_type: Label for log messages (e.g., "training", "RL training").
    """
    pyine.utils.distrib.barrier()
    logger.info(f"starting {training_type}")
    start_time = time.time()
    trainer.train(**train_kwargs)  # type: ignore[reportUnknownMemberType]
    end_time = time.time()
    time_delta_seconds = end_time - start_time
    time_delta_str = pyine.utils.timers.get_human_readable_time(time_delta_seconds)
    logger.info(f"{training_type} finished after {time_delta_str}")


def log_shutdown_status(
    shutdown_manager: pyine.utils.interrupts.GracefulShutdownManager | None,
    training_type: str = "training",
) -> None:
    """Logs if training exited early due to a shutdown request.

    Args:
        shutdown_manager: The shutdown manager instance.
        training_type: Label for log messages.
    """
    if shutdown_manager is not None and shutdown_manager.should_terminate():
        logger.info(f"{training_type} run exited early after honoring shutdown request")


def _resolve_export_rank() -> int:
    """Return a validated global rank for per-rank disk export.

    Raises ValueError in ambiguous distributed setups where LOCAL_RANK is set but no authoritative
    global rank (RANK, SLURM_PROCID, torch.distributed) exists. Allows true single-process runs
    (no LOCAL_RANK) by returning 0.

    Note: does NOT use get_global_rank() directly because that function falls back to LOCAL_RANK
    when no explicit global rank env var exists, which would silently return the local rank and
    cause cross-node path collisions.
    """
    if pyine.utils.distrib.has_explicit_global_rank():
        rank = pyine.utils.distrib.get_global_rank(default=None)
        if rank is None:
            raise RuntimeError("has_explicit_global_rank() returned True but get_global_rank() returned None")
        return rank
    # no explicit global rank; only OK if not in distributed mode at all
    if pyine.utils.distrib.get_local_rank(default=None) is not None:
        raise ValueError(
            "export_all_ranks=True requires an authoritative global rank "
            "(RANK, SLURM_PROCID, or torch.distributed), but only LOCAL_RANK is set; "
            "use a launcher that sets global rank env vars (torchrun, srun, deepspeed)"
        )
    return 0  # single-process, non-distributed


@dataclasses.dataclass
class ModelOrganismRewardComponents:
    """Components produced by reward setup for RL training of model organisms."""

    manager: reward_manager_mod.RewardManager
    """Reward manager that orchestrates term evaluation, aggregation, and logging."""
    adapter: reward_trl.TRLRewardAdapter
    """TRL-compatible adapter wrapping the manager for use as a reward function."""
    _disk_logger: reward_logging.DiskRewardLogger | None = dataclasses.field(default=None, repr=False)
    """Optional LMDB-backed logger for exporting generations to disk; closed via ``close()``."""

    def close(self) -> None:
        """Close resources (e.g. flush disk logger LMDB)."""
        if self._disk_logger is not None:
            self._disk_logger.close()


def create_model_organism_reward_components(
    reward_manager_config: reward_configs.RewardManagerConfig,
    tokenizer: typing.Any,
    *,
    generation_export_config: reward_configs.GenerationExportConfig | None = None,
    wandb_run: typing.Any | None = None,
    prompt_key: str = "prompts",
    sample_data_key: str = "sample_data",
    datamodule: typing.Any,
) -> ModelOrganismRewardComponents:
    """Create a reward manager, adapter, and optional loggers for RL training.

    Args:
        reward_manager_config: Configuration for the reward manager.
        tokenizer: Tokenizer instance passed to the reward manager.
        generation_export_config: Optional config for disk export of generations.
        wandb_run: Optional wandb run object for logging.
        prompt_key: Key under which TRL passes prompt strings to the reward function. Must match
            the column name produced by the data generation pipeline (e.g. TRL's GRPOTrainer
            passes "prompts" by default).
        sample_data_key: Key under which per-sample metadata dicts are passed to the reward
            function. Must match the column name emitted by the datamodule HF dataset creation call.
        datamodule: Datamodule reference. ``RewardManager.reset()`` is called after construction
            with this reference, enabling preflight validation in reward terms (e.g.,
            ``TracedReasoningTerm`` checks ``add_line_numbers``).

    Returns:
        A ModelOrganismRewardComponents instance containing the manager, adapter, and optional
        disk logger.

    Raises:
        ValueError: If datamodule is None, or if generation_export_config is set but logging is
            disabled or log_total is False.
    """
    if datamodule is None:
        raise ValueError("datamodule must be provided to initialize reward terms")
    if generation_export_config is not None and not reward_manager_config.logging.enabled:
        raise ValueError(
            "generation_export_config is set but reward_manager_config.logging.enabled=False; "
            "enable logging for export to work (RewardManager gates all logging on this flag)"
        )
    if generation_export_config is not None and not reward_manager_config.logging.log_total:
        raise ValueError(
            "generation_export_config is set but reward_manager_config.logging.log_total=False; "
            "reward_total will be None in exported records"
        )
    if (
        generation_export_config is not None
        and not generation_export_config.export_all_ranks
        and not reward_manager_config.logging.main_process_only
    ):
        raise ValueError(
            "generation_export_config with export_all_ranks=False requires "
            "logging.main_process_only=True to avoid multiple ranks writing to the same LMDB; "
            "set export_all_ranks=True for per-rank exports, or use main_process_only=True"
        )
    if generation_export_config is not None and generation_export_config.export_all_ranks:
        reward_manager_config = reward_manager_config.model_copy(
            update={"logging": reward_manager_config.logging.model_copy(update={"expect_all_rank_logging": True})}
        )
    is_logging_rank = not reward_manager_config.logging.main_process_only or pyine.utils.distrib.is_main_process()
    loggers: list[reward_types.RewardLogger] = []
    # wandb: only on logging rank (rank 0 when main_process_only=True)
    if wandb_run is not None and reward_manager_config.logging.enabled and is_logging_rank:
        loggers.append(reward_logging.make_wandb_reward_logger(wandb_run, reward_manager_config.logging))
    # disk outputs: on ALL ranks when export_all_ranks=True, else only on logging rank
    disk_logger: reward_logging.DiskRewardLogger | None = None
    if generation_export_config is not None:
        should_create_disk = is_logging_rank or generation_export_config.export_all_ranks
        if should_create_disk:
            rank = _resolve_export_rank() if generation_export_config.export_all_ranks else None
            disk_logger = reward_logging.make_disk_reward_logger(generation_export_config, rank=rank)
            loggers.append(disk_logger)
    reward_logger: reward_types.RewardLogger | None = None
    if len(loggers) == 1:
        reward_logger = loggers[0]
    elif loggers:
        reward_logger = reward_logging.CompositeRewardLogger(loggers)
    manager = reward_manager_mod.RewardManager(
        reward_manager_config,
        logger=reward_logger,
        tokenizer=tokenizer,
    )
    manager.reset(reward_types.RunInitContext(datamodule=datamodule))
    adapter = reward_trl.TRLRewardAdapter(
        manager=manager,
        prompt_key=prompt_key,
        sample_data_key=sample_data_key,
        skip_on_error=True,
    )
    return ModelOrganismRewardComponents(manager=manager, adapter=adapter, _disk_logger=disk_logger)


def get_hardware_training_flags() -> dict[str, typing.Any]:
    """Returns hardware-specific training flags based on available devices.

    Detects CUDA/MPS availability and returns appropriate flags for training configuration.
    This is used to configure hardware-specific settings in both SFT and RL trainer configs.

    Returns:
        Dictionary with keys: use_cpu, fp16, bf16, tf32, dataloader_pin_memory
    """
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()
    return {
        "use_cpu": not (is_cuda or is_mps),
        "fp16": bool(is_cuda and not torch.cuda.is_bf16_supported()),
        "bf16": bool(is_cuda and torch.cuda.is_bf16_supported()),
        "tf32": is_cuda,
        "dataloader_pin_memory": bool(is_cuda),
    }


def get_lora_configs(
    group: str,
    app_description: str = "trainer",
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns LoRA adaptor configs for hydra zen storage.

    This shared function is used by both SFT and RL trainer config modules to avoid
    duplicating the LoRA config definition.

    Args:
        group: The config group name (e.g., "config/lora_config").
        app_description: Description fragment for the config (e.g., "trainer", "RL trainer").

    Returns:
        List containing the default LoRA config description.
    """
    default_lora_config = pyine.configs.utils.make_config_description(
        peft.LoraConfig,
        name="default",
        group=group,
        description=(
            f"Default LoRA settings for all {app_description} configs; applies to all models, "
            "and provides a reasonable default for LoRA adaptation. See `peft.LoraConfig` for more details."
        ),
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    return [default_lora_config]


def get_probe_and_correctness_support_configs(
    group: str,
    split_source: str = "TACO",
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Returns shared probe/correctness Hydra configs for trainer apps.

    This helper keeps trainer config modules aligned when they support both:
    - direct probe training from reward LMDBs via ``ProbeDataModuleConfig``;
    - optional training/evaluation on correctness LMDBs via explicit TACO-oriented presets.

    The correctness presets intentionally use stable, explicit names instead of mutating the
    meaning of a generic base config based on local filesystem state.

    TODO: @@@@@
        Once we 'officialize' the release artifacts, add the paths to pregenerated code exec
        eval records (LMDBs) here? (that way we can have fully defined datamodules, like elsewhere

    Args:
        group: Hydra config group path for the app configs.
        split_source: Dataset name or path used to pre-fill guardrail split resolution for the
            correctness presets. Defaults to ``"TACO"``.

    Returns:
        Probe datamodule configs, correctness datamodule configs, and correctness eval configs.
    """
    probe_dm_configs = pyine.guardrails.data.datamodule_configs.get_datamodule_configs(
        f"{group}/datamodule_config",
    )
    correctness_dm_configs = correctness_configs.get_datamodule_configs(
        group=f"{group}/datamodule_config",
        split_source=split_source,
        base_name="correctness_taco_base",
    )
    correctness_eval_configs = pyine.evals.configs.get_evals_configs(
        eval_type=pyine.evals.common.EvalType.CORRECTNESS,
        group=f"{group}/evals_config",
        split_source=split_source,
        base_name="correctness_taco_base",
        datamodule_name="correctness_taco_dm_base",
    )
    return [*probe_dm_configs, *correctness_dm_configs, *correctness_eval_configs]


def strip_deepspeed_local_rank_arg() -> None:
    """Remove ``--local_rank=N`` from ``sys.argv`` to avoid Hydra parse errors.

    DeepSpeed's native launcher passes ``--local_rank=N`` as a CLI argument, but Hydra doesn't
    recognize it. Distributed training already uses the ``LOCAL_RANK`` environment variable (set
    by DeepSpeed/torchrun), so the CLI argument is redundant. This is a no-op when using
    accelerate's standard launcher which only sets env vars.
    """
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank")]


@torch.distributed.elastic.multiprocessing.errors.record
def hydra_main(
    eval_type: pyine.evals.common.EvalType,
    hydra_config_registration_fn: typing.Callable[[pyine.evals.common.EvalType], typing.Any],
    async_main_wrapper: typing.Callable[..., None],
) -> None:
    """Hydra main entrypoint for trainer apps."""
    strip_deepspeed_local_rank_arg()
    pyine.configs.base.register_searchpath_plugin()
    hydra_config_registration_fn(eval_type)
    hydra_zen.zen(async_main_wrapper).hydra_main(
        config_path=None,
        config_name="entrypoint",
        version_base=pyine.configs.base.target_hydra_version,
    )


def async_hf_trainer_main_wrapper(
    config: AppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Wrapper for the async main function of the hf_trainer app."""
    import pyine.apps.trainers.hf_rl_trainer_configs as hf_rl_trainer_configs
    import pyine.apps.trainers.hf_sft_trainer_configs as hf_sft_trainer_configs
    import pyine.apps.trainers.hf_trainer as hf_trainer_app

    supported_app_config_types = (
        hf_rl_trainer_configs.RLTrainerAppMainConfig,
        hf_sft_trainer_configs.SFTTrainerAppMainConfig,
    )
    if not isinstance(config, supported_app_config_types):
        raise ValueError(f"invalid config type: {type(config)}")
    asyncio.run(hf_trainer_app.main(config=config, runtime=runtime))
