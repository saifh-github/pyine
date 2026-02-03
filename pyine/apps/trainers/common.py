from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import pathlib
import shutil
import time
import typing

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
import pyine.evals.utils
import pyine.utils.distrib
import pyine.utils.interrupts
import pyine.utils.langchain
import pyine.utils.portability
import pyine.utils.reprod
import pyine.utils.timers
import pyine.utils.tokenizers
import pyine.utils.transformers

logger = logging.getLogger(__name__)


def _resolve_attn_implementation(
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
    if is_rl_config(config):
        return  # nothing more to check, can run with/without vllm config
    # if we're doing SFT, make sure everything is compatible
    do_train, _do_eval, do_predict = get_training_flags(config)
    has_vllm_evals_config = config.evals_config.vllm_provider_config is not None
    if do_train and do_predict and has_vllm_evals_config:
        raise ValueError(
            "Invalid configuration: cannot run SFT training and vLLM-based prediction in the same run. "
            "When using vLLM provider, you must manually start the vLLM server with the trained "
            "checkpoint between training and prediction. Please choose one of these options:\n"
            "  Option A: Train only (do_train=True, do_predict=False), then manually start vLLM "
            "server with the checkpoint, then run prediction only (do_train=False, do_predict=True, "
            "vllm_provider_config=<config>)\n"
            "  Option B: Train and predict in one run without vLLM (do_train=True, do_predict=True, "
            "vllm_provider_config=null)",
        )


class ModelTokenizerConfigBase(pydantic.BaseModel):
    """Base configuration providing model and tokenizer fields shared by SFT and RL trainers.

    This base class should be inherited by trainer configs that need to instantiate HuggingFace
    models and tokenizers. Note that subclasses must implement the `target_dtype` property since it
    depends on trainer-specific config fields.

    @@@@ TODO: should we try to make this not specific to causal language models? (e.g. for probing/classifs?)
    """

    # --------------- model settings ---------------

    base_model: str = pydantic.Field(
        ...,  # MISSING! MANDATORY!
        description="Hugging Face model identifier or local path for the base causal LM to fine-tune.",
    )
    auto_model_config: dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: typing.cast("dict[str, typing.Any]", {}),
        description="Model configuration args passed to `transformers.AutoModelForCausalLM.from_pretrained`.",
    )
    quantization_mode: typing.Literal["qlora", "none"] = pydantic.Field(
        default="none",
        description="'Quantization mode; 'qlora' loads the model in 4-bit, and 'none' disables quantization.'",
    )
    lora_config: peft.LoraConfig | pyine.utils.transformers.LoraConfig | None = pydantic.Field(
        default=None,
        description="LoRA adapter configuration; if None, does not apply LoRA.",
    )

    # --------------- tokenizer settings ---------------

    auto_tokenizer_config: dict[str, typing.Any] = pydantic.Field(
        default_factory=lambda: {"use_fast": True},
        description="Tokenizer configuration args passed to `transformers.AutoTokenizer.from_pretrained`.",
    )
    tokenizer_set_padding_to_eos_if_needed: bool = pydantic.Field(
        default=True,
        description="If True and tokenizer has no PAD token, reuse EOS token as PAD for batching.",
    )
    tokenizer_override_padding_to_right_side: bool = pydantic.Field(
        default=True,
        description="Override whichever the tokenizer's default padding side is to 'right'.",
    )
    tokenizer_override_truncation_to_left_side: bool = pydantic.Field(
        default=True,
        description="Override whichever the tokenizer's default truncation side is to 'left'.",
    )

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
        """Returns a model to use for experiments.

        Args:
            checkpoint_path: Optional path to a checkpoint to load from. If provided, loads the
                model from the checkpoint (with LoRA adapters if present). If None, loads the base
                pretrained model.

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
            if config.use_wandb_logging and not wandb_resume_kwargs:
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
        import deepdiff

        current_payload = current_config.normalize_for_resume_overlap_check()
        previous_payload = current_config.normalize_for_resume_overlap_check(previous_config)
        # TODO: config matching/validation needs to be adapted to properly ignore differences due to
        # pydantic runtime fields vs config fields, which is currently causing false positives at resume

        # Use deepdiff for comparison (handles order-independent list comparison and type coercion)
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
    if runtime is not None and persist_to_runtime:
        assert runtime.output_dir_path.is_dir(), "invalid runtime config output dir"
        runtime.metadata["resumed_from_run_dir"] = str(resume_artifacts.run_dir)
        runtime.metadata["resume_checkpoint_path"] = str(resume_artifacts.checkpoint_path)
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
    dm.setup()
    if (
        config.use_wandb_logging
        and runtime is not None
        and runtime.wandb_run is not None
        and pyine.utils.distrib.is_main_process()
    ):
        assert runtime is not None and runtime.wandb_run is not None
        target_subsets = [
            *config.datamodule_config.train_subset_names,
            *config.datamodule_config.valid_subset_names,
            *config.datamodule_config.eval_subset_names,
        ]
        dm_stats = dm.get_stats(target_subsets)
        summary_stats = {f"dataset_stats/{k}": v for k, v in dm_stats.items()}
        runtime.wandb_run.summary.update(summary_stats)  # type: ignore[reportUnknownMemberType]
        for eval_subset_name in config.datamodule_config.eval_subset_names:
            config.evals_config.define_metrics_for_wandb(
                wandb_run=runtime.wandb_run,
                prefix=f"predict/{eval_subset_name}",
            )
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
    import os

    # Check environment variables set by Accelerate
    if os.environ.get("ACCELERATE_USE_DEEPSPEED"):
        return True

    # Try to check Accelerate's state
    try:
        from accelerate import PartialState

        state = PartialState()
        if hasattr(state, "distributed_type"):
            from accelerate.utils import DistributedType

            if state.distributed_type == DistributedType.DEEPSPEED:
                return True
    except (ImportError, RuntimeError):
        # Accelerate not available or not initialized yet
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

    resolved_auto_config = _resolve_attn_implementation(auto_model_config)

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
    model: pyine.evals.utils.InvocableModelChain | transformers.PreTrainedModel | None,
    tokenizer: transformers.PreTrainedTokenizer | None,
    datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
    config: AppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> dict[str, pyine.evals.common.EvalResult]:
    """Evaluates the given model on the specified data subset using the internal evals config.

    Args:
        model: The model to evaluate, in either a langchain runnable or in HF-transformers format.
            Can be None when using vLLM provider (model is served remotely via vLLM server).
        tokenizer: The tokenizer to use for evaluation, if applicable (only for HF-T models).
        datamodule: The datamodule from which to load the evaluation data.
        config: The application configuration, which should contain the evals config.
        runtime: The runtime configuration, which may contain W&B run information.

    Returns:
        The evaluation results as a dictionary indexed by evaluated subset name.
    """
    evaluation_results: dict[str, pyine.evals.common.EvalResult] = {}
    if config.evals_config.eval_type is None:
        return evaluation_results
    start_time = time.time()
    # Check if vLLM provider is configured
    vllm_provider_config = getattr(config.evals_config, "vllm_provider_config", None)
    if vllm_provider_config is not None:
        # vLLM provider mode: use standard runnable chain evaluation
        if model is not None:
            raise ValueError("model should be None when using vLLM provider mode")
        # Get vLLM model from provider config and build prompt chain
        vllm_model = vllm_provider_config.get_model()  # type: ignore[reportUnknownMemberType]
        chain = datamodule.config.get_prompt_chain(vllm_model)
        for eval_subset_name in config.datamodule_config.eval_subset_names:
            logger.info(f"running vLLM chain evaluation on the {eval_subset_name} subset...")
            evaluation_result = await config.evals_config.evaluate_runnable_model(
                chain=chain,
                datamodule=datamodule,
                eval_subset_name=eval_subset_name,
                verbose=True,
            )
            assert isinstance(evaluation_result, pyine.evals.common.EvalResult)
            pyine.evals.utils.print_metrics(evaluation_result.metrics, eval_subset_name, logger.info)
            evaluation_results[eval_subset_name] = evaluation_result
    elif pyine.utils.transformers.is_hf_model(model):
        # Local HF model mode: use model.generate()
        if tokenizer is None or not pyine.utils.transformers.is_hf_tokenizer(tokenizer):
            raise ValueError("invalid tokenizer (need to provide one to evaluate hf model")
        if config.evals_config.eval_padding_side != tokenizer.padding_side:
            new_padding_side = config.evals_config.eval_padding_side
            logger.debug(f"overriding tokenizer padding side to '{new_padding_side}' for evals")
            tokenizer.padding_side = new_padding_side
        hf_model = typing.cast("transformers.PreTrainedModel", model)
        for eval_subset_name in config.datamodule_config.eval_subset_names:
            logger.info(f"running model evaluation on the {eval_subset_name} subset...")
            evaluation_result = await config.evals_config.evaluate_hf_model(
                model=hf_model,
                tokenizer=tokenizer,
                datamodule=datamodule,
                eval_subset_name=eval_subset_name,
                verbose=True,
            )
            assert isinstance(evaluation_result, pyine.evals.common.EvalResult)
            pyine.evals.utils.print_metrics(evaluation_result.metrics, eval_subset_name, logger.info)
            evaluation_results[eval_subset_name] = evaluation_result
    else:
        # Not using vLLM and not an HF model - must be a langchain runnable
        if model is None:
            raise ValueError("model cannot be None when not using vLLM provider mode")
        if not pyine.utils.langchain.is_invocable_chain(model):
            raise ValueError(f"invalid model ({type(model)})")
        if tokenizer is not None:
            raise NotImplementedError("tokenizer support in runnable chain eval is not implemented")
        chain_model = typing.cast("pyine.evals.utils.InvocableModelChain", model)
        for eval_subset_name in config.datamodule_config.eval_subset_names:
            logger.info(f"running chain evaluation on the {eval_subset_name} subset...")
            evaluation_result = await config.evals_config.evaluate_runnable_model(
                chain=chain_model,
                datamodule=datamodule,
                eval_subset_name=eval_subset_name,
                verbose=True,
            )
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

    Handles both SFT configs (with training_args_config) and RL configs (with grpo_config).

    Args:
        config: The trainer app config.

    Returns:
        Tuple of (do_train, do_eval, do_predict) booleans.
    """
    # check for RL config (grpo_config)
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


def get_vllm_provider_model_name(
    config: AppMainConfig,
) -> str | None:
    """Extracts vLLM provider model name from evals config if present.

    Args:
        config: The trainer app config.

    Returns:
        The vLLM provider model name or None.
    """
    if config.evals_config.vllm_provider_config is None:
        return None
    return config.evals_config.vllm_provider_config.model_kwargs.get("model", "default")


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
    """Logs if training exited early due to shutdown request.

    Args:
        shutdown_manager: The shutdown manager instance.
        training_type: Label for log messages.
    """
    if shutdown_manager is not None and shutdown_manager.should_terminate():
        logger.info(f"{training_type} run exited early after honoring shutdown request")


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


@torch.distributed.elastic.multiprocessing.errors.record
def hydra_main(
    eval_type: pyine.evals.common.EvalType,
    hydra_config_registration_fn: typing.Callable[[pyine.evals.common.EvalType], typing.Any],
    async_main_wrapper: typing.Callable[..., None],
) -> None:
    """Hydra main entrypoint for trainer apps."""
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
