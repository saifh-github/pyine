from __future__ import annotations

import json
import logging
import pathlib
import shutil
import time
import typing

import pydantic
import transformers

import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.common
import pyine.evals.utils
import pyine.utils.distrib
import pyine.utils.langchain
import pyine.utils.portability
import pyine.utils.reprod
import pyine.utils.timers
import pyine.utils.transformers

logger = logging.getLogger(__name__)


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


class AppMainConfig(pydantic.BaseModel):
    """Trainer application main entrypoint configuration settings.

    Should apply to all trainers that intend to train/evaluate models.
    """

    model_config = pydantic.ConfigDict(extra="allow")
    """Pydantic model configuration (allow extra fields)."""

    datamodule_config: pydantic.SerializeAsAny[pyine.data.datamodule.BaseDataModuleConfig]
    """Configuration for the datamodule to use."""
    evals_config: pydantic.SerializeAsAny[pyine.evals.common.BaseEvalsConfig]
    """Configuration for the task evaluation strategy to use."""
    use_wandb_logging: bool = False
    """Whether to use W&B logging for the fine-tuning job (via the post-hoc sync approach)."""

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
        current_payload = current_config.normalize_for_resume_overlap_check()
        previous_payload = current_config.normalize_for_resume_overlap_check(previous_config)
        if current_payload == previous_payload:
            return
        diff_keys: list[str] = []
        previous_keys = set(previous_payload.keys())
        for key, current_value in current_payload.items():
            if key not in previous_payload:
                diff_keys.append(key)
                continue
            if previous_payload[key] != current_value:
                diff_keys.append(key)
        for key in previous_keys:
            if key not in current_payload:
                diff_keys.append(key)
        diff_keys = sorted(set(diff_keys))
        diff_keys_str = ", ".join(diff_keys)
        raise ValueError(
            f"resume configuration mismatch detected; differing top-level fields: {diff_keys_str}. "
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
    """Prepares the configured datamodule and returns it.

    Args:
        config: The application configuration, which should contain the datamodule config.
        runtime: The runtime configuration, which may contain W&B run information.

    Returns:
        The instantiated, prepared, and set-up datamodule that is ready to provide data loaders.
    """
    logger.info("preparing datamodule and setting up parsers/loaders...")
    dm = config.datamodule_config.instantiate_datamodule(verbose=True)
    dm.prepare_data()
    pyine.utils.distrib.barrier()  # wait for all processes to finish preparing data
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


async def evaluate_model(
    model: pyine.evals.utils.InvocableModelChain | transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer | None,
    datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
    config: AppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> dict[str, pyine.evals.common.EvalResult]:
    """Evaluates the given model on the specified data subset using the internal evals config.

    Args:
        model: The model to evaluate, in either a langchain runnable or in HF-transformers format.
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
    if pyine.utils.transformers.is_hf_model(model):
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
            logger.info(f"logging predictions for {subset_name} subset to wandb run id: {runtime.wandb_run_id}...")
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
