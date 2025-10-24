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
import pyine.utils.langchain
import pyine.utils.timers
import pyine.utils.transformers

logger = logging.getLogger(__name__)


# @@@@@ TODO: start thinking about hyperparameter sweep impl


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

    # --------------- public / utility / helper functions ---------------

    def is_resuming(self) -> bool:
        """Returns whether the application configuration indicates that a run is being resumed."""
        return self.resume_from_run_dir is not None

    def normalize_for_resume_overlap_check(self) -> dict[str, typing.Any]:
        """Normalizes the config by removing fields that might change without effects on experiments."""
        # derives classes might want to override this and add new pops for other non-important attributes
        data = self.model_dump(mode="json")
        data.pop("use_wandb_logging", None)
        data.pop("resume_from_run_dir", None)
        data.pop("resume_checkpoint_name", None)
        data.pop("resume_wandb_behavior", None)
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
    previous_config: pydantic.SerializeAsAny[AppMainConfig]
    """Configuration object from the previous run, loaded from the logged config file."""
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

    @classmethod
    def create(
        cls,
        config: AppMainConfig,
    ) -> ResumeArtifacts:
        """Creates a ResumeArtifacts object from a config that should contain resume-related args.

        Args:
            config: The application configuration, which should contain the resume settings. If it
                does not, an error will be raised.

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
        previous_config_payload = typing.cast("dict[str, typing.Any] | None", config_payload.get("main_config"))
        if previous_config_payload is None:
            raise ValueError("resume directory is missing the logged main_config payload")
        previous_config = config.__class__.model_validate(previous_config_payload)
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
        return ResumeArtifacts(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            previous_config=previous_config,
            previous_config_path=config_path,
            previous_runtime_dict=runtime_payload,
            previous_runtime_path=runtime_path,
            previous_metadata_dict=metadata_payload,
            previous_metadata_path=metadata_path,
            wandb_resume_kwargs=wandb_resume_kwargs,
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
    def _ensure_resume_config_matches(
        current_config: AppMainConfig,
        previous_config: AppMainConfig,
    ) -> None:
        current_payload = current_config.normalize_for_resume_overlap_check()
        previous_payload = previous_config.normalize_for_resume_overlap_check()
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
) -> ResumeArtifacts | None:
    """Prepares resume artifacts (if resuming is requested) and returns them.

    Will also update the runtime config with the resume artifacts if resuming is requested.
    """
    if not config.is_resuming():
        return None
    resume_artifacts = ResumeArtifacts.create(config)
    logger.info(f"prepared resume artifacts from run dir: {resume_artifacts.run_dir}")
    logger.debug(f"will resume training from checkpoint: {resume_artifacts.checkpoint_path}")
    if resume_artifacts.wandb_resume_kwargs and "id" in resume_artifacts.wandb_resume_kwargs:
        logger.debug(f"will resume wandb run id: {resume_artifacts.wandb_resume_kwargs['id']}")
    elif config.use_wandb_logging:
        logger.debug("will create a new wandb run for resumed training")
    if runtime is not None:
        assert runtime.output_dir_path.is_dir(), "invalid runtime config output dir"
        runtime.metadata["resumed_from_run_dir"] = str(resume_artifacts.run_dir)
        runtime.metadata["resume_checkpoint_path"] = str(resume_artifacts.checkpoint_path)
        file_mappings: list[tuple[pathlib.Path | None, str]] = [
            (resume_artifacts.previous_config_path, "previous_config.json"),
            (resume_artifacts.previous_runtime_path, "previous_runtime.json"),
            (resume_artifacts.previous_metadata_path, "previous_reprod_metadata.json"),
        ]
        for source_path, target_name in file_mappings:
            if source_path is None or not source_path.is_file():
                continue
            target_path = runtime.output_dir_path / target_name
            shutil.copy2(source_path, target_path)
    # @@@@ TODO: add warnings if resuming from different commit/seed?
    return resume_artifacts


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
    dm.setup()
    if config.use_wandb_logging:
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
