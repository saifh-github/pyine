"""Hydra-integrated HuggingFace fine-tuning CLI app.

This app wires together:
- a tokenizer and base model from hugging face;
- a dataset of (potentially multi-turn) conversations;
- a function that converts conversations into supervised examples;
- a Trainer that handles batching/padding/masking + the training loop.
"""

import collections
import logging
import pathlib
import time
import typing

import transformers

import pyine.apps.trainers.common
import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.common
import pyine.evals.utils
import pyine.utils.distrib
import pyine.utils.interrupts
import pyine.utils.reprod
import pyine.utils.timers
import pyine.utils.transformers

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import pyine.apps.trainers.hf_trainer_configs


def sft_train(
    datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
    config: "pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig",
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    resume_artifacts: pyine.apps.trainers.common.ResumeArtifacts | None,
    shutdown_manager: pyine.utils.interrupts.GracefulShutdownManager | None = None,
) -> transformers.Trainer:
    """Run SFT training with HuggingFace Trainer.

    Args:
        datamodule: The datamodule to fetch data from.
        config: The app config object to fetch settings from.
        runtime: The runtime config object to fetch settings from.
        resume_artifacts: Resume artifacts to continue a previous run, when provided.
        shutdown_manager: The shutdown manager to use for graceful shutdown handling.

    Returns:
        The instantiated trainer object containing a model that can be used for predictions.
    """
    assert config.training_args_config.do_train, "do_train must be True for training"
    logger.info("instantiating model and tokenizer...")
    model = config.get_model()
    tokenizer = config.get_tokenizer()
    model_max_seq_len = pyine.utils.transformers.infer_effective_max_seq_len(model, tokenizer)
    logger.info(f"effective max_seq_len={model_max_seq_len}")
    collator = config.get_collator(
        tokenizer=tokenizer,
        max_seq_len=model_max_seq_len,
        wandb_run=runtime.wandb_run if runtime is not None else None,
    )

    # convert conversation-style rows into flat, tokenized examples for training/valid
    logger.info("preparing train dataset...")
    train_ds = datamodule.get_hf_tokenized_examples_dataset(
        subset_name="train",
        tokenizer=tokenizer,
        model_max_seq_len=model_max_seq_len,
    )
    logger.info("preparing validation dataset...")
    valid_ds = datamodule.get_hf_tokenized_examples_dataset(
        subset_name="valid",
        tokenizer=tokenizer,
        model_max_seq_len=model_max_seq_len,
    )
    valid_sample_categories = pyine.evals.utils.extract_sample_categories_from_dataset(
        valid_ds,
        config=config.evals_config.category_extraction_config,
    )
    assert len(valid_sample_categories) == len(valid_ds) and any(c is not None for c in valid_sample_categories), (
        "could not extract sample categories from validation dataset; check that sample data is preserved?"
    )
    category_counts = collections.Counter(cat for cats in valid_sample_categories for cat in cats)
    category_counts_str = "\n\t".join(f"{key}: {val}" for key, val in dict(category_counts).items())
    logger.debug(f"validation data sample category counts:\n\t{category_counts_str}")
    eval_metrics_callback = pyine.evals.utils.build_category_wise_compute_metrics_fn(
        data_sample_categories=valid_sample_categories,
        log_fn=logger.info,
        runtime=runtime,
    )
    training_args_dict = config.training_args_config.model_dump()
    if training_args_dict.get("batch_eval_metrics") is not None:
        logger.warning("batch_eval_metrics is being overridden by the trainer for compatibility with callbacks")
    training_args_dict["batch_eval_metrics"] = True  # for compat w/ the eval_metrics_callback
    if runtime is not None and runtime.wandb_run is not None:
        training_args_dict["report_to"] = ["wandb"]

    # note: if we want to support other trainers (e.g. TRL), update config dict+trainer w/ instantiable classes
    training_args = transformers.TrainingArguments(**training_args_dict)
    save_strategy = getattr(training_args, "save_strategy", None)
    if save_strategy == transformers.trainer_utils.IntervalStrategy.NO:
        raise ValueError(
            "training_args_config.save_strategy cannot be 'no' when graceful interruption handling is enabled",
        )
    save_steps_value = getattr(training_args, "save_steps", None)
    if save_strategy == transformers.trainer_utils.IntervalStrategy.STEPS and (
        save_steps_value is None or save_steps_value <= 0
    ):
        raise ValueError("training_args_config.save_steps must be > 0 when save_strategy='steps'")
    milestone_logger = pyine.utils.transformers.StdoutMilestones(print_fn=logger.info)
    callbacks: list[transformers.TrainerCallback] = [milestone_logger, eval_metrics_callback]
    epoch_callback = pyine.utils.transformers.create_epoch_awareness_callback(
        train_dataset=train_ds,
        datamodule=datamodule,
        subset_names=getattr(config.datamodule_config, "train_subset_names", []),
    )
    if epoch_callback is not None:
        callbacks.append(epoch_callback)
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=eval_metrics_callback,
        callbacks=callbacks,
    )
    train_kwargs: dict[str, typing.Any] = {}
    if resume_artifacts is not None:
        assert resume_artifacts.checkpoint_path.is_dir(), f"invalid ckpt path: {resume_artifacts.checkpoint_path}"
        logger.info(f"resuming from checkpoint: {resume_artifacts.checkpoint_path}")
        train_kwargs["resume_from_checkpoint"] = str(resume_artifacts.checkpoint_path)

    def metadata_writer(checkpoint_dir: pathlib.Path, state: transformers.TrainerState) -> None:
        pyine.utils.transformers.write_checkpoint_metadata(
            checkpoint_dir,
            config=config,
            runtime=runtime,
            state=state,
            shutdown_manager=shutdown_manager,
        )

    if shutdown_manager is not None:
        shutdown_callback = pyine.utils.interrupts.GracefulShutdownCallback(
            shutdown_manager=shutdown_manager,
            metadata_writer=metadata_writer,
        )
        if hasattr(trainer, "add_callback"):
            trainer.add_callback(shutdown_callback)  # type: ignore[reportUnknownMemberType]
        else:
            typing.cast("typing.Any", trainer).callbacks.append(shutdown_callback)
    pyine.utils.distrib.barrier()
    logger.info("starting training")
    start_time = time.time()
    trainer.train(**train_kwargs)  # type: ignore[reportUnknownMemberType]
    end_time = time.time()
    time_delta_seconds = end_time - start_time
    time_delta_str = pyine.utils.timers.get_human_readable_time(time_delta_seconds)
    logger.info(f"training finished after {time_delta_str}")
    if shutdown_manager is not None and shutdown_manager.should_terminate():
        logger.info("training run exited early after honoring shutdown request")
    return trainer


async def rl_train(
    datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
    config: "pyine.apps.trainers.rl_trainer_configs.RLTrainerAppMainConfig",
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    resume_artifacts: pyine.apps.trainers.common.ResumeArtifacts | None,
    shutdown_manager: pyine.utils.interrupts.GracefulShutdownManager | None = None,
) -> typing.Any:  # Returns TRL GRPOTrainer
    """Run RL training with TRL trainers.

    Args:
        datamodule: The datamodule to fetch data from.
        config: The RL trainer config object to fetch settings from.
        runtime: The runtime config object to fetch settings from.
        resume_artifacts: Resume artifacts to continue a previous run, when provided.
        shutdown_manager: The shutdown manager to use for graceful shutdown handling.

    Returns:
        The instantiated TRL trainer object.
    """
    import trl

    import pyine.apps.trainers.rl.data_utils
    import pyine.apps.trainers.rl.rewards

    logger.info("Starting RL training setup...")

    # 1. Setup model and tokenizer
    logger.info("instantiating model and tokenizer...")
    checkpoint_path = resume_artifacts.checkpoint_path if resume_artifacts else None
    model = config.get_model(checkpoint_path=checkpoint_path)
    tokenizer = config.get_tokenizer(checkpoint_path=checkpoint_path)

    # 2. Prepare RL dataset from datamodule
    logger.info("preparing train dataset...")
    train_ds = pyine.apps.trainers.rl.data_utils.prepare_grpo_dataset_from_datamodule(
        datamodule=datamodule,
        subset_name="train",
        tokenizer=tokenizer,
        prompt_version=config.prompt_version,
        include_examples=config.include_prompt_examples,
        use_cache=config.cache_config.use_cache,
        force_regenerate=config.cache_config.force_regenerate,
        cache_dir=config.cache_config.cache_dir,
    )

    eval_ds = None
    if config.grpo_config.do_eval:
        logger.info("preparing validation dataset...")
        eval_ds = pyine.apps.trainers.rl.data_utils.prepare_grpo_dataset_from_datamodule(
            datamodule=datamodule,
            subset_name="valid",
            tokenizer=tokenizer,
            prompt_version=config.prompt_version,
            include_examples=config.include_prompt_examples,
            use_cache=config.cache_config.use_cache,
            force_regenerate=config.cache_config.force_regenerate,
            cache_dir=config.cache_config.cache_dir,
        )

    # 3. Create reward function
    logger.info("creating reward function...")
    reward_fn = pyine.apps.trainers.rl.rewards.create_code_exec_reward_function(
        strip_hard_checks=config.reward_config.strip_whitespace,
        enable_soft_match=config.reward_config.enable_soft_match,
        hard_reward=config.reward_config.hard_match_reward,
        soft_reward=config.reward_config.soft_match_reward,
        fail_reward=config.reward_config.fail_reward,
        expected_outputs_key=config.reward_config.expected_outputs_key,
    )

    # 4. Validate vLLM server (if enabled)
    if config.vllm_config and config.vllm_config.enabled:
        logger.info(f"checking vLLM server at {config.vllm_config.server_url}...")
        import requests

        try:
            response = requests.get(f"{config.vllm_config.server_url}/health")
            assert response.status_code == 200
            logger.info("vLLM server is ready")
        except Exception as e:
            raise RuntimeError(
                f"vLLM server not accessible at {config.vllm_config.server_url}. "
                f"Please start server with: trl vllm-serve --model {config.base_model}"
            ) from e

    # 5. Create TRL trainer
    logger.info("creating GRPO trainer...")
    trainer = trl.GRPOTrainer(
        model=model,  # Pass model object (not string!)
        args=config.grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=reward_fn,
    )

    # 6. Add callbacks (reuse from common.py patterns)
    callbacks: list[transformers.TrainerCallback] = []
    if shutdown_manager is not None:

        def metadata_writer(checkpoint_dir: pathlib.Path, state: transformers.TrainerState) -> None:
            pyine.utils.transformers.write_checkpoint_metadata(
                checkpoint_dir,
                config=config,
                runtime=runtime,
                state=state,
                shutdown_manager=shutdown_manager,
            )

        shutdown_callback = pyine.utils.interrupts.GracefulShutdownCallback(
            shutdown_manager=shutdown_manager,
            metadata_writer=metadata_writer,
        )
        callbacks.append(shutdown_callback)

    for callback in callbacks:
        trainer.add_callback(callback)

    # 7. Train with resume support
    train_kwargs: dict[str, typing.Any] = {}
    if resume_artifacts is not None:
        assert resume_artifacts.checkpoint_path.is_dir(), f"invalid ckpt path: {resume_artifacts.checkpoint_path}"
        logger.info(f"resuming from checkpoint: {resume_artifacts.checkpoint_path}")
        train_kwargs["resume_from_checkpoint"] = str(resume_artifacts.checkpoint_path)

    pyine.utils.distrib.barrier()
    logger.info("starting RL training")
    start_time = time.time()
    trainer.train(**train_kwargs)  # type: ignore[reportUnknownMemberType]
    end_time = time.time()
    time_delta_seconds = end_time - start_time
    time_delta_str = pyine.utils.timers.get_human_readable_time(time_delta_seconds)
    logger.info(f"RL training finished after {time_delta_str}")
    if shutdown_manager is not None and shutdown_manager.should_terminate():
        logger.info("RL training run exited early after honoring shutdown request")
    return trainer


async def main(
    config: typing.Union[
        "pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig",
        "pyine.apps.trainers.rl_trainer_configs.RLTrainerAppMainConfig",
    ],
    runtime: (pyine.configs.schemas.RuntimeConfig | None) = None,  # None unless launched via hydra
) -> None:
    """Main function for the script; performs fine-tuning and evaluation for SFT or RL training.

    Args:
        config: Configuration for the application (SFT or RL trainer config).
        runtime: Configuration for the runtime; available when launched via hydra.
    """
    import pyine.apps.trainers.hf_trainer_configs
    import pyine.apps.trainers.rl_trainer_configs

    pyine.apps.trainers.common.validate_wandb_sweeper_requirements(config)
    # Only validate vLLM compatibility for SFT configs (RL has different vLLM handling)
    if isinstance(config, pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig):
        pyine.apps.trainers.common.validate_training_prediction_vllm_compatibility(config)
    persist_runtime_artifacts = pyine.utils.distrib.is_main_process()
    resume_artifacts = pyine.apps.trainers.common.prepare_resume_artifacts(
        config=config,
        runtime=runtime,
        persist_to_runtime=persist_runtime_artifacts,
    )
    try:
        use_wandb_logging = config.use_wandb_logging and persist_runtime_artifacts
        wandb_init_kwargs = resume_artifacts.wandb_resume_kwargs if resume_artifacts and use_wandb_logging else None
        pyine.utils.reprod.entrypoint_setup(
            runtime_config=runtime,
            main_config=config,
            use_wandb_logging=use_wandb_logging,
            wandb_init_kwargs=wandb_init_kwargs,
            persist_runtime_artifacts=persist_runtime_artifacts,
        )
    except pyine.utils.reprod.DryRunExit:
        return

    datamodule = pyine.apps.trainers.common.prepare_datamodule(config, runtime)
    if not isinstance(datamodule, pyine.data.datamodule.ConversationDataModule):
        raise TypeError(
            f"HuggingFace trainer requires a ConversationDataModule; received {type(datamodule).__name__}",
        )

    with pyine.utils.interrupts.GracefulShutdownManager(log=logger) as shutdown_manager:
        # Dispatch to appropriate trainer based on config type
        if isinstance(config, pyine.apps.trainers.rl_trainer_configs.RLTrainerAppMainConfig):
            # RL training path
            if config.grpo_config.do_train:
                trainer = await rl_train(
                    datamodule=datamodule,
                    config=config,
                    runtime=runtime,
                    resume_artifacts=resume_artifacts,
                    shutdown_manager=shutdown_manager,
                )
                model = typing.cast("transformers.PreTrainedModel", trainer.model)  # type: ignore[reportUnknownMemberType]
                tokenizer = typing.cast("transformers.PreTrainedTokenizer", trainer.processing_class)  # type: ignore[reportUnknownMemberType]
            else:
                # Eval-only mode for RL
                if resume_artifacts is not None:
                    model = config.get_model(checkpoint_path=resume_artifacts.checkpoint_path)
                    tokenizer = config.get_tokenizer(checkpoint_path=resume_artifacts.checkpoint_path)
                else:
                    model = config.get_model()
                    tokenizer = config.get_tokenizer()
        elif isinstance(config, pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig):
            # SFT training path
            if config.training_args_config.do_train:
                trainer = sft_train(
                    datamodule=datamodule,
                    config=config,
                    runtime=runtime,
                    resume_artifacts=resume_artifacts,
                    shutdown_manager=shutdown_manager,
                )
                model = typing.cast(
                    "transformers.PreTrainedModel",
                    trainer.model,  # type: ignore[reportUnknownMemberType]
                )
                tokenizer = typing.cast(
                    "transformers.PreTrainedTokenizer",
                    trainer.processing_class,  # type: ignore[reportUnknownMemberType]
                )
            else:
                # Check if using vLLM provider - if so, skip model loading to save GPU memory
                evals_config = getattr(config, "evals_config", None)
                vllm_provider_config = (
                    getattr(evals_config, "vllm_provider_config", None) if evals_config is not None else None
                )
                if vllm_provider_config is not None:
                    logger.info("vLLM provider enabled - skipping local model and tokenizer loading")
                    model = None  # type: ignore[assignment]
                    tokenizer = None  # type: ignore[assignment]
                    # Note: tokenizer not needed - prompt chain handles formatting internally
                elif resume_artifacts is not None:
                    # Load model and tokenizer from checkpoint
                    model = config.get_model(checkpoint_path=resume_artifacts.checkpoint_path)
                    tokenizer = config.get_tokenizer(checkpoint_path=resume_artifacts.checkpoint_path)
                else:
                    # Load base (pretrained) model and tokenizer
                    model = config.get_model()
                    tokenizer = config.get_tokenizer()
        else:
            raise TypeError(f"Unknown config type: {type(config)}")
        pyine.utils.distrib.barrier()

        # Determine whether to do prediction based on config type
        do_predict = (
            config.training_args_config.do_predict
            if isinstance(config, pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig)
            else config.grpo_config.do_predict
        )

        if do_predict and not shutdown_manager.should_terminate():
            if runtime is not None and runtime.wandb_run is not None and pyine.utils.distrib.is_main_process():
                if model is not None:
                    runtime.wandb_run.summary["model_name"] = model.config.name_or_path
                else:
                    # vLLM provider mode - log the vLLM server model name
                    evals_config = getattr(config, "evals_config", None)
                    vllm_provider_config = (
                        getattr(evals_config, "vllm_provider_config", None) if evals_config is not None else None
                    )
                    assert vllm_provider_config is not None, "vllm_provider_config must be set when model is None"
                    vllm_model_name = vllm_provider_config.model_kwargs.get("model", "default")
                    runtime.wandb_run.summary["model_name"] = vllm_model_name
            if pyine.utils.distrib.is_main_process():
                await pyine.apps.trainers.common.evaluate_model(
                    model=model,
                    tokenizer=tokenizer,  # type: ignore[reportUnknownArgumentType]
                    datamodule=datamodule,
                    config=config,
                    runtime=runtime,
                    # TODO: maybe pass the shutdown_manager to the eval function too? (or let it die?)
                )
            pyine.utils.distrib.barrier()
        elif do_predict:
            logger.info("skipping evaluation because graceful shutdown was requested")

    if runtime is not None and pyine.utils.distrib.is_main_process():
        runtime.finalize()


if __name__ == "__main__":
    import pyine.apps.trainers.hf_trainer_configs

    # TODO: if we ever have more than one eval type, make new entrypoint scripts w/ different eval types
    pyine.apps.trainers.hf_trainer_configs.hydra_main(pyine.evals.common.EvalType.CODE_EXEC)
