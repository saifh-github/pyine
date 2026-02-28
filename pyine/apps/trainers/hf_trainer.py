"""Hydra-integrated HuggingFace fine-tuning CLI app.

This app wires together:
- a tokenizer and base model from hugging face;
- a dataset of (potentially multi-turn) conversations;
- a function that converts conversations into supervised examples;
- a Trainer that handles batching/padding/masking + the training loop.

This trainer supports both SFT and RL training (the latter is based on HuggingFace-TRL), but only
targets the "code execution" task, i.e. training/evaluating models to predict code execution
outcomes. Refer to the probe_trainer or llm_classifier_trainer apps for guardrail training tasks.
"""

from __future__ import annotations

import collections
import logging
import typing

import datasets as hf_datasets
import transformers
import trl

import pyine.apps.trainers.common
import pyine.apps.trainers.hf_rl_trainer_configs as hf_rl_trainer_configs
import pyine.apps.trainers.hf_sft_trainer_configs as hf_sft_trainer_configs
import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.common
import pyine.evals.utils
import pyine.utils.distrib
import pyine.utils.interrupts
import pyine.utils.reprod
import pyine.utils.transformers

logger = logging.getLogger(__name__)

type TRLTrainer = trl.GRPOTrainer  # type: ignore[reportPrivateImportUsage]


def sft_train(
    datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
    config: hf_sft_trainer_configs.SFTTrainerAppMainConfig,
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
    if config.evals_config.category_extraction_config is not None:
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
    else:
        training_args_dict["report_to"] = []  # explicitly disable to prevent auto-detection
    pyine.apps.trainers.common.resolve_save_on_each_node(training_args_dict, runtime)

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
    if config.throughput_logging is not None:
        wandb_run = pyine.apps.trainers.common.get_wandb_run_for_callback(config, runtime, config.throughput_logging)
        if wandb_run is not None:
            throughput_callback = pyine.utils.transformers.ThroughputLoggingCallback(
                config=config.throughput_logging,
                wandb_run=wandb_run,
            )
            callbacks.append(throughput_callback)
    if config.gpu_stats_logging is not None and config.use_wandb_logging:
        wandb_run = pyine.apps.trainers.common.get_wandb_run_for_callback(config, runtime, config.gpu_stats_logging)
        gpu_stats_callback = pyine.utils.transformers.GPUStatsLoggingCallback(
            config=config.gpu_stats_logging,
            wandb_run=wandb_run,
        )
        callbacks.append(gpu_stats_callback)
    train_subset_names = getattr(config.datamodule_config, "train_subset_names", [])
    epoch_callback = pyine.utils.transformers.create_epoch_awareness_callback(
        train_dataset=train_ds,
        datamodule=datamodule,
        subset_names=train_subset_names,
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
    train_kwargs = pyine.apps.trainers.common.prepare_resume_train_kwargs(resume_artifacts)
    shutdown_callback = pyine.apps.trainers.common.create_shutdown_callback(
        config=config,
        runtime=runtime,
        shutdown_manager=shutdown_manager,
    )
    if shutdown_callback is not None:
        pyine.apps.trainers.common.add_callback_to_trainer(trainer, shutdown_callback)
    pyine.apps.trainers.common.run_training_with_timing(trainer, train_kwargs, training_type="SFT training")
    pyine.apps.trainers.common.log_shutdown_status(shutdown_manager, training_type="SFT training")
    return trainer


def rl_train(
    datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
    config: hf_rl_trainer_configs.RLTrainerAppMainConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    resume_artifacts: pyine.apps.trainers.common.ResumeArtifacts | None,
    shutdown_manager: pyine.utils.interrupts.GracefulShutdownManager | None = None,
) -> TRLTrainer:
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
    logger.info("Starting RL training setup...")

    # 1. Setup model and tokenizer
    # For DeepSpeed Zero3: Load the BASE model here, not from checkpoint. The checkpoint weights
    # (sharded across ranks) are restored later by trainer.train(resume_from_checkpoint=...) via
    # DeepSpeed's deepspeed_load_checkpoint(). Loading from checkpoint_path would fail on non-rank-0
    # nodes because only rank 0 has the consolidated safetensors files.
    # For non-DeepSpeed: Load directly from checkpoint if resuming, since all weights are accessible.
    logger.info("instantiating model and tokenizer...")
    is_deepspeed = pyine.apps.trainers.common._is_deepspeed_enabled()  # type: ignore[reportPrivateUsage]
    if resume_artifacts is not None and is_deepspeed:
        logger.info("DeepSpeed detected: loading base model (checkpoint restore handled by trainer)")
        model_checkpoint_path = None  # DeepSpeed handles checkpoint restore via trainer.train()
        tokenizer_checkpoint_path = None  # Load from base model for consistency
    elif resume_artifacts is not None:
        logger.info("Non-DeepSpeed resume: loading model directly from checkpoint")
        model_checkpoint_path = resume_artifacts.checkpoint_path
        tokenizer_checkpoint_path = resume_artifacts.checkpoint_path
    else:
        model_checkpoint_path = None
        tokenizer_checkpoint_path = None
    model = config.get_model(checkpoint_path=model_checkpoint_path)
    tokenizer = config.get_tokenizer(checkpoint_path=tokenizer_checkpoint_path)

    # 2. Prepare RL dataset from datamodule
    # Use the datamodule's native RL dataset method which reuses existing infrastructure
    # Note: prompt configuration (version, include_examples, etc.) comes from datamodule_config.prompt_config
    logger.info("preparing train dataset...")
    train_ds = datamodule.get_hf_messages_dataset(
        subset_name="train",
        append_answer=False,  # no answers for RL
        merge_system_with_user=True,  # to adjust depending on whether we want system messages too
        keep_original_data=True,  # needed to compute rewards
    )
    shuffle_seed = runtime.seed if runtime is not None else config.grpo_config.seed
    train_ds = train_ds.shuffle(seed=shuffle_seed)  # reshuffle, as a precaution, if not done elsewhere
    eval_ds = None
    if config.grpo_config.do_eval:
        logger.info("preparing validation dataset...")
        valid_subset_names = list(config.datamodule_config.valid_subset_names)
        valid_datasets = [
            datamodule.get_hf_messages_dataset(
                subset_name=subset_name,
                append_answer=False,  # no answers for RL
                merge_system_with_user=True,  # to adjust depending on whether we want system messages too
                keep_original_data=True,  # needed to compute rewards
            )
            for subset_name in valid_subset_names
        ]
        eval_ds = valid_datasets[0] if len(valid_datasets) == 1 else hf_datasets.concatenate_datasets(valid_datasets)

    # 3. Create reward manager/function
    logger.info("creating reward function with RewardManager...")
    reward = pyine.apps.trainers.common.create_model_organism_reward_components(
        reward_manager_config=config.reward_manager_config,
        tokenizer=tokenizer,
        generation_export_config=config.generation_export_config,
        wandb_run=runtime.wandb_run if runtime is not None else None,
    )

    # 4. Create TRL trainer
    logger.info("creating GRPO trainer...")
    # override report_to based on wandb availability (similar to SFT trainer logic)
    grpo_config_dict = config.grpo_config.to_dict()
    if runtime is not None and runtime.wandb_run is not None:
        grpo_config_dict["report_to"] = ["wandb"]
    else:
        grpo_config_dict["report_to"] = []  # explicitly disable to prevent auto-detection
    # TRL's __post_init__ auto-computes steps_per_generation from generation_batch_size (or vice versa),
    # but doesn't allow both to be set simultaneously; remove steps_per_generation to avoid conflict
    grpo_config_dict.pop("steps_per_generation", None)
    pyine.apps.trainers.common.resolve_save_on_each_node(grpo_config_dict, runtime)
    grpo_config = trl.GRPOConfig(**grpo_config_dict)  # type: ignore[reportPrivateImportUsage]
    trainer = trl.GRPOTrainer(  # type: ignore[reportPrivateImportUsage]
        model=model,
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=reward.adapter,  # type: ignore[reportArgumentType]  # TRL accepts list[float | None] for skipping
    )

    # 5. Add shutdown callback
    shutdown_callback = pyine.apps.trainers.common.create_shutdown_callback(
        config=config,
        runtime=runtime,
        shutdown_manager=shutdown_manager,
    )
    if shutdown_callback is not None:
        pyine.apps.trainers.common.add_callback_to_trainer(trainer, shutdown_callback)

    # 6. Add reward logging callback for train/eval prefix switching and log flushing
    reward_logging_callback = pyine.utils.transformers.RewardLoggingCallback(
        reward_manager=reward.manager,
        reward_adapter=reward.adapter,
        resume_from_checkpoint=resume_artifacts.checkpoint_path if resume_artifacts else None,
    )
    pyine.apps.trainers.common.add_callback_to_trainer(trainer, reward_logging_callback)

    # 7. Add throughput logging callback if configured
    if config.throughput_logging is not None:
        wandb_run = pyine.apps.trainers.common.get_wandb_run_for_callback(config, runtime, config.throughput_logging)
        if wandb_run is not None:
            throughput_callback = pyine.utils.transformers.ThroughputLoggingCallback(
                config=config.throughput_logging,
                wandb_run=wandb_run,
            )
            pyine.apps.trainers.common.add_callback_to_trainer(trainer, throughput_callback)

    # 8. Add GPU stats logging callback if configured
    if config.gpu_stats_logging is not None and config.use_wandb_logging:
        wandb_run = pyine.apps.trainers.common.get_wandb_run_for_callback(config, runtime, config.gpu_stats_logging)
        gpu_stats_callback = pyine.utils.transformers.GPUStatsLoggingCallback(
            config=config.gpu_stats_logging,
            wandb_run=wandb_run,
        )
        pyine.apps.trainers.common.add_callback_to_trainer(trainer, gpu_stats_callback)

    # 9. Train with resume support
    train_kwargs = pyine.apps.trainers.common.prepare_resume_train_kwargs(resume_artifacts)
    try:
        pyine.apps.trainers.common.run_training_with_timing(trainer, train_kwargs, training_type="RL training")
    finally:
        reward.close()  # to finalize logging, if needed
    pyine.apps.trainers.common.log_shutdown_status(shutdown_manager, training_type="RL training")
    return trainer


async def main(
    config: hf_rl_trainer_configs.RLTrainerAppMainConfig | hf_sft_trainer_configs.SFTTrainerAppMainConfig,
    runtime: (pyine.configs.schemas.RuntimeConfig | None) = None,  # None unless launched via hydra
) -> None:
    """Main function for the script; performs fine-tuning and evaluation for SFT or RL training.

    Args:
        config: Configuration for the application (SFT or RL trainer config).
        runtime: Configuration for the runtime; available when launched via hydra.
    """
    pyine.apps.trainers.common.validate_wandb_sweeper_requirements(config)
    pyine.apps.trainers.common.validate_training_prediction_vllm_compatibility(config)
    persist_runtime_artifacts = pyine.utils.distrib.is_local_main_process()
    is_global_main = pyine.utils.distrib.is_main_process()
    resume_artifacts = pyine.apps.trainers.common.prepare_resume_artifacts(
        config=config,
        runtime=runtime,
        persist_to_runtime=is_global_main,  # fixed-name file copies stay global-rank-0 only
    )
    try:
        # initialize wandb on all ranks only if explicitly requested, otherwise
        # only on global main rank for efficiency. Keep persist_runtime_artifacts separate as it
        # controls file I/O operations (configs, metadata) which should happen on local-rank-0.
        use_wandb_logging = config.use_wandb_logging and (config.wandb_init_on_all_ranks or is_global_main)
        wandb_init_kwargs = resume_artifacts.wandb_resume_kwargs if resume_artifacts and use_wandb_logging else None
        pyine.utils.reprod.entrypoint_setup(
            runtime_config=runtime,
            main_config=config,
            use_wandb_logging=use_wandb_logging,
            wandb_init_kwargs=wandb_init_kwargs,
            wandb_init_on_all_ranks=config.wandb_init_on_all_ranks,
            persist_runtime_artifacts=persist_runtime_artifacts,
            persist_wandb_artifacts=is_global_main,
        )
    except pyine.utils.reprod.DryRunExit:
        return

    datamodule = pyine.apps.trainers.common.prepare_datamodule(config, runtime)
    if not isinstance(datamodule, pyine.data.datamodule.ConversationDataModule):
        raise TypeError(
            f"HuggingFace trainer requires a ConversationDataModule; received {type(datamodule).__name__}",
        )

    with pyine.utils.interrupts.GracefulShutdownManager(log=logger) as shutdown_manager:
        is_rl = pyine.apps.trainers.common.is_rl_config(config)
        do_train, _, do_predict = pyine.apps.trainers.common.get_training_flags(config)
        if do_train:
            if is_rl:
                trainer = rl_train(
                    datamodule=datamodule,
                    config=config,  # type: ignore[arg-type]
                    runtime=runtime,
                    resume_artifacts=resume_artifacts,
                    shutdown_manager=shutdown_manager,
                )
            else:
                trainer = sft_train(
                    datamodule=datamodule,
                    config=config,  # type: ignore[arg-type]
                    runtime=runtime,
                    resume_artifacts=resume_artifacts,
                    shutdown_manager=shutdown_manager,
                )
            model = typing.cast("transformers.PreTrainedModel", trainer.model)  # type: ignore[reportUnknownMemberType]
            tokenizer = typing.cast("transformers.PreTrainedTokenizer", trainer.processing_class)  # type: ignore[reportUnknownMemberType]
        else:
            model, tokenizer = pyine.apps.trainers.common.load_model_and_tokenizer_for_prediction(
                config=config,  # type: ignore[arg-type]
                resume_artifacts=resume_artifacts,
                evals_config=config.evals_config,
            )
        pyine.utils.distrib.barrier()

        # prediction/evaluation phase
        if do_predict and not shutdown_manager.should_terminate():
            if runtime is not None and runtime.wandb_run is not None and pyine.utils.distrib.is_main_process():
                if model is not None:
                    runtime.wandb_run.summary["model_name"] = model.config.name_or_path
                else:  # vLLM provider mode
                    vllm_model_name = pyine.apps.trainers.common.get_vllm_provider_model_name(config)
                    assert vllm_model_name is not None, "could not fetch vllm_provider_config args"
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

    def _register_combined_hydra_configs(
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        # register both SFT and RL configurations in Hydra (main will dispatch based on cfg type)
        hf_sft_trainer_configs.register_hydra_configs(*args, **kwargs)
        hf_rl_trainer_configs.register_hydra_configs(*args, **kwargs)

    # this trainer ONLY supports the code execution task
    pyine.apps.trainers.common.hydra_main(
        eval_type=pyine.evals.common.EvalType.CODE_EXEC,
        hydra_config_registration_fn=_register_combined_hydra_configs,
        async_main_wrapper=pyine.apps.trainers.common.async_hf_trainer_main_wrapper,
    )
