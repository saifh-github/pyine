"""OpenAI API model fine-tuning CLI app."""

import logging
import sys
import typing

import openai
import wandb
import wandb.integration.openai.fine_tuning

import pyine.apps.trainers.common
import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.common
import pyine.utils.llm_providers
import pyine.utils.openai
import pyine.utils.reprod
import pyine.utils.tokenizers

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import pyine.apps.trainers.openai_finetune_configs


def _compute_estimated_train_token_count(
    config: "pyine.apps.trainers.openai_finetune_configs.OpenAIFineTuneAppMainConfig",
    datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
) -> int:
    """Approximate the number of tokens used to train a model on the given dataset."""
    finetuner_params = typing.cast(
        "pyine.utils.openai.OpenAIFineTunerParamsConfig",
        config.openai_finetuner_config.params,
    )
    tokenizer = pyine.utils.tokenizers.get_openai_tokenizer(
        model_id=finetuner_params.base_model,
        raise_if_not_found=False,
    )
    token_count = 0
    # use the same dataset preparation flags as actual training uploads
    append_answer = config.needs_answers_in_train_dataset()
    merge_system_with_user = not config.supports_system_prompt()
    for subset_name in config.datamodule_config.train_subset_names:
        tr_file_path = datamodule.get_openai_messages_dataset(
            subset_name=subset_name,
            append_answer=append_answer,
            merge_system_with_user=merge_system_with_user,
        )
        conversations = pyine.utils.openai.read_dataset_from_jsonl(tr_file_path)
        for conversation in conversations:
            for message in conversation:
                token_count += len(tokenizer.encode(message["content"]))
    return token_count


def train(
    client: openai.OpenAI,
    datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
    config: "pyine.apps.trainers.openai_finetune_configs.OpenAIFineTuneAppMainConfig",
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> str:
    """Fine-tune an OpenAI model on a given dataset and returns the resulting model name.

    Args:
        client: The OpenAI API client to use for fine-tuning.
        datamodule: The datamodule to fetch data from.
        config: The app config object to fetch settings from.
        runtime: The runtime config object to fetch settings from.

    Returns:
        The name of the fine-tuned model.
    """
    approx_tokens = _compute_estimated_train_token_count(config, datamodule)
    logger.info(f"training tokens count estimate: ~{approx_tokens:,}")
    finetuner = config.openai_finetuner_config.instantiate(client)
    if len(config.datamodule_config.train_subset_names) != 1:
        raise ValueError("must provide exactly one train dataset for openai finetuner")
    tr_file_path = datamodule.get_openai_messages_dataset(
        subset_name=config.datamodule_config.train_subset_names[0],
        append_answer=config.needs_answers_in_train_dataset(),
        merge_system_with_user=not config.supports_system_prompt(),
    )
    tr_file_id = finetuner.ensure_uploaded(tr_file_path)
    if len(config.datamodule_config.valid_subset_names) != 1:
        raise ValueError("must provide exactly one valid dataset for openai finetuner")
    va_file_path = datamodule.get_openai_messages_dataset(
        subset_name=config.datamodule_config.valid_subset_names[0],
        append_answer=config.needs_answers_in_train_dataset(),
        merge_system_with_user=not config.supports_system_prompt(),
    )
    va_file_id = finetuner.ensure_uploaded(va_file_path)
    job_id = finetuner.create_job(tr_file_id, va_file_id)
    if config.use_wandb_logging:
        assert runtime is not None and runtime.wandb_run_id is not None
        logger.info(f"using W&B blocking sync under run id: {runtime.wandb_run_id}")
        wandb_sync_kwargs: dict[str, typing.Any] = {"reinit": "return_previous"}
        wandb.integration.openai.fine_tuning.WandbLogger.sync(
            fine_tune_job_id=job_id,
            openai_client=client,
            project="pyine",
            wait_for_job_success=True,
            **wandb_sync_kwargs,
        )
    else:
        try:
            finetuner.stream_job_events(job_id)  # streams events until interrupted (blocking)
        except KeyboardInterrupt:
            logger.info("stopped streaming events; continuing to poll status...")
    model_name = finetuner.wait_for_job(job_id)
    if not model_name:
        logger.error("fine-tune failed or no model name returned")
        sys.exit(1)
    return model_name


async def main(
    config: "pyine.apps.trainers.openai_finetune_configs.OpenAIFineTuneAppMainConfig",
    runtime: (pyine.configs.schemas.RuntimeConfig | None) = None,  # None unless launched via hydra
    skip_fine_tuning: bool = False,  # used to evaluate the base model directly
) -> None:
    """Main function for the script; performs fine-tuning and evaluation for an OpenAI model.

    Args:
        config: Configuration for the application; see `OpenAIFineTuneAppMainConfig` for details.
        runtime: Configuration for the runtime; available when launched via hydra.
        skip_fine_tuning: Whether to skip fine-tuning and just evaluate the base model directly (as
            a reference for performance comparisons).
    """
    if config.is_resuming():
        raise ValueError("run resuming is not supported for OpenAI fine-tuning")
    try:
        pyine.utils.reprod.entrypoint_setup(
            runtime_config=runtime,
            main_config=config,
            use_wandb_logging=config.use_wandb_logging,
        )
    except pyine.utils.reprod.DryRunExit:
        return

    datamodule = pyine.apps.trainers.common.prepare_datamodule(config, runtime)
    if not isinstance(datamodule, pyine.data.datamodule.ConversationDataModule):
        raise TypeError(
            f"OpenAI fine-tuning requires a ConversationDataModule; received {type(datamodule).__name__}",
        )
    client: openai.OpenAI = config.openai_client_config.instantiate()

    if not skip_fine_tuning:
        model_name = train(
            client=client,
            datamodule=datamodule,
            config=config,
            runtime=runtime,
        )
    else:
        # use the base model directly as the target to evaluate
        finetuner_params = typing.cast(
            "pyine.utils.openai.OpenAIFineTunerParamsConfig",
            config.openai_finetuner_config.params,
        )
        model_name = finetuner_params.base_model
        logger.info("skipping fine-tuning, evaluating base model directly")

    if config.use_wandb_logging:
        if runtime is None:
            raise RuntimeError("runtime config must be provided when logging to wandb")
        wandb_run_id = runtime.wandb_run_id
        if wandb_run_id is None:
            raise RuntimeError("wandb run id must be available when logging to wandb")
        if getattr(runtime.wandb_run, "_is_finished", True):
            # the openai integration 'finalized' the run; re-open it to log the last few metrics/summaries
            runtime.wandb_run = wandb.init(
                project=runtime.wandb_run_project,
                entity=runtime.wandb_run_entity,
                id=wandb_run_id,
                resume="must",
            )
        runtime.wandb_run.summary.update({"model_name": model_name})  # type: ignore[reportUnknownMemberType]

    model_for_evals = pyine.utils.llm_providers.get_model_from_provider(
        provider="openai",
        model=model_name,
        client=client.chat.completions,
    )
    text_generation_pipeline_for_evals = datamodule.config.get_prompt_chain(model_for_evals)
    await pyine.apps.trainers.common.evaluate_model(
        model=text_generation_pipeline_for_evals,
        tokenizer=None,
        datamodule=datamodule,
        config=config,
        runtime=runtime,
    )
    if runtime is not None:
        runtime.finalize()


if __name__ == "__main__":
    import pyine.apps.trainers.openai_finetune_configs

    # TODO: if we ever have more than one eval type, make new entrypoint scripts w/ different eval types
    pyine.apps.trainers.openai_finetune_configs.hydra_main(pyine.evals.common.EvalType.CODE_EXEC)
