"""OpenAI API model fine-tuning CLI app."""

import logging
import sys
import typing

import openai
import wandb.integration.openai.fine_tuning

import pyine.apps.trainers.common
import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.common
import pyine.evals.utils
import pyine.organisms.datamodules.utils.samples
import pyine.utils.code.output_compare
import pyine.utils.concurrency
import pyine.utils.llm_providers
import pyine.utils.openai
import pyine.utils.reprod
import pyine.utils.tokenizers
import wandb

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import pyine.apps.trainers.openai_finetune_configs


def _compute_estimated_train_token_count(
    config: "pyine.apps.trainers.openai_finetune_configs.OpenAIFineTuneAppMainConfig",
    dm: pyine.data.datamodule.ConversationDataModule,
) -> int:
    """Approximate the number of tokens used to train a model on the given dataset."""
    tokenizer = pyine.utils.tokenizers.get_openai_tokenizer(
        model_id=config.openai_finetuner_config.params.base_model,
        raise_if_not_found=False,
    )
    token_count = 0
    for subset_name in config.datamodule_config.train_subset_names:
        tr_file_path = dm.get_openai_messages_dataset(subset_name)
        messages = pyine.utils.openai.read_dataset_from_jsonl(tr_file_path)
        for msg in messages:
            if isinstance(msg, dict):
                token_count += len(tokenizer.encode(msg["content"]))
            elif isinstance(msg, list):
                for m in msg:
                    assert isinstance(m, dict), "what kind of structure is this?"
                    token_count += len(tokenizer.encode(m["content"]))
    return token_count


def train(
    client: openai.OpenAI,
    datamodule: pyine.data.datamodule.ConversationDataModule,
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
        wandb.integration.openai.fine_tuning.WandbLogger.sync(
            fine_tune_job_id=job_id,
            openai_client=client,
            project="pyine",
            wait_for_job_success=True,
            reinit="return_previous",  # reuse already-existing run
        )
    else:
        try:
            finetuner.stream_job_events(job_id)  # streams events without blocking
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
    try:
        pyine.utils.reprod.entrypoint_setup(
            runtime_config=runtime,
            main_config=config,
            use_wandb_logging=config.use_wandb_logging,
        )
    except pyine.utils.reprod.DryRunExit:
        return

    dm = pyine.apps.trainers.common.prepare_datamodule(config, runtime)
    client: openai.OpenAI = config.openai_client_config.instantiate()

    if not skip_fine_tuning:
        model_name = train(
            client=client,
            datamodule=dm,
            config=config,
            runtime=runtime,
        )
    else:
        # use the base model directly as the target to evaluate
        model_name = config.openai_finetuner_config.params.base_model
        logger.info("skipping fine-tuning, evaluating base model directly")

    if config.use_wandb_logging:
        run_is_finished = getattr(runtime.wandb_run, "_is_finished", True)
        if run_is_finished:
            # the openai integration 'finalized' the run; re-open it to log the last few metrics/summaries
            # (we replace the original run obj with a re-opened one, hopefully just for summary updates)
            wandb_api = wandb.Api()
            runtime.wandb_run = wandb_api.run(runtime.wandb_run_id)
        runtime.wandb_run.summary.update({"model_name": model_name})

    model_for_evals = pyine.utils.llm_providers.get_model_from_provider(
        provider="openai",
        model=model_name,
        client=client.chat.completions,
    )
    text_generation_pipeline_for_evals = dm.config.get_prompt_chain(model_for_evals)
    await pyine.apps.trainers.common.evaluate_model(
        model=text_generation_pipeline_for_evals,
        tokenizer=None,
        datamodule=dm,
        config=config,
        runtime=runtime,
    )


if __name__ == "__main__":
    import pyine.apps.trainers.openai_finetune_configs

    # TODO: if we ever have more than one eval type, make new entrypoint scripts w/ different eval types
    pyine.apps.trainers.openai_finetune_configs.hydra_main(pyine.evals.common.EvalType.CODE_EXEC)
