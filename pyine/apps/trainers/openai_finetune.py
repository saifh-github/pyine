"""
OpenAI API model fine-tuning CLI.

This is a work-in-progress / demo / reference script for fine-tuning an OpenAI model using a code
execution traces dataset.
"""

import concurrent.futures
import logging
import sys
import typing

import openai
import wandb.integration.openai.fine_tuning

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

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import pyine.apps.trainers.openai_finetune_configs


def _compute_estimated_train_token_count(
    config: "pyine.apps.trainers.openai_finetune_configs.MainConfig",
    dm: pyine.data.datamodule.ConversationDataModule,
) -> int:
    """Approximate the number of tokens used to train a model on the given dataset."""
    tokenizer = pyine.utils.tokenizers.get_openai_tokenizer(
        model_id=config.openai_finetuner_config.params.base_model,
        raise_if_not_found=False,
    )
    tr_file_path = dm.get_openai_messages_dataset("train")
    messages = pyine.utils.openai.read_dataset_from_jsonl(tr_file_path)
    token_count = 0
    for msg in messages:
        if isinstance(msg, dict):
            token_count += len(tokenizer.encode(msg["content"]))
        elif isinstance(msg, list):
            for m in msg:
                assert isinstance(m, dict), "what kind of structure is this?"
                token_count += len(tokenizer.encode(m["content"]))
    return token_count


async def _evaluate(
    model_name: str,
    subset_name: str,
    client: openai.OpenAI,
    dm: pyine.data.datamodule.ConversationDataModule,
    llm_grader_provider_config: pyine.utils.llm_providers.LLMProviderConfig | None,
) -> dict[str, float | int | str]:
    """Evaluates the given model on the specified subset."""
    logger.info(f"running evaluation for {model_name} on the {subset_name} subset...")
    model = pyine.utils.llm_providers.get_model_from_provider(
        provider="openai",
        model=model_name,
        client=client.chat.completions,
    )
    eval_parser = dm.get_parser(subset_name)
    assert isinstance(eval_parser, pyine.organisms.datamodules.utils.samples.SampleBuilder)
    eval_parser = typing.cast(pyine.organisms.datamodules.utils.samples.SampleBuilder, eval_parser)
    metrics = await pyine.evals.common.evaluate_model_on_subset(
        chain=dm.config.get_prompt_chain(model),
        parser=eval_parser,
        llm_grader_provider_config=llm_grader_provider_config,
        verbose=True,
    )
    pyine.evals.utils.print_metrics(metrics, subset_name, logger.info)
    return metrics


async def main(
    config: "pyine.apps.trainers.openai_finetune_configs.MainConfig",
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,  # None unless launched via hydra
    skip_fine_tuning: bool = False,  # used to evaluate the base model directly
) -> None:
    """Main function for the script; performs fine-tuning and evaluation for an OpenAI model.

    Args:
        config: Configuration for the application; see `MainConfig` for details.
        runtime: Configuration for the runtime; available when launched via hydra.
        skip_fine_tuning: Whether to skip fine-tuning and just evaluate the base model directly (as
            a reference for performance comparisons).
    """
    pyine.utils.reprod.entrypoint_setup(
        runtime_config=runtime,
        main_config=config,
        use_wandb_logging=config.use_wandb_logging,
    )

    dm = config.datamodule_config.instantiate_datamodule(verbose=True)
    logger.info("preparing datamodule and setting up parsers/loaders...")
    dm.prepare_data()
    dm.setup()

    if config.use_wandb_logging:
        assert runtime is not None and runtime.wandb_run is not None
        dm_stats = {f"dataset_stats/{k}": v for k, v in dm.get_stats().items()}
        runtime.wandb_run.summary.update(dm_stats)
        for eval_subset_name in config.eval_subset_names:
            pyine.evals.common.define_metrics_for_wandb(
                wandb_run=runtime.wandb_run,
                prefix=f"evals/{eval_subset_name}",
            )

    client: openai.OpenAI = config.openai_client_config.instantiate()
    if not skip_fine_tuning:
        approx_tokens = _compute_estimated_train_token_count(config, dm)
        logger.info(f"training tokens count estimate: ~{approx_tokens:,}")
        finetuner = config.openai_finetuner_config.instantiate(client)
        tr_file_path = dm.get_openai_messages_dataset(
            subset_type="train",
            append_answer=config.needs_answers_in_train_dataset(),
            merge_system_with_user=not config.supports_system_prompt(),
        )
        tr_file_id = finetuner.ensure_uploaded(tr_file_path)
        va_file_path = dm.get_openai_messages_dataset(
            subset_type="valid",
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
                reinit="return_previous",  # noqa; reuse already-existing run
            )
        else:
            try:
                finetuner.stream_job_events(job_id)  # streams events without blocking
            except KeyboardInterrupt:
                logger.info("stopped streaming events; continuing to poll status...")
        model_name = finetuner.wait_for_job(job_id)
        if not model_name:
            logger.error("fine-tune failed or no model name returned")
            sys.exit(-1)
    else:
        # use the base model directly as the target to evaluate
        model_name = config.openai_finetuner_config.params.base_model
        logger.info("skipping fine-tuning, evaluating base model directly")

    if config.use_wandb_logging:
        run_is_finished = getattr(runtime.wandb_run, "_is_finished", True)
        if run_is_finished:
            # the openai integration 'finalized' the run, re-open it to log the last few metrics/summaries
            # (we replace the original run obj with a re-opened one, hopefully just for summary updates)
            wandb_api = wandb.Api()
            runtime.wandb_run = wandb_api.run(runtime.wandb_run_id)
        runtime.wandb_run.summary.update({"model_name": model_name})

    for eval_subset_name in config.eval_subset_names:
        metrics = await _evaluate(
            model_name=model_name,
            subset_name=eval_subset_name,
            client=client,
            dm=dm,
            llm_grader_provider_config=config.llm_grader_provider_config,
        )
        if config.use_wandb_logging:
            prefixed_metrics = {f"evals/{eval_subset_name}/{k}": v for k, v in metrics.items()}
            runtime.wandb_run.summary.update(prefixed_metrics)


if __name__ == "__main__":
    import pyine.apps.trainers.openai_finetune_configs

    pyine.apps.trainers.openai_finetune_configs.hydra_main()
