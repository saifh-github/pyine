"""
OpenAI API model fine-tuning CLI.

This is a work-in-progress / demo / reference script for fine-tuning an OpenAI model using a code
execution traces dataset.
"""

import concurrent.futures
import logging
import sys
import typing

import pydantic

import pyine.configs.schemas
import pyine.data.datamodule
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.evals.common
import pyine.evals.utils
import pyine.organisms.datamodules.shortcuts
import pyine.organisms.datamodules.utils.samples
import pyine.organisms.models.utils.openai
import pyine.organisms.models.utils.tokenizers
import pyine.utils.code.output_compare
import pyine.utils.concurrency
import pyine.utils.llm_providers
import pyine.utils.reprod

logger = logging.getLogger(__name__)


class MainConfig(pydantic.BaseModel):
    """Configuration for the script's main function.

    Assembles the components required to fine-tune a model for code execution using a code execution
    traces datamodule.

    NOTE: this config is intended to be used with the `main` function defined below, and is provided
    here as a demonstration of how to use this app (will be refactored/cleaned when we have a config
    manager).
    """

    datamodule_config: pyine.data.datamodule.ConversationDataModuleConfig
    """Configuration for the datamodule to use."""
    openai_client: pyine.organisms.models.utils.openai.OpenAIClientConfig = (
        pyine.organisms.models.utils.openai.OpenAIClientConfig()
    )
    """Configuration for the OpenAI client to use."""
    openai_finetuner: pyine.organisms.models.utils.openai.OpenAIFineTunerConfig
    """Configuration for the OpenAI fine-tuner to use."""
    llm_grader_provider_config: pyine.utils.llm_providers.LLMProviderConfig | None = None
    """Configuration for the LLM grader provider to use. If not specified, skips LLM grader evaluation."""

    def needs_answers_in_train_dataset(self) -> bool:
        """Returns whether the model needs answers in its training dataset."""
        return self.openai_finetuner.params.method.get("type", "") != "reinforcement"

    def supports_system_prompt(self) -> bool:
        """Returns whether the model to be fine-tuned supports the use of system prompts."""
        models_without_system_prompts = ["o1", "o3", "o4"]
        return not any([self.openai_finetuner.params.base_model.startswith(m) for m in models_without_system_prompts])


def _compute_estimated_train_token_count(
    config: MainConfig,
    dm: pyine.data.datamodule.ConversationDataModule,
) -> int:
    """Approximate the number of tokens used to train a model on the given dataset."""
    tokenizer = pyine.organisms.models.utils.tokenizers.get_openai_tokenizer(
        model_id=config.openai_finetuner.params.base_model,
        raise_if_not_found=False,
    )
    tr_file_path = dm.get_openai_messages_dataset("train")
    messages = pyine.organisms.models.utils.openai.read_dataset_from_jsonl(tr_file_path)
    token_count = 0
    for msg in messages:
        if isinstance(msg, dict):
            token_count += len(tokenizer.encode(msg["content"]))
        elif isinstance(msg, list):
            for m in msg:
                assert isinstance(m, dict), "what kind of structure is this?"
                token_count += len(tokenizer.encode(m["content"]))
    return token_count


def main(
    config: MainConfig,
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
    pyine.utils.reprod.entrypoint_setup(config=runtime)
    dm = config.datamodule_config.instantiate_datamodule(verbose=True)
    dm.prepare_data()
    dm.setup()
    client = config.openai_client.instantiate()
    if not skip_fine_tuning:
        approx_tokens = _compute_estimated_train_token_count(config, dm)
        logger.info(f"training tokens count estimate: ~{approx_tokens:,}")
        finetuner = config.openai_finetuner.instantiate(client)
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
        try:
            finetuner.stream_job_events(job_id)  # streams events without blocking
        except KeyboardInterrupt:
            logger.info("Stopped streaming events; continuing to poll status...")
        model_name = finetuner.wait_for_job(job_id)
        if not model_name:
            logger.error("fine-tune failed or no model name returned")
            sys.exit(-1)
    else:
        # use the base model directly as the target to evaluate
        model_name = config.openai_finetuner.params.base_model

    model = pyine.utils.llm_providers.get_model_from_provider(
        provider="openai",
        model=model_name,
        client=client.chat.completions,
    )
    logger.info("running eval on the valid subset...")
    eval_parser = dm.get_parser("valid")
    assert isinstance(eval_parser, pyine.organisms.datamodules.utils.samples.SampleBuilder)
    eval_parser = typing.cast(pyine.organisms.datamodules.utils.samples.SampleBuilder, eval_parser)
    metrics = pyine.evals.common.evaluate_model_on_subset(
        chain=dm.config.get_prompt_chain(model),
        parser=eval_parser,
        llm_grader_provider_config=config.llm_grader_provider_config,
    )
    pyine.evals.utils.print_metrics(metrics, "valid")


if __name__ == "__main__":
    import pyine.apps.trainers.openai_finetune_configs

    pyine.apps.trainers.openai_finetune_configs.hydra_main()
