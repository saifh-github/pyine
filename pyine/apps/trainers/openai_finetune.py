"""
OpenAI API model fine-tuning CLI.

This is a work-in-progress / demo / reference script for fine-tuning an OpenAI model using a code
execution traces dataset. It is not meant to be used directly yet; TODO! @@@@
"""

import logging
import sys

import langchain_core.messages
import langchain_core.runnables
import pydantic

import pyine.data.datamodule
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.evals.utils
import pyine.organisms.datamodules.shortcuts
import pyine.organisms.datamodules.utils.samples
import pyine.organisms.models.utils.openai
import pyine.organisms.models.utils.tokenizers
import pyine.utils.code.output_compare
import pyine.utils.llm_providers
import pyine.utils.reprod

logger = logging.getLogger(__name__)


default_rl_params_config = pyine.organisms.models.utils.openai.OpenAIFineTunerParamsConfig(
    base_model="o4-mini-2025-04-16",
    method=pyine.organisms.models.utils.openai.PredGraderFineTuneMethodConfig().get_openai_config(),
    seed=0,
    suffix="dummy",
    wandb_integration=None,
    metadata=pyine.utils.reprod.get_reprod_metadata(include_installed_packages=False),  # noqa
    timeout_override=60 * 60,  # 60 min
)
"""Default parameters configuration for OpenAI RL fine-tuning using o4-mini.

Note: as of 2025-09-03, training o4-mini using this approach is VERY COSTLY, even for VERY TINY
datasets (we're talking hundreds of dollars per run here, minimum). Don't use this config unless
you know what you're doing.
"""

default_sft_params_config = pyine.organisms.models.utils.openai.OpenAIFineTunerParamsConfig(
    base_model="gpt-4.1-mini-2025-04-14",
    method=dict(
        type="supervised",
    ),
    seed=0,
    suffix="dummy",
    wandb_integration=None,
    metadata=pyine.utils.reprod.get_reprod_metadata(include_installed_packages=False),  # noqa
    timeout_override=60 * 60,  # 60 min
)
"""Default parameters configuration for OpenAI supervised fine-tuning using gpt-4.1-mini."""


class MainConfig(pydantic.BaseModel):
    """Configuration for the script's main function.

    Assembles the components required to fine-tune a model for code execution using a code execution
    traces datamodule.

    NOTE: this config is intended to be used with the `main` function defined below, and is provided
    here as a demonstration of how to use this app (will be refactored/cleaned when we have a config
    manager).
    """

    seed: int | None = None
    """Seed to use for reproducibility."""
    datamodule_config: pyine.data.datamodule.ConversationDataModuleConfig
    """Configuration for the datamodule to use (NOT SPECIFIED BY DEFAULT!)."""
    openai_client: pyine.organisms.models.utils.openai.OpenAIClientConfig = (
        pyine.organisms.models.utils.openai.OpenAIClientConfig()
    )
    """Configuration for the OpenAI client to use."""
    openai_finetuner: pyine.organisms.models.utils.openai.OpenAIFineTunerConfig = (
        pyine.organisms.models.utils.openai.OpenAIFineTunerConfig(params=default_sft_params_config)
    )
    """Configuration for the OpenAI fine-tuner to use; defaults to an RL fine-tuning config for o4-mini."""
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


def _evaluate(
    config: MainConfig,
    chain: langchain_core.runnables.Runnable,
    dm: pyine.data.datamodule.ConversationDataModule,
    subset: str,
) -> dict[str, float | int | str]:
    """Evaluate a trained model on the specified data subset, returning evaluation metrics."""
    parser = dm.get_parser(subset)
    evaluator = pyine.evals.utils.OutcomeEvaluator(
        llm_provider_config=config.llm_grader_provider_config,
    )
    token_usage = None
    for sample in parser:
        assert isinstance(sample, pyine.organisms.datamodules.utils.samples.SampleData)
        response = chain.invoke(sample._asdict())
        assert isinstance(response, langchain_core.messages.AIMessage)
        evaluator.add_sample(
            identifier=sample.identifier,
            expected=sample.expected_output,
            predicted=response.content,
            tags=sample.get_tag_list(),
        )
        if token_usage is None:
            token_usage = pyine.evals.utils.parse_token_usage_from_response(response)
        else:
            token_usage += pyine.evals.utils.parse_token_usage_from_response(response)
    output_metrics: dict[str, float | int | str] = evaluator.compute_metrics()
    if token_usage is None:
        token_usage = pyine.evals.utils.TokenUsageInfo.get_default()
    output_metrics.update(token_usage.asdict())
    return output_metrics


def main(
    config: MainConfig,
    skip_fine_tuning: bool = False,  # used to evaluate the base model directly
) -> None:
    """Main function for the script; performs fine-tuning and evaluation for an OpenAI model.

    Args:
        config: Configuration for the script; see `MainConfig` for details.
        skip_fine_tuning: Whether to skip fine-tuning and just evaluate the base model directly (as
            a reference for performance comparisons).
    """
    # @@@@@@ TODO: update this main to actually use click or a config/experiment manager
    pyine.utils.reprod.entrypoint_setup(
        seed=config.seed,
    )
    dm = config.datamodule_config.instantiate_datamodule(verbose=True)
    dm.prepare_data()
    dm.setup()
    approx_tokens = _compute_estimated_train_token_count(config, dm)
    logger.info(f"training tokens count estimate: ~{approx_tokens:,}")
    client = config.openai_client.instantiate()
    if not skip_fine_tuning:
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
    eval_chain = dm.config.get_prompt_chain(model)
    logger.info("running eval on the valid subset...")
    metrics = _evaluate(config, eval_chain, dm, "valid")
    eval_output_str = "\n".join([f"\t{key}: {val:.3f}" for key, val in metrics.items()])
    logger.info(f"valid metrics:\n{eval_output_str}")


if __name__ == "__main__":
    _dm_cfg = pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig(
        lmdb_paths=[
            pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO"),
        ],
        max_trace_count=200,  # cap off the max dataset size
        split_file_path=pyine.data.utils.splits.get_dataset_split_file_path("TACO"),
    )
    _main_cfg = MainConfig(
        datamodule_config=_dm_cfg,
        llm_grader_provider_config=pyine.utils.llm_providers.LLMProviderConfig(
            provider="openai",
            model_kwargs=dict(
                model="gpt-4o-mini",
            ),
        ),
    )
    main(_main_cfg, skip_fine_tuning=True)
