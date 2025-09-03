import functools
import logging
import pathlib
import sys
import typing

import langchain_core.prompts
import langchain_core.runnables
import openai
import openai.types
import openai.types.fine_tuning
import pydantic

import pyine.data.datamodule
import pyine.data.traces.dataset_utils
import pyine.organisms.datamodules.shortcuts
import pyine.organisms.datamodules.utils.samples
import pyine.organisms.models.utils.openai
import pyine.organisms.models.utils.tokenizers
import pyine.utils.code.output_compare
import pyine.utils.llm_providers
import pyine.utils.reprod

logger = logging.getLogger(__name__)


class MainConfig(pydantic.BaseModel):

    seed: int | None = None
    datamodule_config: pyine.data.datamodule.ConversationDataModuleConfig = (
        pyine.organisms.datamodules.shortcuts.ShortcutBiasDataModuleConfig(
            lmdb_paths=[
                pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO"),
            ],
            max_trace_count=100,  # cap off the max dataset size
            base_filter_rule="+subset:train",  # conduct all prototyping on train only (w/ internal split)
            subset_filter_rules=dict(),  # reset rule-based assignments for subset (will take random split)
            subset_leftover_split_ratios=dict(
                # take 80-20 split from the original train dataset itself for this prototyping
                train=0.8,
                valid=0.2,
            ),
        )
    )
    openai_client: pyine.organisms.models.utils.openai.OpenAIClientConfig = (
        pyine.organisms.models.utils.openai.OpenAIClientConfig(),
    )
    openai_finetuner: pyine.organisms.models.utils.openai.OpenAIFineTunerConfig = (
        pyine.organisms.models.utils.openai.OpenAIFineTunerConfig(
            params=pyine.organisms.models.utils.openai.OpenAIFineTunerParamsConfig(
                base_model="o4-mini-2025-04-16",
                method=pyine.organisms.models.utils.openai.PredGraderFineTuneMethodConfig().model_dump(),
                seed=0,
                suffix="dummy",
                wandb_integration=None,
                metadata=pyine.utils.reprod.get_reprod_metadata(),  # noqa
                timeout_override=60 * 60,  # 60 min
            )
        )
    )
    pred_output_compare_options: typing.Annotated[
        pyine.utils.code.output_compare.CompareOptions,
        pydantic.Field(
            default=pyine.utils.code.output_compare.get_options_for_code_exec_outputs(),
            description="Options to use for comparing a predicted output with the expected output.",
        ),
    ]


def _prepare_dataset(config: MainConfig) -> pyine.data.datamodule.ConversationDataModule:
    dm = config.datamodule_config.instantiate_datamodule(verbose=True)
    dm.prepare_data()
    dm.setup()
    for subset in ["train", "valid"]:
        _ = dm.get_openai_messages_dataset(subset)  # writes datasets to disk for later
    return dm


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
    token_counts = [len(tokenizer.encode(m["content"])) for m in messages]
    return sum(token_counts)


def _prepare_eval_chain(
    client: openai.OpenAI,
    model_name: str,
    dm: pyine.data.datamodule.ConversationDataModule,
) -> langchain_core.runnables.Runnable:
    """Prepare a chain for evaluating a model using a chat prompt provided by the datamodule."""
    model = pyine.utils.llm_providers.get_model_from_provider(
        provider="openai",
        model=model_name,
        client=client.chat.completions,
    )
    chain = dm.config.get_prompt_chain(model)
    return chain


def _evaluate(
    config: MainConfig,
    chain: langchain_core.runnables.Runnable,
    dm: pyine.data.datamodule.ConversationDataModule,
    subset: str,
) -> dict[str, float]:
    """Evaluate a trained model on the specified data subset, returning evaluation metrics."""
    comp = functools.partial(
        pyine.utils.code.output_compare.compare,
        options=config.pred_output_compare_options,
    )
    parser = dm.get_parser(subset)
    pred_outcomes = []
    for sample in parser:
        assert isinstance(sample, pyine.organisms.datamodules.utils.samples.SampleData)
        response = chain.invoke(sample._asdict())
        comp_result = comp(response.content, sample.output)
        pred_outcomes.append(bool(comp_result))
    accuracy = sum(pred_outcomes) / len(pred_outcomes)
    return {"accuracy": accuracy}


def main(
    config: MainConfig,
    skip_fine_tuning: bool = False,  # used to evaluate the base model directly
) -> None:
    pyine.utils.reprod.entrypoint_setup(
        seed=config.seed,
    )
    dm = _prepare_dataset(config)
    approx_tokens = _compute_estimated_train_token_count(config, dm)
    logger.info(f"training tokens count estimate: ~{approx_tokens:,}")
    client = config.openai_client.instantiate()
    if not skip_fine_tuning:
        finetuner = config.openai_finetuner.instantiate(client)
        tr_file_path = dm.get_openai_messages_dataset("train")
        tr_file_id = finetuner.ensure_uploaded(tr_file_path)
        va_file_path = dm.get_openai_messages_dataset("valid")
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

    eval_chain = _prepare_eval_chain(client, model_name, dm)
    logger.info("running eval on the valid subset...")
    metrics = _evaluate(config, eval_chain, dm, "valid")
    eval_output_str = "\n".join([f"\t{key}: {val:.3f}" for key, val in metrics.items()])
    logger.info(f"valid metrics:\n{eval_output_str}")


if __name__ == "__main__":
    main(MainConfig(), skip_fine_tuning=False)
