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
import typing

import datasets
import transformers

import pyine.apps.trainers.common
import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.common
import pyine.organisms.datamodules.utils.samples
import pyine.utils.llm_providers
import pyine.utils.reprod
import pyine.utils.transformers

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import pyine.apps.trainers.hf_trainer_configs


def _compute_metrics(
    eval_pred: transformers.trainer_utils.EvalPrediction,
    inputs,
    loss,
) -> dict[str, float]:
    # @@@@@@@ TODO do something here?
    # rely on eval loss for model selection; no extra metrics computed
    return {}


def train(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    datamodule: pyine.data.datamodule.ConversationDataModule,
    config: "pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig",
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> transformers.Trainer:
    """Run fine-tuning with HuggingFace Trainer.

    Args:
        model: The model to fine-tune.
        tokenizer: The tokenizer to use for tokenization.
        datamodule: The datamodule to fetch data from.
        config: The app config object to fetch settings from.
        runtime: The runtime config object to fetch settings from.

    Returns:
        The instantiated trainer object that can be used for predictions.
    """
    assert config.training_args_config.do_train, "do_train must be True for training"

    train_ds = [
        datamodule.get_hf_messages_dataset(
            subset_name=subset_name,
            append_answer=True,
        )
        for subset_name in config.datamodule_config.train_subset_names
    ]
    train_ds = train_ds[0] if len(train_ds) == 1 else datasets.concatenate_datasets(train_ds)

    valid_ds = [
        datamodule.get_hf_messages_dataset(
            subset_name=subset_name,
            append_answer=True,
        )
        for subset_name in config.datamodule_config.valid_subset_names
    ]
    valid_ds = valid_ds[0] if len(valid_ds) == 1 else datasets.concatenate_datasets(valid_ds)

    # use some of the dataloader workers for dataset.map to parallelize tokenization
    num_proc = max(1, config.dataloader_num_workers // 2)
    # convert conversation-style rows into flat, tokenized examples for training/valid
    model_max_seq_len = pyine.utils.transformers.infer_effective_max_seq_len(model, tokenizer)
    logger.info(f"effective max_seq_len={model_max_seq_len}")
    train_ds = pyine.utils.transformers.prepare_examples_from_conversations(
        convo_ds=train_ds,
        tokenizer=tokenizer,
        max_seq_len=model_max_seq_len,
        num_proc=num_proc,
    )
    valid_ds = pyine.utils.transformers.prepare_examples_from_conversations(
        convo_ds=valid_ds,
        tokenizer=tokenizer,
        max_seq_len=model_max_seq_len,
        num_proc=num_proc,
    )
    # the collator pads to fixed length and masks labels for prompt tokens
    collator = pyine.utils.transformers.FixedSizePaddingCollatorWithPromptMask(tokenizer, max_length=model_max_seq_len)

    # disable KV cache during training (unnecessary overhead, + helps avoid compat issues w/ checkpointing)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        logger.info("enabling gradient checkpointing")
        model.gradient_checkpointing_enable()

    training_args_dict = config.training_args_config.model_dump()
    if config.use_wandb_logging:
        assert runtime is not None and runtime.wandb_run is not None, "wandb should have been initialized"
        training_args_dict["report_to"] = ["wandb"]

    # note: if we want to support other trainers (e.g. from TRL, upate config dict+trainer w/ instantiable classes)
    training_args = transformers.TrainingArguments(**training_args_dict)
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    logger.info("starting training")
    train_out = trainer.train()
    logger.info("training finished: %s", train_out)
    logger.info("saving adapter and tokenizer to %s", config.output_dir)
    pathlib.Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    return trainer


async def main(
    config: "pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig",
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,  # None unless launched via hydra
) -> None:
    """Main function for the script; performs fine-tuning and evaluation for huggingface model.

    Args:
        config: Configuration for the application; see `HFTrainerAppMainConfig` for details.
        runtime: Configuration for the runtime; available when launched via hydra.
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
    model = config.get_model()
    tokenizer = config.get_tokenizer()

    if config.training_args_config.do_train:
        _ = train(
            model=model,
            tokenizer=tokenizer,
            datamodule=dm,
            config=config,
            runtime=runtime,
        )

    if config.training_args_config.do_predict:
        await pyine.apps.trainers.common.evaluate_model(
            model=model,
            tokenizer=tokenizer,
            datamodule=dm,
            config=config,
            runtime=runtime,
        )


if __name__ == "__main__":
    import pyine.apps.trainers.hf_trainer_configs

    # TODO: if we ever have more than one eval type, make new entrypoint scripts w/ different eval types
    pyine.apps.trainers.hf_trainer_configs.hydra_main(pyine.evals.common.EvalType.CODE_EXEC)
