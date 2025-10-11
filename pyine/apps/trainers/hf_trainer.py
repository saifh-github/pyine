"""Hydra-integrated HuggingFace fine-tuning CLI app.

This app wires together:
- a tokenizer and base model from hugging face;
- a dataset of (potentially multi-turn) conversations;
- a function that converts conversations into supervised examples;
- a Trainer that handles batching/padding/masking + the training loop.
"""

import logging
import time
import typing

import datasets
import transformers

import pyine.apps.trainers.common
import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.common
import pyine.utils.reprod
import pyine.utils.timers
import pyine.utils.transformers

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import pyine.apps.trainers.hf_trainer_configs

#
# def compute_metrics(eval_pred: transformers.trainer_utils.EvalPrediction) -> Dict[str, float]:
#     """Compute metrics for evaluation.
#
#     Args:
#         eval_pred: EvalPrediction object containing predictions and labels
#
#     Returns:
#         Dictionary of computed metrics.
#     """
#     # @@@@@@@ TODO do something here? (training metrics)
#     predictions, labels = eval_pred
#
#     # For causal language modeling, we typically just use perplexity (based on loss)
#     # But here's an example of how you might compute other metrics
#
#     # Shift predictions and labels for next-token prediction
#     shift_predictions = predictions[..., :-1, :].contiguous()
#     shift_labels = labels[..., 1:].contiguous()
#
#     # Get predicted token IDs
#     predicted_ids = np.argmax(shift_predictions, axis=-1)
#
#     # Flatten for metric computation (ignore -100 labels)
#     flat_predictions = predicted_ids.flatten()
#     flat_labels = shift_labels.flatten()
#
#     # Only compute metrics on non-masked tokens
#     mask = flat_labels != -100
#     flat_predictions = flat_predictions[mask]
#     flat_labels = flat_labels[mask]
#
#     if len(flat_labels) > 0:
#         accuracy = accuracy_score(flat_labels, flat_predictions)
#         f1 = f1_score(flat_labels, flat_predictions, average='macro', zero_division=0)
#     else:
#         accuracy = 0.0
#         f1 = 0.0
#
#     return {
#         "accuracy": accuracy,
#         "f1": f1,
#     }


def train(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
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
    num_proc = max(1, config.training_args_config.dataloader_num_workers // 2)
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
    collator = pyine.utils.transformers.PaddingCollatorWithPromptMask(
        tokenizer,
        max_length=model_max_seq_len,
        # pad_to_multiple_of=32,  # @@@@ TODO test speed with and without?
    )

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
        processing_class=tokenizer,
        data_collator=collator,
        # compute_metrics=_compute_metrics,  # @@@@ TODO: update w/ proper callback
        # callbacks=[EarlyStoppingCallback()],  # @@@@@  TODO: update w/ proper callback
    )

    logger.info("starting training")
    start_time = time.time()
    trainer.train(  # type: ignore[reportUnknownMemberType]
        # resume_from_checkpoint=...,  # @@@@@ TODO: add here if needed?
    )
    end_time = time.time()
    time_delta_seconds = end_time - start_time
    time_delta_str = pyine.utils.timers.get_human_readable_time(time_delta_seconds)
    logger.info(f"training finished in {time_delta_str}")
    return trainer


async def main(
    config: "pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig",
    runtime: (pyine.configs.schemas.RuntimeConfig | None) = None,  # None unless launched via hydra
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

    datamodule = pyine.apps.trainers.common.prepare_datamodule(config, runtime)
    if not isinstance(datamodule, pyine.data.datamodule.ConversationDataModule):
        raise TypeError(
            f"HuggingFace trainer requires a ConversationDataModule; received {type(datamodule).__name__}",
        )
    model = config.get_model()
    tokenizer = config.get_tokenizer()

    if config.training_args_config.do_train:
        _ = train(
            model=model,
            tokenizer=tokenizer,
            datamodule=datamodule,
            config=config,
            runtime=runtime,
        )
    # note: if not training, the model+tokenizer states will depend the specified model name/path
    # (those might correspond to the base model or to a local checkpoint from a previous run)

    if config.training_args_config.do_predict:
        if config.use_wandb_logging:
            assert runtime is not None and runtime.wandb_run is not None, "invalid wandb runtime"
            runtime.wandb_run.summary["model_name"] = model.config.name_or_path
        await pyine.apps.trainers.common.evaluate_model(
            model=model,
            tokenizer=tokenizer,
            datamodule=datamodule,
            config=config,
            runtime=runtime,
        )
    if runtime is not None:
        runtime.finalize()


if __name__ == "__main__":
    import pyine.apps.trainers.hf_trainer_configs

    # TODO: if we ever have more than one eval type, make new entrypoint scripts w/ different eval types
    pyine.apps.trainers.hf_trainer_configs.hydra_main(pyine.evals.common.EvalType.CODE_EXEC)
