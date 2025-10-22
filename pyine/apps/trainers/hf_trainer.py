"""Hydra-integrated HuggingFace fine-tuning CLI app.

This app wires together:
- a tokenizer and base model from hugging face;
- a dataset of (potentially multi-turn) conversations;
- a function that converts conversations into supervised examples;
- a Trainer that handles batching/padding/masking + the training loop.
"""

import collections
import logging
import time
import typing

import datasets
import transformers

import pyine.apps.trainers.common
import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.common
import pyine.evals.utils
import pyine.utils.reprod
import pyine.utils.timers
import pyine.utils.transformers

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import pyine.apps.trainers.hf_trainer_configs


def _extract_sample_categories_from_dataset(
    dataset: typing.Any,
) -> list[list[str]]:
    """Return the list of evaluation categories associated with each example of a dataset.

    The returned list will have the same length as the dataset, and each element will be a list
    of categories associated with the corresponding example.
    """
    # @@@@@ TODO: update this to work with target tags instead of just code_type?
    # (@@@ move to datamodule? will need to concat in train func across multiple subsets)
    if dataset is None or not hasattr(dataset, "__len__"):
        return []
    if not hasattr(dataset, "column_names") or "sample_data" not in dataset.column_names:
        return []
    sample_column = dataset["sample_data"]
    if isinstance(sample_column, dict):
        sample_mapping = typing.cast("typing.Mapping[str, typing.Any]", sample_column)
        column = sample_mapping.get("code_type")
        if column is None:
            return [[]] * len(dataset)
        column_iterable = typing.cast("typing.Iterable[typing.Any]", column)
        code_types = list(column_iterable)
        if any(not isinstance(code_type, str) and code_type is not None for code_type in code_types):
            raise ValueError("code_type column must contain only strings or None values")
        return [[c] if c is not None else [] for c in code_types]
    code_types: list[str | None] = []
    sample_iterable = typing.cast("typing.Iterable[typing.Any]", sample_column)
    for sample_data in sample_iterable:
        if isinstance(sample_data, dict):
            sample_mapping = typing.cast("typing.Mapping[str, typing.Any]", sample_data)
            code_type = sample_mapping.get("code_type")
            if code_type is not None and not isinstance(code_type, str):
                raise ValueError("code_type column must contain only strings or None values")
            code_types.append(code_type)
        else:
            code_types.append(None)
    return [[c] if c is not None else [] for c in code_types]


def train(
    datamodule: pyine.data.datamodule.ConversationDataModule[typing.Any],
    config: "pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig",
    runtime: pyine.configs.schemas.RuntimeConfig | None,
    resume_artifacts: pyine.apps.trainers.common.ResumeArtifacts | None,
) -> transformers.Trainer:
    """Run fine-tuning with HuggingFace Trainer.

    Args:
        datamodule: The datamodule to fetch data from.
        config: The app config object to fetch settings from.
        runtime: The runtime config object to fetch settings from.
        resume_artifacts: Resume artifacts to continue a previous run, when provided.

    Returns:
        The instantiated trainer object containing a model that can be used for predictions.
    """
    assert config.training_args_config.do_train, "do_train must be True for training"

    model = config.get_model()
    tokenizer = config.get_tokenizer()

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
            keep_original_data=True,  # for category-wise evals below
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
        keep_extra_fields=["sample_data"],  # for category-wise evals below
    )
    # the collator pads to fixed length and masks labels for prompt tokens
    collator = pyine.utils.transformers.PaddingCollatorWithPromptMask(
        tokenizer,
        max_length=model_max_seq_len,
        # pad_to_multiple_of=32,  # @@@@ TODO test speed with and without?
    )
    valid_sample_categories = _extract_sample_categories_from_dataset(valid_ds)
    assert len(valid_sample_categories) == len(valid_ds) and any(c is not None for c in valid_sample_categories), (
        "could not extract sample categories from validation dataset; check that sample data is preserved?"
    )
    category_counts = collections.Counter(cat for cats in valid_sample_categories for cat in cats)
    category_counts_str = "\n\t".join(f"{key}: {val}" for key, val in dict(category_counts).items())
    logger.debug(f"validation data sample category counts:\n\t{category_counts_str}")
    eval_metrics_callback = pyine.evals.utils.build_category_wise_compute_metrics_fn(
        data_sample_categories=valid_sample_categories,
        metrics_prefix="eval",
        log_fn=logger.info,
    )
    training_args_dict = config.training_args_config.model_dump()
    training_args_dict["batch_eval_metrics"] = True  # for compat w/ the eval_metrics_callback
    if runtime is not None and runtime.wandb_run is not None:
        training_args_dict["report_to"] = ["wandb"]

    # note: if we want to support other trainers (e.g. TRL), update config dict+trainer w/ instantiable classes
    training_args = transformers.TrainingArguments(**training_args_dict)
    milestone_logger = pyine.utils.transformers.StdoutMilestones(print_fn=logger.info)
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=eval_metrics_callback,
        callbacks=[milestone_logger, eval_metrics_callback],
    )
    train_kwargs: dict[str, typing.Any] = {}
    if resume_artifacts is not None:
        assert resume_artifacts.checkpoint_path.is_dir(), f"invalid ckpt path: {resume_artifacts.checkpoint_path}"
        logger.info(f"resuming from checkpoint: {resume_artifacts.checkpoint_path}")
        train_kwargs["resume_from_checkpoint"] = str(resume_artifacts.checkpoint_path)
    logger.info("starting training")
    start_time = time.time()
    trainer.train(**train_kwargs)  # type: ignore[reportUnknownMemberType]
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
    resume_artifacts = pyine.apps.trainers.common.prepare_resume_artifacts(
        config=config,
        runtime=runtime,
    )
    try:
        pyine.utils.reprod.entrypoint_setup(
            runtime_config=runtime,
            main_config=config,
            use_wandb_logging=config.use_wandb_logging,
            wandb_init_kwargs=resume_artifacts.wandb_resume_kwargs if resume_artifacts else None,
        )
    except pyine.utils.reprod.DryRunExit:
        return

    datamodule = pyine.apps.trainers.common.prepare_datamodule(config, runtime)
    if not isinstance(datamodule, pyine.data.datamodule.ConversationDataModule):
        raise TypeError(
            f"HuggingFace trainer requires a ConversationDataModule; received {type(datamodule).__name__}",
        )

    if config.training_args_config.do_train:
        trainer = train(
            datamodule=datamodule,
            config=config,
            runtime=runtime,
            resume_artifacts=resume_artifacts,
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
        if config.is_resuming():
            # reinstantiate based on target checkpoint
            assert resume_artifacts is not None, "resume artifacts must be provided for resuming"
            model: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(  # type: ignore[reportUnknownMemberType]
                resume_artifacts.checkpoint_path,
            )
            tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
                resume_artifacts.checkpoint_path,
            )
        else:
            # get base (pretrained) model/tokenizers directly
            model = config.get_model()
            tokenizer = config.get_tokenizer()

    if config.training_args_config.do_predict:
        if runtime is not None and runtime.wandb_run is not None:
            runtime.wandb_run.summary["model_name"] = model.config.name_or_path
        await pyine.apps.trainers.common.evaluate_model(
            model=model,
            tokenizer=tokenizer,  # type: ignore[reportUnknownArgumentType]
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
