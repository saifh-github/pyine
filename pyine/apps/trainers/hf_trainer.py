"""Hydra-integrated HuggingFace fine-tuning CLI app."""

import json
import logging
import os
import pathlib
import sys
import time
import typing

import datasets
import torch
import transformers

import pyine.configs.schemas
import pyine.data.datamodule
import pyine.utils.reprod

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import pyine.apps.trainers.hf_trainer_configs


def _build_tokenizer(
    base_model: str,
    set_missing_pad_token_as_eos: bool = True,
) -> transformers.PreTrainedTokenizerBase:
    """Load and adapt tokenizer for training."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if set_missing_pad_token_as_eos and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_model_qlora(
    base_model: str,
    torch_dtype: torch.dtype,
    gradient_checkpointing: bool,
) -> transformers.PreTrainedModel:
    """Load a model in 4-bit for LoRA fine-tuning (QLoRA)."""
    quant_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def _build_model_no_quant(
    base_model: str,
    torch_dtype: torch.dtype,
    gradient_checkpointing: bool,
) -> transformers.PreTrainedModel:
    """Load a model without quantization for fine-tuning."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map={"": "mps"} if torch.backends.mps.is_available() else "auto",
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def _apply_chat_template(
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: list[dict],
    add_generation_prompt: bool,
) -> str:
    """Apply the tokenizer chat template if available."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


class ExampleIds(typing.TypedDict):
    prompt_ids: list[int]
    input_ids: list[int]
    prompt_len: int


def _build_example_ids(
    tokenizer: transformers.PreTrainedTokenizerBase,
    history_msgs: list[dict],
    assistant_msg: dict,
    max_seq_len: int,
) -> ExampleIds | None:
    """Turn a (history, assistant) pair into token ids and prompt length.

    Returns:
        ExampleIds: typed structure with prompt token ids, full input ids, and prompt length,
        or None when the assistant message is empty.
    """
    assistant_text = assistant_msg.get("content", "").strip()
    if not assistant_text:
        return None
    # from a conversation history and an assistant message, build the prompt text block
    prompt_text = _apply_chat_template(tokenizer, history_msgs, add_generation_prompt=True)
    full_text = prompt_text + assistant_text
    # tokenize the prompt and full text, and return the prompt ids and full ids
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_seq_len:
        # the full prompt has too many tokens for the model, so we need to truncate it
        response_len = len(full_ids) - len(prompt_ids)
        if response_len >= max_seq_len:
            # the response itself is too long, so we need to truncate it (from the beginning)
            trimmed_full = full_ids[-max_seq_len:]
            return ExampleIds(
                prompt_ids=[],
                input_ids=trimmed_full,
                prompt_len=0,
            )
        # the full response and part of the prompt can fit, so truncate the prompt (from the beginning)
        keep_prompt = max_seq_len - response_len
        truncated_prompt = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
        truncated_full = truncated_prompt + full_ids[-response_len:]
        return ExampleIds(
            prompt_ids=truncated_prompt,
            input_ids=truncated_full,
            prompt_len=len(truncated_prompt),
        )
    # the full prompt fits, so we don't need to truncate anything
    return ExampleIds(
        prompt_ids=prompt_ids,
        input_ids=full_ids,
        prompt_len=len(prompt_ids),
    )


def _prepare_training_views(
    raw_ds: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seq_len: int,
    num_proc: int,
) -> datasets.Dataset:
    """Flatten conversations into (history -> assistant) training examples."""

    def _split_to_examples(example: dict) -> dict[str, typing.Any]:
        messages: list[dict] = example["messages"]
        out_input_ids: list[list[int]] = []
        out_prompt_len: list[int] = []
        for msg_idx, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            history = messages[:msg_idx]
            packed = _build_example_ids(tokenizer, history, message, max_seq_len)
            if not packed:
                continue
            out_input_ids.append(packed["input_ids"])
            out_prompt_len.append(packed["prompt_len"])
        return {"input_ids": out_input_ids, "prompt_len": out_prompt_len}

    dataset = raw_ds.map(
        _split_to_examples,
        batched=False,
        remove_columns=[column for column in raw_ds.column_names if column != "messages"],
        num_proc=num_proc,
        desc="Flatten conversations -> examples",
    )
    dataset = dataset.flatten_indices()
    dataset = dataset.filter(lambda row: isinstance(row["input_ids"], list) and len(row["input_ids"]) > 0)
    return dataset


class DataCollatorPromptMask:
    """Pad inputs and build labels masking out prompt tokens."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(
        self,
        features: list[dict],
    ) -> dict[str, torch.Tensor]:
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        attention_masks: list[list[int]] = []

        for feature in features:
            token_ids = feature["input_ids"][: self.max_length]
            prompt_len = int(feature["prompt_len"])
            pad_len = self.max_length - len(token_ids)
            if pad_len < 0:
                token_ids = token_ids[-self.max_length :]
                prompt_len = max(0, prompt_len - (len(feature["input_ids"]) - self.max_length))
                pad_len = 0

            input_ids = token_ids + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = [1] * len(token_ids) + [0] * pad_len

            labels = input_ids.copy()
            for position in range(min(prompt_len, self.max_length)):
                labels[position] = -100
            for position in range(len(token_ids), self.max_length):
                labels[position] = -100

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_masks.append(attention_mask)

        batch = {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }
        return batch


def train(
    *,
    config: "pyine.apps.trainers.hf_trainer_configs.MainConfig",
    datamodule: pyine.data.datamodule.ConversationDataModule,
) -> None:
    """Run LoRA fine-tuning with HuggingFace Trainer."""
    torch.backends.cuda.matmul.allow_tf32 = True
    transformers.set_seed(config.seed)

    tokenizer = _build_tokenizer(
        config.base_model,
        set_missing_pad_token_as_eos=config.tokenizer_pad_as_eos,
    )
    dtype = torch.bfloat16 if config.use_bf16 else (torch.float16 if config.use_fp16 else torch.float32)
    if config.quantization_mode == "qlora":
        model = _build_model_qlora(config.base_model, dtype, config.gradient_checkpointing)
    elif config.quantization_mode == "none":
        model = _build_model_no_quant(config.base_model, dtype, config.gradient_checkpointing)
    else:
        raise ValueError(f"Unsupported quantization_mode={config.quantization_mode}")
    model = config.lora.apply(model)

    train_raw = datamodule.get_hf_messages_dataset("train")
    eval_raw = datamodule.get_hf_messages_dataset("valid")

    num_proc = max(1, config.dataloader_num_workers // 2)
    train_ds = _prepare_training_views(train_raw, tokenizer, config.max_seq_len, num_proc=num_proc)
    eval_ds = _prepare_training_views(eval_raw, tokenizer, config.max_seq_len, num_proc=num_proc)

    collator = DataCollatorPromptMask(tokenizer, max_length=config.max_seq_len)

    training_args = transformers.TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        logging_steps=config.logging_steps,
        evaluation_strategy="steps" if config.eval_steps > 0 else "no",
        eval_steps=config.eval_steps if config.eval_steps > 0 else None,
        save_steps=config.save_steps,
        save_total_limit=3,
        dataloader_num_workers=config.dataloader_num_workers,
        fp16=config.use_fp16,
        bf16=config.use_bf16,
        report_to=["none"],
        lr_scheduler_type="cosine",
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optimizer,
        load_best_model_at_end=config.eval_steps > 0,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_first_step=True,
        seed=config.seed,
    )

    def _compute_metrics(eval_pred: transformers.trainer_utils.EvalPrediction) -> dict[str, float]:
        _ = eval_pred
        return {}

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if config.eval_steps > 0 else None,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics if config.eval_steps > 0 else None,
    )

    logger.info("starting training")
    train_out = trainer.train()
    logger.info("training finished: %s", train_out)

    logger.info("saving adapter and tokenizer to %s", config.output_dir)
    pathlib.Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    manifest = {
        "base_model": config.base_model,
        "timestamp": int(time.time()),
        "max_seq_len": config.max_seq_len,
        "training_args": training_args.to_dict(),
        "lora": config.lora.model_dump(),
        "seed": config.seed,
    }
    with open(os.path.join(config.output_dir, "training_manifest.json"), "w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, indent=2)
    logger.info("artifacts written to %s", config.output_dir)


def main(
    config: "pyine.apps.trainers.hf_trainer_configs.MainConfig",
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Main entrypoint for HuggingFace model fine-tuning."""
    pyine.utils.reprod.entrypoint_setup(
        runtime_config=runtime,
        main_config=config,
    )

    datamodule = config.datamodule_config.instantiate_datamodule(verbose=True)
    logger.info("preparing datamodule and loading datasets")
    datamodule.prepare_data()
    datamodule.setup()

    try:
        stats = datamodule.get_stats()
        logger.info("datamodule stats: %s", stats)
    except Exception as exc:
        logger.debug("failed to fetch datamodule stats: %s", exc)

    train(config=config, datamodule=datamodule)


if __name__ == "__main__":
    import pyine.apps.trainers.hf_trainer_configs

    try:
        pyine.apps.trainers.hf_trainer_configs.hydra_main()
    except Exception as exc:
        logger.exception("fatal error: %s", exc)
        sys.exit(1)
