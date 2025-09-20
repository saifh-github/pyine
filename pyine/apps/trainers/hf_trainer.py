"""Hydra-integrated HuggingFace fine-tuning CLI app.

This app wires together:
- a tokenizer and base model from hugging face;
- a dataset of (potentially multi-turn) conversations;
- a function that converts conversations into supervised examples;
- a Trainer that handles batching/padding/masking + the training loop.
"""

import collections
import functools
import logging
import pathlib
import typing

import datasets
import langchain_huggingface
import peft
import torch
import transformers

import pyine.apps.trainers.common
import pyine.configs.schemas
import pyine.data.datamodule
import pyine.evals.common
import pyine.organisms.datamodules.utils.samples
import pyine.utils.llm_providers
import pyine.utils.reprod

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import pyine.apps.trainers.hf_trainer_configs


def _build_tokenizer(
    base_model: str,
    set_missing_pad_token_as_eos: bool = True,
) -> transformers.PreTrainedTokenizerBase:
    """Loads and adapts a tokenizer for model training."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if set_missing_pad_token_as_eos and tokenizer.pad_token is None:
        # many causal LMs don't define a PAD token, but the hf Trainer expects one for padding batches
        # reuse EOS as PAD so padding uses a benign but already-known token id
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class _ExampleIds(typing.TypedDict):
    """Typed structure for tokenized example ids."""

    prompt_ids: list[int]
    input_ids: list[int]
    prompt_len: int


class _BatchTensors(typing.TypedDict):
    """Typed batch returned by the collator."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def _build_example_ids(
    tokenizer: transformers.PreTrainedTokenizerBase,
    history_msgs: list[dict],
    assistant_msg: dict,
    max_seq_len: int | None,
) -> _ExampleIds | None:
    """Turns a (history, assistant) pair into token ids and prompt length.

    Returns None when the assistant message is empty.
    """
    assistant_text = assistant_msg.get("content", "").strip()
    if not assistant_text:
        return None
    # from a conversation history and an assistant message, build the prompt text block
    prompt_text = tokenizer.apply_chat_template(
        # this will convert multi-turn messages into a single text block w/ proper role tags and delimiters
        history_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_text + assistant_text
    # tokenize the prompt and full text, and return the prompt ids and full ids
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if max_seq_len is not None and len(full_ids) > max_seq_len:
        # the full prompt has too many tokens for the model, so we need to truncate it
        response_len = len(full_ids) - len(prompt_ids)
        if response_len >= max_seq_len:
            # the response itself is too long, so we need to truncate it (from the beginning)
            trimmed_full = full_ids[-max_seq_len:]
            return _ExampleIds(
                prompt_ids=[],
                input_ids=trimmed_full,
                prompt_len=0,
            )
        # the full response and part of the prompt can fit, so truncate the prompt (from the beginning)
        keep_prompt = max_seq_len - response_len
        truncated_prompt = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
        truncated_full = truncated_prompt + full_ids[-response_len:]
        return _ExampleIds(
            prompt_ids=truncated_prompt,
            input_ids=truncated_full,
            prompt_len=len(truncated_prompt),
        )
    # the full prompt fits, so we don't need to truncate anything
    return _ExampleIds(
        prompt_ids=prompt_ids,
        input_ids=full_ids,
        prompt_len=len(prompt_ids),
    )


def _prepare_training_views(
    raw_ds: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seq_len: int | None,
    num_proc: int,
) -> datasets.Dataset:
    """Flattens conversations into (history -> assistant) training examples.

    The input dataset should have a "messages" column where each row is an entire conversation:
        [{"role": "system"/"user"/"assistant", "content": "..."}, ...]

    We convert each assistant turn into one supervised example whose inputs are:
        [prompt tokens from conversation history] + [assistant response tokens]

    We also keep "prompt_len" so the collator can later mask prompt tokens in labels.
    """

    def _split_to_examples(example: dict) -> dict[str, typing.Any]:
        # called on a single conversation by Dataset.map; must return a dict of column->values
        # (if returned values are lists of equal length, HF "explodes" them into multiple rows)
        messages: list[dict] = example["messages"]
        out_input_ids: list[list[int]] = []
        out_prompt_len: list[int] = []
        for msg_idx, message in enumerate(messages):
            # only model-authored turns are training targets
            if message.get("role") != "assistant":
                continue
            # history is everything before the currently-targeted assistant message
            history = messages[:msg_idx]
            packed = _build_example_ids(tokenizer, history, message, max_seq_len)
            if not packed:
                # skip empty/invalid assistant messages
                continue
            # collect one example per assistant turn; HF will create as many rows as we return here
            out_input_ids.append(packed["input_ids"])
            out_prompt_len.append(packed["prompt_len"])
        # only return columns needed for training; others are discarded via remove_columns below
        return dict(input_ids=out_input_ids, prompt_len=out_prompt_len)

    # @@@@@@@@ TODO: if this map becomes a bottleneck, add batching w/ fast tokenizer, tune num_proc, use cache
    # (worse case scenario, we can switch to a streaming/iterable dataset?)
    dataset = raw_ds.map(
        _split_to_examples,
        batched=False,  # process one conversation at a time for clarity
        # drop original columns so the resulting dataset only has "input_ids" and "prompt_len"
        remove_columns=raw_ds.column_names,
        num_proc=num_proc,  # parallelize if possible
        desc="Preparing examples",
    )
    # mapping with list outputs creates nested rows tied to original row; flatten to simple rows
    dataset = dataset.flatten_indices()
    # keep only rows that actually contain tokenized inputs
    dataset = dataset.filter(lambda row: isinstance(row["input_ids"], list) and len(row["input_ids"]) > 0)
    return dataset


class DataCollatorPromptMask:
    """Pads inputs and build labels masking out prompt tokens.

    The trainer uses the collator to turn a list of examples into fixed-size tensors; this one:
     - pads/truncates sequences to max_length;
     - builds attention_mask (1 for tokens, 0 for padding)
     - builds labels identical to inputs, except:
       - prompt token positions are set to `ignore_index` (ignored by loss)
       - padding positions are set to `ignore_index` (ignored by loss)
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        max_length: int,
        ignore_index: int = -100,  # this is an extremely-commonly-used default in pytorch/huggingface
    ) -> None:
        """Initializes the collator with a tokenizer and max_length."""
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.tokenizer = tokenizer
        assert self.tokenizer.pad_token_id is not None, "tokenizer must have a pad token"

    def __call__(
        self,
        features: list[dict[str, typing.Any]],
    ) -> _BatchTensors:
        """Turns a list of examples into a batch of tensors."""
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for feature in features:
            assert isinstance(feature, collections.abc.Mapping), "unexpected feature type, must be a dict"
            assert "input_ids" in feature, "missing expected input_ids column in feature dict"
            assert "prompt_len" in feature, "missing expected prompt_len column in feature dict"
            # left-truncate if necessary to keep the tail (usually contains the answer)
            orig_ids = feature["input_ids"]
            prompt_len = int(feature["prompt_len"])
            if len(orig_ids) > self.max_length:
                overflow = len(orig_ids) - self.max_length
                token_ids = orig_ids[-self.max_length :]
                # adjust prompt_len when cutting tokens from the left
                prompt_len = max(0, prompt_len - overflow)  # prevents negative if we truncate to response
            else:
                token_ids = orig_ids
            pad_len = self.max_length - len(token_ids)

            # right-pad to a fixed length
            input_ids = token_ids + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = [1] * len(token_ids) + [0] * pad_len

            # labels mirror inputs but ignore loss on prompt and padding positions
            labels = input_ids.copy()
            for position in range(min(prompt_len, self.max_length)):
                labels[position] = self.ignore_index
            for position in range(len(token_ids), self.max_length):
                labels[position] = self.ignore_index

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_masks.append(attention_mask)

        batch = _BatchTensors(
            input_ids=torch.tensor(input_ids_list, dtype=torch.long),
            attention_mask=torch.tensor(attention_masks, dtype=torch.long),
            labels=torch.tensor(labels_list, dtype=torch.long),
        )
        return batch


def infer_effective_max_seq_len(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    default_fallback: int = 2048,
) -> int:
    """Infers the effective max_seq_len for a model and its tokenizer."""
    # check tokenizer-reported cap (may be a very large sentinel if unknown)
    t_max = getattr(tokenizer, "model_max_length", None)
    if t_max is None or t_max > 10**8:  # treat huge sentinels as "unknown"
        t_max = None
    # check model config caps
    cfg = getattr(model, "config", None)
    m_caps: list[int] = []
    if cfg is not None:
        for attr in ("max_position_embeddings", "n_positions", "max_seq_len"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                m_caps.append(val)
        # some models define a smaller sliding window for training efficiency
        sw = getattr(cfg, "sliding_window", None)
        if isinstance(sw, int) and sw > 0:
            m_caps.append(sw)
    # gather all valid candidates we have found
    candidates = [c for c in [t_max, *(m_caps or [])] if isinstance(c, int)]
    if not candidates:
        return default_fallback
    return min(candidates)  # keep the minimum as a conservative choice


def _compute_metrics(
    eval_pred: transformers.trainer_utils.EvalPrediction,
    inputs,
    loss,
) -> dict[str, float]:
    # @@@@@@@ TODO do something here?
    # rely on eval loss for model selection; no extra metrics computed
    return {}


def train(
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    datamodule: pyine.data.datamodule.ConversationDataModule,
    config: "pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig",
    runtime: pyine.configs.schemas.RuntimeConfig | None,
) -> transformers.Trainer:
    """Run fine-tuning with HuggingFace Trainer.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        model: The model to fine-tune.
        datamodule: The datamodule to fetch data from.
        config: The app config object to fetch settings from.
        runtime: The runtime config object to fetch settings from.

    Returns:
        The instantiated trainer object that can be used for predictions.
    """
    assert config.training_args_config.do_train, "do_train must be True for training"
    # the datamodule supplies conversations as HF datasets with a "messages" column
    train_raw = datamodule.get_hf_messages_dataset("train")
    valid_raw = datamodule.get_hf_messages_dataset("valid")
    # use some of the dataloader workers for dataset.map to parallelize tokenization
    num_proc = max(1, config.dataloader_num_workers // 2)
    # convert conversation-style rows into flat, tokenized examples for training/valid
    model_max_seq_len = infer_effective_max_seq_len(model, tokenizer)
    logger.info(f"effective max_seq_len={model_max_seq_len}")
    train_ds = _prepare_training_views(train_raw, tokenizer, model_max_seq_len, num_proc=num_proc)
    valid_ds = _prepare_training_views(valid_raw, tokenizer, model_max_seq_len, num_proc=num_proc)
    # the collator pads to fixed length and masks labels for prompt tokens
    collator = DataCollatorPromptMask(tokenizer, max_length=model_max_seq_len)

    # disable KV cache during training (unnecessary overhead, + helps avoid compat issues w/ checkpointing)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        logger.info("enabling gradient checkpointing")
        model.gradient_checkpointing_enable()

    training_args_dict = config.training_args_config.model_dump()
    if config.use_wandb_logging:
        assert runtime is not None and runtime.wandb_run is not None, "wandb should have been initialized"
        training_args_dict["report_to"] = ["wandb"]
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


async def _evaluate(
    llm: langchain_huggingface.ChatHuggingFace,
    eval_subset_name: str,
    datamodule: pyine.data.datamodule.ConversationDataModule,
    llm_grader_provider_config: pyine.utils.llm_providers.LLMProviderConfig | None,
) -> pyine.evals.common.EvaluationResult:
    """Evaluates the LangChain-wrapped HF model on a given subset."""
    eval_parser = datamodule.get_parser(eval_subset_name)
    assert isinstance(eval_parser, pyine.organisms.datamodules.utils.samples.SampleBuilder)
    eval_parser = typing.cast(pyine.organisms.datamodules.utils.samples.SampleBuilder, eval_parser)
    evaluation_result = await pyine.evals.common.evaluate_langchain_runnable_on_subset(
        chain=datamodule.config.get_prompt_chain(model=llm),
        parser=eval_parser,
        llm_grader_provider_config=llm_grader_provider_config,
        verbose=True,
    )
    return evaluation_result


async def main(
    config: "pyine.apps.trainers.hf_trainer_configs.HFTrainerAppMainConfig",
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,  # None unless launched via hydra
) -> None:
    """Main function for the script; performs fine-tuning and evaluation for huggingface model.

    Args:
        config: Configuration for the application; see `HFTrainerAppMainConfig` for details.
        runtime: Configuration for the runtime; available when launched via hydra.
    """
    pyine.utils.reprod.entrypoint_setup(
        runtime_config=runtime,
        main_config=config,
        use_wandb_logging=config.use_wandb_logging,
    )

    dm = pyine.apps.trainers.common.prepare_code_exec_datamodule(config, runtime)

    logger.info("setting up model and tokenizer...")
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    if config.training_args_config.use_bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16 if config.training_args_config.use_fp16 else torch.float32
    device_map = {"": "mps"} if torch.backends.mps.is_available() else "auto"
    logging.info(f"will use {dtype=} and {device_map=}")
    tokenizer = _build_tokenizer(
        config.base_model,
        set_missing_pad_token_as_eos=config.tokenizer_pad_as_eos,
    )
    if config.quantization_mode == "qlora":
        logger.info("setting up model using QLoRA 4-bit quantization")
        quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=quant_config,
            torch_dtype=dtype,
            device_map=device_map,
        )
    elif config.quantization_mode == "none":
        logger.info("setting up model using no quantization")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.base_model, torch_dtype=dtype, device_map=device_map
        )
    else:
        raise ValueError(f"Unsupported quantization_mode={config.quantization_mode}")
    if config.lora_config is not None:
        logger.info("adding LoRA adapters")
        model = peft.get_peft_model(model, config.lora_config)

    if config.training_args_config.do_train:
        _ = train(tokenizer, model, dm, config, runtime)
    else:
        # @@@@@ TODO: load latest checkpoint? somehow? from path in config?
        raise NotImplementedError("no-training runs not yet supported")

    if config.training_args_config.do_predict:
        model.config.use_cache = True  # enable KV cache for generation
        model.eval()
        # build a LangChain-compatible chat model around the HF model for evaluation
        # model = typing.cast(transformers.PreTrainedModel, model)
        text_gen_pipeline = transformers.pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            # max_new_tokens=...,
            # do_sample=False,
            # pad_token_id=tokenizer.pad_token_id,
            # eos_token_id=tokenizer.eos_token_id,
        )
        eval_callback = functools.partial(
            _evaluate,
            llm=langchain_huggingface.ChatHuggingFace(
                llm=langchain_huggingface.HuggingFacePipeline(
                    pipeline=text_gen_pipeline,
                ),
            ),
        )
        await pyine.apps.trainers.common.evaluate_code_execution_model(eval_callback, dm, config, runtime)


if __name__ == "__main__":
    import pyine.apps.trainers.hf_trainer_configs

    pyine.apps.trainers.hf_trainer_configs.hydra_main()
