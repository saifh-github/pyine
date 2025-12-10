from __future__ import annotations

import collections.abc
import dataclasses
import functools
import logging
import os
import shutil
import typing
import uuid

import datasets as hf_datasets
import filelock

if typing.TYPE_CHECKING:
    import pathlib

    import torch
    import transformers

logger = logging.getLogger(__name__)

__all__ = [
    "ExampleBatchTensors",
    "DataCacheSettings",
    "ConversationMessage",
    "ConversationHistory",
    "apply_model_template_to_messages",
    "prepare_generation_prompts_from_dataset",
    "prepare_examples_from_conversations",
]


class _ExampleData(typing.TypedDict):
    """Typed structure for tokenized example ids. Actual dicts may also contain extra fields."""

    input_ids: list[int]
    """List of token ids for the full example (prompt + potential response)."""
    prompt_ids: list[int]
    """List of token ids for the prompt; may be empty in worst cases of truncation."""


class ExampleBatchTensors(typing.TypedDict):
    """Typed batch returned by training data collators. Actual dicts may also contain extra fields."""

    input_ids: torch.Tensor
    """List of token ids for the full example (prompt + potential response), with padding."""
    attention_mask: torch.Tensor
    """Attention mask for the full example (prompt + potential response), with padding."""
    labels: torch.Tensor
    """Target labels (token ids) that mirror input ids but ignore loss on prompt and padding positions."""
    prompt_len: list[int]
    """Length of the prompt (in number of tokens), prior to padding."""
    input_len: list[int]
    """Length of the full example (prompt + potential response), prior to padding."""


@dataclasses.dataclass(frozen=True)
class DataCacheSettings:
    """Configuration for on-disk data cache."""

    cache_path: pathlib.Path
    """Target directory where the data should be saved."""
    lock_timeout_seconds: float
    """Maximum time to wait when acquiring the cache lock, in seconds."""

    def build_lock(self) -> filelock.BaseFileLock:
        """Returns the file lock guarding this dataset cache."""
        assert self.cache_path.parent.is_dir()
        lock_path = self.cache_path.parent / f"{self.cache_path.name}.lock"
        return filelock.FileLock(str(lock_path), timeout=self.lock_timeout_seconds)


type ConversationMessage = dict[str, typing.Any]
"""Alias for messages in conversation datasets."""

type ConversationHistory = list[ConversationMessage]
"""Alias for ordered conversation histories."""


def _batch_apply_model_template_to_messages(
    batch: collections.abc.Mapping[str, typing.Any],
    tokenizer: transformers.PreTrainedTokenizer,
    append_eos_token: bool,
    strip_output: bool,
    messages_key: str,
    output_key: str,
    keep_original_data: bool,
    apply_chat_template_kwargs: dict[str, typing.Any] | None,
) -> dict[str, typing.Any]:
    """Applies a model template to a batch of (hf-formatted) message dictionaries.

    See `apply_model_template_to_messages` for information on arguments.

    Kept static/top-level-friendly to keep pickling happy.
    """
    assert isinstance(batch, collections.abc.Mapping), f"unexpected input batch type: {type(batch)}"
    assert messages_key in batch, f"missing expected messages key: {messages_key}"
    messages = typing.cast("list[dict[str, str]]", batch[messages_key])
    text_result = tokenizer.apply_chat_template(  # type: ignore[reportUnknownMemberType]
        conversation=messages,
        **(apply_chat_template_kwargs or {}),
    )
    if not isinstance(text_result, list):
        raise TypeError(
            f"expected tokenizer chat template output to be a list of strings, got {type(text_result)}",
        )
    assert all(isinstance(s, str) for s in text_result), "expected output to be a list of strings"
    assert len(text_result) == len(messages), "length mismatch between input messages and output text"
    text_result = typing.cast("list[str]", text_result)
    if strip_output:
        text_result = [item.strip() for item in text_result]
    if append_eos_token:
        assert hasattr(tokenizer, "eos_token"), "tokenizer missing eos token"
        eos_token = typing.cast("str | list[str] | None", tokenizer.eos_token)  # type: ignore[reportUnknownMemberType]
        if eos_token is None:
            raise ValueError("tokenizer has no EOS token configured")
        eos_suffix = "".join(eos_token) if isinstance(eos_token, list) else str(eos_token)
        text_result = [item + eos_suffix for item in text_result]
    output: dict[str, typing.Any] = dict(batch) if keep_original_data else {}
    output[output_key] = text_result
    return output


def apply_model_template_to_messages(
    hf_messages_ds: hf_datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    append_eos_token: bool = False,
    strip_output: bool = False,
    messages_key: str = "messages",
    output_key: str = "text",
    keep_original_data: bool = False,
    apply_chat_template_kwargs: dict[str, typing.Any] | None = None,
    keep_in_memory: bool = False,
) -> hf_datasets.Dataset:
    """Map a HuggingFace messages dataset to a new dataset with text-only samples."""
    transform_batch: typing.Callable[[collections.abc.Mapping[str, typing.Any]], dict[str, typing.Any]] = (
        functools.partial(
            _batch_apply_model_template_to_messages,
            tokenizer=tokenizer,
            append_eos_token=append_eos_token,
            strip_output=strip_output,
            messages_key=messages_key,
            output_key=output_key,
            keep_original_data=keep_original_data,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
    )
    mapped_dataset: hf_datasets.Dataset = hf_messages_ds.map(  # type: ignore[reportUnknownMemberType]
        function=transform_batch,
        batched=True,
        desc="applying tokenizer chat template",
        keep_in_memory=keep_in_memory,
    )
    return mapped_dataset


def prepare_generation_prompts_from_dataset(
    prompts_ds: hf_datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_len: int | None,
    prompt_text_key: str = "text",
    keep_extra_fields: list[str] | bool | None = None,
    keep_in_memory: bool = False,
) -> hf_datasets.Dataset:
    """Prepares and returns a dataset of encoded and generation-ready prompts.

    Args:
        prompts_ds: Input dataset containing message prompts to encode.
        tokenizer: Tokenizer to use for encoding text.
        max_seq_len: Maximum sequence length for truncation.
        prompt_text_key: Key name for the prompt text column (with chat formatting already applied).
        keep_extra_fields: Additional fields to preserve from the input dataset. Can be:
            - A list of field names to keep;
            - True to keep all fields (except messages_key); or
            - None/False to keep only essential fields.
        keep_in_memory: Whether to keep the datasets in memory.

    Returns:
        A HuggingFace Dataset containing encoded prompts ready for generation, with fields:
            - input_ids: List of token ids for the encoded prompts
            - attention_mask: Attention masks for the encoded prompts
            - input_len: Length of each prompt before padding
            - sample_idx: Sequential index for each prompt
            - Any additional fields specified by keep_extra_fields
    """
    # TODO: if the map calls in here become a bottleneck, tune num_proc, use cache
    # (worse case scenario, we can switch to a streaming/iterable dataset?)
    forward_all_fields, keep_extra_fields_list = _parse_keep_extra_fields_config(keep_extra_fields)
    templated_prompts_ds: hf_datasets.Dataset = apply_model_template_to_messages(
        hf_messages_ds=prompts_ds,
        tokenizer=tokenizer,
        keep_original_data=bool(forward_all_fields or keep_extra_fields_list),
        apply_chat_template_kwargs={
            "tokenize": False,
            "add_generation_prompt": True,
        },
        keep_in_memory=keep_in_memory,
    )
    sample_idx = 0

    def _encode_prompts(
        sample: collections.abc.Mapping[str, typing.Any],
    ) -> dict[str, typing.Any]:
        nonlocal sample_idx
        assert isinstance(sample, collections.abc.Mapping), f"unexpected sample data type: {type(sample)}"
        assert prompt_text_key in sample, f"missing '{prompt_text_key}' key for chat-templated text in sample data?"
        assert isinstance(sample[prompt_text_key], str), "expected chat template application to yield string prompts"
        encoded_inputs = tokenizer(sample[prompt_text_key], truncation=True, max_length=max_seq_len)
        input_ids = typing.cast("list[int]", encoded_inputs["input_ids"])
        attention_mask = typing.cast("list[int]", encoded_inputs["attention_mask"])
        output = {
            "sample_idx": sample_idx,
            **sample,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_len": len(input_ids),
        }
        sample_idx += 1
        return output

    if forward_all_fields:
        columns_to_remove: list[str] = []
    elif keep_extra_fields_list:
        columns_to_remove = [col for col in templated_prompts_ds.column_names if col not in keep_extra_fields_list]
    else:
        columns_to_remove = templated_prompts_ds.column_names

    dataset: hf_datasets.Dataset = templated_prompts_ds.map(  # type: ignore[reportUnknownMemberType]
        _encode_prompts,
        batched=False,  # not parallelized since we're adding a nonlocal index above
        remove_columns=columns_to_remove,
        desc="encoding prompts",
        keep_in_memory=keep_in_memory,
    )
    return dataset


def _build_example_ids_from_conversation_parts(
    tokenizer: transformers.PreTrainedTokenizer,
    history_msgs: ConversationHistory,
    assistant_msg: ConversationMessage,
    max_seq_len: int | None,
) -> _ExampleData | None:
    """Turns a (history, assistant) pair into token ids and prompt length.

    Returns None when the assistant message is empty.
    """
    assistant_raw = assistant_msg.get("content", "")
    assistant_text = str(assistant_raw).strip()
    if not assistant_text:
        return None
    # from a conversation history and an assistant message, build the prompt text block
    apply_template_attr = getattr(tokenizer, "apply_chat_template", None)
    if apply_template_attr is None:
        raise AttributeError("tokenizer must support apply_chat_template")
    apply_template = typing.cast("typing.Callable[..., str]", apply_template_attr)
    prompt_text = apply_template(
        # this will convert multi-turn messages into a single text block w/ proper role tags and delimiters
        history_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = f"{prompt_text}{assistant_text}"
    # tokenize the prompt and full text, and return the prompt ids and full ids
    prompt_encoding = tokenizer(prompt_text, add_special_tokens=False)
    full_encoding = tokenizer(full_text, add_special_tokens=False)
    prompt_ids = typing.cast("list[int]", prompt_encoding["input_ids"])
    full_ids = typing.cast("list[int]", full_encoding["input_ids"])
    if max_seq_len is not None and len(full_ids) > max_seq_len:
        # the full prompt has too many tokens for the model, so we need to truncate it
        response_len = len(full_ids) - len(prompt_ids)
        if response_len >= max_seq_len:
            # the response itself is too long, so we need to truncate it (from the beginning)
            trimmed_full = full_ids[-max_seq_len:]
            return _ExampleData(
                input_ids=trimmed_full,
                prompt_ids=[],
            )
        # the full response and part of the prompt can fit, so truncate the prompt (from the beginning)
        keep_prompt = max_seq_len - response_len
        truncated_prompt = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
        truncated_full = truncated_prompt + full_ids[-response_len:]
        return _ExampleData(
            input_ids=truncated_full,
            prompt_ids=truncated_prompt,
        )
    # the full prompt fits, so we don't need to truncate anything
    return _ExampleData(
        input_ids=full_ids,
        prompt_ids=prompt_ids,
    )


def prepare_examples_from_conversations(
    convo_ds: hf_datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_len: int | None,
    num_proc: int,
    messages_key: str = "messages",
    keep_extra_fields: list[str] | bool | None = None,
    keep_in_memory: bool = False,
    cache_settings: DataCacheSettings | None = None,
    force_rebuild: bool = False,
) -> hf_datasets.Dataset:
    """Flattens conversations into tokenizer-encoded (history -> assistant) training examples.

    The input dataset should have a column of messages where each row is an entire conversation:
        [{"role": "system"/"user"/"assistant", "content": "..."}, ...]

    We convert each assistant turn into one supervised example whose inputs are:
        [prompt tokens from conversation history] + [assistant response tokens]

    We also keep "prompt_len" so the collator can later mask prompt tokens in labels.

    Args:
        convo_ds: Input dataset with conversation messages.
        tokenizer: Tokenizer to use for encoding text.
        max_seq_len: Maximum sequence length for truncation.
        num_proc: Number of processes for parallel processing.
        messages_key: Key name for the messages column in the dataset.
        keep_extra_fields: Additional fields to preserve from the input dataset. Can be:
            - A list of field names to keep;
            - True to keep all fields (except messages_key); or
            - None/False to keep only input_ids and prompt_len.
        keep_in_memory: Whether to keep the datasets in memory.
        cache_settings: Optional configuration describing where to cache tokenized datasets. When provided,
            the cache will be checked before recomputing the tokenized dataset, preventing duplicate work.
        force_rebuild: Whether to force rebuilding the cached dataset even if it already exists.
    """
    forward_all_fields, keep_extra_fields_list = _parse_keep_extra_fields_config(keep_extra_fields)

    def _build_dataset() -> hf_datasets.Dataset:
        templated_convo_ds: hf_datasets.Dataset = apply_model_template_to_messages(
            hf_messages_ds=convo_ds,
            tokenizer=tokenizer,
            keep_original_data=bool(forward_all_fields or keep_extra_fields_list),
            apply_chat_template_kwargs={
                "tokenize": False,
                "add_generation_prompt": False,
            },
            keep_in_memory=keep_in_memory,
        )

        def _split_to_examples(batch: dict[str, typing.Any]) -> dict[str, typing.Any]:
            # called on a batch of conversations by Dataset.map; must return dict of column->flat lists
            conversations = typing.cast("list[ConversationHistory]", batch[messages_key])
            out_input_ids: list[list[int]] = []
            out_prompt_len: list[int] = []
            # track extra fields to forward
            extra_fields_data: dict[str, list[typing.Any]] = {}
            if forward_all_fields:
                # initialize lists for all fields except messages_key
                for key in batch:
                    if key != messages_key:
                        extra_fields_data[key] = []
            elif keep_extra_fields_list:
                # initialize lists for specified fields
                for key in keep_extra_fields_list:
                    if key in batch and key != messages_key:
                        extra_fields_data[key] = []
            for convo_idx, messages in enumerate(conversations):
                for msg_idx, message in enumerate(messages):
                    # only model-authored turns are training targets
                    if message.get("role") != "assistant":
                        continue
                    # history is everything before the currently-targeted assistant message
                    history = messages[:msg_idx]
                    result = _build_example_ids_from_conversation_parts(tokenizer, history, message, max_seq_len)
                    if result is None:
                        # skip empty/invalid assistant messages
                        continue
                    # collect one example per assistant turn; HF will create one row per appended item
                    prompt_ids = result["prompt_ids"]
                    out_input_ids.append(result["input_ids"])
                    out_prompt_len.append(len(prompt_ids))
                    # replicate extra fields for this example
                    for key in extra_fields_data:
                        extra_fields_data[key].append(batch[key][convo_idx])

            # return columns needed for training plus any extra fields
            output = {"input_ids": out_input_ids, "prompt_len": out_prompt_len}
            output.update(extra_fields_data)
            return output

        if forward_all_fields:
            columns_to_remove: list[str] = [messages_key]  # always remove messages since it's exploded
        elif keep_extra_fields_list:
            columns_to_remove = [col for col in templated_convo_ds.column_names if col not in keep_extra_fields_list]
            if messages_key not in columns_to_remove:
                columns_to_remove.append(messages_key)  # always remove messages since it's exploded
        else:
            columns_to_remove = templated_convo_ds.column_names
        dataset: hf_datasets.Dataset = templated_convo_ds.map(  # type: ignore[reportUnknownMemberType]
            _split_to_examples,
            batched=True,  # explode list outputs into individual rows automatically
            remove_columns=columns_to_remove,
            num_proc=num_proc,  # parallelize if possible
            desc="preparing examples",
            keep_in_memory=keep_in_memory,
        )
        return dataset

    if cache_settings is None:
        # if caching is not enabled, build and return the dataset directly
        return _build_dataset()

    cache_settings.cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_settings.build_lock():
        if cache_settings.cache_path.exists():
            if force_rebuild:
                logger.info(f"force-regenerating tokenized dataset cache at: {cache_settings.cache_path}")
                shutil.rmtree(cache_settings.cache_path)
            else:
                logger.info(f"loading tokenized dataset from cache: {cache_settings.cache_path}")
                return hf_datasets.Dataset.load_from_disk(  # type: ignore[reportUnknownMemberType]
                    dataset_path=cache_settings.cache_path,
                    keep_in_memory=keep_in_memory,
                )
        logger.info(f"building tokenized dataset cache at: {cache_settings.cache_path}")
        dataset = _build_dataset()
        tmp_path = cache_settings.cache_path.parent / f"{cache_settings.cache_path.name}.tmp.{uuid.uuid4().hex}"
        try:
            dataset.save_to_disk(tmp_path)  # type: ignore[reportUnknownMemberType]
            os.replace(tmp_path, cache_settings.cache_path)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)
        logger.info(f"saved tokenized dataset cache: {cache_settings.cache_path}")
        if keep_in_memory:
            return dataset
        return hf_datasets.Dataset.load_from_disk(  # type: ignore[reportUnknownMemberType]
            dataset_path=cache_settings.cache_path,
            keep_in_memory=keep_in_memory,
        )


def _parse_keep_extra_fields_config(
    keep_extra_fields: list[str] | bool | None,
) -> tuple[bool, list[str]]:
    """Parse keep_extra_fields configuration into forward_all flag and explicit field list.

    Args:
        keep_extra_fields: Configuration specifying which extra fields to forward. Can be:
            - A list of field names to forward;
            - True to forward all fields; or
            - None/False to forward no extra fields.

    Returns:
        A tuple of (forward_all_fields, keep_extra_fields_list)
    """
    if isinstance(keep_extra_fields, list):
        return False, list(keep_extra_fields)
    if keep_extra_fields:
        assert keep_extra_fields is True
        return True, []
    return False, []
