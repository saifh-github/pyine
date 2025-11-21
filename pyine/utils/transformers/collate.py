from __future__ import annotations

import collections.abc
import dataclasses
import logging
import math
import os
import typing

import torch
import wandb

import pyine.utils.distrib
from pyine.utils.transformers.constants import default_ignore_index
from pyine.utils.transformers.data import (
    ExampleBatchTensors,
    _parse_keep_extra_fields_config,  # type: ignore[reportPrivateUsage]
)

if typing.TYPE_CHECKING:
    import transformers

logger = logging.getLogger(__name__)

__all__ = [
    "CollatorBatchLogRecord",
    "CollatorBatchLogHandler",
    "PaddingCollatorWithPromptMask",
]


@dataclasses.dataclass(frozen=True)
class CollatorBatchLogRecord:
    """Record of batch stats for logging."""

    stage: str | None
    """Logical stage of the collator when the batch was prepared, if specified (e.g. train/valid/...)."""
    batch_size: int
    """Batch size."""
    padded_seq_len: int
    """Padded sequence length."""
    padding_ratio: float
    """Ratio of padding tokens in the batch."""
    non_ignored_label_ratio: float
    """Ratio of non-ignored tokens in the batch (i.e., excluding padding)."""


type CollatorBatchLogHandler = collections.abc.Callable[[CollatorBatchLogRecord], None]
"""Callable logging per-batch stats (stage, batch size, padded len, ratios)."""


def _collect_extra_fields_from_batch(
    batch_data: list[dict[str, typing.Any]],
    forward_all_fields: bool,
    keep_extra_fields: list[str],
    expected_field_names: list[str],
) -> dict[str, list[typing.Any]]:
    """Collect extra fields from batch data according to forwarding configuration.

    Args:
        batch_data: List of input data dictionaries.
        forward_all_fields: If True, forward all fields not in expected_field_names.
        keep_extra_fields: Explicit list of field names to forward (ignored if forward_all_fields is True).
        expected_field_names: Field names that should not be forwarded (already handled by collator).

    Returns:
        Dictionary mapping field names to lists of values from the batch
    """
    if not forward_all_fields and not keep_extra_fields:
        return {}
    if forward_all_fields:
        ordered_keys: list[str] = []
        seen: set[str] = set()
        for data in batch_data:
            for key in data:
                if key in expected_field_names or key in seen:
                    continue
                ordered_keys.append(key)
                seen.add(key)
        keys_to_forward = ordered_keys
    else:
        keys_to_forward = [key for key in keep_extra_fields if key not in expected_field_names]
    extra_fields: dict[str, list[typing.Any]] = {}
    for key in keys_to_forward:
        values: list[typing.Any] = []
        for idx, sample in enumerate(batch_data):
            if key not in sample:
                raise ValueError(
                    f"extra field '{key}' is missing from sample index {idx}; "
                    "provide the field for all batched items or disable keep_extra_fields"
                )
            values.append(sample[key])
        extra_fields[key] = values
    return extra_fields


class PaddingCollatorWithPromptMask:
    """Pads/truncates inputs and builds labels masking out prompt tokens.

    This batch collator turns a list of examples (variable-length sequences) into tensors:
     - pads and truncates to a fixed max length or to the effective max length of the batch;
     - builds attention_mask for these padded/truncated sequences (1 for tokens, 0 for padding);
     - builds labels identical to input token ids, except:
       - prompt token positions are set to `ignore_index` (ignored by loss); and
       - padding positions are set to `ignore_index` (ignored by loss).

    Important note: when left-truncating, this collator will NOT allow responses to be truncated; if
    an example is encountered where this would be required, a ValueError is raised. When right-truncating,
    we also do not allow responses to be ENTIRELY truncated; needing to do so will again raise a ValueError.

    Args:
        tokenizer: Tokenizer whose ``pad_token_id`` is used for right-padding.
        max_length: Maximum sequence length to pad or truncate to.
        always_pad_to_max_length: specifies whether to always pad batch samples to the maximum
            length specified as argument or only as necessary (i.e., based on sample lengths).
        pad_to_multiple_of: specifies whether to pad the batch samples to a multiple of a value.
        keep_extra_fields: Additional keys to carry over from input examples to the output batch.
            If a list, only those keys are forwarded. If True, forwards all fields present in
            the inputs. If falsy/None, no extra fields are forwarded.
        drop_overflowing_examples: If True, any examples whose length exceeds the effective max
            sequence length and where truncation would also eliminate the entire prompt/reponse
            will be DROPPED from the batch. Otherwise, an exception is raised.
        ignore_index: Label value used to mask prompt and padding positions in the returned
            ``labels`` tensor.
        batch_log_handler: Optional callable used to log per-batch stats (stage, batch size, padded
            sequence length, and padding/label ratios).
        wandb_run_or_init_kwargs: Optional W&B init kwargs (for a run to resume/connect to) or wandb
            run to log per-batch stats to.
        init_stage: Logical stage where this collator will be applied by default, unless changed;
            could be e.g. 'train', 'valid', etc.
    """

    # @@@@@@ TODO: add support for packing? sort for min-pad batches? (or just toggle group_by_length in trainer args?)

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int,
        *,
        always_pad_to_max_length: bool = False,
        pad_to_multiple_of: int | None = None,
        keep_extra_fields: list[str] | bool | None = None,
        drop_overflowing_examples: bool = False,
        ignore_index: int = default_ignore_index,
        batch_log_handler: CollatorBatchLogHandler | None = None,
        wandb_run_or_init_kwargs: wandb.Run | dict[str, typing.Any] | None = None,
        init_stage: str | None = None,
    ) -> None:
        """Initializes the collator."""
        self.max_length = max_length
        self.always_pad_to_max_length = always_pad_to_max_length
        assert pad_to_multiple_of is None or pad_to_multiple_of > 0, "pad_to_multiple_of must be > 0"
        self.pad_to_multiple_of = int(pad_to_multiple_of) if pad_to_multiple_of else None
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > self.max_length:
            raise ValueError(
                f"pad_to_multiple_of ({self.pad_to_multiple_of}) cannot exceed max_length ({self.max_length})"
            )
        self._forward_all_fields, self._keep_extra_fields = _parse_keep_extra_fields_config(keep_extra_fields)
        self.drop_overflowing_examples = drop_overflowing_examples
        self.ignore_index = ignore_index
        self._batch_log_handler = batch_log_handler
        if isinstance(wandb_run_or_init_kwargs, dict):
            self._wandb_init_kwargs = wandb_run_or_init_kwargs
            self._wandb_run_obj: wandb.Run | None = None  # will be prepared on first use given the init kwargs
        else:
            self._wandb_init_kwargs = None
            self._wandb_run_obj = wandb_run_or_init_kwargs
        self._stage: str | None = init_stage
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if not isinstance(pad_token_id, int):
            raise ValueError("tokenizer must expose an integer pad token")
        self._pad_token_id = pad_token_id
        truncation_side = getattr(tokenizer, "truncation_side", None)
        if not isinstance(truncation_side, str):
            raise ValueError("tokenizer must expose a string truncation side")
        self._truncation_side = truncation_side
        padding_side = getattr(tokenizer, "padding_side", None)
        if not isinstance(padding_side, str):
            raise ValueError("tokenizer must expose a string padding side")
        self._padding_side = padding_side

    @property
    def _metrics_prefix(self) -> str:
        """Returns the prefix to use for logging metrics with this collator."""
        if self._stage is None:
            return "collator"
        return f"collator/{self._stage}"

    @property
    def _use_wandb(self) -> bool:
        """Returns whether this collator is configured to log per-batch stats to wandb."""
        return self._wandb_run_obj is not None or self._wandb_init_kwargs is not None

    @property
    def _wandb_run(self) -> wandb.Run:
        """Returns the wandb run associated with this collator, from the initially provided run id."""
        assert self._use_wandb, "cannot get wandb run; collator is not configured to log per-batch stats to wandb"
        if self._wandb_run_obj is None:
            if self._wandb_init_kwargs is None:
                raise ValueError("cannot get wandb run; _wandb_init_kwargs is None")
            if "id" not in self._wandb_init_kwargs:
                raise ValueError("cannot get wandb run; _wandb_init_kwargs does not contain 'id'")
            wandb_run_id = self._wandb_init_kwargs["id"]
            curr_rank = pyine.utils.distrib.get_global_rank()
            process_label = f"rank_{curr_rank}_pid_{os.getpid()}"
            logger.debug(f"getting wandb run object for id={wandb_run_id} on process {process_label}...")
            init_kwargs = self._wandb_init_kwargs.copy()
            init_kwargs.update(
                {
                    "job_type": "collate",
                    # TODO: @@@@@@ fix this;
                    #       as of 2025-11-20 and wandb 0.22.3, this init seems to hang indefinitely in workers
                    "settings": wandb.Settings(
                        mode="shared",
                        init_timeout=300,
                        x_label=process_label,
                        x_primary=False,
                        x_update_finish_state=False,
                    ),
                }
            )
            self._wandb_run_obj = wandb.init(**init_kwargs)
            logger.debug(f"connected to wandb run on process {process_label} (url={self._wandb_run_obj.url})")
        return self._wandb_run_obj

    def set_stage(
        self,
        stage: str | None,
    ) -> None:
        """Update the logical stage (train/eval) for batch logging."""
        if stage == self._stage:
            return
        logger.debug(f"collator stage set to {stage}")
        self._stage = stage

    def __call__(
        self,
        to_batch: list[dict[str, typing.Any]],
    ) -> ExampleBatchTensors:
        """Turns a list of examples into a batch of tensors.

        The provided list of examples should be a list of dictionaries, where each dictionary should contain at least
        the key "input_ids" with a list of integers representing the input sequence. An optional key "prompt_len"
        can also be provided to indicate the length of the prompt sequence (if not provided, it is assumed to be the
        entire input sequence).

        The output dictionaries will contain the following keys:
            input_ids: the list of token ids for the full example (prompt + potential response).
            attention_mask: mask for the full example (prompt + potential response).
            labels: target labels that mirror input ids but ignore loss on prompt and padding positions.
            prompt_len: length of the prompt (in number of tokens), prior to padding.
            input_len: length of the full example (prompt + potential response), prior to padding.

        The returned dictionary will also carry over any extra fields specified in the initialization.
        """
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        attention_masks: list[list[int]] = []
        prompt_lengths: list[int] = []
        input_lengths: list[int] = []
        # determine the effective padding length for this batch
        if self.always_pad_to_max_length:
            if self.pad_to_multiple_of is None:
                effective_max_length = self.max_length
            else:
                effective_max_length = (self.max_length // self.pad_to_multiple_of) * self.pad_to_multiple_of
        else:
            # compute the maximum input length in the batch (after truncation)
            batch_max_length = 0
            for data in to_batch:
                orig_ids = typing.cast("list[int]", data["input_ids"])
                batch_max_length = max(batch_max_length, min(len(orig_ids), self.max_length))
            if self.pad_to_multiple_of is None:
                effective_max_length = batch_max_length
            else:
                effective_max_length = min(
                    (self.max_length // self.pad_to_multiple_of) * self.pad_to_multiple_of,
                    math.ceil(batch_max_length / self.pad_to_multiple_of) * self.pad_to_multiple_of,
                )
        for data in to_batch:
            assert isinstance(data, collections.abc.Mapping), "unexpected feature type, must be a dict"
            assert "input_ids" in data, "missing expected key 'input_ids' in data dict"
            assert isinstance(data["input_ids"], list), "expected 'input_ids' to be a list of integers"
            # truncate if necessary according to tokenizer's truncation_side
            orig_ids = typing.cast("list[int]", data["input_ids"])
            prompt_len = int(data["prompt_len"]) if "prompt_len" in data else len(orig_ids)
            if prompt_len < 0 or prompt_len > len(orig_ids):
                raise ValueError(
                    f"invalid prompt_len={prompt_len}; "
                    f"value must be between 0 and the sequence length ({len(orig_ids)})"
                )
            response_len = len(orig_ids) - prompt_len
            if len(orig_ids) > effective_max_length:
                overflow = len(orig_ids) - effective_max_length
                if self._truncation_side == "left":
                    if overflow >= prompt_len:
                        if self.drop_overflowing_examples:
                            continue
                        raise ValueError(
                            "found an example with a max length overflow g.e. to the prompt length "
                            f"({overflow=}, {prompt_len=}, {response_len=}, and {effective_max_length=})"
                        )
                    token_ids = orig_ids[-effective_max_length:]
                    prompt_len = prompt_len - overflow
                else:  # truncation_side == "right"
                    if overflow >= response_len:
                        if self.drop_overflowing_examples:
                            continue
                        raise ValueError(
                            "found an example with a max length overflow g.e. to the response length "
                            f"({overflow=}, {prompt_len=}, {response_len=}, and {effective_max_length=})"
                        )
                    token_ids = orig_ids[:effective_max_length]
                    # response_len = response_len - overflow
            else:
                token_ids = orig_ids
            input_len = len(token_ids)
            pad_len = effective_max_length - input_len
            # pad according to tokenizer's padding_side
            if self._padding_side == "left":
                input_ids = [self._pad_token_id] * pad_len + token_ids
                attention_mask = [0] * pad_len + [1] * input_len
            else:  # padding_side == "right"
                input_ids = token_ids + [self._pad_token_id] * pad_len
                attention_mask = [1] * input_len + [0] * pad_len
            # labels mirror input ids but ignore loss on prompt and padding positions
            labels = input_ids.copy()
            if self._padding_side == "left":
                # padding is on the left, prompt is after padding
                labels[:pad_len] = [self.ignore_index] * pad_len
                labels[pad_len : pad_len + min(prompt_len, effective_max_length)] = [self.ignore_index] * min(
                    prompt_len, effective_max_length
                )
            else:  # padding_side == "right"
                # prompt is at the start, padding is at the end
                labels[: min(prompt_len, effective_max_length)] = [self.ignore_index] * min(
                    prompt_len, effective_max_length
                )
                labels[input_len:] = [self.ignore_index] * pad_len
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_masks.append(attention_mask)
            prompt_lengths.append(prompt_len)
            input_lengths.append(input_len)
        if not input_ids_list:
            raise ValueError("no valid examples found in batch; all dropped/overflowing?")
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        batch_dict: dict[str, typing.Any] = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels_tensor,
            "prompt_len": prompt_lengths,
            "input_len": input_lengths,
        }
        self._log_batch_stats(
            batch_size=len(to_batch),
            padded_seq_len=input_ids_tensor.size(1),
            attention_mask=attention_mask_tensor,
            labels=labels_tensor,
        )
        # carry over targeted (or all) metadata fields with the input batch order
        extra_fields = _collect_extra_fields_from_batch(
            batch_data=to_batch,
            forward_all_fields=self._forward_all_fields,
            keep_extra_fields=self._keep_extra_fields,
            expected_field_names=["input_ids", "attention_mask", "labels", "prompt_len", "input_len"],
        )
        batch_dict.update(extra_fields)
        return typing.cast("ExampleBatchTensors", batch_dict)

    def _log_batch_stats(
        self,
        *,
        batch_size: int,
        padded_seq_len: int,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Logs batch stats (stage, batch size, padded sequence length, ratios) if needed."""
        # if we are NOT doing any batch stats logging, just return
        if self._batch_log_handler is None and not self._use_wandb:
            return
        # otherwise, prep the batch stats and log them
        total_tokens = max(batch_size * padded_seq_len, 1)
        valid_tokens = int(attention_mask.sum().item())
        padding_ratio = 1.0 - (valid_tokens / total_tokens)
        non_ignored = int((labels != self.ignore_index).sum().item())
        non_ignored_ratio = non_ignored / total_tokens
        if self._use_wandb:
            metric_prefix = self._metrics_prefix
            self._wandb_run.log(  # type: ignore[reportUnknownMemberType]
                {
                    f"{metric_prefix}/batch_size": batch_size,
                    f"{metric_prefix}/padded_seq_len": padded_seq_len,
                    f"{metric_prefix}/padding_ratio": padding_ratio,
                    f"{metric_prefix}/non_ignored_label_ratio": non_ignored_ratio,
                },
            )
        if self._batch_log_handler is not None:
            record = CollatorBatchLogRecord(
                stage=self._stage,
                batch_size=batch_size,
                padded_seq_len=padded_seq_len,
                padding_ratio=padding_ratio,
                non_ignored_label_ratio=non_ignored_ratio,
            )
            self._batch_log_handler(record)
