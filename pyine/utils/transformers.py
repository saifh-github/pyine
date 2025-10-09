import collections.abc
import contextlib
import math
import typing

import datasets as hf_datasets
import pydantic
import torch
import tqdm
import transformers

import pyine.utils.pydantic

if typing.TYPE_CHECKING:

    class TrainingArgsConfig(pydantic.BaseModel):
        """Stubbed interface for the HuggingFace Trainer Arguments config class defined below."""

        def __getattr__(self, name: str) -> typing.Any: ...

    class GenerationConfig(pydantic.BaseModel):
        """Stubbed interface for the HuggingFace text generation pipeline config class defined below."""

        def __getattr__(self, name: str) -> typing.Any: ...

else:
    TrainingArgsConfig = pyine.utils.pydantic.model_from_callable(
        fn=transformers.TrainingArguments,
        name="TrainingArgsConfig",
        model_config=pydantic.ConfigDict(frozen=True, extra="forbid"),
        default_overrides={
            # we use some updated defaults (low-impact, QoL stuff)
            "load_best_model_at_end": True,  # easy to forget, but important! (also force-saves best ckpt)
            "logging_first_step": True,  # good for plotting/sanity
            "log_level": "info",  # enable info-level logging for models by default
            "report_to": "none",  # disable by default, and enable at runtime if needed
            # we also need to replace some defaults that CANNOT be serialized (factories)
            "lr_scheduler_kwargs": {},  # same behavior as original default
            "include_for_metrics": [],  # same behavior as original default
        },
    )
    """Configuration parameters for the HuggingFace Trainer."""

    GenerationConfig = pyine.utils.pydantic.model_from_callable(
        fn=transformers.GenerationConfig,
        name="GenerationConfig",
        model_config=pydantic.ConfigDict(frozen=True, extra="forbid"),
        default_overrides={
            "max_new_tokens": 128,
        },
    )
    """Configuration parameters for the HuggingFace text generation pipeline."""


default_ignore_index: int = -100  # this is an extremely-commonly-used default in pytorch/huggingface
"""The default index to ignore when computing the loss on example tokens."""


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
    """labels mirror input ids but ignore loss on prompt and padding positions."""
    prompt_len: list[int]
    """Length of the prompt (in number of tokens), prior to padding."""
    input_len: list[int]
    """Length of the full example (prompt + potential response), prior to padding."""


type ConversationMessage = dict[str, typing.Any]
"""Alias for messages in conversation datasets."""

type ConversationHistory = list[ConversationMessage]
"""Alias for ordered conversation histories."""


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
) -> hf_datasets.Dataset:
    """Flattens conversations into (history -> assistant) training examples.

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
    """
    forward_all_fields, keep_extra_fields_list = _parse_keep_extra_fields_config(keep_extra_fields)

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

    # determine which columns to remove (messages_key and any fields we're not keeping)
    if forward_all_fields:
        # keep all columns except messages_key (they'll be handled by _split_to_examples)
        columns_to_remove = [messages_key]
    elif keep_extra_fields_list:
        # keep only specified fields plus input_ids/prompt_len
        columns_to_remove = [
            col for col in convo_ds.column_names if col not in keep_extra_fields_list and col != messages_key
        ]
        columns_to_remove.append(messages_key)
    else:
        # remove all original columns
        columns_to_remove = convo_ds.column_names

    # TODO: if this map becomes a bottleneck, tune num_proc, use cache
    # (worse case scenario, we can switch to a streaming/iterable dataset?)
    dataset: hf_datasets.Dataset = convo_ds.map(  # type: ignore[reportUnknownMemberType]
        _split_to_examples,
        batched=True,  # explode list outputs into individual rows automatically
        remove_columns=columns_to_remove,
        num_proc=num_proc,  # parallelize if possible
        desc="preparing examples",
    )
    return dataset


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
        extra_fields[key] = [data[key] for data in batch_data]
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
        ignore_index: Label value used to mask prompt and padding positions in the returned
            ``labels`` tensor.
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
        ignore_index: int = default_ignore_index,
    ) -> None:
        """Initializes the collator."""
        self.max_length = max_length
        self.always_pad_to_max_length = always_pad_to_max_length
        assert pad_to_multiple_of is None or pad_to_multiple_of > 0, "pad_to_multiple_of must be > 0"
        self.pad_to_multiple_of = int(pad_to_multiple_of) if pad_to_multiple_of else None
        self._forward_all_fields, self._keep_extra_fields = _parse_keep_extra_fields_config(keep_extra_fields)
        self.ignore_index = ignore_index
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

    def __call__(
        self,
        to_batch: list[dict[str, typing.Any]],
    ) -> ExampleBatchTensors:
        """Turns a list of examples into a batch of tensors."""
        expected_field_names = ["input_ids", "prompt_len"]
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
            assert all(k in data for k in expected_field_names), "missing expected key in data dict"
            # truncate if necessary according to tokenizer's truncation_side
            orig_ids = typing.cast("list[int]", data["input_ids"])
            prompt_len = int(data["prompt_len"])
            response_len = len(orig_ids) - prompt_len
            if len(orig_ids) > effective_max_length:
                overflow = len(orig_ids) - effective_max_length
                if self._truncation_side == "left":
                    if overflow >= prompt_len:
                        raise ValueError(
                            "found an example with a max length overflow g.e. to the prompt length "
                            f"({overflow=}, {prompt_len=}, {response_len=}, and {effective_max_length=})"
                        )
                    token_ids = orig_ids[-effective_max_length:]
                    prompt_len = prompt_len - overflow
                else:  # truncation_side == "right"
                    if overflow >= response_len:
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
        batch_dict: dict[str, typing.Any] = {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "prompt_len": prompt_lengths,
            "input_len": input_lengths,
        }
        # carry over targeted (or all) metadata fields with the input batch order
        extra_fields = _collect_extra_fields_from_batch(
            to_batch,
            self._forward_all_fields,
            self._keep_extra_fields,
            expected_field_names,
        )
        batch_dict.update(extra_fields)
        return typing.cast("ExampleBatchTensors", batch_dict)


def infer_effective_max_seq_len(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
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
        raise ValueError(f"can't infer effective max_seq_len for {type(model)} and {type(tokenizer)}")
    return min(candidates)  # keep the minimum as a conservative choice


def _get_base_pretrained_model(obj: typing.Any) -> transformers.PreTrainedModel | None:
    """Tries to recover the underlying `transformers.PreTrainedModel` from common wrappers.

    Should be able to peek through DDP/FSDP/DeepSpeed/Accelerate, PEFT, TRL, pipelines, etc.; if
    the method cannot find a `PreTrainedModel` in the object, returns None.
    """
    seen: set[int] = set()
    cur = obj
    # first, check the simplest case, i.e. if the object itself is what we want
    if isinstance(obj, transformers.PreTrainedModel):
        return obj
    # pipelines have a `.model` that *is* a PreTrainedModel
    if hasattr(cur, "model") and isinstance(cur.model, transformers.PreTrainedModel):
        return cur.model
    while id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, transformers.PreTrainedModel):
            return cur
        # PEFT: try method first (present across versions), then attribute
        if hasattr(cur, "get_base_model"):
            cand = cur.get_base_model()
            if isinstance(cand, transformers.PreTrainedModel):
                return cand
            cur = cand
            continue
        if hasattr(cur, "base_model"):
            cand = cur.base_model
            if isinstance(cand, transformers.PreTrainedModel):
                return cand
            cur = cand
            continue
        # TRL wrapper exposes `.pretrained_model`
        if hasattr(cur, "pretrained_model"):
            cand = cur.pretrained_model
            if isinstance(cand, transformers.PreTrainedModel):
                return cand
            cur = cand
            continue
        # generic wrapper stacks
        if hasattr(cur, "module"):  # nn.DataParallel / DDP / FSDP / DeepSpeed
            cur = cur.module
            continue
        if hasattr(cur, "_orig_mod"):  # accelerate
            cur = cur._orig_mod
            continue
        # be careful with `.model`: unwrap only if it looks like a top-level HF model
        if hasattr(cur, "model"):
            cand = cur.model
            if isinstance(cand, transformers.PreTrainedModel):
                return cand
        break
    return None


def is_hf_model(obj: typing.Any) -> bool:
    """Helper that checks whether a given object is or contains a Hugging Face Transformers model."""
    return _get_base_pretrained_model(obj) is not None


def is_hf_tokenizer(obj: typing.Any) -> bool:
    """Helper that checks whether a given object is a Transformers tokenizer (slow or fast)."""
    base = getattr(transformers, "PreTrainedTokenizerBase", None)
    slow = getattr(transformers, "PreTrainedTokenizer", None)
    fast = getattr(transformers, "PreTrainedTokenizerFast", None)
    candidates = [c for c in (base, slow, fast) if c is not None]
    return any(isinstance(obj, c) for c in candidates)


def supports_text_generation(obj: typing.Any) -> bool:
    """Returns whether the given object supports text generation (`.generate(...)`).

    Prefers verification using the model's own `can_generate()` when available, and falls back to
    capability heuristics on the recovered base model.
    """
    base = _get_base_pretrained_model(obj) or obj
    # preferred signal in modern versions of hf transformers: `PreTrainedModel.can_generate()`
    can_generate = getattr(base, "can_generate", None)
    if callable(can_generate):
        try:
            return bool(can_generate())
        except TypeError:
            return bool(can_generate)
    # fallbacks for older / wrapper cases
    has_generate = callable(getattr(base, "generate", None))
    # looks like a generative architecture: encoder-decoder or has an LM head / output embeddings
    cfg = getattr(base, "config", None)
    is_encdec = bool(getattr(cfg, "is_encoder_decoder", False))
    looks_like_lm = False
    # direct LM heads in many decoder-only models:
    for attr in ("lm_head", "embed_out", "score"):
        if hasattr(base, attr):
            looks_like_lm = True
            break
    # robust check via output embeddings accessor (returns None on non-LM heads)
    get_out = getattr(base, "get_output_embeddings", None)
    if callable(get_out):
        looks_like_lm = looks_like_lm or (get_out() is not None)
    # some nonstandard wrappers implement `prepare_inputs_for_generation` without LM head exposure
    has_pifg = callable(getattr(base, "prepare_inputs_for_generation", None))
    return bool(has_generate and (is_encdec or looks_like_lm or has_pifg))


type GenerationConfigInput = transformers.GenerationConfig | GenerationConfig | collections.abc.Mapping[str, typing.Any]
"""Accepted input types for ``resolve_hf_generation_config``."""


def resolve_hf_generation_config(
    config: GenerationConfigInput,
) -> transformers.GenerationConfig:
    """Return a HuggingFace generation config from supported configuration inputs."""
    if isinstance(config, transformers.GenerationConfig):
        return config
    if isinstance(config, GenerationConfig):
        return transformers.GenerationConfig(**config.model_dump())
    return transformers.GenerationConfig(**dict(config))


def run_text_generation(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    dataloader: torch.utils.data.DataLoader[dict[str, typing.Any]],
    gen_config: transformers.GenerationConfig,
    forward_batch_keys: list[str] | bool | None = None,
    generated_text_key: str = "prediction",
    generated_tokens_key: str = "generated_tokens",
    verbose: bool = False,
) -> list[dict[str, typing.Any]]:
    """Generates continuations for each sample in a dataloader and returns decoded texts.

    Iterates over dict batches containing ``input_ids``, ``attention_mask``, and ``input_len``,
    calls ``model.generate`` under ``no_grad`` (and CUDA autocast when appropriate), and decodes
    only the tokens generated after each sample's prompt length. Optionally forwards selected (or
    all) batch metadata keys alongside the decoded text. A progress bar is shown when ``verbose``
    is True.

    Args:
        model: Pretrained causal language model used to generate continuations. It should already
            be on the desired device and in evaluation mode.
        tokenizer: Tokenizer used to decode the generated token ids into text.
        dataloader: Iterable of batches. Each batch must be a dict with keys
            ``"input_ids"``, ``"attention_mask"``, and ``"input_len"``; lengths in ``input_len``
            are used to strip the prompt from the decoded sequences.
        gen_config: Generation settings already converted into a ``transformers.GenerationConfig``
            object for calls to ``model.generate``.
        forward_batch_keys: Extra batch keys to copy into each output item. If a list, only those
            keys are forwarded. If True, forwards all keys present in the input batch. If None or
            empty, no additional keys are forwarded.
        generated_text_key: Key name used to store the decoded text in each output dictionary.
        generated_tokens_key: Key name used to store the generated token ids in each output dict.
        verbose: If True, displays a progress bar during generation.

    Returns:
        A list of dictionaries, one per input example. Each dict contains the decoded text under
        ``generated_text_key`` and, if requested, any forwarded metadata keys with values aligned
        to the batch order.
    """
    results: list[dict[str, typing.Any]] = []
    device = next(model.parameters()).device
    want_amp = device.type == "cuda" and model.dtype in (torch.bfloat16, torch.float16)
    amp_context: contextlib.AbstractContextManager[typing.Any] = (
        torch.autocast(device_type="cuda", dtype=model.dtype) if want_amp else contextlib.nullcontext()
    )
    expected_field_names = ["input_ids", "attention_mask", "input_len"]
    forward_all_keys = forward_batch_keys is True
    selected_forward_keys: list[str] = list(forward_batch_keys) if isinstance(forward_batch_keys, list) else []
    generate_fn = typing.cast("typing.Callable[..., typing.Any]", model.generate)
    decode_fn = typing.cast("typing.Callable[..., str]", tokenizer.decode)  # type: ignore[reportUnknownMemberType]
    with torch.no_grad():
        prog_bar = tqdm.tqdm(dataloader, desc="generating predictions", smoothing=0.1, disable=not verbose)
        for batch in prog_bar:
            assert isinstance(batch, dict), f"unexpected batch type: {type(batch)}"
            for field_name in expected_field_names:
                if field_name not in batch:
                    raise ValueError(f"batch dictionary is missing expected field: {field_name}")
            input_ids = typing.cast("torch.Tensor", batch["input_ids"]).to(device, non_blocking=True)
            attn_mask = typing.cast("torch.Tensor", batch["attention_mask"]).to(device, non_blocking=True)
            input_len = typing.cast("list[int]", batch["input_len"])
            with amp_context:
                generation_result = generate_fn(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    generation_config=gen_config,
                    return_dict_in_generate=True,
                )
            # note: generation_result.sequences includes the prompt + newly generated tokens
            sequences = typing.cast("torch.Tensor", generation_result.sequences)
            generated_output = sequences.to("cpu")  # [B, prompt+new]
            assert generated_output.ndim == 2 and generated_output.shape[0] == len(input_len)
            for sample_idx in range(generated_output.shape[0]):
                sample_result = generated_output[sample_idx]
                new_tokens_ids = sample_result[input_len[sample_idx] :]
                new_text = decode_fn(
                    new_tokens_ids,
                    skip_special_tokens=True,
                    # clean_up_tokenization_spaces=False,
                )
                assert isinstance(new_text, str), f"unexpected decoder output type: {type(new_text)}"
                curr_output: dict[str, typing.Any] = {
                    generated_tokens_key: new_tokens_ids,
                    generated_text_key: new_text,
                }
                # carry over targeted (or all) metadata fields with the input batch order
                if forward_all_keys or selected_forward_keys:
                    curr_target_keys: list[str]
                    if forward_all_keys:
                        batch_keys = typing.cast("typing.Iterable[str]", batch.keys())
                        curr_target_keys = list(batch_keys)
                    else:
                        curr_target_keys = selected_forward_keys
                    assert generated_tokens_key not in curr_target_keys
                    assert generated_text_key not in curr_target_keys
                    metadata: dict[str, typing.Any] = {k: batch[k][sample_idx] for k in curr_target_keys}
                    curr_output.update(metadata)
                results.append(curr_output)
    return results
