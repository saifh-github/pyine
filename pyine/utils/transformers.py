import collections
import contextlib
import typing

import datasets
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


def _build_example_ids_from_conversation_parts(
    tokenizer: transformers.PreTrainedTokenizer,
    history_msgs: list[dict],
    assistant_msg: dict,
    max_seq_len: int | None,
) -> _ExampleData | None:
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
    convo_ds: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_len: int | None,
    num_proc: int,
    messages_key: str = "messages",
) -> datasets.Dataset:
    """Flattens conversations into (history -> assistant) training examples.

    The input dataset should have a column of messages where each row is an entire conversation:
        [{"role": "system"/"user"/"assistant", "content": "..."}, ...]

    We convert each assistant turn into one supervised example whose inputs are:
        [prompt tokens from conversation history] + [assistant response tokens]

    We also keep "prompt_len" so the collator can later mask prompt tokens in labels.
    """

    def _split_to_examples(example: dict) -> dict[str, typing.Any]:
        # called on a single conversation by Dataset.map; must return a dict of column->values
        # (if returned values are lists of equal length, HF "explodes" them into multiple rows)
        messages: list[dict] = example[messages_key]
        out_input_ids: list[list[int]] = []
        out_prompt_len: list[int] = []
        for msg_idx, message in enumerate(messages):
            # only model-authored turns are training targets
            if message.get("role") != "assistant":
                continue
            # history is everything before the currently-targeted assistant message
            history = messages[:msg_idx]
            result = _build_example_ids_from_conversation_parts(tokenizer, history, message, max_seq_len)
            if not result:
                # skip empty/invalid assistant messages
                continue
            # collect one example per assistant turn; HF will create as many rows as we return here
            out_input_ids.append(result["input_ids"])
            out_prompt_len.append(len(result["prompt_ids"]))
        # only return columns needed for training; others are discarded via remove_columns below
        return {"input_ids": out_input_ids, "prompt_len": out_prompt_len}

    # TODO: if this map becomes a bottleneck, add batching w/ fast tokenizer, tune num_proc, use cache
    # (worse case scenario, we can switch to a streaming/iterable dataset?)
    dataset = convo_ds.map(
        _split_to_examples,
        batched=False,  # process one conversation at a time for clarity
        # drop original columns so the resulting dataset only has "input_ids" and "prompt_len"
        remove_columns=convo_ds.column_names,
        num_proc=num_proc,  # parallelize if possible
        desc="preparing examples",
    )
    # mapping with list outputs creates nested rows tied to original row; flatten to simple rows
    dataset = dataset.flatten_indices()
    # keep only rows that actually contain tokenized inputs
    return dataset.filter(lambda row: isinstance(row["input_ids"], list) and len(row["input_ids"]) > 0)


class FixedSizePaddingCollatorWithPromptMask:
    """Pads inputs to the specified max length and build labels masking out prompt tokens.

    The trainer uses the collator to turn a list of examples into fixed-size tensors; this one:
     - pads/truncates sequences to max_length;
     - builds attention_mask (1 for tokens, 0 for padding)
     - builds labels identical to inputs, except:
       - prompt token positions are set to `ignore_index` (ignored by loss)
       - padding positions are set to `ignore_index` (ignored by loss)
    """

    # @@@@@@ TODO: add support for packing? sort & min-pad batches?
    # current approach w/ static max-length might be annoying for datasets w/ high len variance

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int,
        ignore_index: int = default_ignore_index,
    ) -> None:
        """Initializes the collator."""
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.tokenizer = tokenizer
        assert self.tokenizer.pad_token_id is not None, "tokenizer must have a pad token"

    def __call__(
        self,
        features: list[dict[str, typing.Any]],
    ) -> ExampleBatchTensors:
        """Turns a list of examples into a batch of tensors."""
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        attention_masks: list[list[int]] = []
        prompt_lengths: list[int] = []
        input_lengths: list[int] = []
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
            input_len = len(token_ids)
            pad_len = self.max_length - input_len
            # right-pad to a fixed length
            input_ids = token_ids + [self.tokenizer.pad_token_id] * pad_len
            attention_mask = [1] * input_len + [0] * pad_len
            # labels mirror input ids but ignore loss on prompt and padding positions
            labels = input_ids.copy()
            for position in range(min(prompt_len, self.max_length)):
                labels[position] = self.ignore_index
            for position in range(input_len, self.max_length):
                labels[position] = self.ignore_index
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_masks.append(attention_mask)
            prompt_lengths.append(prompt_len)
            input_lengths.append(input_len)
        return ExampleBatchTensors(
            input_ids=torch.tensor(input_ids_list, dtype=torch.long),
            attention_mask=torch.tensor(attention_masks, dtype=torch.long),
            labels=torch.tensor(labels_list, dtype=torch.long),
            prompt_len=prompt_lengths,
            input_len=input_lengths,
        )


class BatchwisePaddingCollator:
    """Pads each batch to its longest sequence, builds masks, and masks labels for prompt/padding.

    This collator takes variable-length tokenized examples that already include an attention_mask
    and produces a batch with right-padding to the longest sequence in the batch. It also constructs
    labels that mirror input_ids but ignore loss on prompt token positions (derived from ``prompt_ids``
    when present, or treated as zero-length prompt otherwise) and on padding positions. Optional
    metadata fields can be forwarded from the input examples to the output batch without alteration.

    Args:
        tokenizer: Tokenizer whose ``pad_token_id`` is used for right-padding.
        max_allowed_length: If set, validates that no example exceeds this length. Raises a
            ValueError if an input is longer than this cap.
        keep_extra_fields: Additional keys to carry over from input examples to the output batch.
            If a list, only those keys are forwarded. If True, forwards all fields present in
            the inputs. If falsy/None, no extra fields are forwarded.
        ignore_index: Label value used to mask prompt and padding positions in the returned
            ``labels`` tensor.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_allowed_length: int | None = None,
        keep_extra_fields: list[str] | bool | None = None,
        ignore_index: int = default_ignore_index,
    ) -> None:
        """Initializes the collator."""
        self.max_allowed_length = max_allowed_length
        self.keep_extra_fields = keep_extra_fields or []
        self.ignore_index = ignore_index
        self.tokenizer = tokenizer
        assert self.tokenizer.pad_token_id is not None, "tokenizer must have a pad token"

    def __call__(
        self,
        to_batch: list[dict[str, typing.Any]],
    ) -> ExampleBatchTensors:
        """Collates the provided elements into a batch dictionary."""
        expected_field_names = ["input_ids", "attention_mask"]
        unexpected_field_names = ["labels"]
        for data in to_batch:
            for expected_field in expected_field_names:
                if expected_field not in data:
                    raise ValueError(f"input to batch is missing expected field: {expected_field}")
            for unexpected_field in unexpected_field_names:
                if unexpected_field in data:
                    raise ValueError(f"input to batch contains unexpected field: {unexpected_field}")
        input_len, prompt_len = [], []
        for data in to_batch:
            curr_input_len = data.get("input_len", len(data["input_ids"]))
            if "prompt_len" in data or "prompt_ids" in data:
                curr_prompt_len = data.get("prompt_len", len(data["prompt_ids"]))
            else:
                curr_prompt_len = curr_input_len
            if curr_prompt_len > curr_input_len:
                raise ValueError(
                    f"found an element with a prompt length ({curr_prompt_len}) "
                    f"larger than input length ({curr_input_len})"
                )
            if self.max_allowed_length is not None and curr_input_len > self.max_allowed_length:
                raise ValueError(
                    f"found an element with an input length ({curr_input_len}) "
                    f"larger than max_allowed_length ({self.max_allowed_length})"
                )
            input_len.append(curr_input_len)
            prompt_len.append(curr_prompt_len)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(data["input_ids"], dtype=torch.long) for data in to_batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attn_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(data["attention_mask"], dtype=torch.long) for data in to_batch],
            batch_first=True,
            padding_value=0,
        )
        # labels mirror input ids but ignore loss on prompt and padding positions
        labels = input_ids.clone()
        for row_idx, (input_l, prompt_l) in enumerate(zip(input_len, prompt_len, strict=False)):
            labels[row_idx, :prompt_l] = self.ignore_index
            labels[row_idx, input_l:] = self.ignore_index
        output = ExampleBatchTensors(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels,
            prompt_len=prompt_len,
            input_len=input_len,
        )
        # carry over targeted (or all) metadata fields with the input batch order
        if self.keep_extra_fields:
            keep_extra_fields = self.keep_extra_fields
            if not isinstance(keep_extra_fields, list):
                assert keep_extra_fields is True
                keep_extra_fields = {key for data in to_batch for key in data}
            metadata = {k: [data[k] for data in to_batch] for k in keep_extra_fields if k not in expected_field_names}
            output.update(metadata)
        return output


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
    seen = set()
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


@torch.no_grad()
def run_text_generation(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    dataloader: torch.utils.data.DataLoader,
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
    results: list[dict] = []
    device = next(model.parameters()).device
    want_amp = device.type == "cuda" and model.dtype in (torch.bfloat16, torch.float16)
    amp_context = torch.autocast(device_type="cuda", dtype=model.dtype) if want_amp else contextlib.nullcontext()
    expected_field_names = ["input_ids", "attention_mask", "input_len"]
    forward_batch_keys = forward_batch_keys or []
    prog_bar = tqdm.tqdm(dataloader, desc="generating predictions", smoothing=0.1, disable=not verbose)
    for batch in prog_bar:
        assert isinstance(batch, dict), f"unexpected batch type: {type(batch)}"
        for field_name in expected_field_names:
            if field_name not in batch:
                raise ValueError(f"batch dictionary is missing expected field: {field_name}")
        input_ids: torch.Tensor = batch["input_ids"].to(device, non_blocking=True)
        attn_mask: torch.Tensor = batch["attention_mask"].to(device, non_blocking=True)
        input_len: list[int] = batch["input_len"]
        with amp_context:
            generation_result = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                generation_config=gen_config,
                return_dict_in_generate=True,
            )
        # note: generation_result.sequences includes the prompt + newly generated tokens
        generated_output = generation_result.sequences.to("cpu")  # [B, prompt+new]
        assert generated_output.ndim == 2 and generated_output.shape[0] == len(input_len)
        for sample_idx, sample_result in enumerate(generated_output):
            new_tokens_ids = sample_result[input_len[sample_idx] :]
            new_text = tokenizer.decode(
                new_tokens_ids,
                skip_special_tokens=True,
                # clean_up_tokenization_spaces=False,
            )
            curr_output = {
                generated_tokens_key: new_tokens_ids,
                generated_text_key: new_text,
            }
            # carry over targeted (or all) metadata fields with the input batch order
            if forward_batch_keys:
                curr_target_keys = forward_batch_keys
                if not isinstance(curr_target_keys, list):
                    assert curr_target_keys is True
                    curr_target_keys = list(batch)  # forward all preexisting fields
                assert generated_tokens_key not in curr_target_keys
                assert generated_text_key not in curr_target_keys
                metadata = {k: batch[k][sample_idx] for k in curr_target_keys}
                curr_output.update(metadata)
            results.append(curr_output)
    return results
