import collections.abc
import contextlib
import logging
import typing

import torch
import tqdm
import transformers

import pyine.utils.transformers.configs

logger = logging.getLogger(__name__)

__all__ = ["GenerationConfigInput", "resolve_hf_generation_config", "run_text_generation"]

type GenerationConfigInput = (
    transformers.GenerationConfig
    | pyine.utils.transformers.configs.GenerationConfig
    | collections.abc.Mapping[str, typing.Any]
)
"""Accepted input types for ``resolve_hf_generation_config``."""


def resolve_hf_generation_config(
    config: GenerationConfigInput,
) -> transformers.GenerationConfig:
    """Return a HuggingFace generation config from supported configuration inputs."""
    if isinstance(config, transformers.GenerationConfig):
        return config
    if isinstance(config, pyine.utils.transformers.configs.GenerationConfig):
        return transformers.GenerationConfig(**config.model_dump())
    return transformers.GenerationConfig(**dict(config))


def _get_generated_text(
    output_ids: torch.Tensor,  # 1-d array of output token ids produced by the model (combines prompt + generation)
    input_ids: torch.Tensor,  # likely includes padding, so input (prompt) len might be smaller than this tensor's len
    input_len: int,  # length of the input prompt, without counting padding tokens
    tokenizer: transformers.PreTrainedTokenizer,  # to figure out padding side and token ids
) -> tuple[str, torch.Tensor]:  # tuple of (decoded text, generated tokens)
    """Helper that returns decoded text and generated tokens given output and input token ids tensors."""
    assert output_ids.ndim == 1 and input_ids.ndim == 1, "unexpected tensor dimensionality"
    assert input_len <= len(input_ids), "unexpected input (prompt) length greater than input tensor length??"
    # extract the non-padded portion of input_ids based on padding side
    padding_side = tokenizer.padding_side
    if padding_side == "left":
        assert (output_ids[: len(input_ids)] == input_ids).all().item(), (
            "unexpected output ids not overlapping w/ input"
        )
        generated_ids = output_ids[len(input_ids) :]
    else:
        assert padding_side == "right", f"unexpected padding side: {padding_side}"
        raise RuntimeError("text generation should always occur with left-side padding for optimal performance")
    # Compare the prompt portion of output with the non-padded input
    decode_fn = typing.cast("typing.Callable[..., str]", tokenizer.decode)  # type: ignore[reportUnknownMemberType]
    generated_text = decode_fn(
        generated_ids,
        skip_special_tokens=True,
        # clean_up_tokenization_spaces=False,
    )
    assert isinstance(generated_text, str), f"unexpected decoder output type: {type(generated_text)}"
    return generated_text, generated_ids


def run_text_generation(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    dataloader: torch.utils.data.DataLoader[dict[str, typing.Any]],
    gen_config: transformers.GenerationConfig,
    forward_batch_keys: list[str] | bool | None = None,
    generated_text_key: str = "prediction",
    generated_tokens_key: str = "generated_tokens",
    attempt_index_key: str = "attempt_index",
    verbose: bool = False,
) -> list[dict[str, typing.Any]]:
    """Generates continuations for each sample in a dataloader and returns decoded texts.

    Iterates over dict batches containing ``input_ids``, ``attention_mask``, and ``input_len``,
    calls ``model.generate`` under ``no_grad`` (and CUDA autocast when appropriate), and decodes
    only the tokens generated after each sample's prompt length. Optionally forwards selected (or
    all) batch metadata keys alongside the decoded text. A progress bar is shown when ``verbose``
    is True.

    When ``gen_config.num_return_sequences`` is K > 1, each input produces K output dicts (one
    per sequence), for a total of B * K results per batch. Output ordering follows HF convention:
    all K sequences for input 0, then all K for input 1, etc. Each output dict always includes
    an ``attempt_index_key`` entry (0-based index within the K sequences for that input); note
    that this reflects HF output order, not necessarily independent sampling attempts (e.g. with
    beam search the sequences are ordered by score).

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
            empty, no additional keys are forwarded. Must not include ``generated_text_key``,
            ``generated_tokens_key``, or ``attempt_index_key`` (raises ``ValueError``).
        generated_text_key: Key name used to store the decoded text in each output dictionary.
        generated_tokens_key: Key name used to store the generated token ids in each output dict.
        attempt_index_key: Key name used to store the sequence index (0-based) in each output dict.
        verbose: If True, displays a progress bar during generation.

    Returns:
        A list of B * K dictionaries (B = total inputs, K = ``num_return_sequences``). Each dict
        contains the decoded text under ``generated_text_key``, generated token ids under
        ``generated_tokens_key``, the sequence index under ``attempt_index_key``, and any
        forwarded metadata keys with values aligned to the input (not sequence) index.
    """
    # validate output key uniqueness upfront
    output_keys = {generated_text_key, generated_tokens_key, attempt_index_key}
    if len(output_keys) != 3:
        raise ValueError(
            f"output keys must be distinct: generated_text_key={generated_text_key!r}, "
            f"generated_tokens_key={generated_tokens_key!r}, attempt_index_key={attempt_index_key!r}"
        )
    results: list[dict[str, typing.Any]] = []
    device = next(model.parameters()).device
    want_amp = device.type == "cuda" and model.dtype in (torch.bfloat16, torch.float16)
    amp_context: contextlib.AbstractContextManager[typing.Any] = (
        torch.autocast(device_type="cuda", dtype=model.dtype) if want_amp else contextlib.nullcontext()
    )
    expected_field_names = ["input_ids", "attention_mask", "input_len"]
    forward_all_keys = forward_batch_keys is True
    selected_forward_keys: list[str] = list(forward_batch_keys) if isinstance(forward_batch_keys, list) else []
    if selected_forward_keys:
        collisions = output_keys & set(selected_forward_keys)
        if collisions:
            raise ValueError(f"forward_batch_keys contains reserved output key(s): {collisions}")
    generate_fn = typing.cast("typing.Callable[..., typing.Any]", model.generate)
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
            generated_output = typing.cast("torch.Tensor", generation_result.sequences)  # [B*K, prompt+new]
            # with num_return_sequences=K, HF outputs [B*K, seq_len] ordered as
            # [input_0_attempt_0, input_0_attempt_1, ..., input_1_attempt_0, ...]
            # this ordering is guaranteed by HF's GenerationMixin.generate() which loops over
            # beams/sequences per input before moving to the next input. the existing
            # _get_generated_text assertion (which verifies the prompt prefix matches the input)
            # will fail fast if this assumption is ever violated by a future HF version.
            num_return_sequences = getattr(gen_config, "num_return_sequences", 1)
            if num_return_sequences is None:
                num_return_sequences = 1
            if num_return_sequences < 1:
                raise ValueError(f"num_return_sequences must be >= 1, got {num_return_sequences}")
            batch_size = len(input_len)
            expected_rows = batch_size * num_return_sequences
            if generated_output.ndim != 2 or generated_output.shape[0] != expected_rows:
                raise ValueError(
                    f"expected generated output shape [{expected_rows}, ...], got {list(generated_output.shape)}"
                )
            for batch_idx in range(batch_size):
                for attempt_idx in range(num_return_sequences):
                    seq_idx = batch_idx * num_return_sequences + attempt_idx
                    generated_text, generated_ids = _get_generated_text(
                        output_ids=generated_output[seq_idx],
                        input_ids=input_ids[batch_idx],
                        input_len=input_len[batch_idx],
                        tokenizer=tokenizer,
                    )
                    curr_output: dict[str, typing.Any] = {
                        generated_tokens_key: generated_ids.to("cpu", non_blocking=True),
                        generated_text_key: generated_text,
                        attempt_index_key: attempt_idx,
                    }
                    # carry over targeted (or all) metadata fields using batch_idx (not seq_idx)
                    if forward_all_keys or selected_forward_keys:
                        curr_target_keys: list[str]
                        if forward_all_keys:
                            batch_keys = typing.cast("typing.Iterable[str]", batch.keys())
                            curr_target_keys = list(batch_keys)
                            collisions = output_keys & set(curr_target_keys)
                            if collisions:
                                raise ValueError(f"batch keys collide with reserved output key(s): {collisions}")
                        else:
                            curr_target_keys = selected_forward_keys  # already validated upfront
                        metadata: dict[str, typing.Any] = {k: batch[k][batch_idx] for k in curr_target_keys}
                        curr_output.update(metadata)
                    results.append(curr_output)
    return results
