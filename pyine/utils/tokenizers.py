import functools

import tiktoken
import transformers


def get_hf_tokenizer(
    pretrained_model_name_or_path: str,
    set_padding_to_eos_if_needed: bool = False,
    override_padding_to_right_side: bool = False,
    **kwargs: typing.Any,
) -> transformers.PreTrainedTokenizer:
    """Get a hf tokenizer from a pretrained model name or path and additional kwargs.

    Also optionally sets the tokenizer's padding token to the EOS token if missing, and sets
    the padding token to the right side if needed.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
    if set_padding_to_eos_if_needed and tokenizer.pad_token is None:
        # many causal LMs don't define a PAD token, but the hf Trainer expects one for padding batches
        # (reuse EOS as PAD so padding uses a benign but already-known token id)
        tokenizer.pad_token = tokenizer.eos_token
    if override_padding_to_right_side:
        tokenizer.padding_side = "right"
    return tokenizer


@functools.lru_cache(maxsize=64)
def get_openai_tokenizer(
    model_id: str,
    raise_if_not_found: bool = True,
) -> tiktoken.Encoding:
    """Get a tokenizer for an OpenAI model (based on tiktoken).

    If the model is not recognized by tiktoken, will optionally fall back to a common family,
    or raise an error (default behavior).
    """
    try:
        return tiktoken.encoding_for_model(model_id)
    except KeyError as e:
        if raise_if_not_found:
            raise RuntimeError(f"model '{model_id}' not recognized (may be out-of-date)") from e
        # model not recognized by the current tiktoken version; need to fall back to sensible choice
        model_lower = model_id.lower()
        if model_lower.startswith(("gpt-4o", "o4", "gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini")):
            return tiktoken.get_encoding("o200k_base")
        return tiktoken.get_encoding("cl100k_base")
