import functools

import transformers


@functools.wraps(transformers.AutoTokenizer.from_pretrained)
def get_tokenizer(
    pretrained_model_name_or_path: str,
    set_padding_to_eos_if_needed: bool = False,
    override_padding_to_right_side: bool = False,
    **kwargs,
) -> transformers.PreTrainedTokenizerBase:
    """Get a tokenizer from a pretrained model name or path and additional kwargs.

    Also optionally sets the tokenizer's padding token to the EOS token if missing, and sets
    the padding token to the right side if needed.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
    if set_padding_to_eos_if_needed and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if override_padding_to_right_side:
        tokenizer.padding_side = "right"
    return tokenizer
