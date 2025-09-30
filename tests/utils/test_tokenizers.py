import tiktoken
import torch
import transformers

import pyine.utils.tokenizers as toknz


def test_get_hf_tokenizer():
    model_name = "gpt2"
    tokenizer = toknz.get_hf_tokenizer(model_name)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizer)
    assert tokenizer.name_or_path == model_name
    test_str = "Hello, world!"
    output = tokenizer(test_str, return_tensors="pt")
    assert "input_ids" in output and isinstance(output["input_ids"], torch.Tensor)
    assert "attention_mask" in output and isinstance(output["attention_mask"], torch.Tensor)
    # also check if tokenizer options are respected
    tokenizer = toknz.get_hf_tokenizer(
        model_name,
        set_padding_to_eos_if_needed=True,
        override_padding_to_right_side=True,
    )
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.padding_side == "right"


def test_get_openai_tokenizer():
    model_name = "gpt-4o"
    tokenizer = toknz.get_openai_tokenizer(model_name)
    assert isinstance(tokenizer, tiktoken.Encoding)
    test_str = "Hello, world!"
    output = tokenizer.encode(test_str)
    assert isinstance(output, list) and len(output) > 0
    assert all([0 <= tid < tokenizer.n_vocab for tid in output])


def test_unknown_openai_tokenizer():
    model_name = "gpt-unknown"
    tokenizer = toknz.get_openai_tokenizer(model_name, raise_if_not_found=False)
    assert tokenizer is not None
    test_str = "Hello, world!"
    output = tokenizer.encode(test_str)
    assert isinstance(output, list) and len(output) > 0
