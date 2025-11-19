import types

import pytest
import torch
import transformers

import pyine.utils.transformers as utils_transformers
import pyine.utils.transformers.models
import tests.utils.transformers.utils


def test_infer_effective_max_seq_len_uses_minimum_candidate() -> None:
    tokenizer = types.SimpleNamespace(model_max_length=4096)
    config = transformers.GPT2Config(
        n_positions=2048,
        n_ctx=2048,
        n_layer=1,
        n_head=1,
        n_embd=32,
    )
    config.sliding_window = 1024
    result = utils_transformers.infer_effective_max_seq_len(model=config, tokenizer=tokenizer)
    assert result == 1024


def test_infer_effective_max_seq_len_raises_when_unknown() -> None:
    tokenizer = types.SimpleNamespace(model_max_length=None)
    config = transformers.GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=32,
    )
    config.max_position_embeddings = None
    config.n_positions = None
    config.max_seq_len = None
    config.sliding_window = None
    with pytest.raises(ValueError):
        utils_transformers.infer_effective_max_seq_len(model=config, tokenizer=tokenizer)


def test_get_base_pretrained_model_unwraps_module(
    tiny_gpt2_model: transformers.GPT2LMHeadModel,
) -> None:
    class Wrapper(torch.nn.Module):
        def __init__(
            self,
            module: torch.nn.Module,
        ) -> None:
            super().__init__()
            self.module = module

    wrapper = Wrapper(tiny_gpt2_model)
    base = pyine.utils.transformers.models._get_base_pretrained_model(wrapper)
    assert base is tiny_gpt2_model
    assert utils_transformers.is_hf_model(wrapper)
    assert not utils_transformers.is_hf_model(object())


def test_is_hf_tokenizer_detects_transformers_tokenizer() -> None:
    tokenizer = tests.utils.transformers.utils.MinimalPreTrainedTokenizer()
    assert utils_transformers.is_hf_tokenizer(tokenizer)
    assert not utils_transformers.is_hf_tokenizer(object())


def test_supports_text_generation_uses_can_generate(
    tiny_gpt2_model: transformers.GPT2LMHeadModel,
) -> None:
    assert utils_transformers.supports_text_generation(tiny_gpt2_model)
    tiny_gpt2_model.can_generate = lambda: False  # type: ignore[attr-defined]
    assert utils_transformers.supports_text_generation(tiny_gpt2_model) is False


def test_supports_text_generation_heuristics() -> None:
    class EncoderDecoderLike:
        def __init__(self) -> None:
            self.generate = lambda **kwargs: None
            self.config = types.SimpleNamespace(is_encoder_decoder=True)

    class NoGeneration:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(is_encoder_decoder=False)

    assert utils_transformers.supports_text_generation(EncoderDecoderLike()) is True
    assert utils_transformers.supports_text_generation(NoGeneration()) is False
