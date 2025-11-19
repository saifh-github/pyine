import pytest
import transformers

import tests.utils.transformers.utils


@pytest.fixture()
def simple_tokenizer() -> tests.utils.transformers.utils.SimpleTokenizer:
    return tests.utils.transformers.utils.SimpleTokenizer()


@pytest.fixture()
def tiny_gpt2_model() -> transformers.GPT2LMHeadModel:
    config = transformers.GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=32,
        n_positions=16,
        vocab_size=32,
    )
    return transformers.GPT2LMHeadModel(config)
