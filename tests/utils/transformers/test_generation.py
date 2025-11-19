import datasets
import pytest
import torch
import transformers

import pyine.utils.transformers as utils_transformers
import tests.utils.transformers.utils


def test_run_text_generation_decodes_predictions(
    simple_tokenizer: tests.utils.transformers.utils.SimpleTokenizer,
) -> None:
    simple_tokenizer.padding_side = "left"
    model = tests.utils.transformers.utils.DummyGenerationModel()
    dataloader = [
        {
            "input_ids": torch.tensor(
                [[0, ord("A"), ord("B"), ord("C")], [0, 0, ord("D"), ord("E")]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[0, 1, 1, 1], [0, 0, 1, 1]],
                dtype=torch.long,
            ),
            "input_len": [3, 2],
            "meta": ["first", "second"],
        }
    ]
    gen_config = utils_transformers.GenerationConfig()
    results = utils_transformers.run_text_generation(
        model=model,
        tokenizer=simple_tokenizer,
        dataloader=dataloader,
        gen_config=gen_config,
        forward_batch_keys=["meta"],
        generated_text_key="text",
        generated_tokens_key="tokens",
        verbose=False,
    )
    assert len(results) == 2
    assert results[0]["text"] == simple_tokenizer.decode([ord("x"), ord("y")])
    assert results[0]["meta"] == "first"
    second_tokens = results[1]["tokens"]
    assert int(second_tokens[0]) == ord("y")
    assert int(second_tokens[1]) == ord("z")
    assert results[1]["text"] == simple_tokenizer.decode(second_tokens)
    assert results[1]["meta"] == "second"


def test_run_text_generation_with_left_padding() -> None:
    """Test run_text_generation with left-padded inputs (common for decoder-only models)."""
    tokenizer = tests.utils.transformers.utils.SimpleTokenizer(padding_side="left")
    model = tests.utils.transformers.utils.DummyGenerationModel()
    dataloader = [
        {
            "input_ids": torch.tensor(
                [[0, ord("A"), ord("B"), ord("C")], [0, 0, ord("D"), ord("E")]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[0, 1, 1, 1], [0, 0, 1, 1]],
                dtype=torch.long,
            ),
            "input_len": [3, 2],
            "meta": ["first", "second"],
        }
    ]
    gen_config = utils_transformers.GenerationConfig()
    results = utils_transformers.run_text_generation(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        gen_config=gen_config,
        forward_batch_keys=["meta"],
        generated_text_key="text",
        generated_tokens_key="tokens",
        verbose=False,
    )
    assert len(results) == 2
    assert results[0]["text"] == tokenizer.decode([ord("x"), ord("y")])
    assert results[0]["meta"] == "first"
    second_tokens = results[1]["tokens"]
    assert int(second_tokens[0]) == ord("y")
    assert int(second_tokens[1]) == ord("z")
    assert results[1]["text"] == tokenizer.decode(second_tokens)
    assert results[1]["meta"] == "second"


@pytest.mark.slow
def test_run_text_generation_with_real_model(
    tiny_gpt2_model: transformers.GPT2LMHeadModel,
) -> None:
    """Test prepare_examples_from_conversations with a real tokenizer."""
    # Use a real GPT-2 tokenizer and add a chat template
    model_id = "openai-community/gpt2"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    assert tokenizer.pad_token is None
    tokenizer.pad_token = tokenizer.eos_token
    # GPT-2 doesn't have a chat template, so add a simple one for testing
    tokenizer.chat_template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"
    conversations = [
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "something": "hello1",
        },
        {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ],
            "something": "hello2",
        },
    ]
    convo_ds = datasets.Dataset.from_list(conversations)
    # Test that prepare_examples_from_conversations works with a real tokenizer
    examples_ds = utils_transformers.prepare_examples_from_conversations(
        convo_ds=convo_ds,
        tokenizer=tokenizer,
        max_seq_len=128,
        num_proc=1,
        keep_extra_fields=True,
    )
    # Should produce 2 examples (one per assistant turn)
    assert len(examples_ds) == 2
    for example in examples_ds:
        assert "input_ids" in example
        assert "prompt_len" in example
        assert isinstance(example["input_ids"], list)
        assert isinstance(example["prompt_len"], int)
        assert len(example["input_ids"]) > 0
        assert example["prompt_len"] > 0
        # Verify extra fields were preserved
        assert "something" in example
