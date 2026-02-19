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


def test_run_text_generation_multi_return_sequences() -> None:
    """Test run_text_generation with num_return_sequences=K using the dummy model."""
    num_return_sequences = 3
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
    gen_config = transformers.GenerationConfig(
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=0.9,
    )
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
    batch_size = 2
    assert len(results) == batch_size * num_return_sequences
    # verify ordering: results are grouped by input, then by attempt
    for batch_idx in range(batch_size):
        for attempt_idx in range(num_return_sequences):
            result_idx = batch_idx * num_return_sequences + attempt_idx
            result = results[result_idx]
            assert result["attempt_index"] == attempt_idx
            # metadata must map to the batch input, not the sequence index
            expected_meta = ["first", "second"][batch_idx]
            assert result["meta"] == expected_meta
            # generated tokens should be unique per (batch_idx, attempt_idx)
            offset = batch_idx + attempt_idx * 10
            expected_tokens = [ord("x") + offset, ord("y") + offset]
            assert result["text"] == tokenizer.decode(expected_tokens)


class TestRunTextGenerationValidation:
    """Tests for input validation in run_text_generation."""

    def test_duplicate_output_keys_raises(self) -> None:
        tokenizer = tests.utils.transformers.utils.SimpleTokenizer(padding_side="left")
        model = tests.utils.transformers.utils.DummyGenerationModel()
        gen_config = utils_transformers.GenerationConfig()
        with pytest.raises(ValueError, match="output keys must be distinct"):
            utils_transformers.run_text_generation(
                model=model,
                tokenizer=tokenizer,
                dataloader=[],
                gen_config=gen_config,
                generated_text_key="prediction",
                attempt_index_key="prediction",  # collides with generated_text_key
            )

    def test_forward_batch_keys_collision_raises(self) -> None:
        tokenizer = tests.utils.transformers.utils.SimpleTokenizer(padding_side="left")
        model = tests.utils.transformers.utils.DummyGenerationModel()
        gen_config = utils_transformers.GenerationConfig()
        with pytest.raises(ValueError, match="forward_batch_keys contains reserved output key"):
            utils_transformers.run_text_generation(
                model=model,
                tokenizer=tokenizer,
                dataloader=[],
                gen_config=gen_config,
                forward_batch_keys=["meta", "prediction"],  # "prediction" is the default generated_text_key
            )

    def test_num_return_sequences_zero_raises(self) -> None:
        tokenizer = tests.utils.transformers.utils.SimpleTokenizer(padding_side="left")
        model = tests.utils.transformers.utils.DummyGenerationModel()
        gen_config = transformers.GenerationConfig()
        gen_config.num_return_sequences = 0  # bypass HF's own validation
        dataloader = [
            {
                "input_ids": torch.tensor([[0, ord("A")]], dtype=torch.long),
                "attention_mask": torch.tensor([[0, 1]], dtype=torch.long),
                "input_len": [1],
            }
        ]
        with pytest.raises(ValueError, match="num_return_sequences must be >= 1"):
            utils_transformers.run_text_generation(
                model=model,
                tokenizer=tokenizer,
                dataloader=dataloader,
                gen_config=gen_config,
            )


@pytest.mark.slow
@pytest.mark.integration
def test_run_text_generation_multi_return_sequences_real_hf_model() -> None:
    """Integration test: verify HF generate() output ordering with num_return_sequences > 1.

    Uses a real (tiny) GPT-2 model to confirm that HuggingFace groups K outputs per input
    in the expected [input_0_attempt_0, input_0_attempt_1, ..., input_1_attempt_0, ...] order.
    The _get_generated_text assertion inside run_text_generation validates this: it checks that
    the prompt prefix of each output matches the corresponding input, and would fail if HF
    reordered outputs.
    """
    model_id = "openai-community/gpt2"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    assert tokenizer.pad_token is None
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    num_return_sequences = 3
    prompts = ["The quick brown fox", "Hello world, this is a test of"]
    # tokenize with left-padding so all sequences align on the right
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_lens = [encoded["attention_mask"][idx].sum().item() for idx in range(len(prompts))]
    dataloader = [
        {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "input_len": input_lens,
            "prompt_text": prompts,
        }
    ]
    gen_config = transformers.GenerationConfig(
        max_new_tokens=10,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=0.9,
    )
    results = utils_transformers.run_text_generation(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        gen_config=gen_config,
        forward_batch_keys=["prompt_text"],
        verbose=False,
    )
    batch_size = len(prompts)
    assert len(results) == batch_size * num_return_sequences
    for batch_idx in range(batch_size):
        for attempt_idx in range(num_return_sequences):
            result_idx = batch_idx * num_return_sequences + attempt_idx
            result = results[result_idx]
            assert result["attempt_index"] == attempt_idx
            assert result["prompt_text"] == prompts[batch_idx]  # metadata from batch_idx, not seq_idx
            assert isinstance(result["prediction"], str)
            assert len(result["prediction"]) > 0  # model generated something


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
