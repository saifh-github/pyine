import pathlib
import types
import typing

import datasets
import peft
import pytest
import torch
import transformers

import pyine.utils.transformers as utils_transformers


def _compute_token_ids(
    tokenizer: "SimpleTokenizer",
    text: str,
) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


class SimpleTokenizer:
    def __init__(
        self,
        padding_side: str = "right",
        truncation_side: str = "left",
    ) -> None:
        self.pad_token_id = 0
        self.model_max_length = 4096
        self.padding_side = padding_side
        self.truncation_side = truncation_side

    def apply_chat_template(
        self,
        conversation: list[dict[str, typing.Any]] | list[list[dict[str, typing.Any]]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs: typing.Any,
    ) -> str | list[str]:
        # support both single conversation and batched conversations
        # if not conversation:
        #     if add_generation_prompt:
        #         return "assistant:"
        #     return ""
        assert tokenize is False, "test fixture code below does not support tokenization"
        is_batch = isinstance(conversation[0], list)
        if is_batch:
            results = []
            for messages in conversation:
                history = "".join(f"{item['role']}:{item['content']}|" for item in messages)
                if add_generation_prompt:
                    history += "assistant:"
                results.append(history)
            return results
        messages = conversation
        history = "".join(f"{item['role']}:{item['content']}|" for item in messages)
        if add_generation_prompt:
            return history + "assistant:"
        return history

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs: typing.Any,
    ) -> dict[str, list[int]]:
        return {
            "input_ids": [ord(character) for character in text],
            "attention_mask": [1] * len(text),
        }

    def decode(
        self,
        token_ids: typing.Iterable[int] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        values = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        characters: list[str] = []
        for value in values:
            if skip_special_tokens and value == self.pad_token_id:
                continue
            characters.append(chr(int(value)))
        return "".join(characters)


class DummyGenerationOutput:
    def __init__(
        self,
        sequences: torch.Tensor,
    ) -> None:
        self.sequences = sequences


class DummyGenerationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.dtype = torch.float32

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: transformers.GenerationConfig,
        return_dict_in_generate: bool,
    ) -> DummyGenerationOutput:
        sequences: list[torch.Tensor] = []
        for idx in range(input_ids.size(0)):
            new_tokens = torch.tensor(
                [ord("x") + idx, ord("y") + idx],
                dtype=torch.long,
                device=input_ids.device,
            )
            sequences.append(torch.cat((input_ids[idx], new_tokens), dim=0))
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            padding_value=0,
            padding_side="right",
        )
        return DummyGenerationOutput(padded)


class MinimalPreTrainedTokenizer(transformers.PreTrainedTokenizerBase):
    def __init__(self) -> None:
        super().__init__()

    def _tokenize(
        self,
        text: str,
        **kwargs: typing.Any,
    ) -> list[str]:
        del text
        del kwargs
        return []

    def _convert_token_to_id_with_added_voc(
        self,
        token: str,
    ) -> int:
        del token
        return 0

    def _convert_id_to_token(
        self,
        index: int,
    ) -> str:
        del index
        return "token"

    def get_vocab(self) -> dict[str, int]:
        return {"token": 0}

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: str | None = None,
    ) -> tuple[typing.Any, ...]:
        del save_directory
        del filename_prefix
        return ()


@pytest.fixture()
def simple_tokenizer() -> SimpleTokenizer:
    return SimpleTokenizer()


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


def test_build_example_ids_without_truncation(
    simple_tokenizer: SimpleTokenizer,
) -> None:
    history_msgs = [
        {"role": "user", "content": "Hello"},
    ]
    assistant_msg = {"role": "assistant", "content": "Answer"}
    result = utils_transformers._build_example_ids_from_conversation_parts(
        tokenizer=simple_tokenizer,
        history_msgs=history_msgs,
        assistant_msg=assistant_msg,
        max_seq_len=None,
    )
    assert result is not None
    prompt_text = simple_tokenizer.apply_chat_template(history_msgs, tokenize=False, add_generation_prompt=True)
    expected_prompt_ids = _compute_token_ids(simple_tokenizer, prompt_text)
    expected_full_ids = _compute_token_ids(simple_tokenizer, prompt_text + assistant_msg["content"])
    assert result["prompt_ids"] == expected_prompt_ids
    assert result["input_ids"] == expected_full_ids


def test_build_example_ids_truncates_prompt_only(
    simple_tokenizer: SimpleTokenizer,
) -> None:
    history_msgs = [
        {"role": "user", "content": "LongPrompt"},
    ]
    assistant_msg = {"role": "assistant", "content": "OK"}
    result = utils_transformers._build_example_ids_from_conversation_parts(
        tokenizer=simple_tokenizer,
        history_msgs=history_msgs,
        assistant_msg=assistant_msg,
        max_seq_len=12,
    )
    assert result is not None
    prompt_text = simple_tokenizer.apply_chat_template(history_msgs, tokenize=False, add_generation_prompt=True)
    full_ids = _compute_token_ids(simple_tokenizer, prompt_text + assistant_msg["content"])
    response_ids = _compute_token_ids(simple_tokenizer, assistant_msg["content"])
    keep_prompt = 12 - len(response_ids)
    expected_prompt_ids = _compute_token_ids(simple_tokenizer, prompt_text)[-keep_prompt:]
    assert result["prompt_ids"] == expected_prompt_ids
    assert result["input_ids"] == expected_prompt_ids + response_ids
    assert result["input_ids"] == full_ids[-12:]


def test_lora_config_to_peft_preserves_runtime_config() -> None:
    lora_config = utils_transformers.LoraConfig.model_validate({})
    peft_config = lora_config.to_peft_config()
    assert isinstance(peft_config, peft.LoraConfig)
    assert isinstance(peft_config.runtime_config, peft.LoraRuntimeConfig)
    assert hasattr(peft_config.runtime_config, "ephemeral_gpu_offload")


def test_prepare_examples_from_conversations_flattens_assistant_turns(
    simple_tokenizer: SimpleTokenizer,
) -> None:
    conversations = [
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "What"},
                {"role": "assistant", "content": "Answer"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Next"},
                {"role": "assistant", "content": ""},
                {"role": "assistant", "content": "Final"},
            ]
        },
    ]
    dataset = datasets.Dataset.from_list(conversations)
    prepared = utils_transformers.prepare_examples_from_conversations(
        convo_ds=dataset,
        tokenizer=simple_tokenizer,
        max_seq_len=None,
        num_proc=0,
    )
    assert len(prepared) == 3
    expected: list[tuple[list[int], int]] = []
    for conversation in conversations:
        messages = conversation["messages"]
        for idx, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            built = utils_transformers._build_example_ids_from_conversation_parts(
                tokenizer=simple_tokenizer,
                history_msgs=messages[:idx],
                assistant_msg=message,
                max_seq_len=None,
            )
            if not built:
                continue
            expected.append((built["input_ids"], len(built["prompt_ids"])))
    actual: list[tuple[list[int], int]] = []
    for sample_idx in range(len(prepared)):
        sample = prepared[sample_idx]
        actual.append((sample["input_ids"], sample["prompt_len"]))
    assert actual == expected


def test_padding_collator_masks_prompt_and_padding(
    simple_tokenizer: SimpleTokenizer,
) -> None:
    collator = utils_transformers.PaddingCollatorWithPromptMask(
        tokenizer=simple_tokenizer,
        max_length=6,
        ignore_index=-123,
    )
    features = [
        {"input_ids": [1, 2, 3], "prompt_len": 2, "prompt_ids": [8, 9]},
        {"input_ids": [4, 5, 6, 7, 8, 9, 10], "prompt_len": 4},
    ]
    batch = collator(features)
    assert batch["input_ids"].shape == (2, 6)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
    assert batch["prompt_len"] == [2, 3]
    assert batch["input_len"] == [3, 6]
    first_labels = batch["labels"][0].tolist()
    second_labels = batch["labels"][1].tolist()
    assert first_labels == [-123, -123, 3, -123, -123, -123]
    assert second_labels[:3] == [-123, -123, -123]
    assert second_labels[3:] == [8, 9, 10]


def test_padding_collator_both_sides_pad() -> None:
    tokenizer = SimpleTokenizer(padding_side="left", truncation_side="left")
    collator = utils_transformers.PaddingCollatorWithPromptMask(
        tokenizer=tokenizer,
        max_length=6,
        ignore_index=-100,
    )
    features = [
        {"input_ids": [1, 2, 3], "prompt_len": 2},
    ]
    batch = collator(features)
    # if we provide a single sequence as input, no padding should occur
    assert batch["input_ids"].shape == (1, 3)
    assert batch["input_ids"].tolist() == [[1, 2, 3]]
    # the real stuff happens when we have more than one sequence...
    features = [
        {"input_ids": [1, 2, 3], "prompt_len": 2},
        {"input_ids": [4, 5, 6, 7], "prompt_len": 1},
    ]
    batch = collator(features)
    assert batch["input_ids"].shape == (2, 4)
    # left padding: [pad, 1, 2, 3] and [4, 5, 6, 7]
    assert batch["input_ids"].tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    # attention mask: padding=0, content=1
    assert batch["attention_mask"].tolist() == [[0, 1, 1, 1], [1, 1, 1, 1]]
    # labels: mask padding and prompt
    # first row: [pad, pad, pad, prompt, prompt, response]
    assert batch["labels"][0].tolist() == [-100, -100, -100, 3]
    # second row: [pad, pad, prompt, response, response, response]
    assert batch["labels"][1].tolist() == [-100, 5, 6, 7]
    # try again, but with a fixed max-pad size (6) on the other side
    tokenizer.padding_side = "right"
    collator2 = utils_transformers.PaddingCollatorWithPromptMask(
        tokenizer,
        max_length=6,
        always_pad_to_max_length=True,
        ignore_index=-100,
    )
    batch = collator2(features)
    assert batch["input_ids"].shape == (2, 6)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]]
    assert batch["labels"][0].tolist() == [-100, -100, 3, -100, -100, -100]
    assert batch["labels"][1].tolist() == [-100, 5, 6, 7, -100, -100]


def test_padding_collator_both_sides_trunc() -> None:
    tokenizer = SimpleTokenizer(padding_side="right", truncation_side="right")
    collator = utils_transformers.PaddingCollatorWithPromptMask(
        tokenizer=tokenizer,
        max_length=5,
        ignore_index=-100,
    )
    features = [
        {"input_ids": [1, 2, 3, 4, 5, 6, 7], "prompt_len": 3},
    ]
    batch = collator(features)
    # right truncation: keep first 5 tokens [1, 2, 3, 4, 5]
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
    # prompt_len remains 3 since we truncated from right
    assert batch["prompt_len"] == [3]
    # labels: mask first 3 (prompt)
    assert batch["labels"][0].tolist() == [-100, -100, -100, 4, 5]
    # try again, but with truncation from the other side
    tokenizer.truncation_side = "left"
    collator2 = utils_transformers.PaddingCollatorWithPromptMask(
        tokenizer,
        max_length=5,
        ignore_index=-100,
    )
    batch = collator2(features)
    assert batch["input_ids"].tolist() == [[3, 4, 5, 6, 7]]
    assert batch["prompt_len"] == [1]  # shrank since we truncated from left
    assert batch["labels"][0].tolist() == [-100, 4, 5, 6, 7]
    # if we asked for a bit more truncation, the function should raise (no more prompt to truncate)
    collator3 = utils_transformers.PaddingCollatorWithPromptMask(
        tokenizer,
        max_length=4,
        ignore_index=-100,
    )
    with pytest.raises(ValueError):
        _ = collator3(features)


def test_padding_collator_with_multiple_of_32() -> None:
    tokenizer = SimpleTokenizer(padding_side="left", truncation_side="left")
    collator = utils_transformers.PaddingCollatorWithPromptMask(
        tokenizer=tokenizer,
        max_length=80,
        pad_to_multiple_of=32,
        ignore_index=-100,
    )
    features = [
        {"input_ids": list(range(23)), "prompt_len": 23},
        {"input_ids": list(range(31)), "prompt_len": 31},
    ]
    batch = collator(features)
    assert batch["input_ids"].shape == (2, 32)
    assert batch["input_ids"][0][:9].tolist() == [0] * 9
    assert batch["input_ids"][0][9:].tolist() == list(range(23))
    assert batch["input_ids"][1][0].item() == 0
    assert batch["input_ids"][1][1:].tolist() == list(range(31))
    features = [
        {"input_ids": list(range(33)), "prompt_len": 31},
    ]
    batch = collator(features)
    assert batch["input_ids"].shape == (1, 64)
    assert batch["input_ids"][0][:31].tolist() == [0] * 31
    assert batch["input_ids"][0][31:].tolist() == list(range(33))
    features = [
        {"input_ids": list(range(70)), "prompt_len": 31},
    ]
    batch = collator(features)
    assert batch["input_ids"].shape == (1, 64)
    assert batch["input_ids"][0].tolist() == list(range(6, 70))
    assert batch["prompt_len"][0] == 25


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
    base = utils_transformers._get_base_pretrained_model(wrapper)
    assert base is tiny_gpt2_model
    assert utils_transformers.is_hf_model(wrapper)
    assert not utils_transformers.is_hf_model(object())


def test_is_hf_tokenizer_detects_transformers_tokenizer() -> None:
    tokenizer = MinimalPreTrainedTokenizer()
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


def test_run_text_generation_decodes_predictions(
    simple_tokenizer: SimpleTokenizer,
) -> None:
    simple_tokenizer.padding_side = "left"
    model = DummyGenerationModel()
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
    tokenizer = SimpleTokenizer(padding_side="left")
    model = DummyGenerationModel()
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


def test_apply_model_template_to_messages_basic(
    simple_tokenizer: SimpleTokenizer,
) -> None:
    ds = datasets.Dataset.from_dict(
        {
            "messages": [
                [{"role": "user", "content": "Hi"}],
                [{"role": "user", "content": "Bye"}],
            ]
        }
    )
    out_ds = utils_transformers.apply_model_template_to_messages(
        hf_messages_ds=ds,
        tokenizer=simple_tokenizer,
        apply_chat_template_kwargs={"tokenize": False},
    )
    assert isinstance(out_ds, datasets.Dataset)
    # without add_generation_prompt, should not append "assistant:"
    expected_texts = ["user:Hi|", "user:Bye|"]
    assert out_ds["text"] == expected_texts


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


def test_prepare_examples_from_conversations_cache_roundtrip(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = SimpleTokenizer()
    conversations = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]},
        {"messages": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "ciao"}]},
    ]
    convo_ds = datasets.Dataset.from_list(conversations)
    cache_settings = utils_transformers.DataCacheSettings(
        cache_path=tmp_path / "tokenized" / "train_cache",
        lock_timeout_seconds=1.0,
    )
    first_ds = utils_transformers.prepare_examples_from_conversations(
        convo_ds=convo_ds,
        tokenizer=tokenizer,
        max_seq_len=128,
        num_proc=1,
        cache_settings=cache_settings,
    )
    assert cache_settings.cache_path.exists()
    assert len(first_ds) == 2

    def _fail_if_called(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        raise AssertionError("cache path should be reused without regenerating dataset")

    monkeypatch.setattr(utils_transformers, "apply_model_template_to_messages", _fail_if_called)
    second_ds = utils_transformers.prepare_examples_from_conversations(
        convo_ds=convo_ds,
        tokenizer=tokenizer,
        max_seq_len=128,
        num_proc=1,
        cache_settings=cache_settings,
    )
    assert len(second_ds) == len(first_ds)
