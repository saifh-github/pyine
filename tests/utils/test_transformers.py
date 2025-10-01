import types
import typing

import datasets
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
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.model_max_length = 4096

    def apply_chat_template(
        self,
        messages: list[dict[str, typing.Any]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        history = "".join(f"{item['role']}:{item['content']}|" for item in messages)
        if add_generation_prompt:
            return history + "assistant:"
        return history

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [ord(character) for character in text]}

    def decode(
        self,
        token_ids: typing.Iterable[int] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            values = token_ids.tolist()
        else:
            values = list(token_ids)
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
            prompt_length = int(attention_mask[idx].sum().item())
            prompt_tokens = input_ids[idx, :prompt_length]
            new_tokens = torch.tensor(
                [ord("x") + idx, ord("y") + idx],
                dtype=torch.long,
                device=input_ids.device,
            )
            sequences.append(torch.cat((prompt_tokens, new_tokens), dim=0))
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            padding_value=0,
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


def test_build_example_ids_truncates_overlong_response(
    simple_tokenizer: SimpleTokenizer,
) -> None:
    assistant_msg = {"role": "assistant", "content": "ABCDEFGHIJ"}
    result = utils_transformers._build_example_ids_from_conversation_parts(
        tokenizer=simple_tokenizer,
        history_msgs=[],
        assistant_msg=assistant_msg,
        max_seq_len=5,
    )
    assert result is not None
    prompt_text = simple_tokenizer.apply_chat_template([], tokenize=False, add_generation_prompt=True)
    full_ids = _compute_token_ids(simple_tokenizer, prompt_text + assistant_msg["content"])
    assert result["prompt_ids"] == []
    assert result["input_ids"] == full_ids[-5:]


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
        num_proc=1,
    )
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
    for row_idx in range(prepared.num_rows):
        row = prepared[row_idx]
        for input_ids, prompt_len in zip(row["input_ids"], row["prompt_len"], strict=False):
            actual.append((input_ids, prompt_len))
    assert actual == expected


def test_fixed_size_padding_collator_masks_prompt_and_padding(
    simple_tokenizer: SimpleTokenizer,
) -> None:
    collator = utils_transformers.FixedSizePaddingCollatorWithPromptMask(
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


def test_batchwise_padding_collator_pads_batches_and_forwards_metadata(
    simple_tokenizer: SimpleTokenizer,
) -> None:
    collator = utils_transformers.BatchwisePaddingCollator(
        tokenizer=simple_tokenizer,
        keep_extra_fields=["meta"],
        ignore_index=-77,
    )
    batch = collator(
        [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "prompt_len": 2,
                "prompt_ids": [11, 22],
                "meta": "first",
            },
            {
                "input_ids": [4, 5],
                "attention_mask": [1, 1],
                "prompt_ids": [40],
                "meta": "second",
            },
        ]
    )
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["prompt_len"] == [2, 1]
    assert batch["input_len"] == [3, 2]
    assert batch["meta"] == ["first", "second"]
    labels = batch["labels"].tolist()
    assert labels[0] == [-77, -77, 3]
    assert labels[1] == [-77, 5, -77]


def test_batchwise_padding_collator_validates_inputs(
    simple_tokenizer: SimpleTokenizer,
) -> None:
    collator = utils_transformers.BatchwisePaddingCollator(
        tokenizer=simple_tokenizer,
        max_allowed_length=5,
    )
    with pytest.raises(ValueError):
        collator(
            [
                {"attention_mask": [1, 1]},
            ]
        )
    with pytest.raises(ValueError):
        collator(
            [
                {"input_ids": [1, 2, 3, 4, 5, 6], "attention_mask": [1, 1, 1, 1, 1, 1]},
            ]
        )
    with pytest.raises(ValueError):
        collator(
            [
                {
                    "input_ids": [1, 2],
                    "attention_mask": [1, 1],
                    "prompt_len": 3,
                    "prompt_ids": [5, 6, 7],
                }
            ]
        )


def test_infer_effective_max_seq_len_uses_minimum_candidate() -> None:
    tokenizer = types.SimpleNamespace(model_max_length=4096)
    config = types.SimpleNamespace(max_position_embeddings=2048, sliding_window=1024)
    model = types.SimpleNamespace(config=config)
    result = utils_transformers.infer_effective_max_seq_len(model=model, tokenizer=tokenizer)
    assert result == 1024


def test_infer_effective_max_seq_len_raises_when_unknown() -> None:
    tokenizer = types.SimpleNamespace(model_max_length=None)
    model = types.SimpleNamespace(config=types.SimpleNamespace())
    with pytest.raises(ValueError):
        utils_transformers.infer_effective_max_seq_len(model=model, tokenizer=tokenizer)


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
    model = DummyGenerationModel()
    dataloader = [
        {
            "input_ids": torch.tensor(
                [[ord("A"), ord("B"), ord("C"), 0], [ord("D"), ord("E"), 0, 0]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 0], [1, 1, 0, 0]],
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
