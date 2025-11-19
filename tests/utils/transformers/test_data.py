import pathlib
import typing

import datasets
import pytest

import pyine.utils.transformers as utils_transformers
import pyine.utils.transformers.data as data_utils
import tests.utils.transformers.utils


def _compute_token_ids(
    tokenizer: tests.utils.transformers.utils.SimpleTokenizer,
    text: str,
) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def test_build_example_ids_without_truncation(
    simple_tokenizer: tests.utils.transformers.utils.SimpleTokenizer,
) -> None:
    history_msgs = [
        {"role": "user", "content": "Hello"},
    ]
    assistant_msg = {"role": "assistant", "content": "Answer"}
    result = data_utils._build_example_ids_from_conversation_parts(
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
    simple_tokenizer: tests.utils.transformers.utils.SimpleTokenizer,
) -> None:
    history_msgs = [
        {"role": "user", "content": "LongPrompt"},
    ]
    assistant_msg = {"role": "assistant", "content": "OK"}
    result = data_utils._build_example_ids_from_conversation_parts(
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
    simple_tokenizer: tests.utils.transformers.utils.SimpleTokenizer,
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
            built = data_utils._build_example_ids_from_conversation_parts(
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


def test_apply_model_template_to_messages_basic(
    simple_tokenizer: tests.utils.transformers.utils.SimpleTokenizer,
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


def test_prepare_examples_from_conversations_cache_roundtrip(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = tests.utils.transformers.utils.SimpleTokenizer()
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
