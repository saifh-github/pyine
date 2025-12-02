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


def test_prepare_generation_prompts_handles_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
    simple_tokenizer: tests.utils.transformers.utils.SimpleTokenizer,
) -> None:
    """Verify that cache hits don't cause issues.

    When HuggingFace datasets caches the result of a map() call, the map function is not executed
    on subsequent calls. This test simulates a cache hit and verifies the function handles it gracefully.
    """
    prompts = [
        {"messages": [{"role": "user", "content": "Hello"}]},
        {"messages": [{"role": "user", "content": "Hi there"}]},
    ]
    prompts_ds = datasets.Dataset.from_list(prompts)
    # first, build the expected result so we can return it from the cache
    first_result = data_utils.prepare_generation_prompts_from_dataset(
        prompts_ds=prompts_ds,
        tokenizer=simple_tokenizer,
        max_seq_len=128,
        keep_in_memory=True,
    )
    assert len(first_result) == 2
    cached_dataset = first_result
    # now patch the map() method to simulate a cache hit: return the cached dataset without calling map
    original_map = datasets.Dataset.map
    map_call_count = 0

    def patched_map(
        self: datasets.Dataset,
        function: typing.Any,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> datasets.Dataset:
        nonlocal map_call_count
        map_call_count += 1
        if map_call_count == 2:  # second map call is the _encode_prompts one
            return cached_dataset  # simulate cache hit by returning cached result
        return original_map(self, function, *args, **kwargs)

    monkeypatch.setattr(datasets.Dataset, "map", patched_map)
    # should succeed even when cache is hit (no assertion on sample_idx)
    result = data_utils.prepare_generation_prompts_from_dataset(
        prompts_ds=prompts_ds,
        tokenizer=simple_tokenizer,
        max_seq_len=128,
        keep_in_memory=True,
    )
    assert len(result) == 2
