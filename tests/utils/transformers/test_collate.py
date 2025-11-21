import pytest

import pyine.utils.transformers.collate
import tests.utils.transformers.utils


def test_padding_collator_masks_prompt_and_padding(
    simple_tokenizer: tests.utils.transformers.utils.SimpleTokenizer,
) -> None:
    collator = pyine.utils.transformers.collate.PaddingCollatorWithPromptMask(
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
    tokenizer = tests.utils.transformers.utils.SimpleTokenizer(padding_side="left", truncation_side="left")
    collator = pyine.utils.transformers.collate.PaddingCollatorWithPromptMask(
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
    collator2 = pyine.utils.transformers.collate.PaddingCollatorWithPromptMask(
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
    tokenizer = tests.utils.transformers.utils.SimpleTokenizer(padding_side="right", truncation_side="right")
    collator = pyine.utils.transformers.collate.PaddingCollatorWithPromptMask(
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
    collator2 = pyine.utils.transformers.collate.PaddingCollatorWithPromptMask(
        tokenizer,
        max_length=5,
        ignore_index=-100,
    )
    batch = collator2(features)
    assert batch["input_ids"].tolist() == [[3, 4, 5, 6, 7]]
    assert batch["prompt_len"] == [1]  # shrank since we truncated from left
    assert batch["labels"][0].tolist() == [-100, 4, 5, 6, 7]
    # if we asked for a bit more truncation, the function should raise (no more prompt to truncate)
    collator3 = pyine.utils.transformers.collate.PaddingCollatorWithPromptMask(
        tokenizer,
        max_length=4,
        ignore_index=-100,
    )
    with pytest.raises(ValueError):
        _ = collator3(features)


def test_padding_collator_with_multiple_of_32() -> None:
    tokenizer = tests.utils.transformers.utils.SimpleTokenizer(padding_side="left", truncation_side="left")
    collator = pyine.utils.transformers.collate.PaddingCollatorWithPromptMask(
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


def test_padding_collator_rejects_pad_multiple_greater_than_max(
    simple_tokenizer: tests.utils.transformers.utils.SimpleTokenizer,
) -> None:
    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        pyine.utils.transformers.collate.PaddingCollatorWithPromptMask(
            tokenizer=simple_tokenizer,
            max_length=8,
            pad_to_multiple_of=16,
        )


def test_padding_collator_forwards_extra_fields(
    simple_tokenizer: tests.utils.transformers.utils.SimpleTokenizer,
) -> None:
    collator = pyine.utils.transformers.collate.PaddingCollatorWithPromptMask(
        tokenizer=simple_tokenizer,
        max_length=6,
        keep_extra_fields=True,
    )
    features = [
        {"input_ids": [1, 2], "prompt_len": 1, "meta": {"id": "a"}, "identifier": "first"},
        {"input_ids": [3, 4, 5], "prompt_len": 2, "meta": {"id": "b"}, "identifier": "second"},
    ]
    batch = collator(features)
    assert batch["meta"] == [{"id": "a"}, {"id": "b"}]
    assert batch["identifier"] == ["first", "second"]


def test_padding_collator_missing_extra_field_raises(
    simple_tokenizer: tests.utils.transformers.utils.SimpleTokenizer,
) -> None:
    collator = pyine.utils.transformers.collate.PaddingCollatorWithPromptMask(
        tokenizer=simple_tokenizer,
        max_length=6,
        keep_extra_fields=["meta"],
    )
    features = [
        {"input_ids": [1, 2], "prompt_len": 1, "meta": "available"},
        {"input_ids": [3, 4, 5], "prompt_len": 2},
    ]
    with pytest.raises(ValueError, match="extra field 'meta'"):
        collator(features)


def test_padding_collator_invalid_prompt_len_raises(
    simple_tokenizer: tests.utils.transformers.utils.SimpleTokenizer,
) -> None:
    collator = pyine.utils.transformers.collate.PaddingCollatorWithPromptMask(
        tokenizer=simple_tokenizer,
        max_length=4,
    )
    features = [
        {"input_ids": [1, 2, 3], "prompt_len": 5},
    ]
    with pytest.raises(ValueError, match="invalid prompt_len"):
        collator(features)


def test_padding_collator_logs_batch_statistics(
    simple_tokenizer: tests.utils.transformers.utils.SimpleTokenizer,
) -> None:
    records: list[pyine.utils.transformers.collate.CollatorBatchLogRecord] = []

    def _capture(record: pyine.utils.transformers.collate.CollatorBatchLogRecord) -> None:
        records.append(record)

    collator = pyine.utils.transformers.collate.PaddingCollatorWithPromptMask(
        tokenizer=simple_tokenizer,
        max_length=6,
        always_pad_to_max_length=True,
        batch_log_handler=_capture,
    )
    features = [
        {"input_ids": [1, 2, 3], "prompt_len": 2},
        {"input_ids": [4, 5], "prompt_len": 1},
    ]
    collator.set_stage("train")
    collator(features)
    collator(features)
    collator.set_stage("eval")
    collator(
        [
            {"input_ids": [6, 7, 8, 9], "prompt_len": 3},
        ]
    )
    assert len(records) == 3
    assert records[0].stage == "train"
    assert records[0].batch_size == 2
    assert records[0].padded_seq_len == 6
    assert records[0].padding_ratio == pytest.approx(7 / 12)
    assert records[0].non_ignored_label_ratio == pytest.approx(2 / 12)
    assert records[1].stage == "train"
    assert records[2].stage == "eval"
    assert records[2].batch_size == 1
    assert records[2].padded_seq_len == 6
    assert records[2].padding_ratio == pytest.approx(2 / 6)
    assert records[2].non_ignored_label_ratio == pytest.approx(1 / 6)
