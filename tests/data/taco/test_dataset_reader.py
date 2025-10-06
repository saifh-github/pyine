import json
import typing

import datasets as hf_datasets
import pytest

import pyine.data.taco.dataset_reader as taco_reader


class FakeSubset(hf_datasets.Dataset):
    def __init__(
        self,
        samples: list[dict[str, str]],
    ) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, str]:
        return self._samples[idx]


def make_valid_sample(
    solutions: list[str] | None = None,
    input_output: dict[str, typing.Any] | None = None,
    raw_tags: list[str] | None = None,
    tags: list[str] | None = None,
    skill_types: list[str] | None = None,
) -> dict[str, str]:
    if solutions is None:
        solutions = ["print('hi')", "x=1"]
    if input_output is None:
        input_output = {
            "inputs": [[1, 2], {"a": 1}, "x"],
            "outputs": [3, 2, None],
            "fn_name": "foo",
        }
    if raw_tags is None:
        raw_tags = ["raw1", "raw2"]
    if tags is None:
        tags = ["t1"]
    if skill_types is None:
        skill_types = ["s1"]
    return {
        "solutions": json.dumps(solutions),
        "input_output": json.dumps(input_output),
        "raw_tags": str(raw_tags),
        "tags": str(tags),
        "skill_types": str(skill_types),
    }


@pytest.fixture(autouse=True)
def patch_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Tok:
        def encode(
            self,
            s: str,
        ) -> list[str]:
            return list(s)  # token count == len(s)

    monkeypatch.setattr(taco_reader.tiktoken, "encoding_for_model", lambda name: _Tok())


def patch_datasets(
    monkeypatch: pytest.MonkeyPatch,
    train_samples: list[dict[str, str]],
    test_samples: list[dict[str, str]],
) -> None:
    def _fake_load_dataset(
        name: str,
        split: str,
    ) -> FakeSubset:
        assert name == "BAAI/TACO"
        if split == "train":
            return FakeSubset(train_samples)
        if split == "test":
            return FakeSubset(test_samples)
        raise AssertionError("unexpected split")

    monkeypatch.setattr(taco_reader.hf_datasets, "load_dataset", _fake_load_dataset)


def test_len_and_getitem_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = [make_valid_sample()]
    test = [make_valid_sample()]
    patch_datasets(monkeypatch, train, test)
    reader = taco_reader.DatasetReader()
    assert len(reader) == 2
    s0 = reader[0]
    assert s0["subset"] == "train" and s0["subset_idx"] == 0 and s0["idx"] == 0
    assert isinstance(s0["solutions"], list) and len(s0["solutions"]) == 2
    assert isinstance(s0["input_output"], dict)
    assert isinstance(s0["raw_tags"], list) and isinstance(s0["tags"], list) and isinstance(s0["skill_types"], list)


def test_getitem_solutions_json_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = make_valid_sample()
    bad["solutions"] = "[1,2"  # invalid JSON
    patch_datasets(monkeypatch, [bad], [])
    reader = taco_reader.DatasetReader()
    with pytest.raises(ValueError) as ei:
        _ = reader[0]
    assert "cannot parse solutions JSON" in str(ei.value)
    # second access should hit the broken cache
    with pytest.raises(ValueError) as ei2:
        _ = reader[0]
    assert "is broken" in str(ei2.value)


def test_getitem_empty_solutions_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = make_valid_sample(solutions=[])
    patch_datasets(monkeypatch, [bad], [])
    reader = taco_reader.DatasetReader()
    with pytest.raises(ValueError) as ei:
        _ = reader[0]
    assert "no solution" in str(ei.value)


def test_getitem_input_output_json_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = make_valid_sample()
    bad["input_output"] = "{"  # invalid JSON
    patch_datasets(monkeypatch, [bad], [])
    reader = taco_reader.DatasetReader()
    with pytest.raises(ValueError) as ei:
        _ = reader[0]
    assert "cannot parse input_output JSON" in str(ei.value)


def test_getitem_invalid_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = make_valid_sample(
        input_output={"inputs": [1, 2], "outputs": [3]},
    )
    patch_datasets(monkeypatch, [bad], [])
    reader = taco_reader.DatasetReader()
    with pytest.raises(ValueError) as ei:
        _ = reader[0]
    assert "invalid input_output structure" in str(ei.value)


def test_getitem_invalid_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # force type validation failure
    monkeypatch.setattr(taco_reader.DatasetReader, "_validate_types", lambda self, v: False)
    bad = make_valid_sample()
    patch_datasets(monkeypatch, [bad], [])
    reader = taco_reader.DatasetReader()
    with pytest.raises(ValueError) as ei:
        _ = reader[0]
    assert "invalid types in input_output" in str(ei.value)


def test_getitem_invalid_fn_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = make_valid_sample(input_output={"inputs": [1], "outputs": [1], "fn_name": ""})
    patch_datasets(monkeypatch, [bad], [])
    reader = taco_reader.DatasetReader()
    with pytest.raises(ValueError) as ei:
        _ = reader[0]
    assert "invalid fn_name" in str(ei.value)


def test_getitem_ast_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = make_valid_sample()
    bad["raw_tags"] = "not a list"  # literal_eval will fail
    patch_datasets(monkeypatch, [bad], [])
    reader = taco_reader.DatasetReader()
    with pytest.raises(ValueError) as ei:
        _ = reader[0]
    assert "cannot parse raw_tags" in str(ei.value)


def test_get_broken_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = make_valid_sample()
    broken = make_valid_sample()
    broken["solutions"] = "[1,2"  # invalid JSON
    patch_datasets(monkeypatch, [good, broken], [])
    reader = taco_reader.DatasetReader()
    # trigger scanning
    broken_idxs = reader.get_broken_indices()
    assert 1 in broken_idxs


def test_get_statistics_with_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # two samples; one will have a solution filtered out by validation
    s1 = make_valid_sample(solutions=["aa", "bbb"], raw_tags=["a", "b"], tags=["x"], skill_types=["s1"])
    s2 = make_valid_sample(solutions=["cccc"], raw_tags=["a"], tags=["y"], skill_types=["s2"])
    patch_datasets(monkeypatch, [s1, s2], [])
    reader = taco_reader.DatasetReader()

    # mock validate_code: reject strings length 3, accept others
    def _mock_validate_code(code: str) -> None:
        if len(code) == 3:
            raise AssertionError("bad code")

    monkeypatch.setattr(
        taco_reader.pyine.utils.code.validation,
        "validate_code",
        _mock_validate_code,
    )

    stats = reader.get_statistics(validate=True)
    assert stats["total_samples"] == 2
    assert stats["valid_samples"] == 2
    assert stats["broken_samples"] == 0
    assert stats["total_solutions"] == 2  # 'bbb' filtered out
    assert stats["avg_solutions_per_problem"] == 1
    # token lengths equal to string lengths via patched tokenizer
    assert stats["avg_solution_length"] == pytest.approx((2 + 4) / 2)
    assert stats["max_solution_length"] == 4 and stats["min_solution_length"] == 2
    # raw tag distribution: 'a' appears twice, 'b' once
    assert stats["raw_tags_distribution"].get("a", 0) == 2
    assert stats["raw_tags_distribution"].get("b", 0) == 1
    assert stats["tags_distribution"].get("x", 0) == 1
    assert stats["tags_distribution"].get("y", 0) == 1
    assert stats["skill_types_distribution"].get("s1", 0) == 1
    assert stats["skill_types_distribution"].get("s2", 0) == 1
