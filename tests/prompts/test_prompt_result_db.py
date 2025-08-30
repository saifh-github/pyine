import concurrent.futures
import json
import pathlib
import sys
import threading
import types

import pydantic
import pytest

from pyine.prompts.result_db import (
    CreationMeta,
    PromptResultDB,
    PromptResultRecord,
    TypedPromptResultFetcher,
    fetch_or_generate_prompt_results,
)


@pytest.fixture()
def db(tmp_path: pathlib.Path) -> PromptResultDB:
    return PromptResultDB(db_path=tmp_path / "prompt_results.sqlite")


def test_store_and_fetch_by_identifier(db: PromptResultDB):
    v1 = db.store(
        identifier="id1",
        group="g1",
        prompt_name="summary",
        prompt_version="v1",
        prompt="prompt v1",
        result="result v1",
        meta={"model": "dummy", "temperature": 0.1},
        tags=["a", "b"],
    )
    assert v1 == 1
    v2 = db.store(
        identifier="id1",
        group="g1",
        prompt_name="summary",
        prompt_version="v2",
        prompt="prompt v2",
        result="result v2",
        tags=["a", "x"],
    )
    assert v2 == 2
    # fetch all versions for the identifier
    all_for_id = db.get_by_identifier("id1")
    assert len(all_for_id) == 2
    assert all_for_id[1].result == "result v2"
    assert all_for_id[0].prompt == "prompt v1"
    assert all_for_id[0].creation_meta.created_at <= all_for_id[1].creation_meta.created_at
    # fetch constrained by prompt name
    only_summary = db.get_by_identifier("id1", prompt_name="summary")
    assert len(only_summary) == 2
    # fetch constrained by prompt name+version
    only_v1 = db.get_by_identifier("id1", prompt_name="summary", prompt_version="v1")
    assert len(only_v1) == 1
    assert only_v1[0].prompt_version == "v1"
    assert only_v1[0].meta == {"model": "dummy", "temperature": 0.1}
    assert only_v1[0].tags == ["a", "b"]


def test_group_queries_and_tag_filter(db: PromptResultDB):
    # id1 two versions under group g1
    db.store(
        identifier="id1", group="g1", prompt_name="p", prompt_version="1", prompt="p1", result="r1", tags=["a", "b"]
    )  # v1
    db.store(
        identifier="id1", group="g1", prompt_name="p", prompt_version="2", prompt="p2", result="r2", tags=["a"]
    )  # v2
    # id2 one version, contains a wip tag
    db.store(
        identifier="id2",
        group="g1",
        prompt_name="p",
        prompt_version="1",
        prompt="p3",
        result="r3",
        tags=["topic:math", "wip:no"],
    )  # v1
    # group should return 3 records total
    all_items = db.get_by_group("g1")
    assert len(all_items) == 3
    assert {r.identifier for r in all_items} == {"id1", "id2"}
    # filter by prompt name+version
    p1_only = db.get_by_group("g1", prompt_name="p", prompt_version="1")
    assert {(r.identifier, r.prompt_version) for r in p1_only} == {("id1", "1"), ("id2", "1")}
    # apply filter to drop any item with a wip:* tag
    filtered = db.get_by_group("g1", tag_filter_rule="-wip:*")
    assert {r.identifier for r in filtered} == {"id1"}
    # ensure groups listing includes g1
    assert "g1" in db.list_groups()


def test_thread_safety_on_versions(db: PromptResultDB):
    identifier = "id3"
    group = "g2"
    n = 10
    barrier = threading.Barrier(n)

    def insert_one(i: int):
        barrier.wait()
        return db.store(identifier=identifier, group=group, prompt=f"p{i}", result=f"r{i}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(insert_one, i) for i in range(n)]
        concurrent.futures.wait(futures)
        versions = sorted(f.result() for f in futures)
    assert len(versions) == n
    assert len(set(versions)) == n
    assert all(earlier < later for earlier, later in zip(versions, versions[1:]))
    all_for_id = db.get_by_identifier(identifier)
    assert len(all_for_id) == n
    assert all(
        all_for_id[i].creation_meta.created_at <= all_for_id[i + 1].creation_meta.created_at
        for i in range(len(all_for_id) - 1)
    )
    grp = db.get_by_group(group)
    assert len(grp) == n
    assert [r.identifier for r in grp] == [identifier] * n


def test_list_identifiers(db: PromptResultDB):
    db.store(identifier="b", prompt="p", result="r")
    db.store(identifier="a", prompt="p", result="r")
    db.store(identifier="c", prompt="p", result="r")
    ids = db.list_identifiers()
    assert ids == ["a", "b", "c"]


def test_get_by_identifier_max_age_and_tag_filter(db: PromptResultDB):
    import datetime as _dt

    old_cm = CreationMeta(created_at=_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=2))
    db.store(identifier="age_tag", prompt="p", result="old", tags=["wip:yes"], creation_meta=old_cm)
    db.store(identifier="age_tag", prompt="p", result="new", tags=["ok"])
    recent_only = db.get_by_identifier("age_tag", max_result_age=_dt.timedelta(days=1))
    assert [r.result for r in recent_only] == ["new"]
    tag_filtered = db.get_by_identifier("age_tag", tag_filter_rule="-wip:*")
    assert [r.result for r in tag_filtered] == ["new", "new"] or [r.result for r in tag_filtered] == ["new"]


def test_fetch_or_generate_deduplicates_existing(db: PromptResultDB, monkeypatch: pytest.MonkeyPatch):
    db.store(identifier="dup-id", prompt_name="pn", prompt="p", result="X")
    db.store(identifier="dup-id", prompt_name="pn", prompt="p", result="X")

    # use minimal fake prompt manager to avoid heavy deps during tests; not used since no generation needed
    def _tpl(**kwargs):
        return "{name}"

    def _chain(**kwargs):
        return types.SimpleNamespace(invoke=lambda _inputs: "ignored")

    monkeypatch.setattr("pyine.prompts.manager.get_prompt_template", _tpl, raising=False)
    monkeypatch.setattr("pyine.prompts.manager.get_prompt_chain", _chain, raising=False)
    res = fetch_or_generate_prompt_results(
        model=object(),
        identifier="dup-id",
        input_variables={"name": "Bob"},
        prompt_kwargs={"prompt_name": "pn"},
        db=db,
    )
    # deduplication should leave a single record and not generate new ones
    assert len(res) == 1
    assert res[0].result == "X"


def test_fetch_or_generate_generate_until_count_no_log(db: PromptResultDB, monkeypatch: pytest.MonkeyPatch):
    # here, we use a fake prompt manager producing model-like objects with .model_dump_json()

    class ModelLike:
        def __init__(self, data):
            self._data = data

        def model_dump_json(self) -> str:
            return json.dumps(self._data)

    class DummyChain:
        def __init__(self, outputs):
            self._iter = iter(outputs)

        def invoke(self, _inputs):
            try:
                return next(self._iter)
            except StopIteration:
                return ModelLike({"v": 999})

    def get_prompt_template(**_kwargs):
        return "Hello {name}"

    def get_prompt_chain(model=None, **_kwargs):
        return DummyChain([ModelLike({"v": 1}), ModelLike({"v": 2})])

    monkeypatch.setattr("pyine.prompts.manager.get_prompt_template", get_prompt_template, raising=False)
    monkeypatch.setattr("pyine.prompts.manager.get_prompt_chain", get_prompt_chain, raising=False)
    recs = fetch_or_generate_prompt_results(
        model=object(),
        identifier="gen-no-log",
        input_variables={"name": "Bob"},
        prompt_kwargs={"prompt_name": "pn"},
        db=db,
        generate_until_result_count=2,
        log_new_results=False,
    )
    assert len(recs) == 2
    assert all(r.prompt == "Hello Bob" for r in recs)
    # ensure nothing was persisted when log_new_results=False
    assert db.get_by_identifier("gen-no-log") == []


def test_typed_prompt_result_fetcher_decode_record_dict():
    rec = PromptResultRecord(
        identifier="t1",
        prompt="p",
        result=json.dumps({"a": 1}),
        creation_meta=CreationMeta(),
        meta={},
        tags=[],
    )
    fetcher = TypedPromptResultFetcher(dict)
    decoded = fetcher.decode_record(rec)
    assert decoded.result == {"a": 1}


def test_typed_prompt_result_fetcher_fetch_or_generate_with_pydantic(
    db: PromptResultDB, monkeypatch: pytest.MonkeyPatch
):

    class Item(pydantic.BaseModel):
        v: int

    class ModelLike:
        def __init__(self, data):
            self._data = data

        def model_dump_json(self) -> str:
            return json.dumps(self._data)

    class DummyChain:
        def __init__(self, outputs):
            self._iter = iter(outputs)

        def invoke(self, _inputs):
            try:
                return next(self._iter)
            except StopIteration:
                return ModelLike({"v": 999})

    def get_prompt_template(**_kwargs):
        return "Value {x}"

    def get_prompt_chain(model=None, **_kwargs):
        return DummyChain([ModelLike({"v": 10})])

    monkeypatch.setattr("pyine.prompts.manager.get_prompt_template", get_prompt_template, raising=False)
    monkeypatch.setattr("pyine.prompts.manager.get_prompt_chain", get_prompt_chain, raising=False)
    fetcher = TypedPromptResultFetcher(Item)
    items = fetcher.fetch_or_generate(
        model=object(),
        identifier="typed-fetch",
        input_variables={"x": "Z"},
        prompt_kwargs={"prompt_name": "pn"},
        db=db,
        log_new_results=False,
    )
    assert len(items) == 1
    assert isinstance(items[0].result, Item)
    assert items[0].result.v == 10
    assert items[0].record.prompt == "Value Z"
