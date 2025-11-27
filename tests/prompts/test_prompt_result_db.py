import concurrent.futures
import datetime
import json
import pathlib
import threading
import types
import typing

import orjson
import pydantic
import pytest

from pyine.prompts.result_db import (
    CreationMeta,
    PromptResultDB,
    PromptResultRecord,
    TypedPromptResultFetcher,
    ValidationFailedError,
    fetch_or_generate_prompt_results,
)


@pytest.fixture()
def db(tmp_path: pathlib.Path) -> PromptResultDB:
    return PromptResultDB(db_path=tmp_path / "prompt_results.sqlite")


def test_store_and_fetch_by_identifier(db: PromptResultDB) -> None:
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
    assert db.count_entries() == 2
    assert db.list_prompt_names() == ["summary"]
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


def test_creation_meta_requires_timezone() -> None:
    naive_now = datetime.datetime.now()
    with pytest.raises(ValueError):
        CreationMeta(created_at=naive_now)


def test_group_queries_and_tag_filter(db: PromptResultDB) -> None:
    # id1 two versions under group g1
    db.store(
        identifier="id1",
        group="g1",
        prompt_name="p",
        prompt_version="1",
        prompt="p1",
        result="r1",
        tags=["a", "b"],
    )  # v1
    db.store(
        identifier="id1",
        group="g1",
        prompt_name="p",
        prompt_version="2",
        prompt="p2",
        result="r2",
        tags=["a"],
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
    assert {(r.identifier, r.prompt_version) for r in p1_only} == {
        ("id1", "1"),
        ("id2", "1"),
    }
    # apply filter to drop any item with a wip:* tag
    filtered = db.get_by_group("g1", tag_filter_rule="-wip:*")
    assert {r.identifier for r in filtered} == {"id1"}
    # ensure groups listing includes g1
    assert "g1" in db.list_groups()
    # test that prompt-named-based-getter also works OK
    assert len(db.get_by_prompt_name("p")) == 3
    assert len(db.get_by_prompt_name("p", prompt_version="1")) == 2


def test_thread_safety_on_versions(db: PromptResultDB) -> None:
    identifier = "id3"
    group = "g2"
    n = 10
    barrier = threading.Barrier(n)

    def insert_one(
        i: int,
    ) -> int:
        barrier.wait()
        return db.store(identifier=identifier, group=group, prompt=f"p{i}", result=f"r{i}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(insert_one, i) for i in range(n)]
        concurrent.futures.wait(futures)
        versions = sorted(f.result() for f in futures)
    assert len(versions) == n
    assert len(set(versions)) == n
    assert all(earlier < later for earlier, later in zip(versions, versions[1:], strict=False))
    all_for_id = db.get_by_identifier(identifier)
    assert len(all_for_id) == n
    assert all(
        all_for_id[i].creation_meta.created_at <= all_for_id[i + 1].creation_meta.created_at
        for i in range(len(all_for_id) - 1)
    )
    grp = db.get_by_group(group)
    assert len(grp) == n
    assert [r.identifier for r in grp] == [identifier] * n


def test_list_identifiers(db: PromptResultDB) -> None:
    db.store(identifier="b", prompt="p", result="r")
    db.store(identifier="a", prompt="p", result="r")
    db.store(identifier="c", prompt="p", result="r")
    ids = db.list_identifiers()
    assert ids == ["a", "b", "c"]


def test_get_by_identifier_max_age_and_tag_filter(db: PromptResultDB) -> None:
    import datetime as _dt

    old_cm = CreationMeta(created_at=_dt.datetime.now(datetime.UTC) - _dt.timedelta(minutes=30))
    db.store(
        identifier="age_tag",
        prompt="p",
        result="old",
        tags=["wip:yes"],
        creation_meta=old_cm,
    )
    db.store(identifier="age_tag", prompt="p", result="new", tags=["ok"])
    recent_only = db.get_by_identifier("age_tag", max_result_age=_dt.timedelta(minutes=10))
    assert [r.result for r in recent_only] == ["new"]
    tag_filtered = db.get_by_identifier("age_tag", tag_filter_rule="-wip:*")
    assert [r.result for r in tag_filtered] == ["new", "new"] or [r.result for r in tag_filtered] == ["new"]


def test_row_to_record_falls_back_to_created_at_column(db: PromptResultDB) -> None:
    row_id = db.store(identifier="fallback", prompt="prompt", result="result")
    conn = db._connect()
    try:
        created_at_iso = conn.execute("SELECT created_at FROM items WHERE id = ?", (row_id,)).fetchone()[0]
        conn.execute(
            "UPDATE items SET creation_meta = ? WHERE id = ?",
            (
                orjson.dumps(
                    {
                        "created_by": "unit-test",
                        "platform": "test",
                    }
                ),
                row_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    fetched = db.get_by_identifier("fallback")
    assert len(fetched) == 1
    expected_created_at = datetime.datetime.fromisoformat(created_at_iso)
    assert fetched[0].creation_meta.created_at == expected_created_at


def test_fetch_or_generate_deduplicates_existing(
    db: PromptResultDB,
) -> None:
    db.store(identifier="dup-id", prompt_name="pn", prompt="p", result="X")
    db.store(identifier="dup-id", prompt_name="pn", prompt="p", result="X")

    chain_config = _make_prompt_chain_config("{name}")
    res = fetch_or_generate_prompt_results(
        identifier="dup-id",
        input_variables={"name": "Bob"},
        prompt_chain_config=chain_config,
        db=db,
    )
    # deduplication should leave a single record and not generate new ones
    assert len(res) == 1
    assert res[0].result == "X"


class ModelLike:
    def __init__(
        self,
        data: typing.Any,
    ) -> None:
        self._data = data

    def model_dump_json(self) -> str:
        return json.dumps(self._data)


class DummyChain:
    def __init__(
        self,
        outputs: typing.Iterable[typing.Any],
    ) -> None:
        self._iter = iter(outputs)

    def invoke(
        self,
        _inputs: typing.Any,
        **kwargs: typing.Any,
    ) -> ModelLike:
        try:
            return next(self._iter)
        except StopIteration:
            return ModelLike({"v": 999})


class _StringTemplate:
    def __init__(
        self,
        pattern: str,
    ) -> None:
        self._pattern = pattern

    def format(
        self,
        **kwargs: typing.Any,
    ) -> str:
        return self._pattern.format(**kwargs)


def _make_prompt_chain_config(
    template_pattern: str,
    outputs: typing.Iterable[typing.Any] | None = None,
    prompt_name: str = "pn",
    version: str | None = None,
) -> types.SimpleNamespace:
    template = _StringTemplate(template_pattern)
    chain = types.SimpleNamespace(invoke=lambda *_args, **_kwargs: None) if outputs is None else DummyChain(outputs)
    prompt = types.SimpleNamespace(prompt_name=prompt_name, version=version)
    return types.SimpleNamespace(prompt=prompt, template=template, chain=chain)


def test_fetch_or_generate_generate_until_count_no_log(
    db: PromptResultDB,
) -> None:
    chain_config = _make_prompt_chain_config(
        "Hello {name}",
        outputs=[ModelLike({"v": 1}), ModelLike({"v": 2})],
    )
    recs = fetch_or_generate_prompt_results(
        identifier="gen-no-log",
        input_variables={"name": "Bob"},
        prompt_chain_config=chain_config,
        db=db,
        generate_until_result_count=2,
        log_new_results=False,
    )
    assert len(recs) == 2
    assert all(r.prompt == "Hello Bob" for r in recs)
    # ensure nothing was persisted when log_new_results=False
    assert db.get_by_identifier("gen-no-log") == []


def test_typed_prompt_result_fetcher_decode_record_dict() -> None:
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
    db: PromptResultDB,
) -> None:
    class Item(pydantic.BaseModel):
        v: int

    chain_config = _make_prompt_chain_config("Value {x}", outputs=[ModelLike({"v": 10})])
    fetcher = TypedPromptResultFetcher(Item)
    items = fetcher.fetch_or_generate(
        identifier="typed-fetch",
        input_variables={"x": "Z"},
        prompt_chain_config=chain_config,
        db=db,
        log_new_results=False,
    )
    assert len(items) == 1
    assert isinstance(items[0].result, Item)
    assert items[0].result.v == 10
    assert items[0].record.prompt == "Value Z"


def test_validator_with_retries(
    db: PromptResultDB,
) -> None:
    validator_attempts = 0

    def _validator(
        result: str,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> bool:
        nonlocal validator_attempts
        if validator_attempts < 5:
            validator_attempts += 1
            return False
        return True

    with pytest.raises(ValidationFailedError):
        _ = fetch_or_generate_prompt_results(
            identifier="valid-test",
            input_variables={"thang": "woops"},
            prompt_chain_config=_make_prompt_chain_config(
                "thingy {thang}",
                outputs=[ModelLike(f"t{idx}") for idx in range(10)],
            ),
            db=db,
            output_validator=_validator,
            max_unsatisfactory_retries=4,
        )
    validator_attempts = 0
    recs = fetch_or_generate_prompt_results(
        identifier="valid-test",
        input_variables={"thang": "woops"},
        prompt_chain_config=_make_prompt_chain_config(
            "thingy {thang}",
            outputs=[ModelLike(f"t{idx}") for idx in range(10)],
        ),
        db=db,
        output_validator=_validator,
        max_unsatisfactory_retries=5,
    )
    assert len(recs) == 1
    assert recs[0].prompt == "thingy woops"
    assert recs[0].result == '"t5"'
    assert len(db.get_by_identifier("valid-test")) == 1


def test_delete_records_by_identifier(
    db: PromptResultDB,
) -> None:
    # insert two for same identifier and one for another identifier
    db.store(identifier="del-id-1", group="g", prompt="p1", result="r1")
    db.store(identifier="del-id-1", group="g", prompt="p2", result="r2")
    db.store(identifier="keep-id", group="g", prompt="p3", result="r3")
    deleted = db.delete_records(identifier="del-id-1")
    assert deleted == 2
    assert db.get_by_identifier("del-id-1") == []
    assert [r.result for r in db.get_by_identifier("keep-id")] == ["r3"]
    with pytest.raises(ValueError):
        db.delete_records()
    with pytest.raises(ValueError):
        db.delete_records(prompt_version="v1")


def test_delete_records_by_group_and_prompt_version(
    db: PromptResultDB,
) -> None:
    # two in target group (v1 and v2), one in another group
    db.store(
        identifier="x1",
        group="del-group",
        prompt_name="pn",
        prompt_version="v1",
        prompt="p",
        result="r1",
    )
    db.store(
        identifier="x2",
        group="del-group",
        prompt_name="pn",
        prompt_version="v2",
        prompt="p",
        result="r2",
    )
    db.store(
        identifier="x3",
        group="other",
        prompt_name="pn",
        prompt_version="v1",
        prompt="p",
        result="r3",
    )
    deleted = db.delete_records(group="del-group", prompt_name="pn", prompt_version="v1")
    assert deleted == 1
    remaining_in_group = db.get_by_group("del-group")
    assert {(r.identifier, r.prompt_version) for r in remaining_in_group} == {("x2", "v2")}
    other_group = db.get_by_group("other")
    assert {(r.identifier, r.prompt_version) for r in other_group} == {("x3", "v1")}


def test_delete_records_older_than(
    db: PromptResultDB,
) -> None:
    old_cm = CreationMeta(created_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=2))
    new_cm = CreationMeta(created_at=datetime.datetime.now(datetime.UTC))
    db.store(identifier="age-del", prompt="p", result="old", creation_meta=old_cm)
    db.store(identifier="age-del", prompt="p", result="new", creation_meta=new_cm)
    deleted = db.delete_records(older_than=datetime.timedelta(days=1))
    assert deleted == 1
    remaining = db.get_by_identifier("age-del")
    assert [r.result for r in remaining] == ["new"]


def test_get_all_results_ordering_and_filters(
    db: PromptResultDB,
) -> None:
    now = datetime.datetime.now(datetime.UTC)
    cm_old = CreationMeta(created_at=now - datetime.timedelta(minutes=30))
    cm_mid = CreationMeta(created_at=now - datetime.timedelta(minutes=10))
    cm_new = CreationMeta(created_at=now - datetime.timedelta(minutes=5))
    db.store(identifier="id_b", prompt="pb", result="rb", creation_meta=cm_mid, tags=["ok"])
    db.store(
        identifier="id_a",
        prompt="pa_old",
        result="ra_old",
        creation_meta=cm_old,
        tags=["wip:yes"],
    )
    db.store(
        identifier="id_a",
        prompt="pa_new",
        result="ra_new",
        creation_meta=cm_new,
        tags=["ok"],
    )
    all_recs = db.get_all_results()
    assert [(r.identifier, r.prompt) for r in all_recs] == [
        ("id_a", "pa_old"),
        ("id_a", "pa_new"),
        ("id_b", "pb"),
    ]
    recent_only = db.get_all_results(max_result_age=datetime.timedelta(minutes=15))
    assert [(r.identifier, r.prompt) for r in recent_only] == [
        ("id_a", "pa_new"),
        ("id_b", "pb"),
    ]
    no_wip = db.get_all_results(tag_filter_rule="-wip:*")
    assert [(r.identifier, r.prompt) for r in no_wip] == [
        ("id_a", "pa_new"),
        ("id_b", "pb"),
    ]


def test_record_uid() -> None:
    # Full record with prompt_name and prompt_version
    rec = PromptResultRecord(
        identifier="sample_123",
        prompt_name="code_summary",
        prompt_version="v1",
        prompt="Summarize this",
        result="This is a summary",
        creation_meta=CreationMeta(created_at=datetime.datetime(2024, 11, 26, 14, 30, 22, tzinfo=datetime.UTC)),
    )
    uid = rec.record_uid
    assert uid.startswith("sample_123_code_summary_v1_20241126-143022_")
    assert len(uid.split("_")[-1]) == 6  # hash suffix
    # Record without prompt_name/prompt_version
    rec2 = PromptResultRecord(
        identifier="doc_456",
        prompt="Do something",
        result="Done",
        creation_meta=CreationMeta(created_at=datetime.datetime(2024, 11, 26, 15, 0, 0, tzinfo=datetime.UTC)),
    )
    uid2 = rec2.record_uid
    assert uid2.startswith("doc_456_20241126-150000_")
    # Different content = different hash
    rec3 = PromptResultRecord(
        identifier="sample_123",
        prompt_name="code_summary",
        prompt_version="v1",
        prompt="Summarize this",
        result="Different summary",
        creation_meta=CreationMeta(created_at=datetime.datetime(2024, 11, 26, 14, 30, 22, tzinfo=datetime.UTC)),
    )
    assert rec.record_uid != rec3.record_uid  # same metadata, different content


def test_count_entries(db: PromptResultDB) -> None:
    # empty db
    assert db.count_entries() == 0
    assert db.count_entries(identifier="nonexistent") == 0
    # insert test data
    db.store(identifier="id1", group="g1", prompt_name="pn1", prompt_version="v1", prompt="p", result="r")
    db.store(identifier="id1", group="g1", prompt_name="pn1", prompt_version="v2", prompt="p", result="r")
    db.store(identifier="id1", group="g2", prompt_name="pn2", prompt_version="v1", prompt="p", result="r")
    db.store(identifier="id2", group="g1", prompt_name="pn1", prompt_version="v1", prompt="p", result="r")
    db.store(identifier="id3", prompt="p", result="r")  # no group or prompt_name
    # total count
    assert db.count_entries() == 5
    # filter by identifier
    assert db.count_entries(identifier="id1") == 3
    assert db.count_entries(identifier="id2") == 1
    assert db.count_entries(identifier="id3") == 1
    assert db.count_entries(identifier="nonexistent") == 0
    # filter by group
    assert db.count_entries(group="g1") == 3
    assert db.count_entries(group="g2") == 1
    assert db.count_entries(group="nonexistent") == 0
    # filter by prompt_name
    assert db.count_entries(prompt_name="pn1") == 3
    assert db.count_entries(prompt_name="pn2") == 1
    assert db.count_entries(prompt_name="nonexistent") == 0
    # filter by prompt_name + prompt_version
    assert db.count_entries(prompt_name="pn1", prompt_version="v1") == 2
    assert db.count_entries(prompt_name="pn1", prompt_version="v2") == 1
    # combined filters
    assert db.count_entries(identifier="id1", group="g1") == 2
    assert db.count_entries(identifier="id1", prompt_name="pn1") == 2
    assert db.count_entries(identifier="id1", group="g1", prompt_name="pn1", prompt_version="v1") == 1
    assert db.count_entries(group="g1", prompt_name="pn1") == 3
    # error case: prompt_version without prompt_name
    with pytest.raises(ValueError):
        db.count_entries(prompt_version="v1")
    # list filters: identifier
    assert db.count_entries(identifier=["id1", "id2"]) == 4
    assert db.count_entries(identifier=["id1", "id3"]) == 4
    assert db.count_entries(identifier=["id2", "id3"]) == 2
    assert db.count_entries(identifier=["nonexistent1", "nonexistent2"]) == 0
    # list filters: group
    assert db.count_entries(group=["g1", "g2"]) == 4
    assert db.count_entries(group=["g1"]) == 3
    # list filters: prompt_name
    assert db.count_entries(prompt_name=["pn1", "pn2"]) == 4
    assert db.count_entries(prompt_name=["pn1", "nonexistent"]) == 3
    # list filters: prompt_version (with prompt_name)
    assert db.count_entries(prompt_name="pn1", prompt_version=["v1", "v2"]) == 3
    assert db.count_entries(prompt_name=["pn1", "pn2"], prompt_version=["v1"]) == 3
    # combined list filters
    assert db.count_entries(identifier=["id1", "id2"], group=["g1"]) == 3
    assert db.count_entries(identifier=["id1"], prompt_name=["pn1", "pn2"]) == 3
    # mixed: single string + list
    assert db.count_entries(identifier="id1", group=["g1", "g2"]) == 3
    assert db.count_entries(identifier=["id1", "id2"], group="g1") == 3
    assert db.count_entries(identifier=["id1", "id2"], prompt_name="pn1") == 3
    assert db.count_entries(group="g1", prompt_name=["pn1", "pn2"]) == 3
    # empty list = no filter
    assert db.count_entries(identifier=[]) == 5
    assert db.count_entries(identifier=[], group=[]) == 5
    assert db.count_entries(identifier="id1", group=[]) == 3
    # breakdown=True with single list filter
    breakdown = db.count_entries(identifier=["id1", "id2", "id3"], breakdown=True)
    assert breakdown == {("id1",): 3, ("id2",): 1, ("id3",): 1}
    breakdown = db.count_entries(group=["g1", "g2"], breakdown=True)
    assert breakdown == {("g1",): 3, ("g2",): 1}
    breakdown = db.count_entries(prompt_name=["pn1", "pn2"], breakdown=True)
    assert breakdown == {("pn1",): 3, ("pn2",): 1}
    # breakdown=True with multiple list filters
    breakdown = db.count_entries(identifier=["id1", "id2"], group=["g1", "g2"], breakdown=True)
    assert breakdown == {("id1", "g1"): 2, ("id1", "g2"): 1, ("id2", "g1"): 1}
    breakdown = db.count_entries(identifier=["id1"], prompt_name=["pn1", "pn2"], breakdown=True)
    assert breakdown == {("id1", "pn1"): 2, ("id1", "pn2"): 1}
    # breakdown=True with mixed single string + list (only list columns in tuple)
    breakdown = db.count_entries(identifier="id1", group=["g1", "g2"], breakdown=True)
    assert breakdown == {("g1",): 2, ("g2",): 1}
    # breakdown=True with no list filters returns int (falls back to total)
    assert db.count_entries(identifier="id1", breakdown=True) == 3
    # breakdown=True with empty results
    breakdown = db.count_entries(identifier=["nonexistent"], breakdown=True)
    assert breakdown == {}
    # breakdown=False still works (default)
    assert db.count_entries(identifier=["id1", "id2"], breakdown=False) == 4
