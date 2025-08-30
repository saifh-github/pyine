import threading
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import pytest

from pyine.prompts.result_db import PromptResultDB


@pytest.fixture()
def db(tmp_path: Path) -> PromptResultDB:
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

    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(insert_one, i) for i in range(n)]
        wait(futures)
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
