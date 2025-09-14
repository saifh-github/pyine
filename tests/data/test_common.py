import pathlib

import pytest

import pyine.data.common
import pyine.utils.filesystem


def _mk_dirs(tmp_path: pathlib.Path, kind: str, source: str, names: list[str]) -> list[pathlib.Path]:
    """Helper to create dataset directories under tmp data root."""
    base = tmp_path / kind / source
    base.mkdir(parents=True, exist_ok=True)
    out = []
    for n in names:
        p = base / n
        p.mkdir(parents=False, exist_ok=True)
        out.append(p)
    return out


@pytest.mark.parametrize(
    "name,expected",
    [
        ("alpha.2025-09-10.lmdb", (2025, 9, 10)),
        ("x.0001-01-01.lmdb", (1, 1, 1)),
        ("no-date.lmdb", None),
        ("weird.2025-13-40.lmdb", None),  # invalid month/day
        ("prefix.2025-09-10.lmdb.extra", None),
    ],
)
def test_parse_date_from_dirname(name: str, expected):
    fn = pyine.data.common._parse_date_from_dirname
    d = fn(name)
    if expected is None:
        assert d is None
    else:
        y, m, day = expected
        assert d.year == y and d.month == m and d.day == day


@pytest.mark.parametrize("kind", ["traces", "deltas"])
def test_resolve_latest_dataset_path_simple(monkeypatch, tmp_path, kind):
    # create three dataset dirs with different dates and tags
    monkeypatch.setattr(pyine.utils.filesystem, "get_data_root_path", lambda: tmp_path)
    src = "TACO"
    _mk_dirs(
        tmp_path,
        kind,
        src,
        [
            "alpha.2024-12-31.lmdb",
            "beta.2025-01-01.lmdb",
            "zzz.2025-01-03.lmdb",
        ],
    )
    latest = pyine.data.common.resolve_latest_dataset_path(kind=kind, source_dataset_name=src)
    assert latest.name == "zzz.2025-01-03.lmdb"


@pytest.mark.parametrize("kind", ["traces", "deltas"])
def test_resolve_latest_dataset_path_with_filter_regex(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(pyine.utils.filesystem, "get_data_root_path", lambda: tmp_path)
    src = "TACO"
    _mk_dirs(
        tmp_path,
        kind,
        src,
        [
            "tagA.2025-01-01.lmdb",
            "tagB.2025-01-03.lmdb",
        ],
    )
    latest = pyine.data.common.resolve_latest_dataset_path(
        kind=kind,
        source_dataset_name=src,
        filter_rule=r"^tagA\.",
    )
    assert latest.name == "tagA.2025-01-01.lmdb"


@pytest.mark.parametrize("kind", ["traces", "deltas"])
def test_resolve_latest_dataset_path_filter_eliminates_all(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(pyine.utils.filesystem, "get_data_root_path", lambda: tmp_path)
    src = "TACO"
    _mk_dirs(
        tmp_path,
        kind,
        src,
        [
            "tagA.2025-01-01.lmdb",
            "tagB.2025-01-03.lmdb",
        ],
    )
    with pytest.raises(FileNotFoundError):
        pyine.data.common.resolve_latest_dataset_path(
            kind=kind,
            source_dataset_name=src,
            filter_rule=r"^nope",
        )


@pytest.mark.parametrize("kind", ["traces", "deltas"])
def test_resolve_latest_dataset_path_invalid_regex(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(pyine.utils.filesystem, "get_data_root_path", lambda: tmp_path)
    src = "TACO"
    _mk_dirs(tmp_path, kind, src, ["a.2025-01-01.lmdb"])
    with pytest.raises(ValueError):
        pyine.data.common.resolve_latest_dataset_path(
            kind=kind,
            source_dataset_name=src,
            filter_rule=r"[unclosed",
        )


def test_resolve_latest_dataset_path_invalid_kind(tmp_path):
    # no need to create dirs; we validate kind first in code
    with pytest.raises(ValueError):
        pyine.data.common.resolve_latest_dataset_path(kind="unknown", source_dataset_name="TACO")


@pytest.mark.parametrize("kind", ["traces", "deltas"])
def test_resolve_latest_dataset_path_invalid_base_dir(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(pyine.utils.filesystem, "get_data_root_path", lambda: tmp_path)
    # do not create required directories so the base dir is invalid
    with pytest.raises(FileNotFoundError):
        pyine.data.common.resolve_latest_dataset_path(kind=kind, source_dataset_name="TACO")


@pytest.mark.parametrize("kind", ["traces", "deltas"])
def test_resolve_latest_dataset_path_no_valid_date_suffix(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(pyine.utils.filesystem, "get_data_root_path", lambda: tmp_path)
    src = "TACO"
    # name matches numeric pattern but encodes an invalid date so parser will drop it -> ValueError
    _mk_dirs(tmp_path, kind, src, ["foo.2025-13-40.lmdb"])
    with pytest.raises(ValueError):
        pyine.data.common.resolve_latest_dataset_path(kind=kind, source_dataset_name=src)


@pytest.mark.parametrize("kind", ["traces", "deltas"])
def test_resolve_matching_dataset_paths_glob_default(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(pyine.utils.filesystem, "get_data_root_path", lambda: tmp_path)
    src = "TACO"
    _mk_dirs(
        tmp_path,
        kind,
        src,
        [
            "alpha.2025-08-01.lmdb",
            "beta.2025-07-01.lmdb",
            "gamma.2025-08-05.lmdb",
            "undated.lmdb",
        ],
    )
    matched = pyine.data.common.resolve_matching_dataset_paths(
        kind=kind,
        source_dataset_name=src,
        pattern="*.2025-08-*.lmdb",  # shell-style glob
    )
    assert [p.name for p in matched] == ["alpha.2025-08-01.lmdb", "gamma.2025-08-05.lmdb"]


@pytest.mark.parametrize("kind", ["traces", "deltas"])
def test_resolve_matching_dataset_paths_regex_prefix(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(pyine.utils.filesystem, "get_data_root_path", lambda: tmp_path)
    src = "TACO"
    _mk_dirs(
        tmp_path,
        kind,
        src,
        [
            "tag1.2025-01-01.lmdb",
            "other.2025-01-02.lmdb",
            "tag2.2025-01-03.lmdb",
        ],
    )
    matched = pyine.data.common.resolve_matching_dataset_paths(
        kind=kind,
        source_dataset_name=src,
        pattern="re:^tag",
    )
    assert [p.name for p in matched] == ["tag1.2025-01-01.lmdb", "tag2.2025-01-03.lmdb"]


@pytest.mark.parametrize("kind", ["traces", "deltas"])
def test_resolve_matching_dataset_paths_is_regex_param(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(pyine.utils.filesystem, "get_data_root_path", lambda: tmp_path)
    src = "TACO"
    _mk_dirs(
        tmp_path,
        kind,
        src,
        [
            "a.2025-01-01.lmdb",
            "b.2025-01-15.lmdb",
            "c.2025-02-01.lmdb",
        ],
    )
    matched = pyine.data.common.resolve_matching_dataset_paths(
        kind=kind,
        source_dataset_name=src,
        pattern=r"\.2025-01-",
        pattern_is_regex=True,
    )
    assert [p.name for p in matched] == ["a.2025-01-01.lmdb", "b.2025-01-15.lmdb"]


@pytest.mark.parametrize("kind", ["traces", "deltas"])
def test_resolve_matching_dataset_paths_glob_prefix_forces_glob(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(pyine.utils.filesystem, "get_data_root_path", lambda: tmp_path)
    src = "TACO"
    _mk_dirs(
        tmp_path,
        kind,
        src,
        [
            "aa.2025-12-01.lmdb",
            "bb.2025-12-02.lmdb",
            "cc.2025-11-30.lmdb",
        ],
    )
    matched = pyine.data.common.resolve_matching_dataset_paths(
        kind=kind,
        source_dataset_name=src,
        pattern="glob:*.2025-12-*.lmdb",
    )
    assert [p.name for p in matched] == ["aa.2025-12-01.lmdb", "bb.2025-12-02.lmdb"]


def test_resolve_matching_dataset_paths_invalid_kind():
    with pytest.raises(ValueError):
        pyine.data.common.resolve_matching_dataset_paths(
            kind="weird",
            source_dataset_name="TACO",
            pattern="*",
        )


@pytest.mark.parametrize("kind", ["traces", "deltas"])
def test_resolve_matching_dataset_paths_missing_base_dir(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(pyine.utils.filesystem, "get_data_root_path", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        pyine.data.common.resolve_matching_dataset_paths(
            kind=kind,
            source_dataset_name="TACO",
            pattern="*",
        )


@pytest.mark.parametrize("kind", ["traces", "deltas"])
def test_resolve_matching_dataset_paths_no_match(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(pyine.utils.filesystem, "get_data_root_path", lambda: tmp_path)
    src = "TACO"
    _mk_dirs(tmp_path, kind, src, ["a.2025-01-01.lmdb"])
    assert not pyine.data.common.resolve_matching_dataset_paths(
        kind=kind,
        source_dataset_name=src,
        pattern="no-match-*.lmdb",
    )


@pytest.mark.parametrize("kind", ["traces", "deltas"])
def test_resolve_matching_dataset_paths_empty_pattern(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(pyine.utils.filesystem, "get_data_root_path", lambda: tmp_path)
    src = "TACO"
    _mk_dirs(tmp_path, kind, src, ["a.2025-01-01.lmdb"])
    with pytest.raises(ValueError):
        pyine.data.common.resolve_matching_dataset_paths(
            kind=kind,
            source_dataset_name=src,
            pattern="",
        )


@pytest.mark.parametrize("kind", ["traces", "deltas"])
def test_resolve_matching_dataset_paths_invalid_regex(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(pyine.utils.filesystem, "get_data_root_path", lambda: tmp_path)
    src = "TACO"
    _mk_dirs(tmp_path, kind, src, ["a.2025-01-01.lmdb"])
    with pytest.raises(ValueError):
        pyine.data.common.resolve_matching_dataset_paths(
            kind=kind,
            source_dataset_name=src,
            pattern="re:[bad",
        )
