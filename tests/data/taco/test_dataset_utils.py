import datetime
import pathlib

import pytest

import pyine.data.taco.dataset_utils as tdu
import pyine.utils.filesystem as fs_utils


def test_get_latest_repackaged_dataset_path_no_folders(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    # point data root to tmp
    monkeypatch.setattr(fs_utils, "get_data_root_path", lambda: tmp_path)
    rpkg_root = tmp_path / "TACO" / "repackaged"
    rpkg_root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        _ = tdu.get_latest_repackaged_dataset_path()


def test_get_latest_repackaged_dataset_path_selects_latest(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fs_utils, "get_data_root_path", lambda: tmp_path)
    rpkg_root = tmp_path / "TACO" / "repackaged"
    rpkg_root.mkdir(parents=True, exist_ok=True)
    # create dated versioned folders
    (rpkg_root / "2024-12-31-v01").mkdir()
    (rpkg_root / "2025-01-02-v02").mkdir()
    (rpkg_root / "2025-01-02-v03").mkdir()
    latest = tdu.get_latest_repackaged_dataset_path()
    assert latest == rpkg_root / "2025-01-02-v03"


def test_get_new_repackaged_dataset_path_uses_today_and_version(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(fs_utils, "get_data_root_path", lambda: tmp_path)

    class FakeDate(datetime.date):
        @classmethod
        def today(cls):
            return cls(2025, 1, 2)

    # patch the date class used by module's datetime
    monkeypatch.setattr(tdu.datetime, "date", FakeDate)
    path = tdu.get_new_repackaged_dataset_path(dataset_version=7)
    assert path == tmp_path / "TACO" / "repackaged" / "2025-01-02-v07"
