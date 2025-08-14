import pathlib

import pytest

import pyine.utils.filesystem as fs


def test_get_project_root_path_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setenv("PROJECT_ROOT_PATH", str(tmp_path))
    assert fs.get_project_root_path() == tmp_path.resolve()


def test_get_data_root_path_env_and_default(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    # env override
    custom = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT_PATH", str(custom))
    assert fs.get_data_root_path() == custom.resolve()

    # default branch uses project_root/data
    monkeypatch.delenv("DATA_ROOT_PATH", raising=False)
    monkeypatch.setattr(fs, "get_project_root_path", lambda: tmp_path)
    assert fs.get_data_root_path() == tmp_path / "data"


def test_get_relative_path_to_root(tmp_path: pathlib.Path):
    proj = tmp_path
    module_path = proj / "src" / "pkg" / "mod.py"
    rel = fs.get_relative_path_to_root(module_path, proj)
    assert rel == str(pathlib.Path("src") / "pkg" / "mod.py")


def test_get_path_size_file_and_dir(tmp_path: pathlib.Path):
    f1 = tmp_path / "a.txt"
    f1.write_text("hello", encoding="utf-8")  # 5 bytes
    sub = tmp_path / "sub"
    sub.mkdir()
    f2 = sub / "b.bin"
    f2.write_bytes(b"1234567890")  # 10 bytes

    assert fs.get_path_size(f1) == 5
    total = fs.get_path_size(tmp_path)
    assert total >= 15  # at least the two files' bytes


def test_get_human_readable_size_high_units():
    # 1024**7 should fall through to 'Ei' per implementation
    s = fs.get_human_readable_size(1024**7)
    assert s.endswith("EiB")
    assert s.startswith("1024.0")


def test_slugify():
    assert fs.slugify("Hello World") == "hello-world"
    assert fs.slugify("Hello_World") == "hello-world"
    assert fs.slugify("Hello  World") == "hello-world"
    assert fs.slugify("Hello--World") == "hello-world"
    assert fs.slugify("Hello World!") == "hello-world"
    assert fs.slugify("  Hello World  ") == "hello-world"
    assert fs.slugify("UPPER_CASE") == "upper-case"
    assert fs.slugify("special@#characters") == "specialcharacters"


def test_check_output_path_overwrite_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    out = tmp_path / "out"
    out.mkdir()
    # abort branch
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    with pytest.raises(SystemExit):
        fs.check_output_path_overwrite(out)
    assert out.exists()
    # confirm deletion
    (out / "x.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    fs.check_output_path_overwrite(out)
    assert not out.exists()
