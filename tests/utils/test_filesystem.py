import pathlib
import sys
import types

import pytest

import pyine.utils.filesystem as fs


def test_get_project_root_path_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setenv(fs.PROJECT_ROOT_ENV_VAR, str(tmp_path))
    assert fs.get_project_root_path() == tmp_path.resolve()


def test_get_project_root_path_defaults_to_package_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    fake_pkg = tmp_path / "pyine_pkg" / "__init__.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("", encoding="utf-8")
    monkeypatch.delenv(fs.PROJECT_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(fs.pyine, "__file__", str(fake_pkg), raising=True)
    expected_root = fake_pkg.parents[1].resolve()
    assert fs.get_project_root_path() == expected_root


def test_get_data_root_path_env_and_default(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    # env override
    custom = tmp_path / "data_root"
    monkeypatch.setenv(fs.DATA_ROOT_ENV_VAR, str(custom))
    assert fs.get_data_root_path() == custom.resolve()

    # default branch uses project_root/data
    monkeypatch.delenv(fs.DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(fs, "get_project_root_path", lambda: tmp_path)
    assert fs.get_data_root_path() == tmp_path / "data"


def test_get_tmp_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    # 1) TMP_DIR takes precedence
    d1 = tmp_path / "tmp_dir_precedence"
    d1.mkdir()
    monkeypatch.setenv("TMP_DIR", str(d1))
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    assert str(fs.get_tmp_dir()).startswith(str(d1))

    # 2) TMPDIR when TMP_DIR is not set
    monkeypatch.delenv("TMP_DIR", raising=False)
    d2 = tmp_path / "tmpdir_fallback"
    d2.mkdir()
    monkeypatch.setenv("TMPDIR", str(d2))
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    assert str(fs.get_tmp_dir()).startswith(str(d2))

    # 3) SLURM_TMPDIR when neither TMP_DIR nor TMPDIR is set
    monkeypatch.delenv("TMP_DIR", raising=False)
    monkeypatch.delenv("TMPDIR", raising=False)
    d3 = tmp_path / "slurm_tmpdir_fallback"
    d3.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(d3))
    assert str(fs.get_tmp_dir()).startswith(str(d3))

    # 4) default branch when none are set
    monkeypatch.delenv("TMP_DIR", raising=False)
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    default_dir = fs.get_tmp_dir()
    assert default_dir.is_dir()

    # confirm writability by creating a small file
    probe = default_dir / "probe.txt"
    probe.unlink(missing_ok=True)  # in case it already exists
    probe.write_text("ok", encoding="utf-8")
    assert probe.exists()


def test_get_logs_root_path_env_and_default(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    custom = tmp_path / "logs_root"
    monkeypatch.setenv(fs.LOGS_ROOT_ENV_VAR, str(custom))
    assert fs.get_logs_root_path() == custom.resolve()
    monkeypatch.delenv(fs.LOGS_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(fs, "get_project_root_path", lambda: tmp_path)
    assert fs.get_logs_root_path() == tmp_path / "logs"


def test_get_username(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fs.getpass, "getuser", lambda: "tester", raising=True)
    assert fs.get_username() == "tester"
    monkeypatch.setattr(fs.getpass, "getuser", lambda: "", raising=True)
    assert fs.get_username() == "unknown"


def test_find_dotenv_file(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    sub = proj / "app" / "sub"
    sub.mkdir(parents=True)
    proj_env = proj / ".env"
    sub_env = sub / ".env"
    override_env = tmp_path / ".env.override"
    proj_env.write_text("FROM=proj\n")
    sub_env.write_text("FROM=sub\n")
    override_env.write_text("FROM=override\n")
    monkeypatch.chdir(sub)

    # 1) DOTENV_PATH override takes precedence
    monkeypatch.setenv("DOTENV_PATH", str(override_env))
    assert fs.find_dotenv_file() == override_env.resolve()
    monkeypatch.delenv("DOTENV_PATH", raising=False)

    # 2) if python-dotenv is available and start == CWD, delegate to it
    dummy = types.ModuleType("dotenv")

    def _fake_find_dotenv(filename=".env", usecwd=True, raise_error_if_not_found=False):
        p = pathlib.Path.cwd() / filename
        return str(p) if p.exists() else ""

    dummy.find_dotenv = _fake_find_dotenv
    monkeypatch.setitem(sys.modules, "dotenv", dummy)
    # start defaults to CWD -> uses python-dotenv path and finds sub/.env
    assert fs.find_dotenv_file() == sub_env.resolve()

    # 3) fallback: no python-dotenv, walk upward to find nearest .env
    monkeypatch.delitem(sys.modules, "dotenv", raising=False)
    sub_env.unlink()  # force search to climb to /proj/.env
    assert fs.find_dotenv_file(start=sub) == proj_env.resolve()

    # 4) no .env anywhere: return None
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    assert fs.find_dotenv_file(start=empty) is None


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


def test_check_output_path_overwrite_requires_force(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError):
        fs.check_output_path_overwrite(out)
    assert out.exists()


def test_check_output_path_overwrite_deletes_when_forced(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "x.txt").write_text("x", encoding="utf-8")
    fs.check_output_path_overwrite(out, force=True)
    assert not out.exists()
