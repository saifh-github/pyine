import pathlib
import sys
import types

import pytest

import pyine.utils.filesystem as fs


def test_get_project_root_path_env_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
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


def test_get_data_root_path_env_and_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    # env override
    custom = tmp_path / "data_root"
    monkeypatch.setenv(fs.DATA_ROOT_ENV_VAR, str(custom))
    assert fs.get_data_root_path() == custom.resolve()

    # default branch uses project_root/data
    monkeypatch.delenv(fs.DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(fs, "get_project_root_path", lambda: tmp_path)
    assert fs.get_data_root_path() == tmp_path / "data"


def test_get_tmp_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
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


def test_get_data_cache_path(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    custom = tmp_path / "cache"
    monkeypatch.setenv(fs.CACHE_ROOT_ENV_VAR, str(custom))
    data_cache_dir_path = fs.get_data_cache_path()
    assert data_cache_dir_path == custom.resolve()
    assert data_cache_dir_path.exists() and data_cache_dir_path.is_dir()


def test_get_username(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fs.getpass, "getuser", lambda: "tester", raising=True)
    assert fs.get_username() == "tester"
    monkeypatch.setattr(fs.getpass, "getuser", lambda: "", raising=True)
    assert fs.get_username() == "unknown"


def test_find_dotenv_file(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def _fake_find_dotenv(
        filename: str = ".env",
        usecwd: bool = True,
        raise_error_if_not_found: bool = False,
    ) -> str:
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
    # note: this test only works if there's no real .env above tmp_path (e.g. the project's .env)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    real_env_above = any((p / ".env").is_file() for p in tmp_path.parents)
    if real_env_above:
        # can't test "no .env found" case when running from within a project with .env
        result = fs.find_dotenv_file(start=empty)
        assert result is not None  # will find the real project's .env
    else:
        assert fs.find_dotenv_file(start=empty) is None


def test_get_relative_path_to_root(
    tmp_path: pathlib.Path,
) -> None:
    proj = tmp_path
    module_path = proj / "src" / "pkg" / "mod.py"
    rel = fs.get_relative_path_to_root(module_path, proj)
    assert rel == str(pathlib.Path("src") / "pkg" / "mod.py")


def test_get_path_size_file_and_dir(
    tmp_path: pathlib.Path,
) -> None:
    f1 = tmp_path / "a.txt"
    f1.write_text("hello", encoding="utf-8")  # 5 bytes
    sub = tmp_path / "sub"
    sub.mkdir()
    f2 = sub / "b.bin"
    f2.write_bytes(b"1234567890")  # 10 bytes

    assert fs.get_path_size(f1) == 5
    total = fs.get_path_size(tmp_path)
    assert total >= 15  # at least the two files' bytes


def test_get_human_readable_size_high_units() -> None:
    # 1024**7 should fall through to 'Ei' per implementation
    s = fs.get_human_readable_size(1024**7)
    assert s.endswith("EiB")
    assert s.startswith("1024.0")


def test_slugify() -> None:
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


def test_check_output_path_overwrite_deletes_when_forced(
    tmp_path: pathlib.Path,
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "x.txt").write_text("x", encoding="utf-8")
    fs.check_output_path_overwrite(out, force=True)
    assert not out.exists()


class TestNormalizePathTuple:
    def test_none_returns_none(self) -> None:
        assert fs.normalize_path_tuple(None) is None

    def test_single_string(self) -> None:
        result = fs.normalize_path_tuple("/some/path")
        assert result == (pathlib.Path("/some/path"),)

    def test_pathlike(self) -> None:
        result = fs.normalize_path_tuple(pathlib.Path("/a"))
        assert result == (pathlib.Path("/a"),)

    def test_list_of_strings(self) -> None:
        result = fs.normalize_path_tuple(["/a", "/b"])
        assert result == (pathlib.Path("/a"), pathlib.Path("/b"))

    def test_empty_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="empty sequence"):
            fs.normalize_path_tuple([])

    def test_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="bytes"):
            fs.normalize_path_tuple(b"/path")

    def test_bytes_inside_sequence_rejected(self) -> None:
        with pytest.raises(ValueError, match="bytes.*element"):
            fs.normalize_path_tuple([b"/path"])

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(ValueError, match="got int"):
            fs.normalize_path_tuple(42)

    def test_field_name_in_error(self) -> None:
        with pytest.raises(ValueError, match="my_field"):
            fs.normalize_path_tuple(42, field_name="my_field")


class TestSharedFilesystemDetection:
    def test_assumes_shared_on_non_linux(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        # non-Linux should always return True (assume shared)
        assert fs.is_path_on_shared_filesystem(tmp_path) is True

    def test_assumes_shared_on_proc_mounts_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        # mock _parse_proc_mounts to raise an exception
        monkeypatch.setattr(fs, "_parse_proc_mounts", lambda: (_ for _ in ()).throw(Exception("test")))
        assert fs.is_path_on_shared_filesystem(tmp_path) is True

    def test_mount_matching_path_boundary(self) -> None:
        # /data doesn't match /data2
        mounts = [("/data", "nfs", "server:/data"), ("/data2", "ext4", "/dev/sda1")]
        assert fs._find_mount_for_path(pathlib.Path("/data2/foo"), mounts) == "ext4"
        assert fs._find_mount_for_path(pathlib.Path("/data/foo"), mounts) == "nfs"

    def test_longest_mount_match(self) -> None:
        mounts = [
            ("/", "ext4", "/dev/sda1"),
            ("/home", "nfs", "server:/home"),
            ("/home/user", "ext4", "/dev/sdb1"),
        ]
        # /home/user/foo should match /home/user (most specific)
        assert fs._find_mount_for_path(pathlib.Path("/home/user/foo"), mounts) == "ext4"
        # /home/other should match /home
        assert fs._find_mount_for_path(pathlib.Path("/home/other/bar"), mounts) == "nfs"

    def test_unescape_mount_path(self) -> None:
        # test octal sequence unescaping (e.g., \040 = space)
        assert fs._unescape_mount_path("/path\\040with\\040spaces") == "/path with spaces"
        assert fs._unescape_mount_path("/normal/path") == "/normal/path"
        assert fs._unescape_mount_path("/tab\\011here") == "/tab\there"

    def test_shared_fs_types_detected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        # mock mounts to return NFS for tmp_path
        mock_mounts = [(str(tmp_path), "nfs4", "server:/export")]
        monkeypatch.setattr(fs, "_parse_proc_mounts", lambda: mock_mounts)
        assert fs.is_path_on_shared_filesystem(tmp_path) is True

    def test_local_fs_types_detected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        # mock mounts to return ext4 for tmp_path
        mock_mounts = [(str(tmp_path), "ext4", "/dev/sda1")]
        monkeypatch.setattr(fs, "_parse_proc_mounts", lambda: mock_mounts)
        assert fs.is_path_on_shared_filesystem(tmp_path) is False
