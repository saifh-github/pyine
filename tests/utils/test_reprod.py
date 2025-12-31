import builtins
import pathlib
import typing

import dotenv
import pytest

import pyine.utils.reprod as reprod


def test_get_framework_version_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMeta:
        class PackageNotFoundError(Exception):
            pass

        def version(
            self,
            name: str,
        ) -> None:  # noqa: ARG002
            raise FakeMeta.PackageNotFoundError()

    monkeypatch.setattr(reprod.importlib, "metadata", FakeMeta(), raising=True)
    assert reprod.get_framework_version().startswith("0.0.0-unknown")


def test_get_git_repo_regular() -> None:
    # can't really test anything more specific than types by default
    hash = reprod.get_git_revision_hash()
    assert isinstance(hash, str) and len(hash)
    is_clean = reprod.is_git_repo_clean(include_untracked=True)
    assert isinstance(is_clean, bool)


def test_get_git_revision_hash_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # simulate ImportError for git module
    real_import = builtins.__import__

    def fake_import(
        name: str,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        if name == "git":
            raise ImportError("no git")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert reprod.get_git_revision_hash() == "git-import-error"
    assert reprod.is_git_repo_clean() is False


def test_get_git_revision_hash_invalid_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # simulate git module present but repo invalid
    real_import = builtins.__import__

    class FakeGit:
        class InvalidGitRepositoryError(Exception):
            pass

        class Repo:
            def __init__(
                self,
                *args: typing.Any,
                **kwargs: typing.Any,
            ) -> None:  # noqa: ARG002
                raise FakeGit.InvalidGitRepositoryError()

    def fake_import(
        name: str,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        if name == "git":
            return FakeGit
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert reprod.get_git_revision_hash() == "git-revision-unknown"
    assert reprod.is_git_repo_clean() is False


def test_get_installed_packages_fallback_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # force importlib.metadata.distributions to raise ImportError to try pip branch
    real_import = builtins.__import__

    class FakeMeta:
        def distributions(self) -> typing.Any:
            raise ImportError("boom")

    monkeypatch.setattr(reprod.importlib, "metadata", FakeMeta(), raising=True)

    class FakePip:
        def get_installed_distributions(self) -> typing.Any:  # noqa: RUF100
            raise AttributeError("no dist")

    def fake_import(
        name: str,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        if name == "pip":
            return FakePip()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert reprod.get_installed_packages() == []


def test_compute_hash_file_and_dir(
    tmp_path: pathlib.Path,
) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    h_file = reprod.compute_hash(f, algorithm="md5")
    assert len(h_file) == 32

    sub = tmp_path / "d"
    sub.mkdir()
    (sub / "b.txt").write_text("world", encoding="utf-8")
    (sub / "c.bin").write_bytes(b"\x00\x01\x02")
    h_dir = reprod.compute_hash(sub, algorithm="sha1")
    assert len(h_dir) == 40

    # error path w/ raise_on_error=False
    bad = sub / "bad"
    bad.write_bytes(b"x")
    bad.chmod(0)
    try:
        h_dir2 = reprod.compute_hash(sub, algorithm="sha256", raise_on_error=False)
        assert len(h_dir2) == 64
    finally:
        bad.chmod(0o644)


def test_compute_hash_of_bytes_array() -> None:
    b = b"\x00\x01\x02"
    h = reprod.compute_hash(b, algorithm="md5")
    assert len(h) == 32


def test_get_params_hash_stable_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Obj:
        pass

    o = Obj()
    # ensure that including repr with memory addresses yields stable hash due to cleaning
    s1 = reprod.get_params_hash(o)
    s2 = reprod.get_params_hash(o)
    assert s1 == s2


def test_get_versioned_cache_hash_includes_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reprod, "get_framework_version", lambda: "1.0.0")
    versioned_hash = reprod.get_versioned_cache_hash("param1", "param2")
    non_versioned_hash = reprod.get_params_hash("param1", "param2")
    assert versioned_hash != non_versioned_hash


def test_get_versioned_cache_hash_version_change_invalidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reprod, "get_framework_version", lambda: "1.0.0")
    hash_v1 = reprod.get_versioned_cache_hash("param1", "param2")
    monkeypatch.setattr(reprod, "get_framework_version", lambda: "2.0.0")
    hash_v2 = reprod.get_versioned_cache_hash("param1", "param2")
    assert hash_v1 != hash_v2


def test_get_versioned_cache_hash_include_version_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versioned_hash = reprod.get_versioned_cache_hash("param1", include_version=False)
    non_versioned_hash = reprod.get_params_hash("param1")
    assert versioned_hash == non_versioned_hash


def test_get_versioned_cache_hash_deterministic() -> None:
    hash1 = reprod.get_versioned_cache_hash("a", "b", key="value")
    hash2 = reprod.get_versioned_cache_hash("a", "b", key="value")
    assert hash1 == hash2


def test_entrypoint_setup_first_and_second_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # patch logging and pydantic loader to avoid side effects by patching real modules
    calls = {"setup_logging": 0}

    import pyine.utils.logging as real_log_mod

    def fake_setup_logging(
        level: int,
        log_to_file: bool,
        log_path: typing.Any,
    ) -> None:
        calls["setup_logging"] += 1

    monkeypatch.setattr(real_log_mod, "setup_logging", fake_setup_logging, raising=True)
    # ensure dotenv.load_dotenv is a no-op
    monkeypatch.setattr(dotenv, "load_dotenv", lambda x: None, raising=True)
    # reset sentinel so entrypoint_setup runs init branch
    monkeypatch.delattr(reprod.entrypoint_setup, "_executed", raising=False)
    # first call initializes
    reprod.entrypoint_setup()
    # second call should not call setup again
    reprod.entrypoint_setup()
    assert calls["setup_logging"] == 1
