import hashlib
import os
import pathlib
import sys
import types

import pytest

import pyine.utils.reprod as reprod


def test_get_framework_version_package_missing(monkeypatch: pytest.MonkeyPatch):
    class PNF(Exception):
        pass

    class FakeMeta:
        class PackageNotFoundError(Exception):
            pass

        def version(self, name):  # noqa: ARG002
            raise FakeMeta.PackageNotFoundError()

    monkeypatch.setattr(reprod.importlib, "metadata", FakeMeta(), raising=True)
    assert reprod.get_framework_version().startswith("0.0.0-unknown")


def test_get_git_revision_hash_import_error(monkeypatch: pytest.MonkeyPatch):
    # Simulate ImportError for git module
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "git":
            raise ImportError("no git")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert reprod.get_git_revision_hash() == "git-import-error"


def test_get_git_revision_hash_invalid_repo(monkeypatch: pytest.MonkeyPatch):
    # Simulate git module present but repo invalid
    import builtins

    class FakeGit:
        class InvalidGitRepositoryError(Exception):
            pass

        class Repo:
            def __init__(self, *args, **kwargs):  # noqa: ARG002
                raise FakeGit.InvalidGitRepositoryError()

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "git":
            return FakeGit
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert reprod.get_git_revision_hash() == "git-revision-unknown"


def test_get_installed_packages_fallback_empty(monkeypatch: pytest.MonkeyPatch):
    # Force importlib.metadata.distributions to raise ImportError to try pip branch
    class FakeMeta:
        def distributions(self):
            raise ImportError("boom")

    monkeypatch.setattr(reprod.importlib, "metadata", FakeMeta(), raising=True)

    import builtins as _bi

    real_import = _bi.__import__

    class FakePip:
        def get_installed_distributions(self):  # noqa: RUF100
            raise AttributeError("no dist")

    def fake_import(name, *args, **kwargs):
        if name == "pip":
            return FakePip()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_bi, "__import__", fake_import)
    assert reprod.get_installed_packages() == []


def test_compute_hash_file_and_dir(tmp_path: pathlib.Path):
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


def test_get_params_hash_stable_addresses(monkeypatch: pytest.MonkeyPatch):
    class Obj:
        pass

    o = Obj()
    # Ensure that including repr with memory addresses yields stable hash due to cleaning
    s1 = reprod.get_params_hash(o)
    s2 = reprod.get_params_hash(o)
    assert s1 == s2


def test_entrypoint_setup_first_and_second_call(monkeypatch: pytest.MonkeyPatch):
    # Patch logging and pydantic loader to avoid side effects by patching real modules
    calls = {"setup_logging": 0, "register_models": 0}

    import pyine.utils.logging as real_log_mod
    import pyine.utils.pydantic as real_pyd_mod

    def fake_setup_logging(level, log_to_file):  # noqa: ARG002
        calls["setup_logging"] += 1

    def fake_register_models_from_package(pkg):  # noqa: ARG002
        calls["register_models"] += 1

    monkeypatch.setattr(real_log_mod, "setup_logging", fake_setup_logging, raising=True)
    monkeypatch.setattr(
        real_pyd_mod.PydanticYAMLLoader,
        "register_models_from_package",
        classmethod(lambda cls, pkg: fake_register_models_from_package(pkg)),
        raising=True,
    )

    # ensure dotenv.load_dotenv is a no-op
    import dotenv as real_dotenv

    monkeypatch.setattr(real_dotenv, "load_dotenv", lambda x: None, raising=True)

    # reset sentinel so entrypoint_setup runs init branch
    monkeypatch.delattr(reprod.entrypoint_setup, "_executed", raising=False)

    # first call initializes
    reprod.entrypoint_setup(seed=123, log_level=10, log_to_file=False)
    # second call should not call setup again, only reseed
    reprod.entrypoint_setup(seed=456)

    assert calls["setup_logging"] == 1
    assert calls["register_models"] == 1
