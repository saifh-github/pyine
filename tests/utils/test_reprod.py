import hashlib
import importlib
import importlib.metadata
import platform
import random
import sys
import time
import types

import numpy as np
import pytest

import pyine.utils.reprod as reprod


def test_get_python_version() -> None:
    """Test that get_python_version returns the current Python version."""
    assert reprod.get_python_version() == platform.python_version()


def test_get_platform_name(monkeypatch) -> None:
    """Test that get_platform_name returns the platform node name."""
    monkeypatch.setattr(platform, "node", lambda: "test-node")
    assert reprod.get_platform_name() == "test-node"


def test_get_timestamp(monkeypatch) -> None:
    """Test that get_timestamp returns a formatted timestamp string."""
    monkeypatch.setattr(time, "strftime", lambda fmt: "20250101-000000")
    assert reprod.get_timestamp() == "20250101-000000"


def test_get_framework_version_success(monkeypatch) -> None:
    """Test that get_framework_version returns the installed package version."""
    monkeypatch.setattr(importlib.metadata, "version", lambda pkg: "1.2.3")
    assert reprod.get_framework_version() == "1.2.3"


def test_get_framework_version_not_installed(monkeypatch) -> None:
    """Test that get_framework_version raises ValueError when package is not found."""

    def fake_version(pkg):
        raise importlib.metadata.PackageNotFoundError()

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    with pytest.raises(ValueError):
        reprod.get_framework_version()


def test_get_git_revision_hash_unknown_repo(monkeypatch) -> None:
    """Test that get_git_revision_hash returns unknown when repo can't be found."""
    # create dummy git module with Repo raising InvalidGitRepositoryError
    dummy_git = types.SimpleNamespace()

    class DummyRepo:
        def __init__(*args, **kwargs):
            raise importlib.metadata.PackageNotFoundError

    dummy_git.Repo = DummyRepo
    dummy_git.InvalidGitRepositoryError = Exception
    monkeypatch.setitem(sys.modules, "git", dummy_git)
    assert reprod.get_git_revision_hash() == "git-revision-unknown"


def test_get_git_revision_hash_success(monkeypatch) -> None:
    """Test that get_git_revision_hash returns commit hash when repo is found."""
    dummy_git = types.SimpleNamespace()

    class DummyHead:
        object = types.SimpleNamespace(hexsha="deadbeef")

    class DummyRepo:
        def __init__(self, path, search_parent_directories=True):
            pass

        head = DummyHead()

    dummy_git.Repo = DummyRepo
    dummy_git.InvalidGitRepositoryError = Exception
    monkeypatch.setitem(sys.modules, "git", dummy_git)
    # call the function; since repo path is irrelevant, it will return our dummy sha
    assert reprod.get_git_revision_hash() == "deadbeef"


class DummyDist:
    """Dummy class to simulate importlib.metadata.Distribution objects."""

    def __init__(self, name, version):
        self.name = name
        self.version = version


class DummyPkg:
    """Dummy class to simulate pip.get_installed_distributions objects."""

    def __init__(self, key, version):
        self.key = key
        self.version = version


def test_get_installed_packages_metadata(monkeypatch) -> None:
    """Test that get_installed_packages returns packages from importlib.metadata."""
    dists = [DummyDist("pkgA", "0.1"), DummyDist("pkgB", "2.0")]
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: dists)
    result = reprod.get_installed_packages()
    assert result == ["pkgA==0.1", "pkgB==2.0"]


def test_get_installed_packages_pip(monkeypatch) -> None:
    """Test that get_installed_packages uses pip as fallback when importlib.metadata fails."""
    # Simulate importlib.metadata failure
    monkeypatch.setattr(importlib, "metadata", None)
    # Provide pip fallback
    dummy_pip = types.SimpleNamespace()
    dummy_pip.get_installed_distributions = lambda: [DummyPkg("pkgX", "1.0"), DummyPkg("pkgY", "3.5")]
    monkeypatch.setitem(sys.modules, "pip", dummy_pip)
    result = reprod.get_installed_packages()
    assert result == ["pkgX==1.0", "pkgY==3.5"]


def test_get_installed_packages_empty(monkeypatch) -> None:
    """Test that get_installed_packages returns empty list when all methods fail."""
    monkeypatch.setattr(importlib, "metadata", None)
    monkeypatch.setitem(sys.modules, "pip", None)
    result = reprod.get_installed_packages()
    assert result == []


def test_get_params_hash_consistency() -> None:
    """Test that get_params_hash returns consistent hashes for identical inputs."""
    h1 = reprod.get_params_hash(1, "a", foo=3)
    h2 = reprod.get_params_hash(1, "a", foo=3)
    assert isinstance(h1, str) and len(h1) == 40
    assert h1 == h2


def test_get_params_hash_removes_addresses() -> None:
    """Test that get_params_hash removes memory addresses from object representations."""

    class A:
        def __repr__(self):
            return "<A object at 0xABCDEF>"

    h = reprod.get_params_hash(A())
    # Should not include "0x" in the hash input
    assert "0x" not in h


def test_compute_file_hash(tmp_path) -> None:
    """Test that compute_file_hash correctly computes file checksums."""
    content = b"hello world"
    file_path = tmp_path / "test.txt"
    file_path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert reprod.compute_file_hash(str(file_path), "sha256", chunk_size=4) == expected
    expected_md5 = hashlib.md5(content).hexdigest()
    assert reprod.compute_file_hash(str(file_path), "md5", chunk_size=4) == expected_md5


def test_set_seed_reproducibility() -> None:
    """Test that set_seed ensures reproducible random number generation."""
    reprod.set_seed(123)
    r1 = random.random()
    a1 = np.random.rand()
    reprod.set_seed(123)
    r2 = random.random()
    a2 = np.random.rand()
    assert r1 == r2
    assert a1 == a2


def test_get_reprod_metadata(monkeypatch) -> None:
    """Test that get_reprod_metadata returns a dictionary with expected keys."""
    monkeypatch.setattr("pyine.utils.reprod.get_python_version", lambda: "pv")
    monkeypatch.setattr("pyine.utils.reprod.get_platform_name", lambda: "pn")
    monkeypatch.setattr("pyine.utils.reprod.get_timestamp", lambda: "ts")
    monkeypatch.setattr("pyine.utils.reprod.get_framework_version", lambda: "fv")
    monkeypatch.setattr("pyine.utils.reprod.get_git_revision_hash", lambda: "gh")
    monkeypatch.setattr("pyine.utils.reprod.get_installed_packages", lambda: ["p1", "p2"])
    md = reprod.get_reprod_metadata()
    assert md == {
        "python_version": "pv",
        "platform": "pn",
        "timestamp": "ts",
        "framework_version": "fv",
        "git_revision_hash": "gh",
        "installed_packages": ["p1", "p2"],
    }
