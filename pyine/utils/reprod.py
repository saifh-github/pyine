import hashlib
import importlib.metadata
import os
import platform
import random
import re
import time
import typing

import numpy as np


def get_python_version() -> str:
    """Return current Python version, e.g. '3.12.6'."""
    return platform.python_version()


def get_platform_name() -> str:
    """Returns a print-friendly platform name that can be used for logs / data tagging."""
    return str(platform.node())


def get_timestamp() -> str:
    """Returns a print-friendly timestamp (year, month, day, hour, minute, second) for logs."""
    return time.strftime("%Y%m%d-%H%M%S")


def get_framework_version() -> str:
    """Returns the version of the benchmark framework (in MAJOR.MINOR.PATCH format, e.g. 1.2.3)."""
    try:
        package_version = importlib.metadata.version("pyine")
    except importlib.metadata.PackageNotFoundError:
        raise ValueError("Package 'pyine' not installed; cannot determine framework version!")
    return package_version


def get_git_revision_hash() -> str:
    """Returns a print-friendly hash (SHA1 signature) for the underlying git repository (if found).

    If a git repository is not found, the function will return a static string.
    """
    try:
        import git  # noqa
    except (ImportError, AttributeError):
        return "git-import-error"
    try:
        repo = git.Repo(path=os.path.abspath(__file__), search_parent_directories=True)
        sha = repo.head.object.hexsha
        return str(sha)
    except (AttributeError, ValueError, git.InvalidGitRepositoryError):
        return "git-revision-unknown"


def get_installed_packages() -> list[str]:
    """Returns a list of all packages installed in the current environment.

    If the required packages cannot be imported, the returned list will be empty. Note that some
    packages may not be properly detected by this approach, and it is pretty hacky, so use it with a
    grain of salt (i.e. just for logging is fine).
    """
    try:
        import importlib.metadata

        pkgs = [f"{pkg.name}=={pkg.version}" for pkg in importlib.metadata.distributions()]
    except (ImportError, AttributeError):
        try:
            import pip  # noqa

            # noinspection PyUnresolvedReferences
            pkgs = [f"{pkg.key}=={pkg.version}" for pkg in pip.get_installed_distributions()]
        except (ImportError, AttributeError):
            pkgs = []
    return sorted(pkgs, key=str.casefold)  # noqa


def get_params_hash(*args, **kwargs):
    """Computes and returns the hash (md5 checksum) of a given set of parameters.

    Args:
        Any combination of parameters that are hashable via their string representation.

    Returns:
        The hashing result as a string of hexadecimal digits.
    """
    # by default, will use the repr of all params but remove the 'at 0x00000000' addresses
    clean_str = re.sub(r" at 0x[a-fA-F\d]+", "", str(args) + str(kwargs))
    return hashlib.sha1(clean_str.encode(), usedforsecurity=False).hexdigest()


def compute_file_hash(path: str, algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    """Compute checksum of a file using given algorithm."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    """Sets the seed for the random and numpy.random modules."""
    random.seed(seed)
    np.random.seed(seed)


def get_reprod_metadata() -> dict[str, typing.Any]:
    """Returns a dictionary of metadata that can be used to assess reproducibility."""
    return {
        "python_version": get_python_version(),
        "platform": get_platform_name(),
        "timestamp": get_timestamp(),
        "framework_version": get_framework_version(),
        "git_revision_hash": get_git_revision_hash(),
        "installed_packages": get_installed_packages(),
    }
