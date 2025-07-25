import hashlib
import importlib.metadata
import logging
import os
import pathlib
import platform
import random
import re
import time
import typing

import dotenv
import numpy as np

import pyine.utils.logging

logger = logging.getLogger(__name__)


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


def compute_hash(
    path: pathlib.Path | typing.AnyStr,
    algorithm: str = "sha256",
    chunk_size: int = 8192,
    raise_on_error: bool = True,
) -> str:
    """Compute checksum of a file or directory using given hashing algorithm.

    This function handles both individual files and directories. For files, it directly hashes
    the content. For directories, it recursively processes all contained files and combines
    their hashes in a deterministic way that depends only on content and relative paths, not
    on metadata like timestamps or permissions.

    Args:
        path: path to the file or directory to hash.
        algorithm: hash algorithm to use (e.g., "md5", "sha1", "sha256").
        chunk_size: size of chunks to read when hashing large files.
        raise_on_error: if True, raises an exception if any file cannot be read.

    Returns:
        Hexadecimal digest of the hash.
    """
    path_obj = pathlib.Path(path).expanduser().resolve()
    if path_obj.is_file():
        # file case - direct hash of contents
        h = hashlib.new(algorithm)
        with path_obj.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    elif path_obj.is_dir():
        # directory case - combine hashes of all files
        dir_hash = hashlib.new(algorithm)
        all_files = []
        for file_path in sorted(path_obj.glob("**/*")):
            if file_path.is_file():
                rel_path = file_path.relative_to(path_obj)
                all_files.append((str(rel_path), file_path))
        all_files.sort()  # sort by relative path for deterministic ordering
        for rel_path, file_path in all_files:
            path_hash = hashlib.new(algorithm, rel_path.encode()).hexdigest()
            file_hash = hashlib.new(algorithm)
            try:
                with file_path.open("rb") as f:
                    for chunk in iter(lambda: f.read(chunk_size), b""):
                        file_hash.update(chunk)
            except (OSError, PermissionError) as e:
                if raise_on_error:
                    # default behavior: we probably don't expect a dataset to contain bad files
                    raise e
                else:
                    # otherwise, include error information in the hash if we can't read a file
                    file_hash.update(f"ERROR: {str(e)}".encode())
            combined = f"{path_hash}:{file_hash.hexdigest()}".encode()
            dir_hash.update(combined)
        return dir_hash.hexdigest()
    else:
        raise ValueError(f"path does not exist: {path}")


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


def entrypoint_setup(
    seed: int | None = None,
    log_level: int = logging.INFO,
    log_to_file: bool = False,
) -> None:
    """Sets up the framework (env vars, logging, rng) for reproducible experiments."""
    # use a sentinel object to track first execution
    if not hasattr(entrypoint_setup, "_executed"):
        dotenv.load_dotenv()
        pyine.utils.logging.setup_logging(
            level=log_level,
            log_to_file=log_to_file,
        )
        entrypoint_setup._executed = True
    # no matter what execution this is, re-seed if needed
    if seed is not None:
        set_seed(seed)
    logger.info(f"set up entrypoint (seed={seed})")
