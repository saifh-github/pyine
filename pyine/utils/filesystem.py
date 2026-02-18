import collections.abc
import contextlib
import getpass
import logging
import os
import pathlib
import re
import shutil
import sys
import tempfile
import typing

import dotenv

import pyine

logger = logging.getLogger(__name__)

PROJECT_ROOT_ENV_VAR = "PYINE_PROJECT_ROOT"
"""Name of the environment variable used to override the default project root path."""
DATA_ROOT_ENV_VAR = "PYINE_DATA_ROOT"
"""Name of the environment variable used to override the default data root path."""
CACHE_ROOT_ENV_VAR = "PYINE_CACHE_ROOT"
"""Name of the environment variable used to override the default cache root path."""
CONFIGS_ROOT_ENV_VAR = "PYINE_CONFIGS_ROOT"
"""Name of the environment variable used to override the default configs search path."""
LOGS_ROOT_ENV_VAR = "PYINE_LOGS_ROOT"
"""Name of the environment variable used to override the default logs root path."""


def get_project_root_path() -> pathlib.Path:
    """Returns the default path to the project root directory.

    The path can be overridden by setting the 'PROJECT_ROOT_ENV_VAR' environment variable.
    If the variable is not set, it defaults to the parent directory of the `pyine` package.

    Returns:
        pathlib.Path: The absolute, resolved path to the project root directory.
    """
    env_path = os.environ.get(PROJECT_ROOT_ENV_VAR, None)
    if env_path:
        env_path = os.path.expandvars(os.path.expanduser(env_path))
        return pathlib.Path(env_path).resolve()
    # if the environment variable is not set, default to the 'pyine' root directory
    return pathlib.Path(pyine.__file__).parents[1].resolve()


def get_data_root_path() -> pathlib.Path:
    """Returns the root path for storing and loading datasets.

    The path can be overridden by setting the 'DATA_ROOT_ENV_VAR' environment variable.
    If the variable is not set, it defaults to a 'data' directory in the project root.

    Returns:
        pathlib.Path: The absolute, resolved path to the data root directory.
    """
    env_path = os.environ.get(DATA_ROOT_ENV_VAR, None)
    if env_path:
        env_path = os.path.expandvars(os.path.expanduser(env_path))
        return pathlib.Path(env_path).resolve()
    # if the environment variable is not set, default to the 'data' directory
    return get_project_root_path() / "data"


def get_logs_root_path() -> pathlib.Path:
    """Returns the root path for storing logs.

    The path can be overridden by setting the 'LOGS_ROOT_ENV_VAR' environment variable.
    If the variable is not set, it defaults to a 'logs' directory in the project root.

    Returns:
        pathlib.Path: The absolute, resolved path to the logs root directory.
    """
    env_path = os.environ.get(LOGS_ROOT_ENV_VAR, None)
    if env_path:
        env_path = os.path.expandvars(os.path.expanduser(env_path))
        return pathlib.Path(env_path).resolve()
    # if the environment variable is not set, default to the 'logs' directory
    return get_project_root_path() / "logs"


def get_tmp_dir(mode: int = 0o700) -> pathlib.Path:
    """Returns a user-specific temporary directory under the system temp root.

    The returned path should be stable across processes, persist until the OS cleans temp
    (often reboot or periodic cleanup), and it should respect TMPDIR/TEMP/TMP on all
    platforms. That path should also correspond to an already-created directory.

    The default path can be overridden by setting the 'TMP_DIR', 'TMPDIR', or 'SLURM_TMPDIR'
    environment variables (those will be checked in that order). If none of these variables are
    the OS-specified temp dir specified via `tempfile.gettempdir()` will be used.
    """
    env_path = os.getenv("TMP_DIR")
    if env_path is not None:
        env_path = os.path.expandvars(os.path.expanduser(env_path))
        tmpdir_base = pathlib.Path(env_path)
    else:
        env_path = os.getenv("TMPDIR")
        if env_path is not None:
            env_path = os.path.expandvars(os.path.expanduser(env_path))
            tmpdir_base = pathlib.Path(env_path)
        else:
            env_path = os.getenv("SLURM_TMPDIR")
            if env_path is not None:
                env_path = os.path.expandvars(os.path.expanduser(env_path))
                tmpdir_base = pathlib.Path(env_path)
            else:
                tmpdir_base = pathlib.Path(tempfile.gettempdir())
    if not tmpdir_base.is_dir():
        tmpdir_base.mkdir(parents=True, exist_ok=True)
    if not os.access(str(tmpdir_base), os.W_OK):
        raise PermissionError(f"base temporary dir not writable: {tmpdir_base}")
    # create per-user subdir to avoid collisions on multi-user machines
    tmpdir = pathlib.Path(tmpdir_base) / f"pyine-{get_username()}"
    tmpdir.mkdir(mode=mode, exist_ok=True)
    with contextlib.suppress(PermissionError):
        tmpdir.chmod(mode)
    # final sanity check
    if not os.access(str(tmpdir), os.W_OK):
        raise PermissionError(f"temporary dir not writable: {tmpdir_base}")
    return tmpdir


def get_data_cache_path() -> pathlib.Path:
    """Returns the path to the dataset cache directory."""
    env_path = os.environ.get(CACHE_ROOT_ENV_VAR, None)
    if env_path:
        env_path = os.path.expandvars(os.path.expanduser(env_path))
        out_path = pathlib.Path(env_path).resolve()
    else:
        out_path = get_data_root_path() / "cache"
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def get_data_cache_subdir(*parts: str) -> pathlib.Path:
    """Resolve and return a data cache subdirectory rooted under the data cache path."""
    cache_root = get_data_cache_path()
    cache_dir = cache_root.joinpath(*parts)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_configs_root_path() -> pathlib.Path | None:
    """Returns the optional root path for storing configuration files (if defined; None otherwise)."""
    env_path = os.environ.get(CONFIGS_ROOT_ENV_VAR, None)
    if env_path:
        env_path = os.path.expandvars(os.path.expanduser(env_path))
        return pathlib.Path(env_path).resolve()
    return None


def get_username() -> str:
    """Returns the username from the environment or password database.

    If the username cannot be determined, returns 'unknown'.
    """
    return getpass.getuser() or "unknown"


def find_dotenv_file(start: str | pathlib.Path | None = None) -> pathlib.Path | None:
    """Returns the path to the CLOSEST '.env' file by walking upward from `start` (or CWD).

    Respects the `DOTENV_PATH` env var if set. Uses python-dotenv's `find_dotenv` when available
    (no dependency required).

    If not dotenv file is found, returns `None`.
    """
    # first, check explicit override (highest priority)
    override = os.getenv("DOTENV_PATH")
    if override:
        override = os.path.expandvars(os.path.expanduser(override))
        p = pathlib.Path(override).expanduser().resolve()
        if p.is_file():
            return p
    # defer search to python-dotenv (from CWD)
    start = pathlib.Path(start or pathlib.Path.cwd()).resolve()
    if start == pathlib.Path.cwd().resolve():
        path_str = dotenv.find_dotenv(filename=".env", usecwd=True, raise_error_if_not_found=False)
        return pathlib.Path(path_str).resolve() if path_str else None
    # minimal stdlib fallback: walk up to filesystem root
    for folder in (start, *start.parents):
        candidate = folder / ".env"
        if candidate.is_file():
            return candidate.resolve()
    return None  # found nothing


def get_relative_path_to_root(
    module_file: str | pathlib.Path,
    project_root_path: str | pathlib.Path | None = None,
) -> str:
    """Returns the relative path of the given module w.r.t. the project's root directory.

    Parameters:
        module_file (str): The file path of the current module (__file__).
        project_root_path (str): The absolute path of the project's root directory. If `None`, will
            be inferred from the `pyine` package. Default is `None`.

    Returns:
        str: The relative path of the module with respect to the project root.
    """
    module_path = pathlib.Path(module_file).resolve()
    if project_root_path is None:
        project_root_path = get_project_root_path()
    project_root_path = pathlib.Path(project_root_path).resolve()
    return str(module_path.relative_to(project_root_path))


def get_path_size(path: str | pathlib.Path) -> int:
    """Calculate the total size of a file or directory in bytes.

    Parameters:
        path (str | pathlib.Path): Path to the file or directory.

    Returns:
        int: Total size in bytes.
    """
    path = pathlib.Path(path)
    if path.is_file():
        return path.stat().st_size
    total_size = 0
    for item in path.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
    return total_size


def get_human_readable_size(num_bytes: int, suffix: str = "B") -> str:
    """Convert bytes to human-readable string, e.g. 1.2MiB."""
    value = float(num_bytes)
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi"]:
        if abs(value) < 1024.0:
            return f"{value:3.1f}{unit}{suffix}"
        value /= 1024.0
    return f"{value:.1f}Ei{suffix}"


def check_output_path_overwrite(
    output_path: str | pathlib.Path,
    *,
    force: bool = False,
) -> None:
    """Ensure the output location is safe to write to before creating artifacts.

    When the target already exists, the caller must explicitly opt into deletion via
    ``force=True`` (or the CLI equivalent). Otherwise, a ``FileExistsError`` is raised
    so automated workflows cannot hang waiting for user confirmation.
    """
    output_path = pathlib.Path(output_path).resolve()
    if not output_path.exists():
        return
    if not force:
        message = (
            f"Refusing to overwrite existing output at: {output_path.absolute()} — "
            "re-run with force=True or pass --force from the CLI to continue."
        )
        logger.error(message)
        raise FileExistsError(message)
    logger.warning(f"overwriting existing output at: {output_path.absolute()}")
    if output_path.is_file():
        output_path.unlink()
    elif output_path.is_dir():
        shutil.rmtree(output_path)


def slugify(text: str) -> str:
    """Convert text to a filesystem‐safe slug (lowercase, alnum, hyphens)."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_-]+", "-", text).strip("-")


def normalize_path_tuple(
    value: object,
    *,
    field_name: str = "paths",
) -> tuple[pathlib.Path, ...] | None:
    """Normalize a flexible path-or-paths-or-None input into a tuple of ``pathlib.Path``.

    Handles common config input shapes (strings, ``os.PathLike``, sequences including
    ``omegaconf.ListConfig``) and returns a canonicalized tuple. Intended for use in
    pydantic before-validators or manual config normalization.

    Args:
        value: A single path (``str`` or ``os.PathLike``), a sequence of paths, or ``None``.
        field_name: Name of the config field, used in error messages.

    Returns:
        ``None`` if *value* is ``None``, otherwise a non-empty tuple of ``pathlib.Path``.

    Raises:
        ValueError: If *value* is an empty sequence, contains ``bytes``/``bytearray`` elements,
            or is an unsupported type.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        raise ValueError(f"{field_name}: expected path string, not {type(value).__name__}")
    if isinstance(value, str):
        return (pathlib.Path(value),)
    if isinstance(value, os.PathLike):
        return (pathlib.Path(os.fspath(value)),)  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(value, collections.abc.Sequence):
        if not value:
            raise ValueError(f"{field_name} received an empty sequence; use None to disable")
        items: list[pathlib.Path] = []
        for item in value:  # pyright: ignore[reportUnknownVariableType]
            if isinstance(item, (bytes, bytearray)):
                raise ValueError(f"{field_name} sequence contains {type(item).__name__} element; expected path strings")
            items.append(pathlib.Path(typing.cast("str", item)))
        return tuple(items)
    raise ValueError(f"{field_name}: expected path, sequence of paths, or None; got {type(value).__name__}")


_SHARED_FS_TYPES: set[str] = {
    "nfs",
    "nfs4",
    "lustre",
    "gpfs",
    "cifs",
    "smb",
    "smbfs",
    "glusterfs",
    "ceph",
    "beegfs",
    "panfs",
    "pvfs2",
    "orangefs",
}
"""Filesystem types considered shared/network filesystems."""


def is_path_on_shared_filesystem(path: pathlib.Path | str) -> bool:
    """Detect if path is on a shared/network filesystem.

    Uses /proc/mounts on Linux to determine filesystem type. RETURNS TRUE (assume shared) if
    detection fails or on non-Linux. This is the safe default; accidentally disabling per-node prep
    is better than accidentally enabling it on shared storage.

    Why no file visibility fallback: a file visibility test (rank 0 writes, others check) would
    require coordination across ranks. On node-local storage, filesystem-based barriers can't work,
    risking deadlock or divergent decisions. Users can use PYINE_PER_NODE_PREP=on to override.

    Args:
        path: Path to check (will be resolved to absolute path).

    Returns:
        True if path appears to be on a shared filesystem (NFS, Lustre, etc.)
        or if detection fails (conservative default).
    """
    path = pathlib.Path(path).resolve()
    if sys.platform != "linux":
        logger.warning(f"shared FS detection not supported on {sys.platform}, assuming shared")
        return True
    try:
        mounts = _parse_proc_mounts()
        fs_type = _find_mount_for_path(path, mounts)
        if fs_type is None:
            logger.warning(f"could not determine filesystem type for {path}, assuming shared")
            return True
        is_shared = fs_type.lower() in _SHARED_FS_TYPES
        logger.debug(f"path {path} is on {fs_type} filesystem (shared={is_shared})")
        return is_shared
    except Exception as exc:
        logger.warning(f"shared FS detection failed: {exc}, assuming shared")
        return True


def _parse_proc_mounts() -> list[tuple[str, str, str]]:
    """Parse /proc/mounts and return list of (mount_point, fs_type, device).

    Handles escaped characters (e.g., \\040 for space) in mount paths.

    Returns:
        List of tuples containing (mount_point, fs_type, device).
    """
    mounts: list[tuple[str, str, str]] = []
    with open("/proc/mounts") as fd:
        for line in fd:
            parts = line.split()
            if len(parts) >= 3:
                device, mount_point, fs_type = parts[0], parts[1], parts[2]
                mount_point = _unescape_mount_path(mount_point)
                mounts.append((mount_point, fs_type, device))
    return mounts


def _unescape_mount_path(path: str) -> str:
    """Unescape octal sequences in mount paths from /proc/mounts.

    Args:
        path: Mount path that may contain octal escape sequences like \\040.

    Returns:
        Unescaped path string.
    """

    def replace_octal(match: re.Match[str]) -> str:
        return chr(int(match.group(1), 8))

    return re.sub(r"\\([0-7]{3})", replace_octal, path)


def _find_mount_for_path(
    path: pathlib.Path,
    mounts: list[tuple[str, str, str]],
) -> str | None:
    """Find the filesystem type for a path by matching longest mount point prefix.

    Uses path-boundary-safe matching: /data matches /data but not /data2.

    Args:
        path: Path to find mount point for.
        mounts: List of (mount_point, fs_type, device) tuples from _parse_proc_mounts.

    Returns:
        Filesystem type string if found, None otherwise.
    """
    path_str = str(path)
    best_match: tuple[str, str] | None = None  # (mount_point, fs_type)
    for mount_point, fs_type, _ in mounts:
        if path_str == mount_point or path_str.startswith(mount_point + "/"):
            if best_match is None or len(mount_point) > len(best_match[0]):
                best_match = (mount_point, fs_type)
    return best_match[1] if best_match else None
