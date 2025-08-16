import getpass
import logging
import os
import pathlib
import re
import shutil
import tempfile

import pyine

logger = logging.getLogger(__name__)


def get_project_root_path() -> pathlib.Path:
    """Returns the default path to the project root directory.

    The path can be overridden by setting the 'PROJECT_ROOT_PATH' environment variable.
    If the variable is not set, it defaults to the parent directory of the `pyine` package.

    Returns:
        pathlib.Path: The absolute, resolved path to the project root directory.
    """
    env_path = os.environ.get("PROJECT_ROOT_PATH", None)
    if env_path:
        return pathlib.Path(env_path).resolve()
    # if the environment variable is not set, default to the 'pyine' root directory
    return pathlib.Path(pyine.__file__).parents[1].resolve()


def get_data_root_path() -> pathlib.Path:
    """Returns the root path for storing and loading datasets.

    The path can be overridden by setting the 'DATA_ROOT_PATH' environment variable.
    If the variable is not set, it defaults to a 'data' directory in the project root.

    Returns:
        pathlib.Path: The absolute, resolved path to the data root directory.
    """
    env_path = os.environ.get("DATA_ROOT_PATH", None)
    if env_path:
        return pathlib.Path(env_path).resolve()
    # if the environment variable is not set, default to the 'data' directory
    return get_project_root_path() / "data"


def get_tmp_dir(mode: int = 0o700) -> pathlib.Path:
    """Returns a user-specific temporary directory under the system temp root.

    The returned path should be stable across processes, persist until the OS cleans temp
    (often reboot or periodic cleanup), and it shoul respects TMPDIR/TEMP/TMP on all
    platforms.

    The default path can be overridden by setting the 'TMP_DIR', 'TMPDIR', or 'SLURM_TMPDIR'
    environment variables (those will be checked in that order). If none of these variables are
    the OS-specified temp dir specified via `tempfile.gettempdir()` will be used.
    """
    if os.getenv("TMP_DIR") is not None:
        tmpdir_base = pathlib.Path(os.getenv("TMP_DIR"))
    elif os.getenv("TMPDIR") is not None:
        tmpdir_base = pathlib.Path(os.getenv("TMPDIR"))
    elif os.getenv("SLURM_TMPDIR") is not None:
        tmpdir_base = pathlib.Path(os.getenv("SLURM_TMPDIR"))
    else:
        tmpdir_base = pathlib.Path(tempfile.gettempdir())
    if not tmpdir_base.is_dir():
        raise ValueError(f"base temporary dir does not exist: {tmpdir_base}")
    if not os.access(str(tmpdir_base), os.W_OK):
        raise PermissionError(f"base temporary dir not writable: {tmpdir_base}")
    # create per-user subdir to avoid collisions on multi-user machines
    username = getpass.getuser() or "unknown"
    tmpdir = pathlib.Path(tmpdir_base) / f"pyine-{username}"
    tmpdir.mkdir(mode=mode, exist_ok=True)
    try:
        tmpdir.chmod(mode)
    except PermissionError:
        pass  # ignore if filesystem/OS doesn't support chmod
    # final sanity check
    if not os.access(str(tmpdir), os.W_OK):
        raise PermissionError(f"temporary dir not writable: {tmpdir_base}")
    return tmpdir


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
    relative_path = str(module_path.relative_to(project_root_path))
    return relative_path


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
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f}{unit}{suffix}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}Ei{suffix}"


def check_output_path_overwrite(
    output_path: str | pathlib.Path,
) -> None:
    """Check if the output path exists and prompt the user to overwrite it if needed.

    If the output does exist and the user confirms the overwrite, the existing output is deleted.
    Otherwise, the program exits without doing anything.
    """
    output_path = pathlib.Path(output_path).resolve()
    if output_path.exists():
        logger.warning(f"Attempting overwrite at: {output_path.absolute()}")
        overwrite = (
            input(
                f"The output already exists at: {output_path.absolute()}\n"
                "Do you want to delete it so it can be recreated? [y/N]: "
            )
            .strip()
            .lower()
        )
        if overwrite != "y":
            logger.critical("Overwrite operation aborted.")
            exit(0)
        else:
            logger.warning(f"Overwriting existing output at: {output_path.absolute()}")
            shutil.rmtree(output_path)


def slugify(text: str) -> str:
    """Convert text to a filesystem‐safe slug (lowercase, alnum, hyphens)."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text
