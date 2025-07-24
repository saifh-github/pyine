import logging
import pathlib
import re
import shutil

import pyine

logger = logging.getLogger(__name__)


def get_relative_path_to_root(
    module_file: str | pathlib.Path,
    project_root: str | pathlib.Path | None = None,
) -> str:
    """
    Get the relative path of the given module with respect to the project's root directory.

    Parameters:
        module_file (str): The file path of the current module (__file__).
        project_root (str): The absolute path of the project's root directory.

    Returns:
        str: The relative path of the module with respect to the project root.
    """
    module_path = pathlib.Path(module_file).resolve()
    if project_root is None:
        project_root = pathlib.Path(pyine.__file__).parents[1].resolve()
    project_root_path = pathlib.Path(project_root).resolve()
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
