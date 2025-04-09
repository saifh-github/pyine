import pathlib

import pyine


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
