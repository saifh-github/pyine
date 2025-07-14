import pathlib
import shutil

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


def check_output_path_overwrite(
    output_path: str | pathlib.Path,
) -> None:
    """Check if the output path exists and prompt the user to overwrite it if needed.

    If the output does exist and the user confirms the overwrite, the existing output is deleted.
    Otherwise, the program exits without doing anything.
    """
    output_path = pathlib.Path(output_path).resolve()
    if output_path.exists():
        overwrite = (
            input(
                f"The output already exists at: {output_path.absolute()}\n"
                "Do you want to delete it so it can be recreated? [y/N]: "
            )
            .strip()
            .lower()
        )
        if overwrite != "y":
            print("Operation aborted.")
            exit(0)
        shutil.rmtree(output_path)
