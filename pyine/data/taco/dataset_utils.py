import datetime
import pathlib

import pyine.utils.filesystem


def get_latest_repackaged_dataset_path() -> pathlib.Path:
    """Returns the path to the latest repackaged dataset.

    If multiple repackaged datasets are available, the most recent version is returned.
    """
    rpkg_root = pyine.utils.filesystem.get_data_root_path() / "TACO" / "repackaged"
    assert rpkg_root.exists() and rpkg_root.is_dir(), f"invalid repackaged dataset root path: {rpkg_root}"
    dataset_folders = list(rpkg_root.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-v*/"))
    if not dataset_folders:
        raise FileNotFoundError(f"No dataset folders found in {rpkg_root}")
    latest_dataset = max(dataset_folders)
    return pathlib.Path(latest_dataset)


def get_new_repackaged_dataset_path(
    dataset_version: int = 1,
) -> pathlib.Path:
    """Returns the path where a new repackaged dataset should be saved.

    Will be named based on today's date and version number.
    """
    rpkg_root = pyine.utils.filesystem.get_data_root_path() / "TACO" / "repackaged"
    today = datetime.date.today()
    dataset_name = f"{today.strftime('%Y-%m-%d')}-v{dataset_version:02d}"
    return rpkg_root / dataset_name
