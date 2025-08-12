import pyine.data.taco.dataset_utils


def has_taco_dataset():
    """Returns True if the TACO dataset is available."""
    try:
        taco_path = pyine.data.taco.dataset_utils.get_latest_repackaged_dataset_path()
    except (FileNotFoundError, AssertionError):
        return False
    return taco_path.exists()


TACO_DATASET_MISSING = not has_taco_dataset()
