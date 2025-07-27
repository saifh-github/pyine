import pyine.data.traces.dataset_utils


def has_taco_dataset():
    """Returns True if the TACO dataset is available."""
    try:
        taco_path = pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO")
    except (FileNotFoundError, AssertionError):
        return False
    return taco_path.exists()


TACO_DATASET_MISSING = not has_taco_dataset()
