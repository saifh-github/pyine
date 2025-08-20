import pyine.data.taco.dataset_utils
import pyine.data.traces.dataset_utils


def has_taco_dataset():
    """Returns True if the TACO dataset is available."""
    try:
        path = pyine.data.taco.dataset_utils.get_latest_repackaged_dataset_path()
    except (FileNotFoundError, AssertionError):
        return False
    return path.exists()


def has_taco_traces_dataset():
    """Returns True if the TACO traces dataset is available."""
    try:
        path = pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO")
    except (FileNotFoundError, AssertionError):
        return False
    return path.exists()


TACO_DATASET_MISSING = not has_taco_dataset()
TACO_TRACES_DATASET_MISSING = not has_taco_traces_dataset()
