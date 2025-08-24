import os

import pyine.data.taco.dataset_utils
import pyine.data.traces.dataset_utils
import pyine.utils.reprod

pyine.utils.reprod.load_dotenv()


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


def has_hf_access_token():
    """Return True if a Hugging Face user access token is available."""
    token = os.environ.get("HF_TOKEN", None)
    return token is not None and len(token) > 0


HF_ACCESS_TOKEN_MISSING = not has_hf_access_token()
