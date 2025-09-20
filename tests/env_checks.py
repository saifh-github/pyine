import os
import socket

import pyine.data.taco.dataset_utils
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
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


def has_taco_traces_dataset_split():
    """Returns true if the TACO traces dataset split is available."""
    try:
        path = pyine.data.utils.splits.get_dataset_split_file_path("TACO", must_exist=True)
    except (FileNotFoundError, AssertionError):
        return False
    return path.exists()


TACO_DATASET_MISSING = not has_taco_dataset()
TACO_TRACES_DATASET_MISSING = not has_taco_traces_dataset()
TACO_TRACES_DATASET_SPLIT_MISSING = not has_taco_traces_dataset_split()


def has_hf_access_token():
    """Return True if a Hugging Face user access token is available."""
    token = os.environ.get("HF_TOKEN", None)
    return token is not None and len(token) > 0


HF_ACCESS_TOKEN_MISSING = not has_hf_access_token()


def has_openai_api_key():
    """Return True if an OpenAI API key is available."""
    key = os.environ.get("OPENAI_API_KEY", None)
    return key is not None and len(key) > 0


OPENAI_API_KEY_MISSING = not has_openai_api_key()


def _read_bool_env(
    name: str,
) -> bool | None:
    """Reads a boolean from an environment variable if set; returns None otherwise."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw_low = raw.strip().lower()
    if raw_low in {"1", "true", "yes", "on"}:
        return True
    if raw_low in {"0", "false", "no", "off"}:
        return False
    return None


def has_network_access() -> bool:
    """Returns True if outbound network access appears available.

    Override via:
      - PYINE_NETWORK_AVAILABLE=true/false
      - PYINE_DISABLE_NETWORK=true (forces False)

    Otherwise, perform a quick TCP connect probe to api.openai.com:443 with a short timeout.
    """
    # explicit overrides first
    override = _read_bool_env("PYINE_NETWORK_AVAILABLE")
    if override is not None:
        return bool(override)
    disabled = _read_bool_env("PYINE_DISABLE_NETWORK")
    if disabled is True:
        return False
    # fast connectivity probe
    try:
        with socket.create_connection(("api.openai.com", 443), timeout=1.0) as _:
            return True
    except OSError:
        return False


NETWORK_UNAVAILABLE = not has_network_access()
