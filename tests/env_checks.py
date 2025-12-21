import os
import socket

import torch

import pyine.data.taco.dataset_utils
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.utils.reprod

pyine.utils.reprod.load_dotenv()


def has_taco_dataset() -> bool:
    """Returns True if the TACO dataset is available."""
    try:
        path = pyine.data.taco.dataset_utils.get_latest_repackaged_dataset_path()
    except (FileNotFoundError, AssertionError):
        return False
    return path.exists()


def has_taco_traces_dataset() -> bool:
    """Returns True if the TACO traces dataset is available."""
    try:
        path = pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO")
    except (FileNotFoundError, AssertionError):
        return False
    return path.exists()


def has_taco_traces_dataset_split() -> bool:
    """Returns true if the TACO traces dataset split is available."""
    try:
        path = pyine.data.utils.splits.get_dataset_split_file_path("TACO", must_exist=True)
    except (FileNotFoundError, AssertionError):
        return False
    return path.exists()


def has_hf_access_token() -> bool:
    """Return True if a Hugging Face user access token is available."""
    token = os.environ.get("HF_TOKEN", None)
    return token is not None and len(token) > 0


def has_openai_api_key() -> bool:
    """Return True if an OpenAI API key is available."""
    key = os.environ.get("OPENAI_API_KEY", None)
    return key is not None and len(key) > 0


def has_wandb_api_key() -> bool:
    """Return True if a W&B API key is available."""
    key = os.environ.get("WANDB_API_KEY", None)
    return key is not None and len(key) > 0


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


LARGE_GPU_MEMORY_THRESHOLD_GB = 40


def has_large_gpu() -> bool:
    """Returns True if at least one GPU with 40+ GB of memory is available.

    This can be used to determine whether we're on a compute cluster with high-end GPUs suitable
    for memory-intensive tests.

    Override via:
      - PYINE_LARGE_GPU_AVAILABLE=true/false
    """
    override = _read_bool_env("PYINE_LARGE_GPU_AVAILABLE")
    if override is not None:
        return bool(override)
    if not torch.cuda.is_available():
        return False
    for device_idx in range(torch.cuda.device_count()):
        total_memory_bytes = torch.cuda.get_device_properties(device_idx).total_memory
        total_memory_gb = total_memory_bytes / (1024**3)
        if total_memory_gb >= LARGE_GPU_MEMORY_THRESHOLD_GB:
            return True
    return False


def has_any_accelerator() -> bool:
    """Returns True if any accelerator is available.

    This can be used to determine whether we're on a platform with ANY device that PyTorch can
    leverage (whether tiny, e.g. mps, or large, e.g. cluster GPU).
    """
    return torch.cuda.is_available() or torch.backends.mps.is_available()


TACO_DATASET_MISSING = not has_taco_dataset()
TACO_TRACES_DATASET_MISSING = not has_taco_traces_dataset()
TACO_TRACES_DATASET_SPLIT_MISSING = not has_taco_traces_dataset_split()
HF_ACCESS_TOKEN_MISSING = not has_hf_access_token()
OPENAI_API_KEY_MISSING = not has_openai_api_key()
WANDB_API_KEY_MISSING = not has_wandb_api_key()
NETWORK_UNAVAILABLE = not has_network_access()
NO_ACCELERATOR_AVAILABLE = not has_any_accelerator()
LARGE_GPU_UNAVAILABLE = not has_large_gpu()
