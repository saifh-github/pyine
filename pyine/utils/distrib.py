import itertools
import logging
import os
import pathlib
import tempfile
import time
import typing

import torch

logger = logging.getLogger(__name__)

_GLOBAL_RANK_ENV_KEYS: tuple[str, ...] = (
    "RANK",
    "SLURM_PROCID",
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
)
"""Environment variables that may encode the global rank before torch.distributed initializes."""
_LOCAL_RANK_ENV_KEYS: tuple[str, ...] = (
    "LOCAL_RANK",
    "MPI_LOCALRANKID",
)
"""Environment variables that may encode the node-local rank before torch.distributed initializes."""
_WORLD_SIZE_ENV_KEYS: tuple[str, ...] = (
    "WORLD_SIZE",
    "SLURM_NTASKS",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
)
"""Environment variables that may encode the total world size before torch.distributed initializes."""
_LOCAL_WORLD_SIZE_ENV_KEYS: tuple[str, ...] = (
    "LOCAL_WORLD_SIZE",
    "MPI_LOCALNRANKS",
)
"""Environment variables that may encode the per-node world size before torch.distributed initializes."""
_FALLBACK_BARRIER_COUNTER = itertools.count()
"""Monotonic counter ensuring each fallback barrier rendezvous uses a distinct directory."""
_FALLBACK_BARRIER_PREFIX = "pyine_pre_ddp_barrier"
"""Prefix applied to fallback barrier rendezvous directories for deterministic cleanup."""
_FALLBACK_BARRIER_TIMEOUT_SECONDS = float(os.environ.get("PYINE_BARRIER_TIMEOUT_SECONDS", "600"))
"""Maximum number of seconds a rank will wait inside the fallback barrier."""

__all__ = [
    "all_reduce_boolean_or",
    "all_gather_objects",
    "broadcast_boolean",
    "broadcast_object",
    "barrier",
    "get_backend",
    "get_distributed_context",
    "get_global_rank",
    "get_local_rank",
    "get_world_size",
    "is_distributed",
    "is_main_process",
]


def get_global_rank(
    default: int | None = 0,
) -> int | None:
    """Returns the global distributed rank, falling back to `default` when unavailable.

    Args:
        default: Value returned when no distributed backend or environment hints are detected.
            Use ``None`` to propagate the absence of a rank.

    Returns:
        The detected global rank or the provided default.
    """
    rank = _get_torch_rank()
    if rank is not None:
        return rank
    rank = _read_first_env_int(_GLOBAL_RANK_ENV_KEYS)
    if rank is not None and rank >= 0:
        return rank
    local_rank = _read_first_env_int(_LOCAL_RANK_ENV_KEYS)
    if local_rank is not None and local_rank >= 0:
        return local_rank
    return default


def get_local_rank(
    default: int | None = 0,
) -> int | None:
    """Return the process-local rank within the node, falling back to `default`.

    Args:
        default: Value returned when no local rank information is available. Use ``None`` to
            propagate the absence of a rank.

    Returns:
        The detected local rank or the provided default.
    """
    rank = _read_first_env_int(_LOCAL_RANK_ENV_KEYS)
    if rank is not None:
        return rank
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        return int(torch.distributed.get_rank() % max(device_count, 1))
    return default


def get_world_size(
    default: int | None = 1,
) -> int | None:
    """Return the total number of distributed processes, falling back to `default`.

    Args:
        default: Value returned when no world size can be detected. Use ``None`` to propagate the
            absence of a detected world size.

    Returns:
        The detected world size or the provided default.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    world_size = _read_first_env_int(_WORLD_SIZE_ENV_KEYS)
    if world_size is not None:
        return world_size
    local_world_size = _read_first_env_int(_LOCAL_WORLD_SIZE_ENV_KEYS)
    if local_world_size is not None:
        return local_world_size
    return default


def is_distributed() -> bool:
    """Return whether the current execution appears to be distributed."""
    world_size = get_world_size(default=None)
    return world_size is not None and world_size > 1


def is_main_process(
    rank: int | None = None,
) -> bool:
    """Return ``True`` when the provided (or detected) rank corresponds to the main process."""
    if rank is None:
        rank = get_global_rank(default=None)
    return rank in (None, 0)


def barrier() -> None:
    """Synchronize all distributed processes.

    Hugging Face's Trainer (and Accelerate) bring up `torch.distributed` lazily. Several call sites
    in PyINE need a coordination point *before* that happens (most notably datamodule preparation,
    where non-zero ranks must wait for rank 0 to finish e.g. downloads). To keep that code unchanged,
    this helper performs a filesystem-based rendezvous until the process group is live, then falls
    back to the standard CUDA-aware barrier path.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        device_ids = None
        if torch.cuda.is_available():
            backend = torch.distributed.get_backend()
            if backend == "nccl":
                local_rank = get_local_rank(default=None)
                if local_rank is not None and 0 <= local_rank < torch.cuda.device_count():
                    current_device = torch.cuda.current_device()
                    if current_device != local_rank:
                        torch.cuda.set_device(local_rank)
                    device_ids = [local_rank]
        torch.distributed.barrier(device_ids=device_ids)  # type: ignore[reportUnknownMemberType]
        return
    _fallback_barrier_if_needed()


def broadcast_object(
    payload: typing.Any,
    src: int = 0,
) -> typing.Any:
    """Broadcast a picklable payload from the source rank to all ranks."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return payload
    object_list = [payload]
    torch.distributed.broadcast_object_list(object_list, src=src)  # type: ignore[reportUnknownMemberType]
    return object_list[0]


def broadcast_boolean(
    flag: bool,
    src: int = 0,
) -> bool:
    """Broadcast a boolean flag from the source rank to all ranks."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return flag
    tensor = torch.tensor([1 if flag else 0], device="cpu", dtype=torch.int32)
    torch.distributed.broadcast(tensor, src=src)  # type: ignore[reportUnknownMemberType]
    return bool(tensor.item())


def all_reduce_boolean_or(flag: bool) -> bool:
    """Return True when any rank reports True."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return flag
    tensor = torch.tensor([1 if flag else 0], device="cpu", dtype=torch.int32)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)  # type: ignore[reportUnknownMemberType]
    return bool(tensor.item())


def all_gather_objects(
    obj: typing.Any,
) -> list[typing.Any]:
    """Gather picklable objects from all ranks."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return [obj]
    world_size = torch.distributed.get_world_size()  # type: ignore[reportUnknownMemberType]
    gather_list: list[typing.Any] = [None] * world_size
    torch.distributed.all_gather_object(gather_list, obj)  # type: ignore[reportUnknownMemberType]
    return gather_list


def get_distributed_context() -> dict[str, typing.Any]:
    """Return a snapshot of distributed runtime values."""
    return {
        "rank": get_global_rank(default=None),
        "local_rank": get_local_rank(default=None),
        "world_size": get_world_size(default=None),
        "backend": get_backend(),
    }


def _get_torch_rank() -> int | None:
    """Return the rank reported by `torch.distributed`, when ready."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return None
    return int(torch.distributed.get_rank())


def _read_first_env_int(
    keys: tuple[str, ...],
) -> int | None:
    """Return the first environment variable among `keys` that can be parsed as an integer."""
    for key in keys:
        raw_value = os.getenv(key)
        if raw_value is None:
            continue
        try:
            return int(raw_value)
        except ValueError:
            logger.debug(f"failed to parse env var {key}={raw_value} as int")
    return None


def get_backend(
    default: str | None = "n/a",
) -> str | None:
    """Return the distributed backend name if available."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return default
    backend = torch.distributed.get_backend()
    return str(backend)


def _fallback_barrier_if_needed() -> None:
    """Synchronize ranks before the torch process group comes online.

    We rely on the launcher (torchrun, SLURM, etc.) to set rank and world-size environment
    variables. Each worker writes a marker file to a shared rendezvous directory; once all markers
    exist, the barrier completes. Rank 0 then removes the rendezvous directory so repeated barriers
    do not leak files. This provides a safe pre-DDP synchronization point without double-initializing
    the process group Hugging Face owns.
    """
    world_size = get_world_size(default=None)
    rank = get_global_rank(default=None)
    if world_size is None or world_size <= 1 or rank is None:
        return
    barrier_idx = next(_FALLBACK_BARRIER_COUNTER)
    call_token = f"{_FALLBACK_BARRIER_PREFIX}_{barrier_idx}"
    barrier_dir = _resolve_barrier_root() / call_token
    barrier_dir.mkdir(parents=True, exist_ok=True)
    rank_file = barrier_dir / f"rank_{rank}"
    rank_file.touch(exist_ok=False)
    deadline = time.monotonic() + _FALLBACK_BARRIER_TIMEOUT_SECONDS
    while True:
        participants = list(barrier_dir.iterdir())
        if len(participants) >= world_size:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"fallback barrier timed out waiting for {world_size} ranks (saw {len(participants)})",
            )
        time.sleep(0.1)
    rank_file.unlink()
    if rank == 0:
        barrier_dir.rmdir()


def _resolve_barrier_root() -> pathlib.Path:
    """Return the directory used for fallback barrier rendezvous files.

    Preference order:
    1. `PYINE_BARRIER_ROOT` when set explicitly by the user.
    2. Hydra's run directory, which our launch script shares across ranks by default.
    3. A temp folder so that ad-hoc launches (e.g., local tests) still have a rendezvous path.
    """
    env_root = os.environ.get("PYINE_BARRIER_ROOT")
    if env_root:
        return pathlib.Path(env_root).expanduser()
    hydra_root = os.environ.get("HYDRA_RUN_DIR")
    if hydra_root:
        return pathlib.Path(hydra_root)
    temp_root = pathlib.Path(tempfile.gettempdir())
    root_dir = temp_root / "pyine_barriers"
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir
