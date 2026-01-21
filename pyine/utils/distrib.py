import contextlib
import dataclasses
import datetime
import itertools
import logging
import os
import pathlib
import tempfile
import time
import typing

import torch

import pyine.utils.filesystem

logger = logging.getLogger(__name__)

_GLOBAL_RANK_ENV_KEYS: tuple[str, ...] = (
    "RANK",
    "SLURM_PROCID",
    "OMPI_COMM_WORLD_RANK",
    "MV2_COMM_WORLD_RANK",
    "PMI_RANK",
)
"""Environment variables that may encode the global rank before torch.distributed initializes."""
_LOCAL_RANK_ENV_KEYS: tuple[str, ...] = (
    "LOCAL_RANK",
    "MPI_LOCALRANKID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "ACCELERATE_LOCAL_PROCESS_INDEX",
)
"""Environment variables that may encode the node-local rank before torch.distributed initializes."""
_WORLD_SIZE_ENV_KEYS: tuple[str, ...] = (
    "WORLD_SIZE",
    "SLURM_NTASKS",
    "OMPI_COMM_WORLD_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "ACCELERATE_NUM_PROCESSES",
)
"""Environment variables that may encode the total world size before torch.distributed initializes."""
_LOCAL_WORLD_SIZE_ENV_KEYS: tuple[str, ...] = (
    "LOCAL_WORLD_SIZE",
    "MPI_LOCALNRANKS",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "MV2_COMM_WORLD_LOCAL_SIZE",
)
"""Environment variables that may encode the per-node world size before torch.distributed initializes."""
_NODE_RANK_ENV_KEYS: tuple[str, ...] = (
    "NODE_RANK",
    "SLURM_NODEID",
    "GROUP_RANK",
    "ACCELERATE_MACHINE_RANK",
)
"""Environment variables that may encode the node rank before torch.distributed initializes."""
_NNODES_ENV_KEYS: tuple[str, ...] = (
    "NNODES",
    "SLURM_NNODES",
    "ACCELERATE_NUM_MACHINES",
)
"""Environment variables that may encode the number of nodes before torch.distributed initializes."""
_MASTER_ADDR_ENV_KEYS: tuple[str, ...] = (
    "MASTER_ADDR",
    "ACCELERATE_MAIN_PROCESS_IP",
)
"""Environment variables that may encode the master address for torch.distributed rendezvous."""
_MASTER_PORT_ENV_KEYS: tuple[str, ...] = (
    "MASTER_PORT",
    "ACCELERATE_MAIN_PROCESS_PORT",
)
"""Environment variables that may encode the master port for torch.distributed rendezvous."""
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
    "determine_per_node_prep_mode",
    "ensure_torch_distributed_initialized",
    "FingerprintPayload",
    "get_backend",
    "get_distributed_context",
    "get_global_rank",
    "get_local_rank",
    "get_local_world_size",
    "get_node_rank",
    "get_num_nodes",
    "get_world_size",
    "is_distributed",
    "is_local_main_process",
    "is_main_process",
    "is_per_node_prep_enabled",
    "local_barrier",
    "validate_cross_node_fingerprints",
    "validate_node_configuration",
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


def get_local_world_size(
    default: int | None = 1,
) -> int | None:
    """Return the number of processes on the current node, falling back to `default`.

    Args:
        default: Value returned when no local world size can be detected. Use ``None`` to propagate
            the absence of a detected local world size.

    Returns:
        The detected local world size or the provided default.
    """
    local_world_size = _read_first_env_int(_LOCAL_WORLD_SIZE_ENV_KEYS)
    if local_world_size is not None and local_world_size > 0:
        return local_world_size
    return default


def get_node_rank(
    default: int | None = 0,
) -> int | None:
    """Return the index of the current node (machine) among all nodes.

    This is distinct from:
    - global_rank: index of the current process among all processes (0 to world_size-1)
    - local_rank: index of the current process within this node (0 to local_world_size-1)

    Node rank ranges from 0 to num_nodes-1, where num_nodes = world_size // local_world_size.

    Priority order:
    1. NODE_RANK, SLURM_NODEID, GROUP_RANK env vars (explicit)
    2. Derived: global_rank // local_world_size (requires LOCAL_WORLD_SIZE)
    3. Default value

    Using env vars first reduces dependency on LOCAL_WORLD_SIZE being present.

    Args:
        default: Value returned when no node rank can be detected. Use ``None`` to propagate the
            absence of a detected node rank.

    Returns:
        The detected node rank (0 to num_nodes-1) or the provided default.
    """
    node_rank = _read_first_env_int(_NODE_RANK_ENV_KEYS)
    if node_rank is not None and node_rank >= 0:
        return node_rank
    global_rank = get_global_rank(default=None)
    local_world_size = get_local_world_size(default=None)
    if global_rank is not None and local_world_size is not None and local_world_size > 0:
        return global_rank // local_world_size
    return default


def get_num_nodes(
    default: int | None = 1,
) -> int | None:
    """Return the total number of nodes.

    Priority order:
    1. NNODES, SLURM_NNODES env vars (explicit)
    2. Derived: world_size // local_world_size (requires LOCAL_WORLD_SIZE)
    3. Default value

    Args:
        default: Value returned when the number of nodes cannot be detected. Use ``None`` to
            propagate the absence of a detected node count.

    Returns:
        The detected number of nodes or the provided default.
    """
    num_nodes = _read_first_env_int(_NNODES_ENV_KEYS)
    if num_nodes is not None and num_nodes > 0:
        return num_nodes
    world_size = get_world_size(default=None)
    local_world_size = get_local_world_size(default=None)
    if world_size is not None and local_world_size is not None and local_world_size > 0:
        return world_size // local_world_size
    return default


def is_distributed() -> bool:
    """Return whether the current execution appears to be distributed."""
    world_size = get_world_size(default=None)
    return world_size is not None and world_size > 1


def is_main_process(
    rank: int | None = None,
) -> bool:
    """Return ``True`` when the provided (or detected) global rank is 0.

    The main process (global_rank == 0) is the single primary process across all nodes.
    Use this for operations that should happen exactly once globally (e.g., logging,
    saving final checkpoints).

    See also: ``is_local_main_process()`` for operations that should happen once per node.

    Args:
        rank: Optional explicit global rank to check. If None, detects from environment.

    Returns:
        True if global_rank == 0 (or if global_rank cannot be determined).
    """
    if rank is None:
        rank = get_global_rank(default=None)
    return rank in (None, 0)


def is_local_main_process(
    local_rank: int | None = None,
) -> bool:
    """Return ``True`` when the provided (or detected) local rank is 0.

    The local main process (local_rank == 0) is the primary process on each node. Use this for
    operations that should happen once per node (e.g., per-node data preparation when caches are
    on node-local storage).

    See also: ``is_main_process()`` for operations that should happen exactly once globally.

    Args:
        local_rank: Optional explicit local rank to check. If None, detects from environment.

    Returns:
        True if local_rank == 0 (or if local_rank cannot be determined).
    """
    if local_rank is None:
        local_rank = get_local_rank(default=None)
    return local_rank in (None, 0)


def barrier() -> None:
    """Synchronize all distributed processes.

    Hugging Face's Trainer (and Accelerate) bring up `torch.distributed` lazily. Several call sites
    in PyINE need a coordination point *before* that happens (most notably datamodule preparation,
    where non-zero ranks must wait for rank 0 to finish e.g. downloads). To keep that code unchanged,
    this helper performs a filesystem-based rendezvous until the process group is live, then falls
    back to the standard CUDA-aware barrier path.
    """
    logger.debug("synchronizing distributed processes")
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
    """Broadcast a boolean flag from the source rank to all ranks.

    Uses object-based broadcast internally to avoid NCCL/CPU tensor incompatibility.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return flag
    return bool(broadcast_object(flag, src=src))


def all_reduce_boolean_or(flag: bool) -> bool:
    """Return True when any rank reports True.

    Uses object-based gather internally to avoid NCCL/CPU tensor incompatibility.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return flag
    all_flags = all_gather_objects(flag)
    return any(all_flags)


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


@dataclasses.dataclass
class FingerprintPayload:
    """Typed payload for cross-node fingerprint validation.

    Used by ``validate_cross_node_fingerprints()`` to gather and compare fingerprints from all
    nodes. Local rank 0 on each node creates a payload; other ranks pass None.
    """

    ok: bool
    """Whether fingerprint computation succeeded on this node."""
    fingerprint: str | None
    """The computed fingerprint (SHA256 hex digest), or None if computation failed."""
    error: str | None
    """Error message if computation failed, or None on success."""
    node_rank: int = -1
    """Index of the node that produced this payload (set during gather, -1 if unassigned)."""


_LOCAL_BARRIER_COUNTER = itertools.count()
"""Monotonic counter ensuring each local barrier rendezvous uses a distinct directory."""


def local_barrier() -> None:
    """Intra-node synchronization using a filesystem-based rendezvous.

    CRITICAL REQUIREMENTS:
    1. All local ranks on each node MUST call this exactly once per phase;
    2. No conditionals that could cause call-order divergence between local ranks;
    3. job_token MUST be identical across all local ranks (guaranteed by env vars).

    ALWAYS uses tempfile.gettempdir() (/tmp), never PYINE_CACHE_ROOT. This ensures local barriers
    work even if PYINE_CACHE_ROOT is on shared storage.
    """
    if not is_distributed():
        return
    local_world_size = get_local_world_size(default=1)
    if local_world_size is None or local_world_size <= 1:
        return
    _fallback_local_barrier()


def is_per_node_prep_enabled() -> bool:
    """Return True if per-node preparation is enabled.

    Checks in order:
    1. PYINE_PER_NODE_PREP env var override (on/off/auto);
    2. num_nodes > 1 (multi-node setup required);
    3. Cache path is NOT on a shared filesystem (auto-detected).

    Override values:
    - "on" or "1": Force enable per-node prep (FAILS LOUD if prerequisites missing);
    - "off" or "0": Force disable per-node prep;
    - "auto" or unset: Auto-detect based on filesystem type.

    Note:
        Per-node prep implies cross-node fingerprint validation, which requires torch.distributed
        to be initializable (MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_WORLD_SIZE must be
        set). Use torchrun or accelerate.

    Returns:
        True if per-node data preparation should be used.
    """
    override = os.environ.get("PYINE_PER_NODE_PREP", "auto").lower()
    if override in ("on", "1", "true"):
        num_nodes = get_num_nodes(default=None)
        if num_nodes is None:
            raise RuntimeError(
                "PYINE_PER_NODE_PREP=on but cannot determine num_nodes. "
                "Ensure WORLD_SIZE and LOCAL_WORLD_SIZE are set, or use SLURM/torchrun."
            )
        if num_nodes <= 1:
            raise RuntimeError(
                f"PYINE_PER_NODE_PREP=on but num_nodes={num_nodes} (single-node). "
                "Per-node prep requires multi-node setup."
            )
        logger.info("per-node prep forced ON via PYINE_PER_NODE_PREP")
        return True
    if override in ("off", "0", "false"):
        logger.info("per-node prep forced OFF via PYINE_PER_NODE_PREP")
        return False
    if get_num_nodes(default=1) <= 1:  # type: ignore[reportOptionalOperand]
        return False
    cache_path = pyine.utils.filesystem.get_data_cache_path()
    if pyine.utils.filesystem.is_path_on_shared_filesystem(cache_path):
        logger.info(f"cache path {cache_path} is on shared FS, per-node prep disabled")
        return False
    logger.info(f"cache path {cache_path} is on local FS, per-node prep enabled")
    return True


def determine_per_node_prep_mode() -> bool:
    """Determine per-node prep mode with cross-rank synchronization requiring unanimity.

    For multi-node setups, this function:
    1. Initializes torch.distributed early (before any divergent logic);
    2. Computes each rank's local per-node prep decision;
    3. Gathers all decisions and requires unanimity across all ranks.

    If nodes disagree (e.g., due to heterogeneous filesystem mounts), this fails loudly to avoid
    silent correctness issues. To force a specific behavior, set PYINE_PER_NODE_PREP=on or
    PYINE_PER_NODE_PREP=off for all ranks.

    For single-node setups, returns False immediately without initialization.

    Returns:
        True if per-node data preparation should be used (all ranks unanimous), False otherwise
        (single-node, or unanimous disable).
    """
    num_nodes = get_num_nodes(default=1)
    if num_nodes is None or num_nodes <= 1:
        return False
    # multi-node: initialize DDP early so we can gather decisions
    ensure_torch_distributed_initialized()
    # compute local decision (may differ across nodes due to /proc/mounts)
    local_decision = is_per_node_prep_enabled()
    node_rank = get_node_rank(default=-1)
    cache_path = pyine.utils.filesystem.get_data_cache_path()
    # gather decisions from all ranks with diagnostic info
    local_info = (local_decision, node_rank, str(cache_path))
    all_info = all_gather_objects(local_info)
    # check for unanimity - all ranks must agree
    all_decisions = [info[0] for info in all_info]
    if all(all_decisions):
        # all ranks want per-node prep enabled
        logger.info("per-node prep enabled (unanimous agreement across all ranks)")
        return True
    if not any(all_decisions):
        # all ranks want per-node prep disabled
        logger.info("per-node prep disabled (unanimous agreement across all ranks)")
        return False
    # disagreement detected: fail loudly
    node_decisions: dict[int, set[bool]] = {}
    for decision, n_rank, _ in all_info:
        if n_rank not in node_decisions:
            node_decisions[n_rank] = set()
        node_decisions[n_rank].add(decision)
    node_summary = ", ".join(
        f"node {n}: {'enable' if True in decisions else 'disable'}" for n, decisions in sorted(node_decisions.items())
    )
    raise RuntimeError(
        "per-node prep decision disagreement across ranks. "
        f"Decisions by node: [{node_summary}]. "
        "This typically means some nodes see the cache path as shared storage while others see it as local. "
        "Set PYINE_PER_NODE_PREP=on or PYINE_PER_NODE_PREP=off for all ranks to force a deterministic behavior."
    )


def validate_node_configuration(use_per_node_prep: bool) -> None:
    """Validate node configuration for per-node data preparation.

    When per-node prep is enabled:
    - LOCAL_WORLD_SIZE MUST be available (FAIL LOUDLY if missing);
    - WORLD_SIZE % LOCAL_WORLD_SIZE == 0 (homogeneous nodes required);
    - torch.distributed must be initializable via env:// so cross-node validation can run.

    Args:
        use_per_node_prep: The synchronized per-node prep decision from determine_per_node_prep_mode().
            This MUST be the already-synchronized decision, not a local re-computation.

    Raises:
        RuntimeError: If per-node prep is enabled but configuration is invalid.
    """
    if not use_per_node_prep:
        return
    world_size = get_world_size(default=None)
    local_world_size = get_local_world_size(default=None)
    if local_world_size is None:
        raise RuntimeError(
            "Per-node data preparation enabled but LOCAL_WORLD_SIZE is not available. "
            "Set LOCAL_WORLD_SIZE env var or ensure your launcher provides it."
        )
    if world_size is None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            raise RuntimeError(
                "Per-node data preparation enabled but WORLD_SIZE is not available. "
                "Set WORLD_SIZE env var or ensure torch.distributed is initialized."
            )
    if local_world_size <= 0:
        raise RuntimeError(f"Invalid LOCAL_WORLD_SIZE: {local_world_size}")
    if world_size % local_world_size != 0:
        raise RuntimeError(
            f"Heterogeneous node configuration detected: WORLD_SIZE={world_size} is not "
            f"divisible by LOCAL_WORLD_SIZE={local_world_size}. Per-node data preparation "
            "requires homogeneous nodes (same number of processes on each node)."
        )
    num_nodes = world_size // local_world_size
    logger.info(f"per-node data preparation enabled: {num_nodes} nodes, {local_world_size} processes per node")
    if not torch.distributed.is_available():
        raise RuntimeError(
            "per-node data preparation is enabled but torch.distributed is not available",
        )
    if not torch.distributed.is_initialized():
        master_addr = _read_first_env_str(_MASTER_ADDR_ENV_KEYS)
        master_port = _read_first_env_str(_MASTER_PORT_ENV_KEYS)
        if master_addr is None or master_port is None:
            raise RuntimeError(
                "per-node data preparation requires cross-node fingerprint validation, "
                "but torch.distributed is not initialized and MASTER_ADDR/MASTER_PORT are missing; "
                "use torchrun/accelerate or initialize torch.distributed before calling distrib utils.",
            )


def ensure_torch_distributed_initialized() -> None:
    """Ensure torch.distributed is initialized when distributed execution is detected.

    Per-node preparation requires cross-node fingerprint validation. That validation uses
    collectives, so we initialize the default process group early (before HF Trainer/Accelerate).

    HF Trainer/Accelerate/DeepSpeed compatibility:
        These libraries check ``torch.distributed.is_initialized()`` before initializing,
        so early initialization here is safe - they will reuse our process group. We use
        standard defaults (nccl on CUDA, gloo otherwise, env:// init method) that match
        typical launcher configurations. If you encounter issues with non-standard distributed
        settings, set ``PYINE_SKIP_DDP_INIT=1`` to skip early initialization (cross-node
        fingerprint validation will then require the downstream library to initialize first).

    This intentionally fails loudly when required launcher environment variables are missing.
    """
    if os.environ.get("PYINE_SKIP_DDP_INIT", "").lower() in ("1", "true", "on"):
        logger.info("skipping early DDP init due to PYINE_SKIP_DDP_INIT")
        return
    if not is_distributed():
        return
    if not torch.distributed.is_available():
        raise RuntimeError(
            "distributed execution detected (WORLD_SIZE>1) but torch.distributed is not available",
        )
    if torch.distributed.is_initialized():
        return
    rank = get_global_rank(default=None)
    world_size = get_world_size(default=None)
    if rank is None or world_size is None:
        raise RuntimeError(
            "distributed execution detected but rank/world_size could not be determined; "
            "ensure your launcher sets RANK and WORLD_SIZE (or equivalent)",
        )
    master_addr = _read_first_env_str(_MASTER_ADDR_ENV_KEYS)
    master_port = _read_first_env_str(_MASTER_PORT_ENV_KEYS)
    if master_addr is None or master_port is None:
        raise RuntimeError(
            "per-node data preparation requires torch.distributed to be initializable via env://, "
            "but MASTER_ADDR or MASTER_PORT is missing; use torchrun/accelerate or set these env vars",
        )
    # ensure MASTER_ADDR/MASTER_PORT are set for init_method="env://" (it only reads these specific vars)
    if os.environ.get("MASTER_ADDR") is None:
        os.environ["MASTER_ADDR"] = master_addr
    if os.environ.get("MASTER_PORT") is None:
        os.environ["MASTER_PORT"] = master_port
    backend = os.environ.get("PYINE_TORCH_DISTRIBUTED_BACKEND")
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    backend = backend.lower()
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("PYINE_TORCH_DISTRIBUTED_BACKEND=nccl but CUDA is not available")
        local_rank = get_local_rank(default=None)
        if local_rank is None:
            raise RuntimeError("NCCL backend requires LOCAL_RANK to be set by the launcher")
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError("CUDA is available but torch.cuda.device_count() returned 0")
        if not (0 <= local_rank < device_count):
            raise RuntimeError(
                f"invalid LOCAL_RANK={local_rank} for device_count={device_count}",
            )
        torch.cuda.set_device(local_rank)
    timeout_seconds_raw = os.environ.get("PYINE_DDP_INIT_TIMEOUT_SECONDS", "600")
    try:
        timeout_seconds = float(timeout_seconds_raw)
    except ValueError as exc:
        raise RuntimeError(f"invalid PYINE_DDP_INIT_TIMEOUT_SECONDS={timeout_seconds_raw!r}") from exc
    torch.distributed.init_process_group(  # type: ignore[reportUnknownMemberType]
        backend=backend,
        init_method="env://",
        rank=int(rank),
        world_size=int(world_size),
        timeout=datetime.timedelta(seconds=timeout_seconds),
    )


def validate_cross_node_fingerprints(
    payload: FingerprintPayload | None,
    description: str = "prepared data",
) -> None:
    """Validate all nodes prepared identical data.

    MUST be called by ALL ranks to avoid deadlock. Non-local-rank-0 processes pass payload=None.
    Local-rank-0 passes FingerprintPayload(ok, fingerprint, error). RAISES on ALL ranks if
    validation fails OR if any node had an error.

    REQUIRES torch.distributed.is_initialized() for multi-node.

    Args:
        payload: FingerprintPayload from local-rank-0, None for other ranks.
        description: Human-readable label for error messages.

    Raises:
        RuntimeError: If validation fails or torch.distributed is not initialized.
    """
    if not is_distributed():
        return
    num_nodes = get_num_nodes(default=1)
    if num_nodes is None or num_nodes <= 1:
        return
    ensure_torch_distributed_initialized()
    local_rank = get_local_rank(default=0)
    node_rank = get_node_rank(default=0)
    gather_payload: FingerprintPayload | None = None
    if local_rank == 0 and payload is not None:
        gather_payload = FingerprintPayload(
            ok=payload.ok,
            fingerprint=payload.fingerprint,
            error=payload.error,
            node_rank=node_rank if node_rank is not None else -1,
        )
    all_payloads = all_gather_objects(gather_payload)
    validation_passed = True
    error_message = ""
    if is_main_process():
        node_payloads = {p.node_rank: p for p in all_payloads if p is not None}
        errors = [(n, p.error) for n, p in node_payloads.items() if not p.ok]
        if errors:
            validation_passed = False
            error_info = "; ".join(f"node {n}: {e}" for n, e in errors)
            error_message = f"Fingerprint computation failed: {error_info}"
        elif len(node_payloads) != num_nodes:
            validation_passed = False
            error_message = (
                f"Expected {num_nodes} node payloads, got {len(node_payloads)}. "
                f"Observed node ranks: {sorted(node_payloads.keys())}"
            )
        else:
            fingerprints = {n: p.fingerprint for n, p in node_payloads.items()}
            sorted_nodes = sorted(fingerprints.keys())
            reference_node = sorted_nodes[0]
            reference_fp = fingerprints[reference_node]
            mismatches = [(n, fp) for n, fp in fingerprints.items() if fp != reference_fp]
            if mismatches:
                validation_passed = False
                mismatch_info = ", ".join(
                    f"node {n}: {fp[:32]}..." if fp else f"node {n}: None" for n, fp in mismatches
                )
                ref_display = f"{reference_fp[:32]}..." if reference_fp else "None"
                error_message = (
                    f"Cross-node {description} mismatch! "
                    f"Reference (node {reference_node}): {ref_display}, "
                    f"mismatches: {mismatch_info}"
                )
            else:
                logger.info(f"cross-node {description} validation passed ({len(fingerprints)} nodes)")
    validation_passed = broadcast_boolean(validation_passed, src=0)
    error_message = broadcast_object(error_message, src=0)
    if not validation_passed:
        raise RuntimeError(f"Cross-node validation failed: {error_message}")


def _resolve_local_barrier_root() -> pathlib.Path:
    """Return node-local path for intra-node barrier rendezvous.

    ALWAYS uses tempfile.gettempdir() (typically /tmp). Never uses PYINE_CACHE_ROOT because that
    could be shared storage.
    """
    return pathlib.Path(tempfile.gettempdir()) / "pyine_local_barriers"


def _fallback_local_barrier() -> None:
    """Synchronize local ranks on a node using filesystem-based rendezvous.

    Similar to _fallback_barrier_if_needed but scoped to local ranks only.
    """
    local_world_size = get_local_world_size(default=None)
    local_rank = get_local_rank(default=None)
    if local_world_size is None or local_world_size <= 1 or local_rank is None:
        return
    barrier_idx = next(_LOCAL_BARRIER_COUNTER)
    job_dir = _resolve_local_barrier_root() / _fallback_job_token()
    job_dir.mkdir(parents=True, exist_ok=True)
    node_rank = get_node_rank(default=0) or 0
    barrier_dir = job_dir / f"local_barrier_node{node_rank}_{barrier_idx}"
    barrier_dir.mkdir(parents=True, exist_ok=True)
    rank_file = barrier_dir / f"local_rank_{local_rank}"
    rank_file.touch(exist_ok=False)
    deadline = time.monotonic() + _FALLBACK_BARRIER_TIMEOUT_SECONDS
    while True:
        participants = list(barrier_dir.iterdir())
        if len(participants) >= local_world_size:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"local barrier timed out waiting for {local_world_size} local ranks (saw {len(participants)})",
            )
        time.sleep(0.1)
    grace_period = 0.5
    time.sleep(grace_period)
    rank_file.unlink()
    with contextlib.suppress(OSError):
        barrier_dir.rmdir()
    with contextlib.suppress(OSError):
        job_dir.rmdir()


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


def _read_first_env_str(
    keys: tuple[str, ...],
) -> str | None:
    """Return the first environment variable among `keys` that is set and non-empty."""
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
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
    exist, the barrier completes. We scope rendezvous directories under a per-job token derived from
    launch environment variables (or `PYINE_BARRIER_TOKEN`) so concurrent jobs on the same node do not
    interfere. Cleanup is best-effort—whichever rank finishes last removes the directory tree. This
    provides a safe pre-DDP synchronization point without double-initializing the process group
    Hugging Face owns.
    """
    world_size = get_world_size(default=None)
    rank = get_global_rank(default=None)
    if world_size is None or world_size <= 1 or rank is None:
        return
    barrier_idx = next(_FALLBACK_BARRIER_COUNTER)
    job_dir = _resolve_barrier_root() / _fallback_job_token()
    job_dir.mkdir(parents=True, exist_ok=True)
    barrier_dir = job_dir / f"{_FALLBACK_BARRIER_PREFIX}_{barrier_idx}"
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
    # Grace period: Give all ranks time to observe the completion state before anyone deletes
    # This prevents a race where the first rank to delete causes others to see incomplete state
    grace_period = 1.0  # 1 second should be more than enough for all ranks to see completion
    time.sleep(grace_period)
    rank_file.unlink()
    with contextlib.suppress(OSError):
        barrier_dir.rmdir()
    with contextlib.suppress(OSError):
        job_dir.rmdir()


def _fallback_job_token() -> str:
    """Return a per-job identifier to avoid cross-run rendezvous collisions.

    Priority order:
    1. User-provided `PYINE_BARRIER_TOKEN` (guaranteed shared across ranks).
    2. `TORCHELASTIC_RUN_ID`, populated by torchrun/elastic launches.
    3. Tuple of `MASTER_ADDR` and `MASTER_PORT`, which differ across concurrent standalone torchrun jobs.
    4. A `_default` sentinel when no information is available. Users should set
       `PYINE_BARRIER_TOKEN` (or run under Hydra, which gives us a distinct root) if they expect to
       run multiple pre-DDP barriers concurrently in the same temp directory.
    """
    explicit = os.environ.get("PYINE_BARRIER_TOKEN")
    if explicit:
        return _sanitize_token(explicit)
    torchelastic_run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    if torchelastic_run_id:
        return _sanitize_token(torchelastic_run_id)
    master_addr = _read_first_env_str(_MASTER_ADDR_ENV_KEYS)
    master_port = _read_first_env_str(_MASTER_PORT_ENV_KEYS)
    if master_addr or master_port:
        token = f"master_{master_addr or 'n'}_{master_port or '0'}"
        return _sanitize_token(token)
    return "_default"


def _sanitize_token(token: str) -> str:
    """Return a filesystem-safe token derived from the provided string."""
    allowed: list[str] = []
    for char in token:
        if char.isalnum() or char in ("-", "_"):
            allowed.append(char)
        else:
            allowed.append("_")
    sanitized = "".join(allowed).strip("_")
    return sanitized or "_default"


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
