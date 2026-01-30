"""GPU statistics collection utilities.

This module provides utilities for collecting GPU statistics using pynvml (nvidia-ml-py) and PyTorch.
It handles the mapping between PyTorch device indices (which can be remapped via CUDA_VISIBLE_DEVICES)
and NVML physical device handles using PCI bus ID matching.
"""

import atexit
import contextlib
import dataclasses
import logging
import typing

import torch

logger = logging.getLogger(__name__)

# lazy-loaded pynvml module reference (set by _init_nvml)
_pynvml: typing.Any = None
_nvml_initialized: bool = False
_nvml_init_error: Exception | None = None  # stores non-ImportError init failure


def _init_nvml() -> bool:
    """Initializes NVML library lazily. Returns True if successful.

    This function is idempotent and thread-safe for single-threaded initialization.
    """
    global _pynvml, _nvml_initialized, _nvml_init_error
    if _nvml_initialized:
        return _pynvml is not None
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()  # type: ignore[reportUnknownMemberType]
        _pynvml = pynvml
        _nvml_initialized = True
        atexit.register(_shutdown_nvml)
        logger.debug("NVML initialization successfull")
        return True
    except ImportError:
        logger.info("pynvml not installed; NVML stats will be unavailable")
        _nvml_initialized = True
        return False
    except Exception as exc:  # noqa: BLE001
        # non-ImportError failure (e.g., broken NVML install, driver mismatch)
        logger.warning(f"NVML initialization failed: {exc}")
        _nvml_init_error = exc
        _nvml_initialized = True
        return False


def _shutdown_nvml() -> None:
    """Shutdown NVML library (registered via atexit)."""
    global _pynvml, _nvml_initialized, _nvml_init_error
    if _pynvml is not None:
        try:
            _pynvml.nvmlShutdown()
            logger.debug("NVML shutdown successfull")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"failed to properly shutdown NVML: {exc}")
    _pynvml = None
    _nvml_initialized = False
    _nvml_init_error = None


def _parse_pci_bus_id(pci_bus_id: str) -> tuple[int, int, int, int] | None:
    """Parse PCI bus ID into numeric components.

    PCI bus IDs have the format domain:bus:device.function (e.g., "0000:3b:00.0"). Different
    sources may use different domain padding (e.g., "00000000:3b:00.0" vs "0000:3b:00.0"), so we
    parse into integers for reliable comparison.

    Args:
        pci_bus_id: Raw PCI bus ID string.

    Returns:
        Tuple of (domain, bus, device, function) as integers, or None if parsing fails.
    """
    try:
        cleaned = pci_bus_id.strip().upper()
        # format: domain:bus:device.function
        parts = cleaned.replace(".", ":").split(":")
        if len(parts) != 4:
            return None
        domain = int(parts[0], 16)
        bus = int(parts[1], 16)
        device = int(parts[2], 16)
        function = int(parts[3], 16)
        return domain, bus, device, function
    except (ValueError, IndexError):
        return None


def _pci_bus_ids_match(pci_bus_id_a: str, pci_bus_id_b: str) -> bool:
    """Check if two PCI bus IDs refer to the same device.

    Compares PCI bus IDs by parsing into numeric components, handling format differences like
    domain padding (e.g., "00000000:3b:00.0" vs "0000:3b:00.0").

    Args:
        pci_bus_id_a: First PCI bus ID string.
        pci_bus_id_b: Second PCI bus ID string.

    Returns:
        True if both IDs refer to the same device, False otherwise.
    """
    parsed_a = _parse_pci_bus_id(pci_bus_id_a)
    parsed_b = _parse_pci_bus_id(pci_bus_id_b)
    if parsed_a is None or parsed_b is None:
        # fallback to string comparison if parsing fails
        return pci_bus_id_a.strip().upper() == pci_bus_id_b.strip().upper()
    return parsed_a == parsed_b


def _get_nvml_handle_for_torch_device(
    torch_device_index: int,
) -> typing.Any | None:
    """Get NVML handle for a PyTorch device using PCI bus ID mapping.

    This handles the case where CUDA_VISIBLE_DEVICES remaps device indices. PyTorch uses logical
    indices (0, 1, 2...) while NVML uses physical indices. We map via PCI bus ID to find the
    correct physical device.

    Args:
        torch_device_index: PyTorch logical device index.

    Returns:
        NVML device handle, or None if mapping fails or NVML unavailable.
    """
    if _pynvml is None:
        return None
    try:
        props = torch.cuda.get_device_properties(torch_device_index)  # type: ignore[reportUnknownMemberType]
        pci_bus_id = getattr(props, "pci_bus_id", None)  # type: ignore[reportUnknownArgumentType]
        if pci_bus_id is None:
            logger.warning(f"torch device {torch_device_index} has no pci_bus_id attribute")
            return None
        # try direct lookup first (most efficient) when pci_bus_id is str/bytes
        # note: nvidia-ml-py>=12.0.0 handles str/bytes conversion internally
        if isinstance(pci_bus_id, (str, bytes)):
            try:
                return _pynvml.nvmlDeviceGetHandleByPciBusId(pci_bus_id)
            except _pynvml.NVMLError:
                pass
        else:
            logger.debug(
                f"torch device {torch_device_index} has unexpected pci_bus_id type "
                f"{type(pci_bus_id)}; falling back to NVML PCI matching"
            )
        # optional UUID-based lookup before fallback PCI matching
        cuda_uuid = getattr(props, "uuid", None)
        if cuda_uuid is not None:
            cuda_uuid_str = str(cuda_uuid)
            with contextlib.suppress(_pynvml.NVMLError):
                return _pynvml.nvmlDeviceGetHandleByUUID(cuda_uuid_str)
        # fallback: iterate NVML devices and match by PCI info (handles format differences)
        pci_domain_id = getattr(props, "pci_domain_id", None)
        pci_device_id = getattr(props, "pci_device_id", None)
        pci_bus_id_int = pci_bus_id if isinstance(pci_bus_id, int) else None
        target_pci_tuple: tuple[int, int, int, int] | None = None
        if (
            pci_domain_id is not None
            and pci_bus_id_int is not None
            and pci_device_id is not None
            and all(isinstance(value, int) for value in [pci_domain_id, pci_bus_id_int, pci_device_id])
        ):
            target_pci_tuple = (pci_domain_id, pci_bus_id_int, pci_device_id, 0)
        device_count = _pynvml.nvmlDeviceGetCount()
        for nvml_idx in range(device_count):
            handle = _pynvml.nvmlDeviceGetHandleByIndex(nvml_idx)
            nvml_pci = _pynvml.nvmlDeviceGetPciInfo(handle)
            nvml_bus_id = nvml_pci.busId
            if isinstance(nvml_bus_id, bytes):
                nvml_bus_id = nvml_bus_id.decode()
            if isinstance(pci_bus_id, (str, bytes)) and _pci_bus_ids_match(nvml_bus_id, pci_bus_id):
                return handle
            if target_pci_tuple is not None:
                nvml_tuple = _parse_pci_bus_id(nvml_bus_id)
                if nvml_tuple is not None and nvml_tuple == target_pci_tuple:
                    return handle
        logger.warning(
            f"NVML initialized but no device found matching PyTorch device {torch_device_index} "
            f"(PCI bus ID: {pci_bus_id}); NVML stats will be unavailable for this device"
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"failed to get NVML handle for torch device {torch_device_index}: {exc}")
        return None


def _get_device_key(
    torch_device_index: int,
    nvml_handle: typing.Any | None,
) -> str:
    """Get a unique device key for identifying GPUs across nodes.

    Uses UUID if NVML available, else pci_bus_id, else cuda:idx fallback.

    Args:
        torch_device_index: PyTorch logical device index.
        nvml_handle: NVML device handle, or None.

    Returns:
        Unique device key string.
    """
    if nvml_handle is not None and _pynvml is not None:
        try:
            uuid = _pynvml.nvmlDeviceGetUUID(nvml_handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            return uuid
        except _pynvml.NVMLError:
            pass
    # fallback: PCI bus ID if available
    try:
        props = torch.cuda.get_device_properties(torch_device_index)  # type: ignore[reportUnknownMemberType]
        pci_bus_id = getattr(props, "pci_bus_id", None)  # type: ignore[reportUnknownArgumentType]
        if pci_bus_id:
            return f"pci:{pci_bus_id}"
    except Exception:  # noqa: BLE001, S110
        pass  # GPU properties lookup failed; fall through to cuda:idx fallback
    # last resort: cuda index (not globally unique, but works locally)
    return f"cuda:{torch_device_index}"


@dataclasses.dataclass(slots=True, frozen=True)
class GPUStats:
    """Snapshot of GPU statistics for a single device."""

    torch_device_index: int
    """PyTorch device index (logical)."""
    device_key: str
    """Unique key (UUID, pci:..., or cuda:idx)."""

    # NVML stats (None if NVML unavailable or unsupported)
    utilization_gpu_percent: float | None
    """GPU compute utilization (0-100)."""
    utilization_mem_controller_percent: float | None
    """Memory controller utilization (0-100)."""
    vram_used_bytes: int | None
    """VRAM currently used (bytes)."""
    vram_total_bytes: int | None
    """Total VRAM available (bytes)."""
    vram_used_percent: float | None
    """VRAM used as percentage of total (0-100)."""
    power_watts: float | None
    """Current power consumption (watts)."""
    temperature_celsius: float | None
    """GPU temperature (Celsius)."""

    # PyTorch stats (always available if CUDA)
    pytorch_allocated_bytes: int
    """Memory currently allocated by PyTorch tensors."""
    pytorch_reserved_bytes: int
    """Memory reserved by PyTorch caching allocator."""
    pytorch_max_allocated_bytes: int
    """Peak memory allocated since last reset."""
    pytorch_max_reserved_bytes: int
    """Peak memory reserved since last reset."""
    pytorch_total_bytes: int
    """Total device memory from get_device_properties."""


def _collect_stats_for_device(
    torch_device_index: int,
    nvml_handle: typing.Any | None = None,
    *,
    skip_handle_lookup: bool = False,
) -> GPUStats:
    """Collect GPU stats for a specific PyTorch device.

    Args:
        torch_device_index: PyTorch logical device index.
        nvml_handle: Pre-fetched NVML handle (for caching). If None and skip_handle_lookup is
            False, will look up the handle.
        skip_handle_lookup: If True and nvml_handle is None, skip NVML stats entirely instead of
            looking up the handle.

    Returns:
        GPUStats snapshot for the device.
    """
    if nvml_handle is None and not skip_handle_lookup:
        nvml_handle = _get_nvml_handle_for_torch_device(torch_device_index)
    device_key = _get_device_key(torch_device_index, nvml_handle)
    utilization_gpu: float | None = None
    utilization_mem_ctrl: float | None = None
    vram_used: int | None = None
    vram_total: int | None = None
    vram_used_pct: float | None = None
    power_watts: float | None = None
    temperature: float | None = None
    if nvml_handle is not None and _pynvml is not None:
        with contextlib.suppress(_pynvml.NVMLError):
            util = _pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
            utilization_gpu = float(util.gpu)
            utilization_mem_ctrl = float(util.memory)
        with contextlib.suppress(_pynvml.NVMLError):
            mem_info = _pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            vram_used = int(mem_info.used)
            vram_total = int(mem_info.total)
            if vram_total > 0:
                vram_used_pct = 100.0 * vram_used / vram_total
        with contextlib.suppress(_pynvml.NVMLError):
            # power in milliwatts, convert to watts
            power_mw = _pynvml.nvmlDeviceGetPowerUsage(nvml_handle)
            power_watts = float(power_mw) / 1000.0
        with contextlib.suppress(_pynvml.NVMLError):
            temperature = float(_pynvml.nvmlDeviceGetTemperature(nvml_handle, _pynvml.NVML_TEMPERATURE_GPU))
    # collect PyTorch stats (always available for CUDA devices)
    pytorch_allocated = torch.cuda.memory_allocated(torch_device_index)
    pytorch_reserved = torch.cuda.memory_reserved(torch_device_index)
    pytorch_max_allocated = torch.cuda.max_memory_allocated(torch_device_index)
    pytorch_max_reserved = torch.cuda.max_memory_reserved(torch_device_index)
    props = torch.cuda.get_device_properties(torch_device_index)  # type: ignore[reportUnknownMemberType]
    pytorch_total: int = props.total_memory  # type: ignore[reportUnknownMemberType]
    return GPUStats(
        torch_device_index=torch_device_index,
        device_key=device_key,
        utilization_gpu_percent=utilization_gpu,
        utilization_mem_controller_percent=utilization_mem_ctrl,
        vram_used_bytes=vram_used,
        vram_total_bytes=vram_total,
        vram_used_percent=vram_used_pct,
        power_watts=power_watts,
        temperature_celsius=temperature,
        pytorch_allocated_bytes=pytorch_allocated,
        pytorch_reserved_bytes=pytorch_reserved,
        pytorch_max_allocated_bytes=pytorch_max_allocated,
        pytorch_max_reserved_bytes=pytorch_max_reserved,
        pytorch_total_bytes=int(pytorch_total),  # type: ignore[reportUnknownArgumentType]
    )


class GPUStatsCollector:
    """Collects GPU statistics for the current process's device(s).

    This collector handles both NVML (nvidia-ml-py) and PyTorch memory statistics. It gracefully
    degrades when NVML is unavailable, providing PyTorch-only stats.

    Example:
        ```python
        collector = GPUStatsCollector()
        if collector.is_enabled():
            stats = collector.collect_current_device()
            if stats:
                print(f"GPU utilization: {stats.utilization_gpu_percent}%")
        ```
    """

    def __init__(
        self,
        *,
        require_nvml: bool = False,
    ) -> None:
        """Initialize the collector.

        Args:
            require_nvml: If True, raise RuntimeError if NVML unavailable.

        Raises:
            RuntimeError: If require_nvml=True and CUDA or NVML unavailable.
        """
        self._enabled = False
        self._nvml_available = False
        self._handle_cache: dict[int, typing.Any | None] = {}  # torch device index -> NVML handle
        if not torch.cuda.is_available():
            if require_nvml:
                raise RuntimeError("CUDA not available; cannot collect GPU stats")
            logger.warning("CUDA not available; GPU stats collector disabled (no GPU metrics will be logged)")
            return
        self._enabled = True
        self._nvml_available = _init_nvml()
        if require_nvml and not self._nvml_available:
            if _nvml_init_error is not None:
                raise RuntimeError(f"NVML initialization failed: {_nvml_init_error}") from _nvml_init_error
            raise RuntimeError(
                "NVML not available (pynvml not installed); install with: uv sync --extra gpu-monitoring"
            )
        if self._nvml_available:
            logger.debug("GPU stats collector initialized with NVML support")
        else:
            logger.debug("GPU stats collector initialized (PyTorch-only, no NVML)")

    def is_enabled(self) -> bool:
        """Return True if collector can collect stats (CUDA available)."""
        return self._enabled

    def is_nvml_available(self) -> bool:
        """Return True if NVML is available for detailed stats."""
        return self._nvml_available

    def _get_cached_nvml_handle(self, torch_device_index: int) -> typing.Any | None:
        """Get NVML handle for a device, caching for repeated lookups.

        Args:
            torch_device_index: PyTorch logical device index.

        Returns:
            Cached NVML device handle, or None if unavailable.
        """
        if not self._nvml_available:
            return None
        if torch_device_index not in self._handle_cache:
            self._handle_cache[torch_device_index] = _get_nvml_handle_for_torch_device(torch_device_index)
        return self._handle_cache[torch_device_index]

    def collect_current_device(self) -> GPUStats | None:
        """Collect stats for torch.cuda.current_device().

        Returns:
            GPUStats for current device, or None if collector disabled.
        """
        if not self._enabled:
            return None
        device_idx = torch.cuda.current_device()
        nvml_handle = self._get_cached_nvml_handle(device_idx)
        return _collect_stats_for_device(device_idx, nvml_handle, skip_handle_lookup=True)

    def collect_all_visible_devices(self) -> list[GPUStats]:
        """Collect stats for all CUDA_VISIBLE_DEVICES.

        Returns:
            List of GPUStats for all visible devices, or empty list if disabled.
        """
        if not self._enabled:
            return []
        device_count = torch.cuda.device_count()
        results: list[GPUStats] = []
        for idx in range(device_count):
            nvml_handle = self._get_cached_nvml_handle(idx)
            results.append(_collect_stats_for_device(idx, nvml_handle, skip_handle_lookup=True))
        return results

    def reset_pytorch_peak_stats(
        self,
        device_indices: list[int] | None = None,
    ) -> None:
        """Reset torch.cuda.reset_peak_memory_stats() for specified devices.

        Args:
            device_indices: List of device indices to reset, or None for current device only.
        """
        if not self._enabled:
            return
        if device_indices is None:
            device_indices = [torch.cuda.current_device()]
        for idx in device_indices:
            torch.cuda.reset_peak_memory_stats(idx)
