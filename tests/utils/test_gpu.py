import pytest
import pytest_mock

import pyine.utils.gpu as gpu_module


class TestParsePciBusId:
    def test_parse_standard_format(self) -> None:
        result = gpu_module._parse_pci_bus_id("0000:3b:00.0")
        assert result == (0, 0x3B, 0, 0)

    def test_parse_with_whitespace(self) -> None:
        result = gpu_module._parse_pci_bus_id("  0000:3B:00.0  ")
        assert result == (0, 0x3B, 0, 0)

    def test_parse_long_domain(self) -> None:
        # NVML sometimes uses 8-digit domain padding
        result = gpu_module._parse_pci_bus_id("00000000:3b:00.0")
        assert result == (0, 0x3B, 0, 0)

    def test_parse_invalid_format(self) -> None:
        assert gpu_module._parse_pci_bus_id("invalid") is None
        assert gpu_module._parse_pci_bus_id("0000:3b:00") is None  # missing function
        assert gpu_module._parse_pci_bus_id("") is None


class TestPciBusIdsMatch:
    def test_match_identical(self) -> None:
        assert gpu_module._pci_bus_ids_match("0000:3B:00.0", "0000:3B:00.0")

    def test_match_case_insensitive(self) -> None:
        assert gpu_module._pci_bus_ids_match("0000:3b:00.0", "0000:3B:00.0")

    def test_match_different_domain_padding(self) -> None:
        # key test: PyTorch uses 4-digit domain, NVML might use 8-digit
        assert gpu_module._pci_bus_ids_match("0000:3b:00.0", "00000000:3b:00.0")
        assert gpu_module._pci_bus_ids_match("00000000:3B:00.0", "0000:3B:00.0")

    def test_match_with_whitespace(self) -> None:
        assert gpu_module._pci_bus_ids_match("  0000:3b:00.0  ", "0000:3B:00.0")

    def test_no_match_different_bus(self) -> None:
        assert not gpu_module._pci_bus_ids_match("0000:3b:00.0", "0000:4c:00.0")

    def test_no_match_different_device(self) -> None:
        assert not gpu_module._pci_bus_ids_match("0000:3b:00.0", "0000:3b:01.0")

    def test_fallback_string_compare_on_parse_failure(self) -> None:
        # if one can't be parsed, falls back to string comparison
        assert gpu_module._pci_bus_ids_match("INVALID", "INVALID")
        assert gpu_module._pci_bus_ids_match("invalid", "INVALID")  # case-insensitive fallback
        assert not gpu_module._pci_bus_ids_match("INVALID", "OTHER")


class TestGPUStatsDataclass:
    def test_gpustats_frozen(self) -> None:
        stats = gpu_module.GPUStats(
            torch_device_index=0,
            device_key="test-uuid",
            utilization_gpu_percent=50.0,
            utilization_mem_controller_percent=30.0,
            vram_used_bytes=1000,
            vram_total_bytes=2000,
            vram_used_percent=50.0,
            power_watts=100.0,
            temperature_celsius=60.0,
            pytorch_allocated_bytes=500,
            pytorch_reserved_bytes=1000,
            pytorch_max_allocated_bytes=800,
            pytorch_max_reserved_bytes=1200,
            pytorch_total_bytes=2000,
        )
        assert stats.torch_device_index == 0
        assert stats.device_key == "test-uuid"
        with pytest.raises(AttributeError):
            stats.device_key = "new-uuid"  # type: ignore[misc]

    def test_gpustats_none_nvml_fields(self) -> None:
        stats = gpu_module.GPUStats(
            torch_device_index=0,
            device_key="cuda:0",
            utilization_gpu_percent=None,
            utilization_mem_controller_percent=None,
            vram_used_bytes=None,
            vram_total_bytes=None,
            vram_used_percent=None,
            power_watts=None,
            temperature_celsius=None,
            pytorch_allocated_bytes=0,
            pytorch_reserved_bytes=0,
            pytorch_max_allocated_bytes=0,
            pytorch_max_reserved_bytes=0,
            pytorch_total_bytes=1000,
        )
        assert stats.utilization_gpu_percent is None
        assert stats.power_watts is None
        assert stats.pytorch_total_bytes == 1000


class TestGetDeviceKey:
    def test_device_key_cuda_fallback(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """When NVML unavailable and no pci_bus_id, use cuda:idx fallback."""
        mock_props = mocker.MagicMock()
        mock_props.pci_bus_id = None
        mocker.patch("torch.cuda.get_device_properties", return_value=mock_props)
        result = gpu_module._get_device_key(0, None)
        assert result == "cuda:0"

    def test_device_key_pci_fallback(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """When NVML unavailable but pci_bus_id exists, use pci: prefix."""
        mock_props = mocker.MagicMock()
        mock_props.pci_bus_id = "0000:3B:00.0"
        mocker.patch("torch.cuda.get_device_properties", return_value=mock_props)
        result = gpu_module._get_device_key(0, None)
        assert result == "pci:0000:3B:00.0"


class TestGPUStatsCollector:
    def test_collector_disabled_when_no_cuda(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("torch.cuda.is_available", return_value=False)
        collector = gpu_module.GPUStatsCollector(require_nvml=False)
        assert collector.is_enabled() is False
        assert collector.collect_current_device() is None
        assert collector.collect_all_visible_devices() == []

    def test_collector_raises_when_require_nvml_and_no_cuda(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("torch.cuda.is_available", return_value=False)
        with pytest.raises(RuntimeError, match="CUDA not available"):
            gpu_module.GPUStatsCollector(require_nvml=True)

    def test_collector_enabled_with_cuda_no_nvml(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch.object(gpu_module, "_init_nvml", return_value=False)
        collector = gpu_module.GPUStatsCollector(require_nvml=False)
        assert collector.is_enabled() is True
        assert collector.is_nvml_available() is False

    def test_collector_raises_when_require_nvml_but_nvml_unavailable(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch.object(gpu_module, "_init_nvml", return_value=False)
        with pytest.raises(RuntimeError, match="NVML not available"):
            gpu_module.GPUStatsCollector(require_nvml=True)

    def test_reset_pytorch_peak_stats_disabled(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Reset should be no-op when collector disabled."""
        mocker.patch("torch.cuda.is_available", return_value=False)
        collector = gpu_module.GPUStatsCollector(require_nvml=False)
        collector.reset_pytorch_peak_stats()  # should not raise


class TestCollectStatsForDevice:
    def test_collect_stats_basic(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test collecting stats without NVML (PyTorch-only)."""
        mocker.patch.object(gpu_module, "_pynvml", None)
        mock_props = mocker.MagicMock()
        mock_props.pci_bus_id = None
        mock_props.total_memory = 8_000_000_000
        mocker.patch("torch.cuda.get_device_properties", return_value=mock_props)
        mocker.patch("torch.cuda.memory_allocated", return_value=1_000_000_000)
        mocker.patch("torch.cuda.memory_reserved", return_value=2_000_000_000)
        mocker.patch("torch.cuda.max_memory_allocated", return_value=1_500_000_000)
        mocker.patch("torch.cuda.max_memory_reserved", return_value=2_500_000_000)
        stats = gpu_module._collect_stats_for_device(0)
        assert stats.torch_device_index == 0
        assert stats.device_key == "cuda:0"
        assert stats.utilization_gpu_percent is None  # no NVML
        assert stats.power_watts is None
        assert stats.pytorch_allocated_bytes == 1_000_000_000
        assert stats.pytorch_total_bytes == 8_000_000_000


class TestNVMLInitialization:
    def test_init_nvml_import_error(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test _init_nvml handles ImportError gracefully."""
        # reset global state
        gpu_module._pynvml = None
        gpu_module._nvml_initialized = False
        mocker.patch.dict("sys.modules", {"pynvml": None})
        mocker.patch("builtins.__import__", side_effect=ImportError("no pynvml"))
        result = gpu_module._init_nvml()
        assert result is False
        assert gpu_module._nvml_initialized is True  # marked as attempted

    def test_init_nvml_idempotent(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test _init_nvml is idempotent after first call."""
        gpu_module._pynvml = None
        gpu_module._nvml_initialized = True  # already initialized
        result = gpu_module._init_nvml()
        assert result is False  # _pynvml is None
