"""Tests for ProbeDataModule."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

import datasets
import pytest

import pyine.probes.data.datamodule
import pyine.probes.data.debug_dataset
from pyine.probes.data.datamodule import ProbeDataModule
from pyine.probes.data.datamodule_configs import ProbeDataModuleConfig


@pytest.fixture
def debug_lmdb(tmp_path: Path) -> Path:
    """Create a debug LMDB at tmp_path and return its path."""
    lmdb_path = tmp_path / "debug.lmdb"
    pyine.probes.data.debug_dataset.create_debug_probe_lmdb(lmdb_path, n_train=20, n_eval_families=10, seed=42)
    return lmdb_path


@pytest.fixture
def dm_config(debug_lmdb: Path) -> ProbeDataModuleConfig:
    return ProbeDataModuleConfig(lmdb_path=str(debug_lmdb))


@pytest.fixture
def datamodule(dm_config: ProbeDataModuleConfig) -> ProbeDataModule:
    return ProbeDataModule(dm_config)


class TestProbeDataModuleLifecycle:
    def test_prepare_data_creates_cache(self, datamodule: ProbeDataModule) -> None:
        datamodule.prepare_data()
        cache_path = datamodule._get_cache_path()
        assert cache_path.exists()

    def test_setup_loads_dataset(self, datamodule: ProbeDataModule) -> None:
        datamodule.prepare_data()
        datamodule.setup()
        ds = datamodule.get_probe_dataset()
        assert isinstance(ds, datasets.DatasetDict)
        assert "train" in ds
        assert "valid" in ds

    def test_get_probe_dataset_before_setup_raises(self, datamodule: ProbeDataModule) -> None:
        with pytest.raises(RuntimeError, match="not set up"):
            datamodule.get_probe_dataset()

    def test_teardown_clears_state(self, datamodule: ProbeDataModule) -> None:
        datamodule.prepare_data()
        datamodule.setup()
        datamodule.teardown()
        with pytest.raises(RuntimeError, match="not set up"):
            datamodule.get_probe_dataset()

    def test_full_lifecycle(self, datamodule: ProbeDataModule) -> None:
        datamodule.prepare_data()
        datamodule.setup()
        ds = datamodule.get_probe_dataset()
        assert len(ds["train"]) == 20
        assert len(ds["valid"]) == 30  # 10 families * 3 code types
        datamodule.teardown()


class TestProbeDataModuleAccessors:
    def test_code_type_mappings_populated(self, datamodule: ProbeDataModule) -> None:
        datamodule.prepare_data()
        datamodule.setup()
        ct_to_id = datamodule.code_type_to_id
        id_to_ct = datamodule.id_to_code_type
        assert len(ct_to_id) > 0
        assert len(id_to_ct) == len(ct_to_id)
        for ct, ct_id in ct_to_id.items():
            assert id_to_ct[ct_id] == ct

    def test_code_type_to_id_before_setup_raises(self, datamodule: ProbeDataModule) -> None:
        with pytest.raises(RuntimeError, match="not set up"):
            _ = datamodule.code_type_to_id

    def test_get_stats_returns_split_info(self, datamodule: ProbeDataModule) -> None:
        datamodule.prepare_data()
        datamodule.setup()
        stats = datamodule.get_stats()
        assert "train/num_samples" in stats
        assert "valid/num_samples" in stats
        assert stats["train/num_samples"] == 20
        assert stats["valid/num_samples"] == 30

    def test_get_fingerprint_inputs_returns_valid(self, datamodule: ProbeDataModule) -> None:
        datamodule.prepare_data()
        fp = datamodule.get_fingerprint_inputs()
        assert fp.extra_content is not None


class TestProbeDataModuleCache:
    def test_cache_is_reused(self, datamodule: ProbeDataModule) -> None:
        """Second prepare_data call is a no-op (cache already exists)."""
        datamodule.prepare_data()
        cache_path = datamodule._get_cache_path()
        mtime_first = cache_path.stat().st_mtime_ns
        # Second call should not recreate the cache
        datamodule.prepare_data()
        mtime_second = cache_path.stat().st_mtime_ns
        assert mtime_first == mtime_second

    def test_cache_path_missing_lmdb_raises(self, tmp_path: Path) -> None:
        cfg = ProbeDataModuleConfig(lmdb_path=str(tmp_path / "nonexistent.lmdb"))
        dm = ProbeDataModule(cfg)
        with pytest.raises(FileNotFoundError, match="does not exist"):
            dm._get_cache_path()

    def test_cache_fingerprints_data_mdb(self, debug_lmdb: Path, dm_config: ProbeDataModuleConfig) -> None:
        """Cache path is based on data.mdb stat, not directory stat."""
        dm = ProbeDataModule(dm_config)
        path1 = dm._get_cache_path()

        # Touch data.mdb to simulate a data change
        data_mdb = debug_lmdb / "data.mdb"
        assert data_mdb.exists()
        original_content = data_mdb.read_bytes()
        data_mdb.write_bytes(original_content + b"\x00")  # append a byte to change size

        path2 = dm._get_cache_path()
        assert path1 != path2, "Cache path should change when data.mdb changes"


class TestProbeDataModuleStaleCacheValidation:
    def test_stale_cache_with_text_column_raises(self, datamodule: ProbeDataModule) -> None:
        """If a cached dataset has 'text' instead of 'messages', setup() raises."""
        import shutil

        datamodule.prepare_data()
        cache_path = datamodule._get_cache_path()

        # tamper with cached dataset: load -> rename -> save to tmp -> replace original
        stale_ds = datasets.DatasetDict.load_from_disk(str(cache_path))
        for split_name in stale_ds:
            stale_ds[split_name] = stale_ds[split_name].rename_column("messages", "text")
        # save to a temporary path, then replace the original (HF prevents in-place overwrite)
        tmp_stale_path = cache_path.parent / f"{cache_path.name}_stale"
        stale_ds.save_to_disk(str(tmp_stale_path))
        shutil.rmtree(cache_path)
        tmp_stale_path.rename(cache_path)

        # setup() should detect the stale schema and raise
        with pytest.raises(ValueError, match="missing columns"):
            datamodule.setup()

    def test_valid_cache_is_not_regenerated(self, datamodule: ProbeDataModule) -> None:
        """A cache with correct schema is loaded without regeneration."""
        datamodule.prepare_data()
        cache_path = datamodule._get_cache_path()
        mtime_before = cache_path.stat().st_mtime_ns
        datamodule.setup()
        ds = datamodule.get_probe_dataset()
        assert "messages" in ds["train"].column_names
        mtime_after = cache_path.stat().st_mtime_ns
        assert mtime_before == mtime_after


class TestProbeDataModuleNotImplemented:
    def test_train_dataloader_raises(self, datamodule: ProbeDataModule) -> None:
        with pytest.raises(NotImplementedError):
            datamodule.train_dataloader()

    def test_val_dataloader_raises(self, datamodule: ProbeDataModule) -> None:
        with pytest.raises(NotImplementedError):
            datamodule.val_dataloader()

    def test_test_dataloader_raises(self, datamodule: ProbeDataModule) -> None:
        with pytest.raises(NotImplementedError):
            datamodule.test_dataloader()

    def test_predict_dataloader_raises(self, datamodule: ProbeDataModule) -> None:
        with pytest.raises(NotImplementedError):
            datamodule.predict_dataloader()
