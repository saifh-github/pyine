import multiprocessing
import os
import pathlib
import typing
from unittest.mock import patch

import pytest

import pyine.utils.distrib
import pyine.utils.filesystem
import pyine.utils.reprod


@pytest.fixture(autouse=True)
def clear_distrib_env(
    monkeypatch: pytest.MonkeyPatch,
) -> typing.Iterator[None]:
    """Ensure distributed-related environment variables are unset for each test."""
    keys = (
        set(pyine.utils.distrib._GLOBAL_RANK_ENV_KEYS)
        | set(pyine.utils.distrib._LOCAL_RANK_ENV_KEYS)
        | set(pyine.utils.distrib._WORLD_SIZE_ENV_KEYS)
        | set(pyine.utils.distrib._LOCAL_WORLD_SIZE_ENV_KEYS)
        | set(pyine.utils.distrib._NODE_RANK_ENV_KEYS)
        | set(pyine.utils.distrib._NNODES_ENV_KEYS)
        | set(pyine.utils.distrib._MASTER_ADDR_ENV_KEYS)
        | set(pyine.utils.distrib._MASTER_PORT_ENV_KEYS)
        | {"PYINE_PER_NODE_PREP", "PYINE_CACHE_ROOT"}
    )
    original_env = {key: os.environ.get(key) for key in keys}
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    try:
        yield
    finally:
        for key, value in original_env.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)


def test_get_global_rank_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "3")
    assert pyine.utils.distrib.get_global_rank() == 3


def test_get_global_rank_default_none() -> None:
    assert pyine.utils.distrib.get_global_rank(default=None) is None


def test_is_main_process_with_explicit_rank() -> None:
    assert pyine.utils.distrib.is_main_process(rank=0) is True
    assert pyine.utils.distrib.is_main_process(rank=5) is False


def test_is_main_process_with_negative_local_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "-1")
    assert pyine.utils.distrib.get_global_rank(default=None) is None
    assert pyine.utils.distrib.is_main_process() is True


def test_get_world_size_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "8")
    assert pyine.utils.distrib.get_world_size() == 8


def test_is_distributed_true_when_world_size_gt_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert pyine.utils.distrib.is_distributed() is True


def test_distributed_context_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv("WORLD_SIZE", "16")
    context = pyine.utils.distrib.get_distributed_context()
    assert context["rank"] == 4
    assert context["world_size"] == 16
    assert "local_rank" in context


def _barrier_worker(
    rank: int,
    world_size: int,
    barrier_root: str,
) -> None:
    """Exercise the filesystem-based barrier in a subprocess before DDP init."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["PYINE_BARRIER_ROOT"] = barrier_root
    pyine.utils.distrib.barrier()


def test_barrier_fallback_rendezvous(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    barrier_root = tmp_path_factory.mktemp("barrier")
    world_size = 2
    ctx = multiprocessing.get_context("spawn")
    workers = [
        ctx.Process(
            target=_barrier_worker,
            args=(rank, world_size, str(barrier_root)),
        )
        for rank in range(world_size)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
    for worker in workers:
        assert worker.exitcode == 0
    assert not any(barrier_root.iterdir())


class TestIsLocalMainProcess:
    def test_explicit_local_rank_zero(self) -> None:
        assert pyine.utils.distrib.is_local_main_process(local_rank=0) is True

    def test_explicit_local_rank_nonzero(self) -> None:
        assert pyine.utils.distrib.is_local_main_process(local_rank=1) is False
        assert pyine.utils.distrib.is_local_main_process(local_rank=5) is False

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert pyine.utils.distrib.is_local_main_process() is True
        monkeypatch.setenv("LOCAL_RANK", "2")
        assert pyine.utils.distrib.is_local_main_process() is False


class TestGetNodeRank:
    def test_derived_from_global_rank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RANK", "5")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
        assert pyine.utils.distrib.get_node_rank() == 2  # 5 // 2 = 2

    def test_explicit_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NODE_RANK", "3")
        assert pyine.utils.distrib.get_node_rank() == 3

    def test_slurm_nodeid_takes_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURM_NODEID", "7")
        monkeypatch.setenv("RANK", "10")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        # SLURM_NODEID=7 should be used, not 10//4=2
        assert pyine.utils.distrib.get_node_rank() == 7

    def test_default_when_not_available(self) -> None:
        assert pyine.utils.distrib.get_node_rank(default=0) == 0
        assert pyine.utils.distrib.get_node_rank(default=None) is None


class TestGetNumNodes:
    def test_derived_from_world_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        assert pyine.utils.distrib.get_num_nodes() == 2  # 8 // 4 = 2

    def test_explicit_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NNODES", "4")
        assert pyine.utils.distrib.get_num_nodes() == 4

    def test_slurm_nnodes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLURM_NNODES", "6")
        assert pyine.utils.distrib.get_num_nodes() == 6


class TestGetLocalWorldSize:
    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        assert pyine.utils.distrib.get_local_world_size() == 4

    def test_default_when_not_available(self) -> None:
        assert pyine.utils.distrib.get_local_world_size(default=1) == 1


class TestIsPerNodePrepEnabled:
    def test_single_node_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")  # 1 node
        assert pyine.utils.distrib.is_per_node_prep_enabled() is False

    def test_multi_node_with_shared_fs_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")  # 2 nodes
        with patch.object(pyine.utils.filesystem, "is_path_on_shared_filesystem", return_value=True):
            assert pyine.utils.distrib.is_per_node_prep_enabled() is False

    def test_multi_node_with_local_fs_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")  # 2 nodes
        with patch.object(pyine.utils.filesystem, "is_path_on_shared_filesystem", return_value=False):
            assert pyine.utils.distrib.is_per_node_prep_enabled() is True

    def test_override_on_forces_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PYINE_PER_NODE_PREP", "on")
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        # even if FS detection would say shared, override should enable
        assert pyine.utils.distrib.is_per_node_prep_enabled() is True

    def test_override_off_forces_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PYINE_PER_NODE_PREP", "off")
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        # even if FS detection would say local, override should disable
        assert pyine.utils.distrib.is_per_node_prep_enabled() is False

    def test_override_on_fails_with_single_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PYINE_PER_NODE_PREP", "on")
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")  # 1 node
        with pytest.raises(RuntimeError, match="single-node"):
            pyine.utils.distrib.is_per_node_prep_enabled()


class TestDeterminePerNodePrepMode:
    def test_single_node_returns_false_without_ddp_init(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")  # 1 node
        with patch.object(pyine.utils.distrib, "ensure_torch_distributed_initialized") as mock_init:
            result = pyine.utils.distrib.determine_per_node_prep_mode()
            assert result is False
            mock_init.assert_not_called()  # single-node skips DDP init

    def test_multi_node_unanimous_enable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All ranks agree to enable per-node prep."""
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")  # 2 nodes
        monkeypatch.setenv("RANK", "0")
        # Simulate all ranks returning (True, node_rank, cache_path)
        all_decisions = [(True, 0, "/local/cache")] * 4 + [(True, 1, "/local/cache")] * 4
        with (
            patch.object(pyine.utils.distrib, "ensure_torch_distributed_initialized") as mock_init,
            patch.object(pyine.utils.distrib, "all_gather_objects", return_value=all_decisions),
            patch.object(pyine.utils.filesystem, "is_path_on_shared_filesystem", return_value=False),
        ):
            result = pyine.utils.distrib.determine_per_node_prep_mode()
            assert result is True
            mock_init.assert_called_once()

    def test_multi_node_unanimous_disable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All ranks agree to disable per-node prep."""
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")  # 2 nodes
        monkeypatch.setenv("RANK", "0")
        # Simulate all ranks returning (False, node_rank, cache_path)
        all_decisions = [(False, 0, "/shared/cache")] * 4 + [(False, 1, "/shared/cache")] * 4
        with (
            patch.object(pyine.utils.distrib, "ensure_torch_distributed_initialized"),
            patch.object(pyine.utils.distrib, "all_gather_objects", return_value=all_decisions),
            patch.object(pyine.utils.filesystem, "is_path_on_shared_filesystem", return_value=True),
        ):
            result = pyine.utils.distrib.determine_per_node_prep_mode()
            assert result is False

    def test_multi_node_disagreement_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If nodes disagree, per-node prep fails loudly to avoid silent correctness issues."""
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")  # 2 nodes
        monkeypatch.setenv("RANK", "0")
        # Node 0 wants enable (local FS), node 1 wants disable (shared FS)
        all_decisions = [(True, 0, "/local/cache")] * 4 + [(False, 1, "/shared/cache")] * 4
        with (
            patch.object(pyine.utils.distrib, "ensure_torch_distributed_initialized"),
            patch.object(pyine.utils.distrib, "all_gather_objects", return_value=all_decisions),
            patch.object(pyine.utils.filesystem, "is_path_on_shared_filesystem", return_value=False),
            pytest.raises(RuntimeError, match="decision disagreement"),
        ):
            pyine.utils.distrib.determine_per_node_prep_mode()


class TestValidateNodeConfiguration:
    def test_passes_with_valid_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "16")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29500")
        pyine.utils.distrib.validate_node_configuration(use_per_node_prep=True)  # should not raise

    def test_fails_on_heterogeneous_nodes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "10")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")  # 10 % 4 != 0
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29500")
        with pytest.raises(RuntimeError, match="Heterogeneous node configuration"):
            pyine.utils.distrib.validate_node_configuration(use_per_node_prep=True)

    def test_skips_when_per_node_prep_disabled(self) -> None:
        # even with invalid config, validation should pass when use_per_node_prep=False
        pyine.utils.distrib.validate_node_configuration(use_per_node_prep=False)


class TestFingerprintInputs:
    def test_fingerprint_changes_with_metadata(self, tmp_path: pathlib.Path) -> None:
        metadata = tmp_path / "metadata.json"
        metadata.write_text('{"key": "value1"}')
        inputs1 = pyine.utils.reprod.FingerprintInputs(metadata_paths=[metadata])
        fp1 = pyine.utils.reprod.compute_data_fingerprint(inputs=inputs1)
        metadata.write_text('{"key": "value2"}')
        inputs2 = pyine.utils.reprod.FingerprintInputs(metadata_paths=[metadata])
        fp2 = pyine.utils.reprod.compute_data_fingerprint(inputs=inputs2)
        assert fp1 != fp2  # different metadata -> different fingerprint

    def test_fingerprint_changes_with_binary_metadata(self, tmp_path: pathlib.Path) -> None:
        metadata = tmp_path / "metadata.msgspec"
        metadata.write_bytes(b"\x01\x02\x03")
        inputs1 = pyine.utils.reprod.FingerprintInputs(metadata_paths=[metadata])
        fp1 = pyine.utils.reprod.compute_data_fingerprint(inputs=inputs1)
        metadata.write_bytes(b"\x01\x02\x04")
        inputs2 = pyine.utils.reprod.FingerprintInputs(metadata_paths=[metadata])
        fp2 = pyine.utils.reprod.compute_data_fingerprint(inputs=inputs2)
        assert fp1 != fp2

    def test_fingerprint_includes_config_hash(self, tmp_path: pathlib.Path) -> None:
        metadata = tmp_path / "metadata.json"
        metadata.write_text('{"key": "value"}')
        inputs = pyine.utils.reprod.FingerprintInputs(metadata_paths=[metadata])
        fp1 = pyine.utils.reprod.compute_data_fingerprint(inputs=inputs, config_hash="hash1")
        fp2 = pyine.utils.reprod.compute_data_fingerprint(inputs=inputs, config_hash="hash2")
        assert fp1 != fp2

    def test_fingerprint_handles_multiple_paths(self, tmp_path: pathlib.Path) -> None:
        meta1 = tmp_path / "meta1.json"
        meta2 = tmp_path / "meta2.json"
        meta1.write_text('{"a": 1}')
        meta2.write_text('{"b": 2}')
        inputs = pyine.utils.reprod.FingerprintInputs(metadata_paths=[meta1, meta2])
        fp = pyine.utils.reprod.compute_data_fingerprint(inputs=inputs)
        assert fp  # should not be empty


class TestCachePath:
    def test_cache_path_has_no_rank_subdir(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "5")  # node_rank = 5 // 4 = 1
        monkeypatch.setenv("PYINE_CACHE_ROOT", str(tmp_path))
        cache_path = pyine.utils.filesystem.get_data_cache_path()
        # cache path should be exactly PYINE_CACHE_ROOT, not a node-namespaced subdir
        assert cache_path == tmp_path
        # verify no "node_N" subdir was appended (where N is a digit)
        parts = cache_path.parts
        node_parts = [p for p in parts if p.startswith("node_") and len(p) > 5 and p[5:].isdigit()]
        assert len(node_parts) == 0


class TestFingerprintPayload:
    def test_dataclass_fields(self) -> None:
        payload = pyine.utils.distrib.FingerprintPayload(ok=True, fingerprint="abc123", error=None, node_rank=0)
        assert payload.ok is True
        assert payload.fingerprint == "abc123"
        assert payload.error is None
        assert payload.node_rank == 0

    def test_error_payload(self) -> None:
        payload = pyine.utils.distrib.FingerprintPayload(ok=False, fingerprint=None, error="test error")
        assert payload.ok is False
        assert payload.fingerprint is None
        assert payload.error == "test error"


class TestRequireInitializedProcessGroup:
    def test_passes_when_not_distributed(self) -> None:
        """No error when WORLD_SIZE <= 1 (not distributed)."""
        pyine.utils.distrib.require_initialized_process_group()  # should not raise

    def test_raises_when_distributed_but_not_initialized(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Raises RuntimeError when WORLD_SIZE > 1 but torch.distributed not initialized."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        with pytest.raises(RuntimeError, match="process group is not initialized"):
            pyine.utils.distrib.require_initialized_process_group()


class TestAllGatherObjects:
    def test_returns_single_element_when_not_distributed(self) -> None:
        """Returns [obj] when not in distributed mode."""
        result = pyine.utils.distrib.all_gather_objects({"key": "value"})
        assert result == [{"key": "value"}]

    def test_raises_when_distributed_but_not_initialized(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Raises if WORLD_SIZE > 1 but process group not initialized."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        with pytest.raises(RuntimeError, match="process group is not initialized"):
            pyine.utils.distrib.all_gather_objects({"key": "value"})

    def test_raises_when_env_changes_to_distributed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Check still runs when environment changes from non-distributed to distributed."""
        # first call: not distributed, should succeed and return [obj]
        result = pyine.utils.distrib.all_gather_objects({"key": "value"})
        assert result == [{"key": "value"}]
        # now set WORLD_SIZE=2 (but don't init process group)
        monkeypatch.setenv("WORLD_SIZE", "2")
        # second call should raise because we're now distributed but not initialized
        with pytest.raises(RuntimeError, match="process group is not initialized"):
            pyine.utils.distrib.all_gather_objects({"key": "value"})
