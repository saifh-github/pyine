"""Distributed tests for RewardManager.

These tests validate that the RewardManager correctly handles distributed training scenarios, particularly around
barrier synchronization and distributed stats gathering. They spawn multiple processes using torch.multiprocessing.
Some tests require GPUs and will be skipped if unavailable.
"""

import os

import pytest
import torch
import torch.distributed
import torch.multiprocessing as mp

import pyine.organisms.models.rewards.core.configs
import pyine.organisms.models.rewards.core.logging
import pyine.organisms.models.rewards.core.manager
import tests.organisms.models.rewards.conftest as rewards_conftest


def _init_distributed_process(rank: int, world_size: int, backend: str = "gloo") -> None:
    """Initialize torch.distributed for a worker process.

    Args:
        rank: The rank of this process.
        world_size: Total number of processes.
        backend: Backend to use (gloo for CPU, nccl for GPU).
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _cleanup_distributed() -> None:
    """Clean up torch.distributed after test."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# worker functions for GPU tests (must be at module level for pickling)
def _worker_flush_stats_gpu(rank: int, world_size: int) -> None:
    """Worker for test_flush_stats_with_barrier_no_deadlock_gpu."""
    _init_distributed_process(rank, world_size, backend="nccl")
    torch.cuda.set_device(rank)
    # only rank 0 gets a logger (simulates real distributed training setup)
    logger = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger() if rank == 0 else None
    config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
        terms=[
            pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                name="parseable",
                type="parseable_answer",
                params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
            )
        ],
        parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
        logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
            enabled=True,
            main_process_only=True,
            barrier_before_finalize=True,
            gather_distributed_summaries=False,  # focus on barrier deadlock
        ),
    )
    manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger)
    # compute some rewards on each rank
    for i in range(5):
        ctx = manager.build_sample_context(
            prompt="test",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data(f"s{rank}_{i}"),
        )
        manager.compute(ctx, log=False)
    # this should NOT deadlock (barrier is before early returns)
    manager.flush_stats()
    # verify rank 0 logged stats
    if rank == 0:
        assert logger is not None
        assert len(logger.runs) == 1
    _cleanup_distributed()


def _worker_gather_summaries_gpu(rank: int, world_size: int, result_queue: mp.Queue) -> None:  # type: ignore[type-arg]
    """Worker for test_gather_distributed_summaries_aggregates_correctly_gpu."""
    _init_distributed_process(rank, world_size, backend="nccl")
    torch.cuda.set_device(rank)
    # only rank 0 gets a logger
    logger = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger() if rank == 0 else None
    config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
        terms=[
            pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                name="parseable",
                type="parseable_answer",
                params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
            )
        ],
        parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
        logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
            enabled=True,
            main_process_only=True,
            barrier_before_finalize=True,
            gather_distributed_summaries=True,
            scope_prefix="reward",
        ),
    )
    manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger)
    # each rank computes different number of rewards with different values
    # rank 0: 2 samples with reward 1.0 each
    # rank 1: 3 samples with reward 1.0 each
    num_samples = 2 + rank
    for i in range(num_samples):
        ctx = manager.build_sample_context(
            prompt="test",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data(f"s{rank}_{i}"),
        )
        manager.compute(ctx, log=False)
    # flush stats (triggers gather on all ranks)
    manager.flush_stats()
    # only rank 0 should have logged the aggregated results
    if rank == 0:
        assert logger is not None
        assert len(logger.runs) == 1
        run_entry = logger.runs[0]
        reward_totals = run_entry["reward_totals"]
        # expected: (2 samples from rank0 + 3 from rank1) = 5 total samples
        # all rewards are 1.0, so mean should be 1.0
        assert "reward/run/sample_count" in reward_totals
        assert reward_totals["reward/run/sample_count"] == 5.0
        assert "reward/run/mean" in reward_totals
        assert reward_totals["reward/run/mean"] == pytest.approx(1.0)
        result_queue.put({"success": True, "reward_totals": reward_totals})
    else:
        result_queue.put({"success": True})
    _cleanup_distributed()


# worker functions for CPU tests (must be at module level for pickling)
def _worker_flush_stats_cpu(rank: int, world_size: int) -> None:
    """Worker for test_flush_stats_with_barrier_no_deadlock_cpu."""
    _init_distributed_process(rank, world_size, backend="gloo")
    # only rank 0 gets a logger
    logger = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger() if rank == 0 else None
    config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
        terms=[
            pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                name="parseable",
                type="parseable_answer",
                params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
            )
        ],
        parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
        logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
            enabled=True,
            main_process_only=True,
            barrier_before_finalize=True,
            gather_distributed_summaries=False,
        ),
    )
    manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger)
    # compute some rewards
    for i in range(5):
        ctx = manager.build_sample_context(
            prompt="test",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data(f"s{rank}_{i}"),
        )
        manager.compute(ctx, log=False)
    # this should NOT deadlock
    manager.flush_stats()
    # verify rank 0 logged
    if rank == 0:
        assert logger is not None
        assert len(logger.runs) == 1
    _cleanup_distributed()


def _worker_finalize_run_cpu(rank: int, world_size: int) -> None:
    """Worker for test_finalize_run_with_gather_no_deadlock_cpu."""
    _init_distributed_process(rank, world_size, backend="gloo")
    # only rank 0 gets a logger
    logger = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger() if rank == 0 else None
    config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
        terms=[
            pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                name="parseable",
                type="parseable_answer",
            )
        ],
        parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
        logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
            enabled=True,
            main_process_only=True,
            barrier_before_finalize=True,
            gather_distributed_summaries=True,
        ),
    )
    manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger)
    # only rank 0 computes rewards (simulates imbalanced workload)
    if rank == 0:
        for i in range(3):
            ctx = manager.build_sample_context(
                prompt="test",
                model_output="<final>ok</final>",
                sample_data=rewards_conftest.make_sample_data(f"s{rank}_{i}"),
            )
            manager.compute(ctx, log=False)
    # all ranks call finalize_run (should not deadlock even though rank1 has no stats)
    manager.finalize_run()
    # verify rank 0 logged its stats
    if rank == 0:
        assert logger is not None
        assert len(logger.runs) == 1
        # should only see rank 0's stats (3 samples)
        run_entry = logger.runs[0]
        reward_totals = run_entry["reward_totals"]
        assert reward_totals["reward/run/sample_count"] == 3.0
    _cleanup_distributed()


def _worker_mixed_logger_states_cpu(rank: int, world_size: int, result_queue: mp.Queue) -> None:  # type: ignore[type-arg]
    """Worker for test_mixed_logger_states_with_gather_cpu."""
    _init_distributed_process(rank, world_size, backend="gloo")
    # only rank 0 gets a logger (typical setup)
    logger = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger() if rank == 0 else None
    config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
        terms=[
            pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                name="parseable",
                type="parseable_answer",
                params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
            )
        ],
        parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
        logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
            enabled=True,
            main_process_only=True,
            barrier_before_finalize=True,
            gather_distributed_summaries=True,
            scope_prefix="",
        ),
    )
    manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger)
    # all ranks compute same number of rewards
    for i in range(4):
        ctx = manager.build_sample_context(
            prompt="test",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data(f"s{rank}_{i}"),
        )
        manager.compute(ctx, log=False)
    # gather and log (all ranks participate in gather)
    manager.flush_stats()
    if rank == 0:
        assert logger is not None
        assert len(logger.runs) == 1
        run_entry = logger.runs[0]
        reward_totals = run_entry["reward_totals"]
        # should aggregate stats from all ranks: 2 ranks * 4 samples = 8 total
        # note: scope_prefix="" means no prefix at all (not even /run/)
        assert reward_totals["sample_count"] == 8.0
        assert reward_totals["mean"] == pytest.approx(1.0)
        result_queue.put({"rank": rank, "success": True})
    else:
        # non-main ranks should complete without error
        result_queue.put({"rank": rank, "success": True})
    _cleanup_distributed()


@pytest.mark.distributed
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires at least 2 GPUs for distributed testing",
)
class TestRewardManagerDistributedGPU:
    """Distributed tests that require GPUs (use NCCL backend)."""

    def test_flush_stats_with_barrier_no_deadlock_gpu(self) -> None:
        """Test that flush_stats with barrier doesn't deadlock when non-main ranks lack logger."""
        world_size = min(2, torch.cuda.device_count())
        mp.spawn(_worker_flush_stats_gpu, args=(world_size,), nprocs=world_size, join=True)

    def test_gather_distributed_summaries_aggregates_correctly_gpu(self) -> None:
        """Test that gather_distributed_summaries correctly aggregates stats across ranks.

        Validates that when gather_distributed_summaries=True, all ranks participate in the
        gather operation and rank 0 receives correctly merged statistics.
        """
        world_size = min(2, torch.cuda.device_count())
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        mp.spawn(_worker_gather_summaries_gpu, args=(world_size, result_queue), nprocs=world_size, join=True)
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        rank0_result = next((r for r in results if "reward_totals" in r), None)
        assert rank0_result is not None
        assert rank0_result["success"]


@pytest.mark.distributed
class TestRewardManagerDistributedCPU:
    """Distributed tests that work on CPU (use gloo backend).

    These tests are more portable and can run in CI environments without GPUs.
    """

    def test_flush_stats_with_barrier_no_deadlock_cpu(self) -> None:
        """Test that flush_stats with barrier doesn't deadlock (CPU version).

        This is the CPU-only version of the GPU test, using gloo backend.
        """
        world_size = 2
        mp.spawn(_worker_flush_stats_cpu, args=(world_size,), nprocs=world_size, join=True)

    def test_finalize_run_with_gather_no_deadlock_cpu(self) -> None:
        """Test that finalize_run with gather doesn't deadlock when ranks have no stats.

        This validates that even when some ranks have empty stats (count=0), the gather
        operation completes successfully without deadlock.
        """
        world_size = 2
        mp.spawn(_worker_finalize_run_cpu, args=(world_size,), nprocs=world_size, join=True)

    def test_mixed_logger_states_with_gather_cpu(self) -> None:
        """Test gather operation when ranks have different logger states.

        Validates that the distributed gather correctly handles the case where only
        some ranks have loggers, which is the typical distributed training setup.
        """
        world_size = 2
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        mp.spawn(_worker_mixed_logger_states_cpu, args=(world_size, result_queue), nprocs=world_size, join=True)
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        assert len(results) == world_size
        assert all(r["success"] for r in results)
