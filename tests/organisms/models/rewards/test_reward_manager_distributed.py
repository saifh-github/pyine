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
import pyine.organisms.models.rewards.core.registry
import pyine.organisms.models.rewards.core.types
import tests.organisms.models.rewards.conftest as rewards_conftest

pytestmark = pytest.mark.slow


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


class _RankAsFloatTerm:
    """Simple term for distributed tests that returns the current rank as the reward."""

    def reset(
        self,
        run_init_ctx: pyine.organisms.models.rewards.core.types.RunInitContext,
    ) -> None:
        del run_init_ctx

    def __call__(
        self,
        sample_ctx: pyine.organisms.models.rewards.core.types.SampleContext,
    ) -> pyine.organisms.models.rewards.core.types.TermResult:
        del sample_ctx
        rank_str = os.environ.get("RANK", "0")
        return pyine.organisms.models.rewards.core.types.TermResult(value=float(int(rank_str)))


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
        assert "reward/run/total/count" in reward_totals
        assert reward_totals["reward/run/total/count"] == 5.0
        assert "reward/run/total/mean" in reward_totals
        assert reward_totals["reward/run/total/mean"] == pytest.approx(1.0)
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
    # simulate imbalanced workload: only rank 0 has samples, but ALL ranks must participate
    # in compute_batch() to avoid distributed deadlock.
    for i in range(3):
        if rank == 0:
            ctx = manager.build_sample_context(
                prompt="test",
                model_output="<final>ok</final>",
                sample_data=rewards_conftest.make_sample_data(f"s{rank}_{i}"),
            )
            manager.compute(ctx, log=False)
        else:
            manager.compute_batch([], log=False)
    # all ranks call finalize_run (should not deadlock even though rank1 has no stats)
    manager.finalize_run()
    # verify rank 0 logged its stats
    if rank == 0:
        assert logger is not None
        assert len(logger.runs) == 1
        # should only see rank 0's stats (3 samples)
        run_entry = logger.runs[0]
        reward_totals = run_entry["reward_totals"]
        assert reward_totals["reward/run/total/count"] == 3.0
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
        assert reward_totals["reward/run/total/count"] == 8.0
        assert reward_totals["reward/run/total/mean"] == pytest.approx(1.0)
        result_queue.put({"rank": rank, "success": True})
    else:
        # non-main ranks should complete without error
        result_queue.put({"rank": rank, "success": True})
    _cleanup_distributed()


def _worker_batch_stats_gather_logs_on_all_ranks_cpu(  # type: ignore[type-arg]
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
) -> None:
    """Worker for test_batch_stats_gather_logs_on_all_ranks_cpu."""
    _init_distributed_process(rank, world_size, backend="gloo")
    logger = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger(
        log_every_n_generations=9999,
        log_tables=False,
    )
    registry = pyine.organisms.models.rewards.core.registry.RewardRegistry()

    def factory(
        spec: pyine.organisms.models.rewards.core.configs.RewardTermSpec,
        *,
        parser: pyine.organisms.models.rewards.core.types.OutputParser | None,
    ) -> pyine.organisms.models.rewards.core.types.RewardTerm:
        del spec
        del parser
        return _RankAsFloatTerm()

    registry.register_term("test_rank_as_float", factory)
    config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
        terms=[
            pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                name="rank",
                type="test_rank_as_float",
            )
        ],
        logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
            enabled=True,
            main_process_only=False,
            gather_distributed_summaries=True,
            log_batch_stats=True,
            log_every_n_generations=9999,
            log_tables=False,
        ),
    )
    manager = pyine.organisms.models.rewards.core.manager.RewardManager(
        config,
        logger=logger,
        registry=registry,
    )
    manager.compute_batch(
        [
            rewards_conftest.make_sample_context(identifier=f"s{rank}_0"),
            rewards_conftest.make_sample_context(identifier=f"s{rank}_1"),
        ]
    )
    assert len(logger.batch_stats) == 1
    entry = logger.batch_stats[0]
    assert entry["batch_count"] == 1
    assert entry["batch_mean"] == pytest.approx(0.5)
    assert entry["batch_std"] == pytest.approx(0.5)
    result_queue.put({"rank": rank, "success": True})
    _cleanup_distributed()


def _worker_global_generation_counts_unique_across_ranks_cpu(  # type: ignore[type-arg]
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
) -> None:
    """Worker for test_global_generation_counts_unique_across_ranks_cpu.

    Verifies that when both ranks have samples, each rank gets unique, non-overlapping
    global generation indices via the prefix-sum computation.
    """
    _init_distributed_process(rank, world_size, backend="gloo")
    logger = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger(
        log_every_n_generations=1,  # log every sample
        log_tables=True,
    )
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
            main_process_only=False,  # all ranks log
            gather_distributed_summaries=True,
            log_batch_stats=True,
            log_every_n_generations=1,
            log_tables=True,
        ),
    )
    manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger)
    manager.set_key_prefix("train/")
    # rank 0 has 3 samples, rank 1 has 2 samples
    # expected global indices:
    #   rank 0: [1, 2, 3]
    #   rank 1: [4, 5]
    num_samples = 3 if rank == 0 else 2
    sample_ctxs = [rewards_conftest.make_sample_context(identifier=f"s{rank}_{i}") for i in range(num_samples)]
    manager.compute_batch(sample_ctxs, log=True)
    # extract generation_count from logged table rows (InMemoryRewardLogger stores dicts)
    if not logger.table_rows:
        result_queue.put({"rank": rank, "success": False, "error": "no table rows logged"})
        _cleanup_distributed()
        return
    generation_counts = [row["generation_count"] for row in logger.table_rows]
    result_queue.put(
        {
            "rank": rank,
            "success": True,
            "generation_counts": generation_counts,
            "global_generation_count": manager._global_generation_counts.get("train/", 0),
            "global_batch_count": manager._global_batch_counts.get("train/", 0),
        }
    )
    _cleanup_distributed()


def _worker_global_completion_idx_unique_across_ranks_cpu(  # type: ignore[type-arg]
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
) -> None:
    """Worker for test_global_completion_idx_unique_across_ranks_cpu.

    Verifies that completion_idx is globally computed across ranks. When the same identifier
    appears on multiple ranks, completion indices should be unique across the entire batch.
    """
    _init_distributed_process(rank, world_size, backend="gloo")
    logger = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger(
        log_every_n_generations=1,
        log_tables=True,
    )
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
            main_process_only=False,
            gather_distributed_summaries=True,
            log_batch_stats=True,
            log_every_n_generations=1,
            log_tables=True,
        ),
    )
    manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger)
    manager.set_key_prefix("train/")
    # both ranks have samples with the SAME identifier ("shared_prompt")
    # this simulates GRPO-style batching where multiple completions for the same prompt
    # might be distributed across ranks
    # rank 0: [shared_prompt, shared_prompt] -> should get completion_idx [0, 1]
    # rank 1: [shared_prompt, unique_b]      -> should get completion_idx [2, 0]
    if rank == 0:
        sample_ctxs = [
            rewards_conftest.make_sample_context(identifier="shared_prompt"),
            rewards_conftest.make_sample_context(identifier="shared_prompt"),
        ]
    else:
        sample_ctxs = [
            rewards_conftest.make_sample_context(identifier="shared_prompt"),
            rewards_conftest.make_sample_context(identifier="unique_b"),
        ]
    manager.compute_batch(sample_ctxs, log=True)
    if not logger.table_rows:
        result_queue.put({"rank": rank, "success": False, "error": "no table rows logged"})
        _cleanup_distributed()
        return
    completion_indices = [row["completion_idx"] for row in logger.table_rows]
    result_queue.put(
        {
            "rank": rank,
            "success": True,
            "completion_indices": completion_indices,
        }
    )
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

    def test_batch_stats_gather_logs_on_all_ranks_cpu(self) -> None:
        """Test that batch stats are logged on all ranks when main_process_only=False."""
        world_size = 2
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        mp.spawn(
            _worker_batch_stats_gather_logs_on_all_ranks_cpu,
            args=(world_size, result_queue),
            nprocs=world_size,
            join=True,
        )
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        assert len(results) == world_size
        assert all(r["success"] for r in results)

    def test_global_generation_counts_unique_across_ranks_cpu(self) -> None:
        """Test that global generation counts are unique and non-overlapping across ranks.

        When both ranks have samples, the prefix-sum computation should assign:
        - rank 0 (3 samples): global indices [1, 2, 3]
        - rank 1 (2 samples): global indices [4, 5]

        This verifies the core distributed counting invariant: no two samples across
        any rank should have the same global generation_count.
        """
        world_size = 2
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        mp.spawn(
            _worker_global_generation_counts_unique_across_ranks_cpu,
            args=(world_size, result_queue),
            nprocs=world_size,
            join=True,
        )
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        assert len(results) == world_size
        assert all(r["success"] for r in results), [r.get("error") for r in results]
        # sort by rank to make assertions easier
        results_by_rank = {r["rank"]: r for r in results}
        rank0_counts = results_by_rank[0]["generation_counts"]
        rank1_counts = results_by_rank[1]["generation_counts"]
        # rank 0 should have indices [1, 2, 3]
        assert rank0_counts == [1, 2, 3], f"rank 0 got {rank0_counts}"
        # rank 1 should have indices [4, 5]
        assert rank1_counts == [4, 5], f"rank 1 got {rank1_counts}"
        # verify no overlap (belt and suspenders)
        all_counts = set(rank0_counts) | set(rank1_counts)
        assert len(all_counts) == len(rank0_counts) + len(rank1_counts), "duplicate generation counts across ranks"
        # verify global counters are updated correctly on all ranks
        for rank_result in results:
            assert rank_result["global_generation_count"] == 5, (
                f"rank {rank_result['rank']} has wrong global_generation_count: "
                f"{rank_result['global_generation_count']}"
            )
            assert rank_result["global_batch_count"] == 1, (
                f"rank {rank_result['rank']} has wrong global_batch_count: {rank_result['global_batch_count']}"
            )

    def test_global_completion_idx_unique_across_ranks_cpu(self) -> None:
        """Test that completion_idx is globally computed across ranks.

        When the same identifier appears on multiple ranks, the completion indices
        should be globally unique within the batch. This validates the distributed
        gathering of sample identifiers for global completion index computation.

        Setup:
        - rank 0: [shared_prompt, shared_prompt] -> completion_idx [0, 1]
        - rank 1: [shared_prompt, unique_b]      -> completion_idx [2, 0]

        Note: "unique_b" starts at 0 because it's a different identifier.
        """
        world_size = 2
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        mp.spawn(
            _worker_global_completion_idx_unique_across_ranks_cpu,
            args=(world_size, result_queue),
            nprocs=world_size,
            join=True,
        )
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        assert len(results) == world_size
        assert all(r["success"] for r in results), [r.get("error") for r in results]
        results_by_rank = {r["rank"]: r for r in results}
        rank0_completion = results_by_rank[0]["completion_indices"]
        rank1_completion = results_by_rank[1]["completion_indices"]
        # rank 0 has [shared_prompt, shared_prompt] -> [0, 1]
        assert rank0_completion == [0, 1], f"rank 0 got {rank0_completion}"
        # rank 1 has [shared_prompt, unique_b] -> [2, 0]
        # shared_prompt continues from rank 0's count (2), unique_b starts fresh (0)
        assert rank1_completion == [2, 0], f"rank 1 got {rank1_completion}"
