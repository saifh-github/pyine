"""Distributed tests for HuggingFace Trainer callback assumptions.

These tests validate that the HuggingFace Trainer calls callback hooks on all ranks,
which is a prerequisite for safe distributed gathering in GPUStatsLoggingCallback.

The tests spawn multiple processes using torch.multiprocessing and use the gloo backend
(CPU-only, no GPUs required).
"""

import os
import pathlib
import socket
import typing

import pytest
import torch
import torch.distributed
import torch.multiprocessing as mp
import transformers


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def _init_distributed_process(rank: int, world_size: int, port: int) -> None:
    """Initialize torch.distributed for a worker process."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _cleanup_distributed() -> None:
    """Clean up torch.distributed after test."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


class _SentinelCallback(transformers.TrainerCallback):
    """Callback that performs all_gather in on_log to verify all ranks enter together.

    If any rank doesn't call on_log, the all_gather will hang, causing the test to timeout.
    """

    def __init__(self, marker_dir: pathlib.Path, rank: int) -> None:
        self._marker_dir = marker_dir
        self._rank = rank
        self._on_log_count = 0

    @typing.override
    def on_log(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        logs: dict[str, typing.Any] | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Perform all_gather to verify all ranks enter on_log together."""
        self._on_log_count += 1
        # all_gather_object requires all ranks to participate
        gathered: list[dict[str, typing.Any]] = [{} for _ in range(torch.distributed.get_world_size())]
        my_data = {"rank": self._rank, "on_log_count": self._on_log_count}
        torch.distributed.all_gather_object(gathered, my_data)
        # write marker file to indicate this rank reached on_log
        marker_file = self._marker_dir / f"rank_{self._rank}_on_log_{self._on_log_count}.txt"
        marker_file.write_text(f"rank={self._rank}, on_log_count={self._on_log_count}, gathered={gathered}")


class _DummyDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
    """Minimal dataset for testing."""

    def __init__(self, size: int = 16) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.zeros(4, dtype=torch.long), "labels": torch.zeros(4, dtype=torch.long)}


def _worker_on_log_all_ranks(rank: int, world_size: int, port: int, marker_dir: str) -> None:
    """Worker that verifies on_log is called on all ranks.

    This worker:
    1. Initializes distributed with gloo backend
    2. Creates a minimal Trainer with a SentinelCallback
    3. Runs a few training steps (triggers on_log)
    4. If on_log isn't called on all ranks, the all_gather in SentinelCallback will hang
    """
    _init_distributed_process(rank, world_size, port)
    marker_path = pathlib.Path(marker_dir)
    # create minimal model config (hermetic, no network access)
    config = transformers.GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=32,
        vocab_size=100,
    )
    model = transformers.AutoModelForCausalLM.from_config(config)
    # create training args with logging enabled
    training_args = transformers.TrainingArguments(
        output_dir=str(marker_path / f"output_rank_{rank}"),
        per_device_train_batch_size=2,
        max_steps=4,
        logging_steps=2,  # log every 2 steps
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        ddp_backend="gloo",
        local_rank=rank,
        dataloader_num_workers=0,
    )
    # create trainer with sentinel callback
    sentinel_callback = _SentinelCallback(marker_path, rank)
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=_DummyDataset(),
        callbacks=[sentinel_callback],
    )
    # run training (this will trigger on_log on logging_steps)
    trainer.train()
    # verify this rank's callback was called
    assert sentinel_callback._on_log_count > 0, f"rank {rank} on_log was never called"
    _cleanup_distributed()


@pytest.mark.distributed
@pytest.mark.slow
@pytest.mark.timeout(60)  # fail fast if deadlock occurs
class TestCallbackDistributedAssumptions:
    """Tests validating HuggingFace Trainer callback behavior in distributed settings.

    These tests verify that callback hooks (especially on_log) are called on all ranks,
    which is required for safe distributed gathering operations.
    """

    def test_on_log_called_on_all_ranks(self, tmp_path: pathlib.Path) -> None:
        """Test that Trainer.on_log is called on all ranks at the same step.

        This test validates the core assumption that GPUStatsLoggingCallback relies on:
        that on_log is called synchronously on all ranks when logging_steps is reached.

        If on_log is not called on all ranks, the all_gather_object inside the sentinel
        callback will hang, and this test will timeout.
        """
        world_size = 2
        port = _find_free_port()
        mp.spawn(
            _worker_on_log_all_ranks,
            args=(world_size, port, str(tmp_path)),
            nprocs=world_size,
            join=True,
        )
        # verify marker files from all ranks exist
        for rank in range(world_size):
            markers = list(tmp_path.glob(f"rank_{rank}_on_log_*.txt"))
            assert len(markers) > 0, f"rank {rank} did not write any on_log markers"
