import os
import typing

import pytest

import pyine.utils.distrib


@pytest.fixture(autouse=True)
def clear_distrib_env(
    monkeypatch: pytest.MonkeyPatch,
) -> typing.Iterator[None]:
    """Ensure distributed-related environment variables are unset for each test."""
    keys = {
        "RANK",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "WORLD_SIZE",
        "SLURM_NTASKS",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "LOCAL_WORLD_SIZE",
        "MPI_LOCALNRANKS",
    }
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
