import tracemalloc
import typing

import pytest

import pyine.utils.profiling as profiling


@pytest.fixture(autouse=True)
def clean_tracemalloc() -> typing.Iterator[None]:
    # ensure tracemalloc is stopped before and after each test
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    try:
        yield
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()


class TestStartMemoryBaseline:
    def test_starts_tracing_and_returns_snapshot(self) -> None:
        baseline = profiling.start_memory_baseline(n_frames=5)
        assert isinstance(baseline, tracemalloc.Snapshot)
        assert tracemalloc.is_tracing()


class TestReportMemoryDiff:
    def test_logs_header_and_entries_with_allocations(self) -> None:
        baseline = profiling.start_memory_baseline()
        # allocate some memory to create noticeable diffs
        data = [bytearray(1024) for _ in range(100)]  # ~100 KiB
        logs: list[str] = []
        top = 5
        profiling.report_memory_diff(baseline, top=top, logger=logs.append)
        # header line and at least one entry should be present
        assert len(logs) >= 1
        assert logs[0] == f"Top {top} allocation changes since baseline (by size):"
        assert any("blocks" in line for line in logs[1:])
        # use the allocated data to avoid being optimized away
        assert sum(len(buf) for buf in data) == 1024 * 100
