import tracemalloc
import typing

import pyine.utils.filesystem


def start_memory_baseline(n_frames: int = 25) -> tracemalloc.Snapshot:
    """Start tracemalloc and capture a baseline snapshot.

    Args:
        n_frames: Number of frames to keep in tracebacks.

    Returns:
        tracemalloc.Snapshot: Baseline snapshot to compare against later.
    """
    tracemalloc.start(n_frames)  # start early (ideally at program entry)
    return tracemalloc.take_snapshot()


def report_memory_diff(
    baseline: tracemalloc.Snapshot,
    top: int = 20,
    logger: typing.Callable[[str], None] | None = None,
) -> None:
    """Report the top memory allocation changes since a baseline snapshot.

    Args:
        baseline: The baseline snapshot previously captured via start_memory_baseline.
        top: The number of entries to display.
        logger: Optional logger to use for printing. If None, prints to stdout.

    Returns:
        None.
    """
    if logger is None:
        logger = print
    snap = tracemalloc.take_snapshot()
    # group by file:line; other valid keys include "filename" and "traceback"
    stats = snap.compare_to(baseline, key_type="lineno")
    # sort by size difference (descending)
    stats.sort(key=lambda s: s.size_diff, reverse=True)
    logger(f"Top {top} allocation changes since baseline (by size):")
    for stat in stats[:top]:
        size_str = pyine.utils.filesystem.get_human_readable_size(stat.size_diff)
        logger(f"{size_str:>12}  ({stat.count_diff:+5d} blocks)  {stat.traceback}")
