"""Pure measurement helpers for guardrail inference metrics.

This module contains only timing/counting logic - no data generation or model loading.
"""

from __future__ import annotations

import gc
import statistics
import time
import typing

import numpy
import torch
from torch.utils.flop_counter import FlopCounterMode

if typing.TYPE_CHECKING:
    from collections.abc import Callable


def measure_flops(
    model: torch.nn.Module,
    sample_input: dict[str, torch.Tensor] | tuple[torch.Tensor, ...],
    forward_fn: Callable[..., typing.Any],
) -> int:
    """Measure FLOPs for a single forward pass.

    FLOPs are deterministic - called once per (model, batch_size, seq_length)
    configuration, NOT inside the benchmark iteration loop.

    Args:
        model: The model to measure (must be in eval mode).
        sample_input: Input tensors for the forward pass.
        forward_fn: Callable that runs ``forward_fn(model, sample_input)``.

    Returns:
        Total FLOPs as an integer.
    """
    assert not model.training, "Model must be in eval mode for FLOPs measurement"
    with FlopCounterMode(display=False) as flop_counter:
        forward_fn(model, sample_input)
    return flop_counter.get_total_flops()


def measure_memory(
    model: torch.nn.Module,
    sample_input: dict[str, torch.Tensor] | tuple[torch.Tensor, ...],
    forward_fn: Callable[..., typing.Any],
    device: torch.device,
) -> dict[str, int | None]:
    """Measure parameter memory and peak activation memory.

    Uses ``torch.cuda.memory_stats()`` for reliable peak measurement
    (avoids fragile peak-minus-before delta).

    Args:
        model: The model to measure (must be in eval mode).
        sample_input: Input tensors for the forward pass.
        forward_fn: Callable that runs ``forward_fn(model, sample_input)``.
        device: Device the model runs on.

    Returns:
        Dict with ``param_count``, ``param_memory_bytes``, and
        ``peak_activation_memory_bytes`` (None on CPU).
    """
    assert not model.training, "Model must be in eval mode for memory measurement"

    # Static: parameter memory (always computable)
    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # Dynamic: peak activation memory during forward
    peak_activation_bytes: int | None = None

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        with torch.inference_mode():
            forward_fn(model, sample_input)

        torch.cuda.synchronize()
        peak_activation_bytes = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    elif device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        # Best-effort MPS measurement
        with torch.inference_mode():
            forward_fn(model, sample_input)
        peak_activation_bytes = torch.mps.current_allocated_memory()

    # else: CPU - only parameter memory is reported

    return {
        "param_count": param_count,
        "param_memory_bytes": param_bytes,
        "peak_activation_memory_bytes": peak_activation_bytes,
    }


def measure_latency(
    model: torch.nn.Module,
    sample_input: dict[str, torch.Tensor] | tuple[torch.Tensor, ...],
    forward_fn: Callable[..., typing.Any],
    num_warmup: int,
    num_iterations: int,
    device: torch.device,
) -> dict[str, float]:
    """Measure forward pass latency with warmup and CUDA event timing.

    Measurement hygiene:
    - Pre-allocated CUDA event pairs (no allocation inside measurement loop)
    - GC disabled during measurement to prevent GC-pause variance
    - ``torch.inference_mode()`` for maximum performance
    - ``cudnn.benchmark`` disabled during measurement to avoid auto-tuning variance

    Args:
        model: The model to measure (must be in eval mode).
        sample_input: Input tensors for the forward pass.
        forward_fn: Callable that runs ``forward_fn(model, sample_input)``.
        num_warmup: Number of warmup forward passes (not timed).
        num_iterations: Number of timed forward passes.
        device: Device the model runs on.

    Returns:
        Dict with ``mean_ms``, ``std_ms``, ``median_ms``,
        ``p90_ms``, ``p95_ms``, ``p99_ms``, ``min_ms``, ``max_ms``.
    """
    assert not model.training, "Model must be in eval mode for latency measurement"

    use_cuda = device.type == "cuda"

    # Warmup (absorbs cudnn auto-tuning, JIT compilation, etc.)
    with torch.inference_mode():
        for _ in range(num_warmup):
            forward_fn(model, sample_input)
            if use_cuda:
                torch.cuda.synchronize()

    # Pre-allocate CUDA events
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] | None = None
    if use_cuda:
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(num_iterations)
        ]

    # Disable GC and cudnn.benchmark during measurement
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    gc.disable()

    try:
        latencies_ms: list[float] = []
        with torch.inference_mode():
            for i in range(num_iterations):
                if use_cuda:
                    assert events is not None
                    start_evt, end_evt = events[i]
                    start_evt.record()
                    forward_fn(model, sample_input)
                    end_evt.record()
                    torch.cuda.synchronize()
                    latencies_ms.append(start_evt.elapsed_time(end_evt))  # pyright: ignore[reportUnknownMemberType]  # CUDA event stubs
                else:
                    t0 = time.perf_counter()
                    forward_fn(model, sample_input)
                    t1 = time.perf_counter()
                    latencies_ms.append((t1 - t0) * 1000)
    finally:
        gc.enable()
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark

    return {
        "mean_ms": statistics.mean(latencies_ms),
        "std_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "median_ms": statistics.median(latencies_ms),
        "p90_ms": float(numpy.percentile(latencies_ms, 90)),
        "p95_ms": float(numpy.percentile(latencies_ms, 95)),
        "p99_ms": float(numpy.percentile(latencies_ms, 99)),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }
