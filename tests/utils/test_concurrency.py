import asyncio
import time
import typing

import pytest

import pyine.utils.concurrency


def fast_job() -> str:
    """Return quickly after a brief sleep."""
    time.sleep(0.2)
    return "fast"


def slow_job() -> str:
    """Return after a longer sleep to simulate slow work."""
    time.sleep(1.5)
    return "slow"


def failing_job() -> str:
    """Raise a runtime error after a short delay."""
    time.sleep(0.2)
    raise RuntimeError("boom")


@pytest.mark.slow
def test_run_in_parallel() -> None:
    """Ensure parallel execution gathers all results and surfaces failures without crashing."""
    funcs = [
        fast_job,
        slow_job,
        failing_job,
        fast_job,
    ]
    for use_processes in [True, False]:
        results, errors = pyine.utils.concurrency.run_in_parallel(
            callables=funcs,
            use_processes=use_processes,
            max_workers=4,
        )
        # results list is aligned with funcs; one failing entry should be None
        assert len(results) == len(funcs)
        assert "fast" in results  # at least one fast completed
        assert "slow" in results  # slow completed
        # exactly one failure expected from failing_job
        assert sum([r is None for r in results]) == 1
        # errors list also aligned with funcs; contains the failing job's exception
        assert len(errors) == len(funcs)
        assert results[2] is None and errors[2] is not None
        assert isinstance(errors[2], RuntimeError) and str(errors[2]) == "boom"


@pytest.mark.slow
def test_run_in_parallel_with_shared_pool() -> None:
    """Ensure parallel execution gathers all results and surfaces failures without crashing."""
    funcs = [fast_job for _ in range(50)] + [failing_job]
    start_time = time.time()
    results, errors = pyine.utils.concurrency.run_in_parallel(
        callables=funcs,
        use_processes=True,
        use_shared_pool=True,
    )
    end_time = time.time()
    # results list is aligned with funcs; one failing entry should be None
    assert len(results) == len(funcs)
    assert sum(1 for r in results if r is None) == 1
    # exactly one failure expected from failing_job
    assert sum([r is None for r in results]) == 1
    # errors list also aligned with funcs; contains the failing job's exception
    assert len(errors) == len(funcs)
    assert results[-1] is None and errors[-1] is not None
    assert isinstance(errors[-1], RuntimeError) and str(errors[-1]) == "boom"
    # should have seen some speedup too
    assert end_time - start_time < len(funcs) * 0.2


class DummyRunnable(pyine.utils.concurrency.SupportsAInvoke):
    """
    Minimal async runnable for tests.
    - Returns `result` after `delay` unless `exc` is set.
    - If `expect_config` is provided, asserts equality with the received config.
    - Calls `on_call()` (awaitable) if provided, useful to track concurrency.
    """

    def __init__(
        self,
        *,
        result: typing.Any = None,
        delay: float = 0.0,
        exc: BaseException | None = None,
        expect_config: typing.Mapping[str, typing.Any] | None = None,
        on_call: typing.Any = None,  # async callable or None
    ):
        self.result = result
        self.delay = delay
        self.exc = exc
        self.expect_config = expect_config
        self.on_call = on_call
        self.calls = 0

    async def ainvoke(  # mimics the ainvoke signature from langchain runnables
        self,
        input: typing.Any,
        *,
        config: typing.Mapping[str, typing.Any] | None = None,
    ) -> typing.Any:
        self.calls += 1
        if self.expect_config is not None:
            assert config == self.expect_config
        if self.on_call:
            await self.on_call()
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        # if result is None, echo input (handy in some tests)
        return self.result if self.result is not None else input


@pytest.mark.asyncio
async def test_all_success_preserves_order():
    jobs = [
        pyine.utils.concurrency.Job(DummyRunnable(result="A", delay=0.3), {"q": 1}),
        pyine.utils.concurrency.Job(DummyRunnable(result="B", delay=0.1), {"q": 2}),
        pyine.utils.concurrency.Job(DummyRunnable(result="C", delay=0.2), {"q": 3}),
    ]
    out = await pyine.utils.concurrency.run_independent(jobs, max_inflight=2, timeout=1.0)

    assert [r.index for r in out] == [0, 1, 2]
    assert [r.ok for r in out] == [True, True, True]
    assert [r.value for r in out] == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_per_request_timeout_marks_error_only_for_slow_tasks():
    slow = DummyRunnable(result="too slow", delay=0.2)
    fast = DummyRunnable(result="ok", delay=0.01)
    out = await pyine.utils.concurrency.run_independent(
        jobs=[
            pyine.utils.concurrency.Job(slow, {}),
            pyine.utils.concurrency.Job(fast, {}),
        ],
        max_inflight=2,
        timeout=0.1,
    )
    # slow timed out
    assert out[0].ok is False
    assert out[0].error is not None and isinstance(out[0].error, asyncio.TimeoutError)
    # fast succeeded
    assert out[1].ok is True
    assert out[1].value == "ok"


@pytest.mark.asyncio
async def test_exceptions_are_captured_per_job():
    boom = DummyRunnable(exc=ValueError("boom"))
    ok = DummyRunnable(result=42)
    out = await pyine.utils.concurrency.run_independent(
        jobs=[
            pyine.utils.concurrency.Job(boom, {}),
            pyine.utils.concurrency.Job(ok, {}),
        ],
        timeout=1.0,
    )
    assert out[0].ok is False
    assert isinstance(out[0].error, ValueError)
    assert out[1].ok is True
    assert out[1].value == 42


@pytest.mark.asyncio
async def test_global_concurrency_cap_is_respected():
    # track peak concurrency with an async critical section
    current = 0
    peak = 0
    lock = asyncio.Lock()

    async def on_call():
        nonlocal current, peak
        async with lock:
            current += 1
            peak = max(peak, current)
        # hold the slot briefly to let others overlap
        await asyncio.sleep(0.05)
        async with lock:
            current -= 1

    jobs = [pyine.utils.concurrency.Job(DummyRunnable(result=i, on_call=on_call), {"i": i}) for i in range(12)]
    max_inflight = 3
    out = await pyine.utils.concurrency.run_independent(jobs, max_inflight=max_inflight, timeout=1.0)

    assert all(r.ok for r in out)
    # peak should never exceed the cap
    assert peak <= max_inflight
    # in practice we usually hit the cap at least once
    assert peak >= 1


@pytest.mark.asyncio
async def test_config_is_passed_to_each_runnable():
    cfg = {"max_concurrency": 7}
    r = DummyRunnable(result="ok", expect_config=cfg)
    out = await pyine.utils.concurrency.run_independent(
        jobs=[pyine.utils.concurrency.Job(r, {"x": 1}, cfg)],
        timeout=1.0,
    )
    assert out[0].ok is True
    assert out[0].value == "ok"
    assert r.calls == 1


@pytest.mark.asyncio
async def test_empty_jobs_returns_empty_list():
    out = await pyine.utils.concurrency.run_independent([])
    assert out == []


@pytest.mark.asyncio
async def test_run_with_sliding_window_respects_cap_and_processes_all_results():
    n_items = 10
    cap = 3
    items = list(range(n_items))

    def worker(x: int) -> int:
        time.sleep(0.05)
        return x * 2

    def submit_one(item, executor):
        return executor.submit(worker, item)

    results: dict[int, int] = {}

    def process_result(item, result):
        results[item] = result

    callback_calls = 0
    peak_inflight = 0

    def progress_callback(in_flight, completed):
        nonlocal callback_calls, peak_inflight
        callback_calls += 1
        if in_flight is not None:
            peak_inflight = max(peak_inflight, len(in_flight))

    await pyine.utils.concurrency.run_with_sliding_window(
        input_items=items,
        submit_one=submit_one,
        process_result=process_result,
        progress_callback=progress_callback,
        max_workers=8,
        max_in_flight_jobs=cap,
    )
    assert len(results) == n_items
    assert results == {i: i * 2 for i in items}
    assert peak_inflight <= cap
    assert peak_inflight >= min(n_items, cap)
    assert callback_calls == 2 * n_items


@pytest.mark.asyncio
async def test_run_with_sliding_window_collects_failures_without_aborting():
    items = [1, 2, 3]

    def worker(x: int) -> int:
        if x == 2:
            raise ValueError("boom")
        time.sleep(0.01)
        return x * 10

    def submit_one(item, executor):
        return executor.submit(worker, item)

    results: dict[int, int] = {}

    def process_result(item, result):
        results[item] = result

    with pytest.raises(pyine.utils.concurrency.SlidingWindowExecutionError) as exc_info:
        await pyine.utils.concurrency.run_with_sliding_window(
            input_items=items,
            submit_one=submit_one,
            process_result=process_result,
            progress_callback=None,
            max_workers=2,
            max_in_flight_jobs=2,
        )

    assert results == {1: 10, 3: 30}
    failures = exc_info.value.failures
    assert len(failures) == 1
    failed_item, error = failures[0]
    assert failed_item == 2
    assert isinstance(error, ValueError)
