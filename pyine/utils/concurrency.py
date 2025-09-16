import asyncio
import atexit
import concurrent.futures
import dataclasses
import inspect
import threading
import typing

__all__ = [
    "SupportsAInvoke",
    "Job",
    "JobResult",
    "get_shared_executor",
    "run_in_parallel",
    "run_independent",
    "SlidingWindowExecutionError",
]

T = typing.TypeVar("T")
P = typing.ParamSpec("P")
MaybeAsyncCallable = typing.Callable[P, typing.Awaitable[T]] | typing.Callable[P, T]

# module-level shared executors (lazy-initialized)
_shared_proc_pool: concurrent.futures.ProcessPoolExecutor | None = None
_shared_thread_pool: concurrent.futures.ThreadPoolExecutor | None = None
_shared_lock = threading.Lock()


def get_shared_executor(
    use_processes: bool = True,
) -> concurrent.futures.Executor:
    """Return a module-scoped shared executor (process or thread).

    The first call for a given executor type creates it; subsequent calls return the same instance.

    Note:
      - On some platforms (e.g., macOS), process-based pools require task functions to be defined
        at module top-level and the caller to be protected by an `if __name__ == "__main__":` guard.
      - The shared pools are shut down automatically on interpreter exit.

    Args:
        use_processes: Whether to return a ProcessPoolExecutor (True) or ThreadPoolExecutor (False).

    Returns:
        A shared concurrent.futures.Executor instance.
    """
    global _shared_proc_pool, _shared_thread_pool
    with _shared_lock:
        if use_processes:
            if _shared_proc_pool is None:
                _shared_proc_pool = concurrent.futures.ProcessPoolExecutor()
                atexit.register(lambda: _shared_proc_pool.shutdown(cancel_futures=True))
            return _shared_proc_pool
        else:
            if _shared_thread_pool is None:
                _shared_thread_pool = concurrent.futures.ThreadPoolExecutor()
                atexit.register(lambda: _shared_thread_pool.shutdown(cancel_futures=True))
            return _shared_thread_pool


def run_in_parallel(
    callables: list[typing.Callable[[], T]],
    use_processes: bool = True,
    max_workers: int | None = None,
    executor: concurrent.futures.Executor | None = None,
    use_shared_pool: bool = False,
) -> tuple[list[T | None], list[BaseException | None]]:
    """Execute a list of zero-arg callables in parallel using a thread/worker pool and collect results.

    This helper can:
      - create a fresh pool (default behavior),
      - reuse a shared, module-scoped pool (set `use_shared_pool=True`), or
      - use a caller-provided executor instance (`executor=...`).

    Args:
        callables: Functions to run. Each should take no arguments and return a value.
        use_processes: If True, use ProcessPoolExecutor (CPU-bound). If False, ThreadPoolExecutor (I/O-bound).
            Ignored when `executor` is provided.
        max_workers: Maximum parallel workers. Used only if no `executor` is provided and if not
            using the shared module-level pool (which has a machine-specific limit on workers).
        executor: Optional pre-created executor to submit tasks to (not shut down here).
        use_shared_pool: If True, submit to a shared module-level pool (created lazily). Only used
            if no `executor` is provided.

    Returns:
        A tuple (results, errors):
          - results: list aligned with `callables`; each is the return value or None on failure.
          - errors: list aligned with `callables`; each is the raised exception or None on success.

    Notes:
        - Exceptions are captured and returned; they do not crash the whole run.
        - If you need arguments, wrap your function with a lambda/partial that binds them.
        - With process pools, tasks must be picklable and defined at module top-level.
    """
    results: list[T | None] = [None] * len(callables)
    errors: list[BaseException | None] = [None] * len(callables)

    # pick an executor according to options
    local_executor: concurrent.futures.Executor | None = None
    if executor is not None:
        exec_inst = executor  # caller-managed
    elif use_shared_pool:
        exec_inst = get_shared_executor(use_processes=use_processes)
    else:
        exec_cls: type[concurrent.futures.Executor]
        if use_processes:
            exec_cls = concurrent.futures.ProcessPoolExecutor
        else:
            exec_cls = concurrent.futures.ThreadPoolExecutor
        local_executor = exec_cls(max_workers=max_workers)
        exec_inst = local_executor

    try:
        future_to_idx: dict[concurrent.futures.Future[T], int] = {}
        for idx, fn in enumerate(callables):
            future = exec_inst.submit(fn)
            future_to_idx[future] = idx
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except BaseException as e:  # capture any error without crashing
                errors[idx] = e
    finally:
        # only shut down if we created a local executor here
        if local_executor is not None:
            # cancel_futures ensures pending tasks are canceled on shutdown if exceptions occur upstream
            if isinstance(
                local_executor, (concurrent.futures.ThreadPoolExecutor, concurrent.futures.ProcessPoolExecutor)
            ):
                local_executor.shutdown(cancel_futures=True)
            else:
                local_executor.shutdown()  # generic fallback

    return results, errors


class SupportsAInvoke(typing.Protocol):
    """Minimal protocol for LangChain-like runnables."""

    async def ainvoke(
        self,
        input: typing.Any,
        *,
        config: typing.Mapping[str, typing.Any] | None = ...,
    ) -> typing.Any: ...


@dataclasses.dataclass(frozen=True)
class Job:
    """One independent job: (runnable, input, optional config)."""

    runnable: SupportsAInvoke
    input: typing.Any
    config: typing.Mapping[str, typing.Any] | None = None
    id: typing.Any | None = None


@dataclasses.dataclass(frozen=True)
class JobResult(typing.Generic[T]):
    """Per-job outcome (aligned with the input order)."""

    index: int
    ok: bool
    value: T | None = None
    error: BaseException | None = None
    job_id: typing.Any | None = None


async def run_independent(
    jobs: typing.Sequence[Job],
    *,
    max_inflight: int = 32,
    timeout: float | None = 60.0,
) -> list[JobResult[T]]:
    """Execute many independent runnables concurrently (compatible with langchain).

    - Uses an asyncio.Semaphore to cap total in-flight requests (across all chains/models).
    - Applies a *per-request* wall-clock timeout via `asyncio.wait_for` (if provided).
    - Never raises; each job's exception is captured in its JobResult.

    Args:
        jobs: Sequence of Job(runnable, input, config).
        max_inflight: Upper bound on concurrent in-flight requests.
        timeout: Per-request timeout in seconds (None disables).

    Returns:
        List of JobResult, in the same order as `jobs`.
        For each result:
          - ok=True and `value` set on success
          - ok=False and `error` set on failure (including TimeoutError)

    Notes:
        - To respect provider RPM/TPM, give all underlying chat models the *same*
          rate limiter (e.g., LangChain's InMemoryRateLimiter). This function
          only controls *concurrency*, not rate limits.
        - `asyncio.wait_for` attempts to cancel timed-out tasks. Whether the
          underlying HTTP request is aborted promptly depends on the client.
    """
    sem = asyncio.Semaphore(max_inflight)
    results: list[JobResult[T]] = [
        JobResult(index=job_idx, ok=False, job_id=job.id) for job_idx, job in enumerate(jobs)
    ]

    async def run_one(i: int, job: Job) -> None:
        async with sem:
            try:
                coro = job.runnable.ainvoke(job.input, config=job.config)
                val = await (asyncio.wait_for(coro, timeout) if timeout else coro)
                results[i] = JobResult(index=i, ok=True, value=val, job_id=job.id)
            except BaseException as e:  # capture any error, including TimeoutError
                results[i] = JobResult(index=i, ok=False, error=e, job_id=job.id)

    tasks = [asyncio.create_task(run_one(i, job)) for i, job in enumerate(jobs)]
    # drain tasks as they finish; this avoids head-of-line blocking on slow jobs
    for t in asyncio.as_completed(tasks):
        await t
    return results


InputItemType = typing.Hashable
"""Type used to represent an input item; should be a lightweight hashable object."""
OutputItemType = typing.TypeVar("OutputItemType")
"""Type used to represent an output item; can be any type, and will not be kept in memory."""
SubmissionFuncType = typing.Callable[[InputItemType, concurrent.futures.Executor], concurrent.futures.Future]
"""Type used to represent job submission callable.

Gets an input item and an executor as arguments (where the input item tied to the job that must be
submitted) and returns a future (which should be generated by the executor).
"""
ProcessorFuncType = MaybeAsyncCallable[[InputItemType, OutputItemType], None]
"""Type used to represent job result processor callable.

Gets an input item and the result of a completed job as arguments; returns nothing. This is called
after the future tied to the input item is resolved.
"""
ProgressCallbackType = MaybeAsyncCallable[[list[InputItemType], list[InputItemType]], None]
"""Type used to represent a progress callback callable.

Gets the list of submitted jobs and the list of completed jobs as arguments; returns nothing.
This is called after any job is submitted or completed.
"""


class SlidingWindowExecutionError(RuntimeError):
    """Aggregate failure raised when one or more jobs crash during execution."""

    def __init__(self, failures: list[tuple[InputItemType, BaseException]]) -> None:
        self.failures = failures
        msg = ", ".join(f"item={item!r} error={exc!r}" for item, exc in failures) or "unknown failure"
        super().__init__(msg)


async def run_with_sliding_window(
    input_items: typing.Iterable[InputItemType],
    submit_one: SubmissionFuncType,
    process_result: ProcessorFuncType,
    progress_callback: ProgressCallbackType | None = None,
    max_workers: int | None = None,
    max_in_flight_jobs: int | None = 32,
) -> None:
    """Run a job submission function based on a thread pool with a sliding window of in-flight jobs.

    Args:
        input_items: Iterable of input items that will be passed to the job submission function.
        submit_one: Function that, given an item, submits one job to an executor and returns a future.
        process_result: Function that, given an item and the result of a completed job, processes the result.
        progress_callback: Optional function that gets called after each job submission or completion.
        max_workers: Maximum number of workers in the thread pool executor. IF None, uses the number of CPUs.
        max_in_flight_jobs: Maximum number of in-flight jobs.
    """
    in_flight: dict[concurrent.futures.Future, InputItemType] = dict()
    completed: list[InputItemType] = []
    iter_items = iter(input_items)
    progress_callback = progress_callback or (lambda x, y: None)
    failures: list[tuple[InputItemType, BaseException]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # initially fill the window to its max size
        while len(in_flight) < max_in_flight_jobs:
            try:
                item = next(iter_items)
            except StopIteration:
                break
            future = submit_one(item, executor)
            assert future not in in_flight
            in_flight[future] = item
            out = progress_callback(list(in_flight.values()), completed)
            if inspect.isawaitable(out):
                await out
        # update output report and continue filling window if needed
        while in_flight:
            done = next(concurrent.futures.as_completed(in_flight))
            assert done in in_flight
            item = in_flight.pop(done)
            try:
                result_value = done.result()
            except BaseException as exc:  # capture failures and keep draining the queue
                failures.append((item, exc))
            else:
                out = process_result(item, result_value)
                if inspect.isawaitable(out):
                    await out
            completed.append(item)
            out = progress_callback(list(in_flight.values()), completed)
            if inspect.isawaitable(out):
                await out
            # refill the window with the next item (if any)
            try:
                item = next(iter_items)
            except StopIteration:
                continue
            future = submit_one(item, executor)
            assert future not in in_flight
            in_flight[future] = item
            out = progress_callback(list(in_flight.values()), completed)
            if inspect.isawaitable(out):
                await out
    if failures:
        # surface the first error while preserving the full list for callers that inspect the exception
        raise SlidingWindowExecutionError(failures) from failures[0][1]
