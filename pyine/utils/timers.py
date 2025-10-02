import contextlib
import datetime
import functools
import logging
import re
import signal
import time
import types
import typing

FuncP = typing.ParamSpec("FuncP")
FuncR = typing.TypeVar("FuncR")


@typing.overload  # noqa: UP047
def timeit(  # noqa: UP047
    _func: typing.Callable[FuncP, FuncR],  # noqa: UP047
    *,
    name: str | None = None,
    logger: logging.Logger | None = None,
) -> typing.Callable[FuncP, FuncR]: ...


@typing.overload  # noqa: UP047
def timeit(  # noqa: UP047
    _func: None = None,
    *,
    name: str | None = None,
    logger: logging.Logger | None = None,
) -> typing.ContextManager[None]: ...


def timeit(  # noqa: UP047
    _func: typing.Callable[FuncP, FuncR] | None = None,  # noqa: UP047
    *,
    name: str | None = None,
    logger: logging.Logger | None = None,
) -> typing.Callable[FuncP, FuncR] | typing.ContextManager[None]:
    """Measures execution time of a function (via decoration) or block (via context-manager).

    For blocks, use as a context manager:

      with timeit(name="block", logger=my_logger):
          ...

    For functions, use as a decorator:

      @timeit
      def foo(): ...

      @timeit(name="foo", logger=my_logger)
      def bar(): ...

    By default, logs to stdout via print(); pass in a Logger to redirect.
    """

    @contextlib.contextmanager
    def _ctx(label: str) -> typing.Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            time_str = get_human_readable_time(elapsed)
            msg = f"Time [{label}]: {time_str}"
            if logger:
                logger.info(msg)
            else:
                print(msg)

    def _decorate(fn: typing.Callable[FuncP, FuncR]) -> typing.Callable[FuncP, FuncR]:
        lbl = name or fn.__name__

        @functools.wraps(fn)
        def _wrapped(*args: FuncP.args, **kwargs: FuncP.kwargs) -> FuncR:
            with _ctx(lbl):
                return fn(*args, **kwargs)

        return _wrapped

    if callable(_func):
        return _decorate(_func)

    if _func is None:
        return _ctx(name or "block")

    raise TypeError(f"invalid func argument: {_func}")


class TimeLimit:
    """
    A context manager that limits the execution time of the enclosed code.

    When the specified time limit is reached, it raises a TimeoutError exception,
    interrupting the execution of the code within the context.

    Args:
        seconds: Maximum number of seconds to allow the code to run.
        timeout_message: Custom message for the TimeoutError exception.
        on_timeout: Optional callback function to execute when timeout occurs.

    Examples:
        >>> with TimeLimit(1.5):
        ...     time.sleep(1)  # This will complete normally
        >>> with TimeLimit(1.5):
        ...     time.sleep(10)  # This will raise TimeoutError after 1.5 seconds
        Traceback (most recent call last):
            ...
        TimeoutError: code execution timed out after 1.5 seconds
    """

    def __init__(
        self,
        seconds: float,
        timeout_message: str | None = None,
        on_timeout: typing.Callable[[], typing.Any] | None = None,
    ) -> None:
        """
        Initializes the TimeLimit context manager.

        Args:
            seconds: Maximum number of seconds to allow the code to run.
            timeout_message: Custom message for the TimeoutError exception.
            on_timeout: Optional callback function to execute when timeout occurs.
        """
        self.seconds = seconds
        self.timeout_message = timeout_message or f"timed out after {seconds:.3f} seconds"
        self.on_timeout = on_timeout
        self._old_handler: typing.Callable | None = None
        self._start_time: float = 0

    def _timeout_handler(
        self,
        signum: int,
        frame: types.FrameType | None,
    ) -> None:
        """
        Signal handler for the SIGALRM signal.

        Args:
            signum: Signal number.
            frame: Current execution frame.

        Raises:
            TimeoutError: Always raised to indicate the timeout.
        """
        elapsed = time.time() - self._start_time
        if self.on_timeout:
            self.on_timeout()  # executes the on_timeout callback if provided
        if elapsed > self.seconds * 1.1:  # if we bust the cap by at least 10%, print the actual time
            raise TimeoutError(f"{self.timeout_message} (actual time: {elapsed:.3f}s)")
        raise TimeoutError(self.timeout_message)

    def __enter__(self) -> "TimeLimit":
        """
        Enters the context and starts the timer.

        Returns:
            The TimeLimit instance.

        Raises:
            ValueError: If signal.SIGALRM is not available on this platform.
        """
        if not hasattr(signal, "SIGALRM"):  # not supported on platforms like Windows
            raise ValueError(
                "TimeLimit requires signal.SIGALRM, which is not available on this platform. "
                "This context manager is not supported on Windows."
            )

        self._old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._timeout_handler)
        self._start_time = time.time()
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: typing.Any | None,
    ) -> bool:
        """
        Exits the context and cancels the timer.

        Args:
            exc_type: Type of exception that was raised, if any.
            exc_val: Exception instance that was raised, if any.
            exc_tb: Traceback of the exception, if any.

        Returns:
            False to propagate exceptions, True to suppress them.
        """
        signal.setitimer(signal.ITIMER_REAL, 0)  # cancels the alarm
        if self._old_handler:
            signal.signal(signal.SIGALRM, self._old_handler)
        return False


def get_human_readable_time(seconds: int | float) -> str:
    """Convert a duration in seconds to a human-readable string."""
    secs = float(seconds)
    units = [
        ("y", 365 * 24 * 60 * 60.0),
        ("d", 24 * 60 * 60.0),
        ("h", 60 * 60.0),
        ("m", 60.0),
        ("s", 1.0),
        ("ms", 1e-3),
        ("µs", 1e-6),
        ("ns", 1e-9),
    ]
    for unit, factor in units:
        if abs(secs) >= factor:
            value = secs / factor
            return f"{value:.3f}{unit}"
    # fallback (shouldn’t really be reached)
    return f"{secs / 1e-9:.1f}ns"


_TIMEDELTA_PATTERN = re.compile(
    r"(?:(?P<years>\d*\.?\d+)y)?"
    r"(?:(?P<days>\d*\.?\d+)d)?"
    r"(?:(?P<hours>\d*\.?\d+)h)?"
    r"(?:(?P<mins>\d*\.?\d+)m)?"
    r"(?:(?P<secs>\d*\.?\d+)s)?"
    r"(?:(?P<millis>\d*\.?\d+)ms)?"
    r"(?:(?P<micros>\d*\.?\d+)µs)?"
    r"(?:(?P<nanos>\d*\.?\d+)ns)?"
)


def parse_timedelta(delta_str: str) -> datetime.timedelta:
    """Parse a string like '2h30m', '45s', '1d2h', '10m5s' into a timedelta.

    Supports y (years), d (days), h (hours), m (minutes), s (seconds), ms (milliseconds),
    µs (microseconds), ns (nanoseconds).
    """
    m = _TIMEDELTA_PATTERN.fullmatch(delta_str.strip())
    if not m:
        raise ValueError(f"invalid timedelta string: '{delta_str}'")
    parts = {name: float(val) for name, val in m.groupdict(default="0").items()}
    return datetime.timedelta(
        days=parts["years"] * 365 + parts["days"],
        hours=parts["hours"],
        minutes=parts["mins"],
        seconds=parts["secs"],
        milliseconds=parts["millis"],
        microseconds=parts["micros"] + parts["nanos"] / 1000,
    )
