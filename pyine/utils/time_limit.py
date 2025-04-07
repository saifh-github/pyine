import signal
import time
import types
import typing


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
        ...
        >>> with TimeLimit(1.5):
        ...     time.sleep(10)  # This will raise TimeoutError after 1.5 seconds
        ...
        Traceback (most recent call last):
            ...
        TimeoutError: Code execution timed out after 1.5 seconds.
    """

    def __init__(
        self,
        seconds: float,
        timeout_message: str | None = None,
        on_timeout: typing.Callable[[], typing.Any] | None = None,
    ):
        """
        Initializes the TimeLimit context manager.

        Args:
            seconds: Maximum number of seconds to allow the code to run.
            timeout_message: Custom message for the TimeoutError exception.
            on_timeout: Optional callback function to execute when timeout occurs.
        """
        self.seconds = seconds
        self.timeout_message = timeout_message or f"Code execution timed out after {seconds} seconds."
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
        raise TimeoutError(f"{self.timeout_message} (Actual time: {elapsed:.2f}s)")

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
