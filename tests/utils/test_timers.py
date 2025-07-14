import signal
import time

import pytest

import pyine.utils.timers


def test_timeit_function() -> None:
    """Test that timeit function correctly measures execution time."""

    def sample_function():
        time.sleep(0.1)
        return 42

    class LocalLogger:
        def __init__(self):
            self.messages = []

        def info(self, message):
            self.messages.append(message)

    local_logger = LocalLogger()
    with pyine.utils.timers.timeit(name="test-block", logger=local_logger):  # noqa
        assert sample_function() == 42
    assert len(local_logger.messages) == 1
    expected_message_prefix = "Time [test-block]: "
    assert local_logger.messages[0].startswith(expected_message_prefix)
    time_str = local_logger.messages[0][len(expected_message_prefix) :]
    measured_time = pyine.utils.timers.parse_timedelta(time_str).total_seconds()
    assert measured_time >= 0.1
    assert measured_time < 0.15


def test_timeout_error_initialization() -> None:
    """Test that a TimeoutError exception is initialized correctly."""
    error_message = "Test timeout message"
    with pytest.raises(TimeoutError, match=error_message):
        raise TimeoutError(error_message)


def test_time_limit_executes_within_time() -> None:
    """Test that code executes successfully within the time limit."""
    with pyine.utils.timers.TimeLimit(0.3):  # should not raise TimeoutError
        time.sleep(0.1)


def test_time_limit_raises_timeout_error() -> None:
    """Test that a timeout error is raised when code exceeds the time limit."""
    with pytest.raises(TimeoutError):
        with pyine.utils.timers.TimeLimit(0.1):
            time.sleep(0.3)  # exceeds the time limit


def test_time_limit_with_custom_message() -> None:
    """Test that a custom timeout message is used when provided."""
    custom_message = "Execution exceeded allowed time"
    with pytest.raises(TimeoutError, match=custom_message):
        with pyine.utils.timers.TimeLimit(0.1, timeout_message=custom_message):
            time.sleep(0.3)


def test_time_limit_on_timeout_callback() -> None:
    """Test that the on_timeout callback is invoked when timeout occurs."""
    callback_triggered = False

    def on_timeout_callback() -> None:
        nonlocal callback_triggered
        callback_triggered = True  # mark callback as triggered

    with pytest.raises(TimeoutError):
        with pyine.utils.timers.TimeLimit(0.1, on_timeout=on_timeout_callback):
            time.sleep(0.3)

    assert callback_triggered is True


def test_time_limit_exit_restores_signal() -> None:
    """Test that TimeLimit restores the previous signal handler upon exit."""
    prev_signal_handler = signal.getsignal(signal.SIGALRM)

    with pytest.raises(TimeoutError):
        with pyine.utils.timers.TimeLimit(0.1):
            assert signal.getsignal(signal.SIGALRM) is not prev_signal_handler
            time.sleep(0.3)

    assert signal.getsignal(signal.SIGALRM) is prev_signal_handler


def test_time_limit_no_timeout_on_short_execution() -> None:
    """Test that TimeLimit works without timeout for a quick operation."""
    with pyine.utils.timers.TimeLimit(0.1):
        sum(range(100000))  # a quick operation
