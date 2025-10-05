import signal
import time

import pytest

import pyine.utils.timers as timers


def test_timeit_function() -> None:
    """Test that timeit function correctly measures execution time."""

    def sample_function() -> int:
        time.sleep(0.1)
        return 42

    class LocalLogger:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def info(
            self,
            message: str,
        ) -> None:
            self.messages.append(message)

    local_logger = LocalLogger()
    with timers.timeit(name="test-block", logger=local_logger.info):
        assert sample_function() == 42
    assert len(local_logger.messages) == 1
    expected_message_prefix = "Time [test-block]: "
    assert local_logger.messages[0].startswith(expected_message_prefix)
    time_str = local_logger.messages[0][len(expected_message_prefix) :]
    measured_time = timers.parse_timedelta(time_str).total_seconds()
    assert measured_time >= 0.1
    assert measured_time < 0.15


def test_timeout_error_initialization() -> None:
    """Test that a TimeoutError exception is initialized correctly."""
    error_message = "Test timeout message"
    with pytest.raises(TimeoutError, match=error_message):
        raise TimeoutError(error_message)


def test_time_limit_executes_within_time() -> None:
    """Test that code executes successfully within the time limit."""
    with timers.TimeLimit(0.3):  # should not raise TimeoutError
        time.sleep(0.1)


def test_time_limit_raises_timeout_error() -> None:
    """Test that a timeout error is raised when code exceeds the time limit."""
    with pytest.raises(TimeoutError), timers.TimeLimit(0.1):
        time.sleep(0.3)  # exceeds the time limit


def test_time_limit_with_custom_message() -> None:
    """Test that a custom timeout message is used when provided."""
    custom_message = "Execution exceeded allowed time"
    with pytest.raises(TimeoutError, match=custom_message), timers.TimeLimit(0.1, timeout_message=custom_message):
        time.sleep(0.3)


def test_time_limit_on_timeout_callback() -> None:
    """Test that the on_timeout callback is invoked when timeout occurs."""
    callback_triggered = False

    def on_timeout_callback() -> None:
        nonlocal callback_triggered
        callback_triggered = True  # mark callback as triggered

    with pytest.raises(TimeoutError), timers.TimeLimit(0.1, on_timeout=on_timeout_callback):
        time.sleep(0.3)

    assert callback_triggered is True


def test_time_limit_exit_restores_signal() -> None:
    """Test that TimeLimit restores the previous signal handler upon exit."""
    prev_signal_handler = signal.getsignal(signal.SIGALRM)

    with pytest.raises(TimeoutError), timers.TimeLimit(0.1):
        assert signal.getsignal(signal.SIGALRM) is not prev_signal_handler
        time.sleep(0.3)

    assert signal.getsignal(signal.SIGALRM) is prev_signal_handler


def test_time_limit_no_timeout_on_short_execution() -> None:
    """Test that TimeLimit works without timeout for a quick operation."""
    with timers.TimeLimit(0.1):
        sum(range(100000))  # a quick operation


def test_timeit_decorator_stdout_default_name(capsys: pytest.CaptureFixture) -> None:
    @timers.timeit
    def greet(
        x: int,
    ) -> int:
        time.sleep(0.01)
        return x

    assert greet(5) == 5
    out = capsys.readouterr().out
    assert "Time [greet]: " in out


def test_timeit_decorator_with_name_and_logger() -> None:
    class LocalLogger:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def info(
            self,
            message: str,
        ) -> None:
            self.messages.append(message)

    local_logger = LocalLogger()

    @timers.timeit(name="custom", logger=local_logger.info)
    def f() -> int:
        time.sleep(0.01)
        return 7

    assert f() == 7
    assert any(msg.startswith("Time [custom]: ") for msg in local_logger.messages)


def test_timeit_invalid_func_argument_typeerror() -> None:
    with pytest.raises(TypeError):
        timers.timeit(123)  # type: ignore[arg-type]


def test_get_human_readable_time_units() -> None:
    assert timers.get_human_readable_time(0.0) == "0.0ns"
    assert timers.get_human_readable_time(0.5) == "500.000ms"
    assert timers.get_human_readable_time(90.0) == "1.500m"
    assert timers.get_human_readable_time(3600.0) == "1.000h"


def test_parse_timedelta_valid_and_invalid() -> None:
    td = timers.parse_timedelta("1h30m")
    assert td.total_seconds() == 5400
    td2 = timers.parse_timedelta("2.5s10ms2µs3ns")
    # datetime.timedelta has microsecond resolution; nanos are effectively dropped
    assert td2.total_seconds() == pytest.approx(2.510002, abs=1e-6)
    td3 = timers.parse_timedelta("1y")
    assert td3.days == 365
    with pytest.raises(ValueError):
        _ = timers.parse_timedelta("not-a-duration")


def test_time_limit_raises_when_sigalrm_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate platform without SIGALRM
    monkeypatch.delattr(signal, "SIGALRM", raising=False)
    with pytest.raises(ValueError), timers.TimeLimit(0.01):
        pass
