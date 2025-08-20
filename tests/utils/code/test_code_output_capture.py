import contextlib
import sys

import pytest

from pyine.utils.code.output_capture import StdStreamCapture


@pytest.fixture
def capture_stdout() -> StdStreamCapture:
    return StdStreamCapture(
        stream_name="stdout",
        encoding="utf-8",
        errors="replace",
        fileno_value=1,
    )


@pytest.fixture
def capture_stderr() -> StdStreamCapture:
    return StdStreamCapture(
        stream_name="stderr",
        encoding="utf-8",
        errors="replace",
        fileno_value=2,
    )


class TestStdStreamCapture:

    def test_text_and_bytes_write_with_redirect_stdout(
        self,
        capture_stdout: StdStreamCapture,
    ) -> None:
        with contextlib.redirect_stdout(capture_stdout):  # noqa
            print("hello", end="")
            sys.stdout.buffer.write(b" world")
            sys.stdout.write("!")
        assert capture_stdout.getvalue() == "hello world!"
        assert capture_stdout.buffer.getvalue() == b"hello world!"

    def test_seek_and_truncate_clear(
        self,
        capture_stdout: StdStreamCapture,
    ) -> None:
        capture_stdout.write("abc")
        capture_stdout.seek(0)
        capture_stdout.truncate(0)
        assert capture_stdout.getvalue() == ""
        assert capture_stdout.buffer.getvalue() == b""

    def test_writelines(self) -> None:
        cap = StdStreamCapture(stream_name="stdout")
        cap.writelines(["a", "b", "c"])
        assert cap.getvalue() == "abc"
        assert cap.buffer.getvalue() == b"abc"

    def test_stderr_redirect_bytes(
        self,
        capture_stderr: StdStreamCapture,
    ) -> None:
        with contextlib.redirect_stderr(capture_stderr):  # noqa
            sys.stderr.buffer.write(b"oops")
            sys.stderr.write("!")
        assert capture_stderr.getvalue() == "oops!"
        assert capture_stderr.buffer.getvalue() == b"oops!"

    def test_non_str_write_coercion(self) -> None:
        cap = StdStreamCapture(stream_name="stdout")
        cap.write(123)
        cap.write(None)
        assert cap.getvalue() == "123None"
        assert cap.buffer.getvalue().endswith(b"123None")

    def test_invalid_bytes_decoding_replacement(self) -> None:
        cap = StdStreamCapture(
            stream_name="stdout",
            encoding="utf-8",
            errors="replace",
        )
        cap.buffer.write(b"\xff")
        val = cap.getvalue()
        assert isinstance(val, str)
        assert len(val) == 1
        assert cap.buffer.getvalue() == b"\xff"
        cap.seek(0)
        cap.truncate(0)
        assert cap.getvalue() == ""
        assert cap.buffer.getvalue() == b""

    def test_fileno_and_isatty(self) -> None:
        out_cap = StdStreamCapture(stream_name="stdout", fileno_value=1)
        err_cap = StdStreamCapture(stream_name="stderr", fileno_value=2)
        assert out_cap.fileno() == 1
        assert err_cap.fileno() == 2
        assert not out_cap.isatty()
        assert not err_cap.isatty()

    def test_print_builtin(
        self,
        capture_stdout: StdStreamCapture,
    ) -> None:
        with contextlib.redirect_stdout(capture_stdout):  # noqa
            print("test1")
            print("test2", end="")
            print("test3", flush=True)
        assert capture_stdout.getvalue() == "test1\ntest2test3\n"
        assert capture_stdout.buffer.getvalue() == b"test1\ntest2test3\n"

    def test_print_builtin_with_file(self) -> None:
        cap = StdStreamCapture(stream_name="stdout")
        print("test3", end="", file=cap)
        print("test4", flush=True, file=cap)
        assert cap.getvalue() == "test3test4\n"
        assert cap.buffer.getvalue() == b"test3test4\n"

    def test_multi_print_inside_exec(self) -> None:
        cap = StdStreamCapture(stream_name="stdout")
        with contextlib.redirect_stdout(cap):  # noqa
            exec("print('test1')")
            assert cap.getvalue() == "test1\n"
            assert cap.buffer.getvalue() == b"test1\n"
            exec("print('test2')")
            assert cap.getvalue() == "test1\ntest2\n"
            assert cap.buffer.getvalue() == b"test1\ntest2\n"

    def test_multi_print_with_reset(self) -> None:
        cap = StdStreamCapture(stream_name="stdout")
        with contextlib.redirect_stdout(cap):  # noqa
            print("test1")
            cap.flush()
            assert cap.getvalue() == "test1\n"
            cap.seek(0)
            cap.truncate(0)
            print("test2")
            cap.flush()
            assert cap.getvalue() == "test2\n"
            cap.seek(0)
            cap.truncate(0)
            print("test3")
            cap.flush()
            assert cap.getvalue() == "test3\n"
            cap.seek(0)
            cap.truncate(0)
