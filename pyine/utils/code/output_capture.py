import collections.abc
import typing


class StdStreamCapture:
    """
    A robust stdout/stderr capture stream.

    - Supports text writes (write(str)) and provides a `.buffer` for binary writes (write(bytes)).
    - Exposes .encoding, .errors, .fileno(), .isatty(), and basic file-like methods used in code.
    - Implements getvalue(), flush(), seek(0), and truncate(0) to be compatible with stepwise draining.
    - Decodes bytes written to `.buffer` into text using the configured encoding/errors to keep
      a consistent text representation for tracing/reporting.
    """

    def __init__(
        self,
        stream_name: str = "stdout",
        encoding: str = "utf-8",
        errors: str = "replace",
        fileno_value: int | None = None,
    ) -> None:
        self.stream_name = stream_name
        self.encoding = encoding
        self.errors = errors
        self._closed = False
        # default to 1 for stdout and 2 for stderr if not provided
        self._fileno = fileno_value if fileno_value is not None else (1 if stream_name == "stdout" else 2)
        self.buffer = self._BytesBuffer(self)

    class _BytesBuffer:
        def __init__(self, parent: "StdStreamCapture") -> None:
            self._parent = parent
            self._buf = bytearray()
            self._closed = False

        def write(
            self,
            data: bytes | bytearray | memoryview | collections.abc.Buffer,
        ) -> int:
            if isinstance(data, memoryview):
                chunk = data.tobytes()
            else:
                try:
                    chunk = bytes(data)
                except TypeError as error:
                    raise TypeError(f"a bytes-like object is required, not '{type(data).__name__}'") from error
            self._buf.extend(chunk)
            return len(chunk)

        def writelines(self, lines: typing.Iterable[collections.abc.Buffer | bytes | bytearray | memoryview]) -> int:
            total = 0
            for line in lines:
                total += self.write(line)
            return total

        def getvalue(self) -> bytes:
            return bytes(self._buf)

        def flush(self) -> None:
            # no-op for in-memory buffer
            return None

        def close(self) -> None:
            self._closed = True

        @property
        def closed(self) -> bool:
            return self._closed

        def seek(self, offset: int, whence: int = 0) -> int:
            # do not mutate the buffer on seek; callers should use truncate(0) explicitly to clear.
            return 0

        def truncate(self, size: int | None = None) -> int:
            if size is None or size == 0:
                self._buf.clear()
            elif size < len(self._buf):
                del self._buf[size:]
            return len(self._buf)

        def readable(self) -> bool:
            return False

        def writable(self) -> bool:
            return True

        def seekable(self) -> bool:
            return False

        def append(self, data: bytes) -> None:
            self._buf.extend(data)

        def to_bytes(self) -> bytes:
            return bytes(self._buf)

        def clear(self) -> None:
            self._buf.clear()

    def write(self, s: typing.Any) -> int:
        text = s if isinstance(s, str) else str(s)
        # mirror to the bytes buffer
        try:
            data = text.encode(self.encoding, errors=self.errors)
        except Exception:
            data = text.encode("utf-8", errors="replace")
        self.buffer.append(data)
        return len(text)

    def writelines(self, lines: typing.Iterable[typing.Any]) -> int:
        total = 0
        for line in lines:
            total += self.write(line)
        return total

    def flush(self) -> None:
        # no-op for in-memory capture
        return None

    def getvalue(self) -> str:
        raw = self.buffer.to_bytes()
        try:
            return raw.decode(self.encoding, errors=self.errors)
        except Exception:
            return raw.decode("utf-8", errors="replace")

    def seek(self, offset: int, whence: int = 0) -> int:
        # our caller uses seek(0) just before truncate(0); we accept it as a no-op
        return 0

    def truncate(self, size: int = 0) -> int:
        # used by caller to clear content between steps
        if size == 0:
            self.buffer.truncate(0)
            return 0
        # partial truncate not supported; return current length
        return len(self.getvalue())

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def fileno(self) -> int:
        return self._fileno

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed
