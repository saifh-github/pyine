import os
import shutil
import subprocess
import sys
import tempfile
import uuid

_DEFAULT_LINE_LENGTH = 120


class FormatterUnavailableError(RuntimeError):
    """Raised when the requested formatter cannot be used."""


def format_code_with_ruff(
    code_string: str,
    line_length: int = _DEFAULT_LINE_LENGTH,
    is_pyi: bool = False,
) -> str:
    """Format Python code using Ruff.

    Ruff currently exposes formatting behaviour through its CLI. This helper shells out to ``ruff
    format`` using a temporary file while mirroring the ``--line-length`` flag.

    Args:
        code_string: Raw Python source that needs formatting.
        line_length: Maximum allowed line length for the formatter.
        is_pyi: Whether the source represents a ``.pyi`` stub.

    Returns:
        The formatted Python source.

    Raises:
        FormatterUnavailableError: If Ruff cannot be located or fails to format the code.
    """
    suffix = ".pyi" if is_pyi else ".py"
    try:
        return _run_cli_formatter(
            code_string=code_string,
            command=_build_ruff_cli_command(
                line_length=line_length,
            ),
            suffix=suffix,
        )
    except ValueError as error:
        raise FormatterUnavailableError("ruff CLI failed to format the provided code") from error


def format_code_with_black(
    code_string: str,
    line_length: int = _DEFAULT_LINE_LENGTH,
    is_pyi: bool = False,
    string_normalization: bool = True,
    use_black_api: bool = True,
) -> str:
    """Format Python code using Black.

    Args:
        code_string: Raw Python source that needs formatting.
        line_length: Maximum allowed line length for the formatter.
        is_pyi: Flag indicating the code is a ``.pyi`` stub.
        string_normalization: Whether string normalization should occur.
        use_black_api: When True, use Black's Python API; otherwise shell out to the CLI.

    Returns:
        The formatted Python source.

    Raises:
        ValueError: Raised when Black fails to format the provided code.
        FormatterUnavailableError: Raised when the Black API cannot be imported.
    """
    if use_black_api:
        try:
            import black
        except Exception as error:
            raise FormatterUnavailableError(
                "black Python API unavailable; call with `use_black_api=False` instead"
            ) from error
        mode = black.Mode(
            line_length=line_length,
            is_pyi=is_pyi,
            string_normalization=string_normalization,
        )
        try:
            return black.format_str(code_string, mode=mode)
        except black.NothingChanged:
            return code_string
        except black.InvalidInput as error:
            raise ValueError(f"error formatting code: {error}") from error
    suffix = ".pyi" if is_pyi else ".py"
    return _run_cli_formatter(
        code_string=code_string,
        command=_build_black_cli_command(
            line_length=line_length,
            is_pyi=is_pyi,
            string_normalization=string_normalization,
        ),
        suffix=suffix,
    )


def format_code(
    code_string: str,
    line_length: int = _DEFAULT_LINE_LENGTH,
    is_pyi: bool = False,
    string_normalization: bool = True,
    formatter: str = "ruff",
    use_black_api: bool = True,
) -> str:
    """Format Python code using Ruff (default) or Black.

    Ruff is preferred when available; if the Ruff CLI cannot be used, the helper automatically falls
    back to Black to preserve the historical behavior of ``format_code``.

    Args:
        code_string: Raw Python source that needs formatting.
        line_length: Maximum allowed line length for the formatter.
        is_pyi: Flag indicating the code is a ``.pyi`` stub.
        string_normalization: Whether string normalization should occur (Black-only feature).
        formatter: Name of the formatter to use; accepted values are ``"ruff"`` and ``"black"``.
        use_black_api: When True and ``formatter="black"``, use Black's Python API.

    Returns:
        The formatted Python source.

    Raises:
        FormatterUnavailableError: Raised when the requested formatter cannot be used.
        ValueError: Raised when the formatter fails to parse or format the provided code.
    """
    formatter_normalized = formatter.lower()
    if formatter_normalized == "ruff":
        try:
            return format_code_with_ruff(
                code_string=code_string,
                line_length=line_length,
                is_pyi=is_pyi,
            )
        except FormatterUnavailableError:
            return format_code_with_black(
                code_string=code_string,
                line_length=line_length,
                is_pyi=is_pyi,
                string_normalization=string_normalization,
                use_black_api=use_black_api,
            )
    if formatter_normalized == "black":
        return format_code_with_black(
            code_string=code_string,
            line_length=line_length,
            is_pyi=is_pyi,
            string_normalization=string_normalization,
            use_black_api=use_black_api,
        )
    raise ValueError(f"unsupported formatter '{formatter}'")


def _build_ruff_cli_command(
    line_length: int,
) -> list[str]:
    ruff_executable = shutil.which("ruff")
    if ruff_executable:
        base_cmd = [ruff_executable, "format"]
    else:
        base_cmd = [sys.executable, "-m", "ruff", "format"]
    base_cmd += ["--line-length", str(line_length), "--quiet"]
    return base_cmd


def _build_black_cli_command(
    line_length: int,
    is_pyi: bool,
    string_normalization: bool,
) -> list[str]:
    black_executable = shutil.which("black")
    if black_executable:
        base_cmd = [black_executable]
    else:
        base_cmd = [sys.executable, "-m", "black"]
    base_cmd += ["--line-length", str(line_length)]
    if is_pyi:
        base_cmd.append("--pyi")
    if not string_normalization:
        base_cmd.append("--skip-string-normalization")
    base_cmd.append("--quiet")
    return base_cmd


def _run_cli_formatter(
    code_string: str,
    command: list[str],
    suffix: str,
) -> str:
    with tempfile.TemporaryDirectory(prefix="format_code_") as tmpdir:
        filename = f"snippet_{uuid.uuid4().hex}{suffix}"
        file_path = os.path.join(tmpdir, filename)
        with open(file_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(code_string)
        process = subprocess.run(
            [*command, file_path],
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            stderr = process.stderr.strip()
            raise ValueError(f"error formatting code (exit code {process.returncode}): {stderr}")
        with open(file_path, encoding="utf-8") as handle:
            return handle.read()


if __name__ == "__main__":
    unformatted_code = """\
def very_long_function_nameeeeeeeeeeeeeeeeeeee(
    argument_one,
    argument_two, argument_three,
    arg4,
    arg5,arg6,arg7_and_another_very_long_name_for_an_argument,
):
    print(
        "This is a poorly formatted line")
    a=1; b=2;
    if a<b:
        print('hello world')
    return a+b

class MyClass:
    def another_method(self, x,y,z):
        return ((x*y)  + z)
"""
    print("Original Code:")
    print("```python")
    print(unformatted_code)
    print("```")
    formatted_code = format_code(unformatted_code)
    print("\n" + "=" * 20 + "\n")
    print("Formatted Code:")
    print("```python")
    print(formatted_code)
    print("```")
