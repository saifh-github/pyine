import os
import shutil
import subprocess
import sys
import tempfile
import uuid


def format_code(
    code_string: str,
    line_length: int = 120,
    is_pyi: bool = False,
    string_normalization: bool = True,
    use_black_api: bool = True,
) -> str:
    """
    Formats a given string of Python code using the Black formatter.

    By default (use_black_api=False), this function avoids using Black's Python API. As of
    2025-09-06, this API is not yet totally stable it seems, and might even be the source of
    memory leaks. Instead, here, we write the code to be formatted to a uniquely-named temporary
    file, invoke Black via the command line to reformat it in-place, and then read the
    formatted code back. Set use_black_api=True to use Black's Python API instead.

    Args:
        code_string: The string of Python code to format.
        line_length: The maximum line length to allow.
        is_pyi: If True, formats the code as a .pyi stub file.
        string_normalization: If True, normalizes string quotes and prefixes.
        use_black_api: If True, use Black's Python API; otherwise use the CLI (default).

    Returns:
        A string containing the formatted Python code.

    Raises:
        ValueError: If Black fails to format the input (CLI or API path).
        RuntimeError: If Black API is requested but not available.
    """
    if use_black_api:
        try:
            import black  # lazy import to avoid requiring the API by default
        except Exception as e:
            raise RuntimeError("black Python API unavailable; call with `use_black_api=False` instead") from e
        mode = black.Mode(
            line_length=line_length,
            is_pyi=is_pyi,
            string_normalization=string_normalization,
        )
        try:
            return black.format_str(code_string, mode=mode)
        except black.NothingChanged:
            # if the code is already formatted, return the original string
            return code_string
        except black.InvalidInput as e:
            raise ValueError(f"error formatting code: {e}") from e
    # default path: use Black CLI with a unique temporary file
    suffix = ".pyi" if is_pyi else ".py"
    with tempfile.TemporaryDirectory(prefix="format_code_") as tmpdir:
        filename = f"snippet_{uuid.uuid4().hex}{suffix}"
        file_path = os.path.join(tmpdir, filename)
        with open(file_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(code_string)
        # prefer 'black' executable if available, otherwise use 'python -m black'
        black_exe = shutil.which("black")
        if black_exe:
            cmd = [black_exe]
        else:
            cmd = [sys.executable, "-m", "black"]
        cmd += ["--line-length", str(line_length)]
        if is_pyi:
            cmd.append("--pyi")
        if not string_normalization:
            cmd.append("--skip-string-normalization")
        cmd += ["--quiet", file_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ValueError(
                f"error formatting code with Black CLI (exit code {proc.returncode}): {proc.stderr.strip()}"
            )
        with open(file_path, encoding="utf-8") as f:
            return f.read()


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
