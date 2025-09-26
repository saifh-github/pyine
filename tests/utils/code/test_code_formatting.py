import builtins
import pathlib
import subprocess
import sys
import types

import black
import pytest

import pyine.utils.code.formatting as fmt


def test_format_code_ruff_cli_invocation(monkeypatch: pytest.MonkeyPatch):
    captured_command: dict[str, list[str]] = {}

    def fake_which(executable_name: str) -> str | None:
        return "/usr/bin/ruff" if executable_name == "ruff" else None

    def fake_run(cmd, capture_output, text):
        captured_command["cmd"] = cmd
        target_file = pathlib.Path(cmd[-1])
        target_file.write_text("cli formatted\n", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("pyine.utils.code.formatting.shutil.which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)
    formatted = fmt.format_code("print('hi')", line_length=99)
    assert formatted == "cli formatted\n"
    cmd = captured_command["cmd"]
    assert cmd[0] == "/usr/bin/ruff"
    assert cmd[1:5] == ["format", "--line-length", "99", "--quiet"]


def test_format_code_ruff_cli_failure_falls_back_to_black(monkeypatch: pytest.MonkeyPatch):
    def fake_which(executable_name: str) -> str | None:
        return "/usr/bin/ruff" if executable_name == "ruff" else None

    def fake_run(cmd, capture_output, text):
        return types.SimpleNamespace(returncode=2, stderr="boom")

    def fake_black_formatter(code_string, mode):
        return "black fallback\n"

    monkeypatch.setattr("pyine.utils.code.formatting.shutil.which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(black, "format_str", fake_black_formatter)
    formatted = fmt.format_code("print('hi')")
    assert formatted == "black fallback\n"


def test_format_code_black_api_returns_original_on_nothing_changed(monkeypatch: pytest.MonkeyPatch):
    def _raise_nothing_changed(code_string, mode):
        raise black.NothingChanged

    monkeypatch.setattr(black, "format_str", _raise_nothing_changed)
    code = "def f(x):\n    return x\n"
    assert fmt.format_code(code, formatter="black") == code


def test_format_code_black_api_raises_value_error_on_invalid_input(monkeypatch: pytest.MonkeyPatch):
    def _raise_invalid_input(code_string, mode):
        raise black.InvalidInput("bad code")

    monkeypatch.setattr(black, "format_str", _raise_invalid_input)
    with pytest.raises(ValueError):
        _ = fmt.format_code("def bad(:", formatter="black")


def test_format_code_black_cli_uses_black_binary(monkeypatch: pytest.MonkeyPatch):
    captured_command: dict[str, list[str]] = {}

    def fake_which(executable_name: str) -> str | None:
        return "/usr/bin/black" if executable_name == "black" else None

    def fake_run(cmd, capture_output, text):
        captured_command["cmd"] = cmd
        target_file = pathlib.Path(cmd[-1])
        target_file.write_text("formatted via cli\n", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("pyine.utils.code.formatting.shutil.which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)
    formatted = fmt.format_code(
        "print('hi')",
        line_length=88,
        is_pyi=True,
        string_normalization=False,
        formatter="black",
        use_black_api=False,
    )
    assert formatted == "formatted via cli\n"
    cmd = captured_command["cmd"]
    assert cmd[0] == "/usr/bin/black"
    assert cmd[1:5] == ["--line-length", "88", "--pyi", "--skip-string-normalization"]


def test_format_code_black_cli_fallbacks_to_module(monkeypatch: pytest.MonkeyPatch):
    captured_command: dict[str, list[str]] = {}

    def fake_which(executable_name: str) -> None:
        return None

    def fake_run(cmd, capture_output, text):
        captured_command["cmd"] = cmd
        target_file = pathlib.Path(cmd[-1])
        target_file.write_text("module runner\n", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("pyine.utils.code.formatting.shutil.which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)
    formatted = fmt.format_code("print('hi')", formatter="black", use_black_api=False)
    assert formatted == "module runner\n"
    cmd = captured_command["cmd"]
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-m", "black", "--line-length"]


def test_format_code_black_cli_failure_raises_value_error(monkeypatch: pytest.MonkeyPatch):
    def fake_which(executable_name: str) -> str | None:
        return "/usr/bin/black" if executable_name == "black" else None

    def fake_run(cmd, capture_output, text):
        return types.SimpleNamespace(returncode=1, stderr="boom")

    monkeypatch.setattr("pyine.utils.code.formatting.shutil.which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError) as exc_info:
        _ = fmt.format_code("print('hi')", formatter="black", use_black_api=False)
    assert "error formatting code" in str(exc_info.value)


def test_format_code_black_api_unavailable(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "black":
            raise ImportError("missing black")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(fmt.FormatterUnavailableError) as exc_info:
        _ = fmt.format_code("print('hi')", formatter="black", use_black_api=True)
    assert "black Python API unavailable" in str(exc_info.value)


def test_format_code_unsupported_formatter():
    with pytest.raises(ValueError):
        _ = fmt.format_code("print('hi')", formatter="yapf")
