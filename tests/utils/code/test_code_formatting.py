import pathlib
import subprocess
import types

import pytest

import pyine.utils.code.formatting as fmt


def test_format_code_ruff_cli_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_command: dict[str, list[str]] = {}

    def fake_which(executable_name: str) -> str | None:
        return "/usr/bin/ruff" if executable_name == "ruff" else None

    def fake_run(
        cmd: list[str],
        capture_output: bool,
        text: bool,
    ) -> types.SimpleNamespace:
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


def test_format_code_unsupported_formatter() -> None:
    with pytest.raises(ValueError):
        _ = fmt.format_code("print('hi')", formatter="yapf")
