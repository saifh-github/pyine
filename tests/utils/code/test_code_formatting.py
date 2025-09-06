import black
import pytest

import pyine.utils.code.formatting as fmt


def test_reformatting_basic():
    unformatted_code = "def my_func( a,b ):\n  return a+b"
    expected_formatted_code = "def my_func(a, b):\n    return a + b\n"
    actual_formatted_code = fmt.format_code(unformatted_code)
    assert actual_formatted_code == expected_formatted_code


def test_format_code_returns_original_on_nothing_changed(monkeypatch: pytest.MonkeyPatch):
    # simulate black.format_str raising NothingChanged

    def _raise_nothing_changed(code_string, mode):
        raise black.NothingChanged

    monkeypatch.setattr(black, "format_str", _raise_nothing_changed)
    code = "def f(x):\n    return x\n"
    assert fmt.format_code(code) == code


def test_format_code_raises_value_error_on_invalid_input(monkeypatch: pytest.MonkeyPatch):
    # simulate black.format_str raising InvalidInput

    def _raise_invalid_input(code_string, mode):
        raise black.InvalidInput("bad code")

    monkeypatch.setattr(black, "format_str", _raise_invalid_input)
    with pytest.raises(ValueError):
        _ = fmt.format_code("def bad(:")
