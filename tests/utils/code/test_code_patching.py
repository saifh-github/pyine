import typing

import pytest

import pyine.utils.code.patching as patching


def test_compute_and_apply_patch_success() -> None:
    original = """line1\nline2\nline3\n"""
    modified = """line1\nLINE TWO\nline3\n"""
    diff = patching.compute_patch(original, modified)
    result, ok = patching.apply_patch(original, diff)
    assert ok is True
    assert result == modified


def test_apply_patch_empty_patch_fails() -> None:
    original = "a\n"
    result, ok = patching.apply_patch(original, "")
    assert ok is False
    assert result == original


def test_apply_patch_multi_file_fails() -> None:
    a1 = "a\n"
    a2 = "b\n"
    d1 = patching.compute_patch(a1, "A\n")
    d2 = patching.compute_patch(a2, "B\n")
    # Force second patch to be for a different file by changing headers
    d2 = d2.replace("--- original", "--- original2").replace("+++ modified", "+++ modified2")
    multi = d1 + d2
    result, ok = patching.apply_patch(a1, multi)
    assert ok is False
    assert result == a1


def test_apply_patch_context_mismatch_fails() -> None:
    original = "one\nTWO\nthree\n"
    modified = "one\nTWO!\nthree\n"
    diff = patching.compute_patch(original, modified)
    wrong_original = "one\nTWO DIFF\nthree\n"
    result, ok = patching.apply_patch(wrong_original, diff)
    assert ok is False
    assert result == wrong_original


def test_apply_patch_out_of_order_hunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Build a fake PatchSet with a single file containing two hunks out of order
    class FakeHunk:
        def __init__(
            self,
            source_start: int,
            source_length: int,
        ) -> None:
            self.source_start = source_start
            self.source_length = source_length

        def __iter__(self) -> typing.Iterator[typing.Any]:
            return iter(())  # no lines to validate context

    class FakePatchedFile:
        def __iter__(self) -> typing.Iterator[FakeHunk]:
            # first hunk moves index forward by 2, second hunk starts earlier -> out of order
            return iter((FakeHunk(5, 2), FakeHunk(1, 1)))

    class FakePatchSet:
        def __len__(self) -> int:
            return 1

        def __getitem__(
            self,
            idx: int,
        ) -> FakePatchedFile:
            assert idx == 0
            return FakePatchedFile()

    monkeypatch.setattr(patching.unidiff.PatchSet, "from_string", staticmethod(lambda s: FakePatchSet()))

    original = "a\n b\n c\n"
    result, ok = patching.apply_patch(original, "irrelevant")
    assert ok is False
    assert result == original


def test_apply_patch_parsing_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(s: str) -> typing.NoReturn:
        raise RuntimeError("boom")

    monkeypatch.setattr(patching.unidiff.PatchSet, "from_string", staticmethod(boom))
    original = "a\n"
    result, ok = patching.apply_patch(original, "patch")
    assert ok is False
    assert result == original


def test_show_colored_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    # Prepare a small diff with various line prefixes
    diff = """--- original\n+++ modified\n@@ -1,2 +1,2 @@\n-line\n+line!\n context\n"""

    captured = {"html": None}

    class FakeHTML:
        def __init__(
            self,
            html_content: str,
        ) -> None:
            captured["html"] = html_content

    def fake_display(
        obj: typing.Any,
    ) -> None:  # noqa: ARG001
        # object is FakeHTML instance; nothing to do
        return None

    # monkeypatch the display functions on the actual module object
    import IPython.display as real_disp

    monkeypatch.setattr(real_disp, "HTML", FakeHTML, raising=True)
    monkeypatch.setattr(real_disp, "display", fake_display, raising=True)

    patching.show_colored_diff(diff)

    assert captured["html"] is not None
    # expect colored spans for -, +, @@ and context lines
    assert "color:#b31d28" in captured["html"]  # red for removals
    assert "color:#22863a" in captured["html"]  # green for additions
    assert "color:#8250df" in captured["html"]  # purple for hunk header
    assert "color:#6a737d" in captured["html"]  # grey for context
