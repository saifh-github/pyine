"""Lightweight utilities for comparing Python program outputs.

This module provides utilities to compare outputs that may be:
- strings (possibly printed representations of Python literals/containers)
- ints/floats
- already-parsed containers of plain Python types

It supports tolerant float comparisons, deep container comparisons, and
numeric-aware text comparison for printed outputs.
"""

import ast
import dataclasses
import math
import re
import typing

import pydantic

import pyine.utils.portability

__all__ = [
    "CompareOptions",
    "CompareResult",
    "compare",
]

Number = typing.Union[int, float]
auto = typing.Literal["auto"]


class CompareOptions(pydantic.BaseModel):
    """Options that control how outputs are compared."""

    rel_tol: float | auto = 1e-9
    """Relative tolerance for float comparisons."""
    abs_tol: float | auto = 0.0
    """Absolute tolerance for float comparisons."""
    normalize_whitespace: bool = True
    """Normalize internal whitespace and line endings when comparing text."""
    strip: bool = True
    """Strip leading/trailing whitespace before comparing text."""
    case_sensitive: bool = True
    """Whether text comparison is case-sensitive."""
    numeric_token_tolerance: bool = True
    """If True, text comparison tokenizes numbers and compares them with tolerances."""
    list_order_matters: bool = True
    """Whether list element order matters."""
    tuple_order_matters: bool = True
    """Whether tuple element order matters."""
    array_type_matters: bool = True
    """Whether list vs tuple type must match. If False, lists and tuples are compared as sequences regardless of type."""
    # note: dict and set are order-insensitive by definition; nested lists/tuples obey above flags
    nan_equal: bool = True
    """If True, NaN is considered equal to NaN."""


@dataclasses.dataclass
class CompareResult:
    """Result of a comparison.

    Attributes:
        equal: True if compared values are considered equal.
        reason: Explanation when not equal (empty if equal).
        path: Breadcrumb path indicating where a mismatch occurred.
    """

    equal: bool
    reason: str = ""
    path: str = ""  # breadcrumb path for where the mismatch occurred

    def __bool__(self) -> bool:
        """Return True if comparison succeeded."""
        return self.equal

    def with_path(self, segment: str) -> "CompareResult":
        """Return a copy of this result with a path segment prepended when unequal.

        Args:
            segment: Path segment to prepend.

        Returns:
            CompareResult: Updated result (or self if already equal).
        """
        if self.equal:
            return self
        new_path = segment if not self.path else f"{segment}.{self.path}"
        return CompareResult(False, self.reason, new_path)


def compare(
    a: typing.Any,
    b: typing.Any,
    options: CompareOptions | None = None,
) -> CompareResult:
    """Compare two outputs that may be text, numbers, or containers.

    Args:
        a: First value.
        b: Second value.
        options: Optional comparison options. If None, defaults are used.

    Returns:
        CompareResult: Outcome with equality flag, reason, and mismatch path.
    """
    if options is None:
        options = CompareOptions()
    # fast path when both are numeric (int/float)
    if _is_number(a) and _is_number(b):
        return _compare_numbers(a, b, options)
    # if either is string, attempt to interpret as Python literal to compare structured data
    if isinstance(a, str) or isinstance(b, str):
        return _compare_strings_or_literals(a, b, options)
    # otherwise, compare as structured Python objects (lists, dicts, sets, tuples, POD, ...)
    return _compare_objects(a, b, options, path="")


def _compare_strings_or_literals(
    a: typing.Any,
    b: typing.Any,
    opt: CompareOptions,
) -> CompareResult:
    """Compare two strings or Python literals."""
    a_is_str = isinstance(a, str)
    b_is_str = isinstance(b, str)
    # if both strings, try literal_eval both
    if a_is_str and b_is_str:
        a_lit = _try_literal_eval(a)
        b_lit = _try_literal_eval(b)
        if a_lit.success and b_lit.success:
            return _compare_objects(a_lit.value, b_lit.value, opt, path="")
        # if only one is successfully parsed, fall back to text compare
        return _compare_text(a, b, opt)
    # if one is str and the other is not, try parsing the string and compare
    if a_is_str and not b_is_str:
        a_lit = _try_literal_eval(a)
        if a_lit.success:
            return _compare_objects(a_lit.value, b, opt, path="")
        # if not a literal, compare string to repr of b as text
        return _compare_text(a, repr(b), opt)
    if b_is_str and not a_is_str:
        b_lit = _try_literal_eval(b)
        if b_lit.success:
            return _compare_objects(a, b_lit.value, opt, path="")
        return _compare_text(repr(a), b, opt)
    # fallback (should not reach)
    raise NotImplementedError("unexpected string/literals comparison case")


@dataclasses.dataclass
class _ParseResult:
    success: bool
    value: typing.Any = None


def _try_literal_eval(s: str) -> _ParseResult:
    """Try to parse a string as a Python literal."""
    try:
        value = ast.literal_eval(s)
        return _ParseResult(True, value)
    except Exception:
        return _ParseResult(False, None)


def _compare_text(a: str, b: str, opt: CompareOptions) -> CompareResult:
    """Compare two strings with tolerances for numeric values."""
    if not opt.case_sensitive:
        a, b = a.lower(), b.lower()
    if opt.strip:
        a, b = a.strip(), b.strip()
    if opt.normalize_whitespace:
        a = _normalize_ws(a)
        b = _normalize_ws(b)
    if a == b:
        # we're done, strings matched
        return _ok()
    # otherwise, it might be a tolerance problem (comparing numbers as strings)
    if opt.numeric_token_tolerance:
        # compare by tokenizing numbers and comparing numerically with tolerance.
        return _compare_text_with_numeric_tolerance(a, b, opt)
    # if we get here, nothing worked, the text is considered different
    return _fail(f"Text differs: {a!r} != {b!r}")


_NUM_RE = re.compile(
    r"""
    (?P<num>
        [+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?   # decimal like 1.23, .5, 10, 10. with optional exponent
        |
        [+-]?(?:inf|nan)  # inf/nan spellings with optional sign
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _compare_text_with_numeric_tolerance(a: str, b: str, opt: CompareOptions) -> CompareResult:
    """Compare two strings with numeric tolerances."""
    # - split strings into alternating text and numeric tokens
    # - non-numeric substrings must match exactly (after earlier normalization)
    # - numeric tokens are compared with float tolerance (including handling truncated prints)
    a_parts = _split_with_numbers(a)
    b_parts = _split_with_numbers(b)
    if len(a_parts) != len(b_parts):
        return _fail(f"Token structure differs: {len(a_parts)} tokens vs {len(b_parts)} tokens")
    for idx, (ak, av), (bk, bv) in zip(range(len(a_parts)), a_parts, b_parts):
        if ak != bk:
            return _fail(f"Token kind mismatch at token {idx}: {ak} vs {bk}")
        if ak == "text":
            if av != bv:
                return _fail(f"Text segment differs at token {idx}: {av!r} != {bv!r}")
        else:
            # check numeric match
            an = _parse_number_token(av)
            bn = _parse_number_token(bv)
            res = _compare_numbers(an, bn, opt)
            if not res.equal:
                return res.with_path(f"num[{idx}]")
    return _ok()


def _split_with_numbers(s: str) -> list[tuple[str, str]]:
    """Split a string into alternating text and numeric tokens."""
    parts: list[tuple[str, str]] = []
    pos = 0
    for m in _NUM_RE.finditer(s):
        if m.start() > pos:
            parts.append(("text", s[pos : m.start()]))
        parts.append(("num", m.group("num")))
        pos = m.end()
    if pos < len(s):
        parts.append(("text", s[pos:]))
    if not parts:
        parts = [("text", s)]
    return parts


def _parse_number_token(tok: str) -> float:
    """Parse a numeric token into a float."""
    tl = tok.lower()
    if "inf" in tl:
        return math.inf if not tl.startswith("-") else -math.inf
    if "nan" in tl:
        return math.nan
    try:
        return float(tl)
    except Exception:
        # mark as difference later via exact compare
        return float("nan")


def _normalize_ws(s: str) -> str:
    """Normalize internal whitespace and line endings."""
    # normalize internal whitespace and line endings; keep newlines as separators but trim trailing spaces per line
    lines = [ln.strip() for ln in s.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    norm_lines = [" ".join(ln.split()) for ln in lines]
    return "\n".join(norm_lines)


def _compare_objects(
    a: typing.Any,
    b: typing.Any,
    opt: CompareOptions,
    path: str,
) -> CompareResult:
    """Compare two structured Python objects."""
    if _is_number(a) and _is_number(b):
        res = _compare_numbers(a, b, opt)
        return res if res.equal else res.with_path(path or "value")
    if isinstance(a, str) and isinstance(b, str):
        res = _compare_text(a, b, opt)
        return res if res.equal else res.with_path(path or "str")
    if isinstance(a, bool) and isinstance(b, bool):
        return _ok() if a is b else _fail_path(path, f"Bool differs: {a} != {b}")
    if a is None or b is None:
        return _ok() if a is b else _fail_path(path, f"None differs: {a} != {b}")
    if type(a) is not type(b):
        # allow comparing lists and tuples when configured to ignore array type differences.
        if not opt.array_type_matters and isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            order_a = opt.list_order_matters if isinstance(a, list) else opt.tuple_order_matters
            order_b = opt.list_order_matters if isinstance(b, list) else opt.tuple_order_matters
            order_matters = order_a and order_b
            return _compare_sequences(a, b, opt, path, order_matters=order_matters)
        return _fail_path(path, f"Type differs: {type(a).__name__} != {type(b).__name__}")
    if isinstance(a, list):
        return _compare_sequences(a, b, opt, path, order_matters=opt.list_order_matters)
    if isinstance(a, tuple):
        return _compare_sequences(a, b, opt, path, order_matters=opt.tuple_order_matters)
    if isinstance(a, set):
        return _compare_sets(a, b, opt, path)
    if isinstance(a, dict):
        return _compare_dicts(a, b, opt, path)
    # fallback: direct equality
    return _ok() if a == b else _fail_path(path, f"Values differ: {a!r} != {b!r}")


def _compare_numbers(a: Number, b: Number, opt: CompareOptions) -> CompareResult:
    """Compare two numbers."""
    if isinstance(a, float) or isinstance(b, float):  # noqa
        if math.isnan(a) or math.isnan(b):
            if opt.nan_equal and math.isnan(a) and math.isnan(b):
                return _ok()
            return _fail("NaN mismatch")
        if math.isinf(a) or math.isinf(b):
            return _ok() if (a == b) else _fail("Infinity differs")
    abs_tol, rel_tol = opt.abs_tol, opt.rel_tol
    try:
        if abs_tol == "auto" or rel_tol == "auto":
            a_rtol, a_atol = pyine.utils.portability.estimate_tolerance(str(a))
            b_rtol, b_atol = pyine.utils.portability.estimate_tolerance(str(b))
            if abs_tol == "auto":
                abs_tol = max(a_atol, b_atol)
            if rel_tol == "auto":
                rel_tol = max(a_rtol, b_rtol)
        if math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol):
            return _ok()
    except Exception:
        pass
    return _fail(f"Numbers differ: {a} != {b} (rel_tol={rel_tol}, abs_tol={abs_tol})")


def _compare_sequences(
    a: typing.Iterable[typing.Any],
    b: typing.Iterable[typing.Any],
    opt: CompareOptions,
    path: str,
    order_matters: bool,
) -> CompareResult:
    """Compare two sequences."""
    a_list = list(a)
    b_list = list(b)
    if len(a_list) != len(b_list):
        return _fail_path(path, f"Length differs: {len(a_list)} != {len(b_list)}")
    if order_matters:
        for i, (ai, bi) in enumerate(zip(a_list, b_list)):
            res = _compare_objects(ai, bi, opt, path=f"{path}[{i}]" if path else f"[{i}]")
            if not res.equal:
                return res
        return _ok()
    else:
        # order-insensitive multiset comparison via matching
        used = [False] * len(b_list)
        for i, ai in enumerate(a_list):
            matched = False
            for j, bj in enumerate(b_list):
                if used[j]:
                    continue
                res = _compare_objects(ai, bj, opt, path="")
                if res.equal:
                    used[j] = True
                    matched = True
                    break
            if not matched:
                return _fail_path(path, f"No match for element at index {i}: {ai!r}")
        return _ok()


def _compare_sets(a: set, b: set, opt: CompareOptions, path: str) -> CompareResult:
    """Compare two sets."""
    if len(a) != len(b):
        return _fail_path(path, f"Set size differs: {len(a)} != {len(b)}")
    # convert to lists and do order-insensitive matching using deep compare
    b_remaining = list(b)
    for i, av in enumerate(a):
        found = False
        for j, bv in enumerate(b_remaining):
            res = _compare_objects(av, bv, opt, path="")
            if res.equal:
                b_remaining.pop(j)
                found = True
                break
        if not found:
            return _fail_path(path, f"Set element not found: {av!r}")
    return _ok()


def _compare_dicts(
    a: typing.Mapping[typing.Any, typing.Any],
    b: typing.Mapping[typing.Any, typing.Any],
    opt: CompareOptions,
    path: str,
) -> CompareResult:
    """Compare two dicts."""
    if len(a) != len(b):
        return _fail_path(path, f"Dict size differs: {len(a)} != {len(b)}")
    # Keys compared by value (with deep compare). For simplicity, require exact key equality by equality semantics.
    if set(a.keys()) != set(b.keys()):
        return _fail_path(path, f"Dict keys differ: {set(a.keys()) ^ set(b.keys())}")
    # Compare values
    for k in a.keys():
        res = _compare_objects(a[k], b[k], opt, path=f"{path}[{k!r}]" if path else f"[{k!r}]")
        if not res.equal:
            return res
    return _ok()


def _is_number(x: typing.Any) -> bool:
    """Check if a value is a number (i.e. an integer/float but not a boolean)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _ok() -> CompareResult:
    """Helper to return an equal result."""
    return CompareResult(True, "")


def _fail(reason: str) -> CompareResult:
    """Helper to return a non-equal result."""
    return CompareResult(False, reason, "")


def _fail_path(path: str, reason: str) -> CompareResult:
    """Helper to return a non-equal result with a path."""
    return CompareResult(False, reason, path or "")
