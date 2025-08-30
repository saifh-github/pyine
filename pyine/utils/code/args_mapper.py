import ast
import collections.abc
import inspect
import typing

import orjson


def map_inputs_to_callable(
    callable_obj: typing.Callable[..., typing.Any],
    inputs: typing.Any,
) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]]:
    """Map arbitrary "inputs" to a callable's arguments.

    The goal is to construct positional and keyword arguments that best match the
    signature of ``callable_obj`` given a wide variety of input shapes and
    representations, including strings like "a=2, b=4" or "1, 2".

    This function never discards any provided arguments. If an input contains
    arguments that cannot be represented for the target's signature (e.g., extra
    positional arguments without ``*args`` or unknown keyword arguments without
    ``**kwargs``), it raises a ValueError rather than dropping them. Positional-only
    parameters are handled specially: if a mapping provides such names, they are
    extracted and placed in the positional arguments in the correct order.

    Args:
        callable_obj: The callable to be invoked with mapped arguments.
        inputs: Arbitrary input describing the arguments. Supported forms include:
            - dict-like mappings (treated as kwargs),
            - sequences (treated as positional args),
            - strings describing args/kwargs (e.g. "a=2, b=4" or "1, 2"),
            - JSON strings ("{\"a\": 2}" or "[1, 2]"),
            - query-string like ("a=1&b=2"),
            - objects with attributes (mapped via vars(obj)),
            - scalars (best-effort if the callable has a single parameter).

    Returns:
        A tuple (args, kwargs) that can be used as ``callable_obj(*args, **kwargs)``.
    """
    sig = inspect.signature(callable_obj)
    params = sig.parameters
    pos_only: list[str] = []
    pos_or_kw: list[str] = []
    kw_only: list[str] = []
    has_var_pos = False
    has_var_kw = False
    for param in params.values():
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            pos_only.append(param.name)
        elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            pos_or_kw.append(param.name)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            kw_only.append(param.name)
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            has_var_pos = True
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_kw = True
    args: list[typing.Any] = []
    kwargs: dict[str, typing.Any] = {}
    if inputs is None:
        # if there are required (non-default) parameters, refuse to silently pass nothing
        required_missing = [
            p.name
            for p in params.values()
            if p.default is inspect._empty
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        if required_missing:
            raise ValueError(
                f"cannot map None to callable requiring arguments {required_missing}; missing required inputs"
            )
        # otherwise, nothing to pass
        pass
    elif isinstance(inputs, collections.abc.Mapping):
        # treat as kwargs; we'll relocate positional-only ones to args
        kwargs = dict(inputs)  # shallow copy
    elif isinstance(inputs, str):
        parsed_args, parsed_kwargs = _parse_args_string(inputs)
        args = parsed_args
        kwargs = parsed_kwargs
    elif isinstance(inputs, collections.abc.Sequence):
        args = list(inputs)
    else:
        # try to coerce objects with attributes to kwargs
        obj_vars = _coerce_to_mapping_if_object(inputs)
        if obj_vars is not None:
            kwargs = obj_vars
        else:
            # scalar: if callable has one param and does not demand keyword, pass as single arg
            total_params = sum(
                1
                for param in params.values()
                if param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
            if total_params == 1 or has_var_pos:
                args = [inputs]
            else:
                raise ValueError(
                    "cannot map scalar input to callable without a single "
                    "positional parameter or *args; refusing to discard input"
                )
    # next, for mappings, relocate positional-only to args in correct order
    if kwargs:
        # move positional-only parameters from kwargs into args by order
        for name in pos_only:
            if name in kwargs:
                args.append(kwargs.pop(name))
        # enforce no-discard: if extra kwargs present and no **kwargs, raise
        if not has_var_kw:
            allowed = set(pos_or_kw) | set(kw_only)
            extras = {k for k in kwargs.keys() if k not in allowed}
            if extras:
                raise ValueError(
                    f"cannot pass unknown keyword arguments {sorted(extras)} to "
                    f"{getattr(callable_obj, '__name__', repr(callable_obj))} without "
                    f"**kwargs; refusing to discard extras"
                )
    # validate no-discard policy and no early errors:
    # - do not allow more positional args than the callable can accept unless it has *args
    # - do not allow duplicate values for the same parameter between args and kwargs
    # - ensure no positional-only parameters remain in kwargs
    max_positional = len(pos_only) + len(pos_or_kw)
    if not has_var_pos and len(args) > max_positional:
        raise ValueError(
            f"cannot pass {len(args)} positional arguments to "
            f"{getattr(callable_obj, '__name__', repr(callable_obj))} which accepts "
            f"at most {max_positional} without *args; refusing to discard extras"
        )
    # detect duplicates (names represented positionally and also provided as kwargs)
    # ...determine which parameter names are bound by current args in order (pos-only then positional-or-keyword)
    bound_by_position: list[str] = []
    for idx, name in enumerate(pos_only):
        if idx < len(args):
            bound_by_position.append(name)
    remaining_slots = max(0, len(args) - len(bound_by_position))
    for idx, name in enumerate(pos_or_kw):
        if idx < remaining_slots:
            bound_by_position.append(name)
    duplicates = set(bound_by_position) & set(kwargs.keys())
    if duplicates:
        raise ValueError(
            f"arguments for parameters {sorted(duplicates)} provided "
            f"both positionally and by keyword; ambiguous mapping"
        )
    # ensure no positional-only parameters are left in kwargs (should have been relocated)
    leftover_pos_only = set(pos_only) & set(kwargs.keys())
    if leftover_pos_only:
        raise ValueError(f"positional-only parameters {sorted(leftover_pos_only)} cannot be passed by keyword")
    # final validation: ensure the signature can be bound with the prepared args/kwargs
    try:
        sig.bind(*args, **kwargs)
    except TypeError as e:
        raise ValueError("failed to bind arguments to callable") from e
    return tuple(args), kwargs


def _coerce_to_mapping_if_object(obj: typing.Any) -> dict[str, typing.Any] | None:
    """Coerce an object's attributes to a mapping if reasonable.

    Returns None if ``obj`` does not look like a plain object with attributes.
    """
    # don't treat basic builtins as objects to map
    if isinstance(obj, (int, float, bool, complex, str, bytes, bytearray)):
        return None
    # if it has a __dict__ or vars returns something
    try:
        obj_dict = vars(obj)
        if isinstance(obj_dict, dict) and obj_dict:
            return {k: v for k, v in obj_dict.items() if not k.startswith("_")}
    except TypeError:
        pass
    return None


def _parse_args_string(
    text: str,
) -> tuple[list[typing.Any], dict[str, typing.Any]]:
    """Parse a string representation of arguments.

    Supports the following forms:
    - JSON objects/arrays: "{\"a\":1}" or "[1, 2]"
    - Comma- or ampersand-separated tokens: "a=1, b=2", "1, 2", "a=1&b=2"

    Returns positional args list and kwargs dict.
    """
    text = text.strip()
    if not text:
        return [], {}
    # strip optional surrounding parentheses
    if (text.startswith("(") and text.endswith(")")) or (text.startswith("[") and text.endswith("]")):
        text = text[1:-1].strip()
    # try JSON first for robustness with quoted strings and nested structures
    if text and text[0] in "[{":
        try:
            parsed = orjson.loads(text)
            if isinstance(parsed, collections.abc.Mapping):
                return [], dict(parsed)
            if isinstance(parsed, collections.abc.Sequence):
                return list(parsed), {}
        except Exception:
            # fall back to manual parsing below
            pass
    tokens = _split_top_level(text, delimiters={",", "&"})
    pos_args: list[typing.Any] = []
    kwargs: dict[str, typing.Any] = {}
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        # key=value or key: value
        key, val, is_kv = _maybe_split_kv(tok)
        if is_kv:
            kwargs[key] = _parse_value(val)
        else:
            pos_args.append(_parse_value(tok))
    return pos_args, kwargs


def _maybe_split_kv(tok: str) -> tuple[str, str, bool]:
    depth = 0
    in_str: str | None = None
    for idx, ch in enumerate(tok):
        if in_str:
            if ch == in_str and tok[idx - 1] != "\\":
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            continue
        if ch in "([{":
            depth += 1
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            continue
        if depth == 0 and ch in "=:":
            key = tok[:idx].strip()
            val = tok[idx + 1 :].strip()
            return key, val, True
    return tok, tok, False


def _split_top_level(text: str, delimiters: set[str]) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_str: str | None = None
    for ch in text:
        if in_str:
            buf.append(ch)
            if ch == in_str and (len(buf) < 2 or buf[-2] != "\\"):
                in_str = None
            continue
        if ch in ('"', "'"):
            buf.append(ch)
            in_str = ch
            continue
        if ch in "([{":
            buf.append(ch)
            depth += 1
            continue
        if ch in ")]}":
            buf.append(ch)
            depth = max(0, depth - 1)
            continue
        if depth == 0 and ch in delimiters:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def _parse_value(text: str) -> typing.Any:
    # quick literals
    low = text.strip().lower()
    if low in {"none", "null"}:
        return None
    if low in {"true", "false"}:
        return low == "true"
    # try literal_eval for numbers, strings, lists, dicts, tuples, etc.
    try:
        return ast.literal_eval(text)
    except Exception:
        pass
    # try JSON for unquoted strings like {a:1} won't work; but keep fast-fail
    try:
        return orjson.loads(text)
    except Exception:
        pass
    # as a last resort, return the stripped string
    return text.strip()
