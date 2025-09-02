import datetime
import functools
import importlib
import inspect
import os
import re
import site
import types
import typing

import numpy as np
import pandas as pd


def get_portable_representation(
    obj: typing.Any,
    max_length: int | None = None,
) -> str:
    """Convert any Python object to a portable, environment-independent representation.

    Produces a stable, parse-friendly string representation of Python objects suitable for
    tracing reproducibility across different environments/runs.

    Notes: This function is NOT invertible. If max_length is set and exceeded, the final
    string is truncated (with ellipsis when possible).

    Args:
        obj: Any Python object encountered during code tracing.
        max_length: Optional maximum length for the returned string. If provided and the
            generated representation exceeds this length, it will be truncated. For
            max_length > 3 the truncation uses an ellipsis suffix "..."; otherwise it
            simply slices the string.

    Returns:
        A string containing structured information about the object with stable identifiers.
    """

    def _truncate(s: str) -> str:
        if max_length is None:
            return s
        if max_length < 0:
            return s  # ignore negative
        if len(s) <= max_length:
            return s
        if max_length <= 3:
            return s[:max_length]
        return s[: max_length - 3] + "..."

    def _clean_lingering_addresses_and_paths(s: str) -> str:
        s = re.sub(r" at 0x[0-9a-f]+", "", s)  # remove pointers
        s = re.sub(r" from '.*?'", "", s)  # remove file paths
        return s

    base_types = (int, float, bool, str, bytes, list, tuple, set, dict, BaseException)
    if isinstance(obj, base_types) or obj is None:
        return _truncate(repr(obj))  # all these base types need no special handling
    elif isinstance(obj, np.ndarray):
        return _truncate(f"numpy.{repr(obj)}")  # prefix numpy package name before 'array'
    elif isinstance(obj, pd.DataFrame):
        return _truncate(f"pandas.DataFrame({obj.to_json()})")  # get a serializable output
    elif isinstance(obj, pd.Series):
        return _truncate(f"pandas.Series({obj.to_json()})")  # get a serializable output
    elif inspect.ismodule(obj):
        return _truncate(f"<module '{obj.__name__}'>")
    elif callable(obj):
        return _truncate(f"<callable '{get_portable_function_name(obj)}'>")
    elif hasattr(obj, "__class__") and not isinstance(obj, type):
        return _truncate(f"<instance '{get_fully_qualified_name(obj)}'>")
    else:
        # fallback for any other object; clean the default repr of memory addresses
        # noinspection PyBroadException
        try:
            return _truncate(_clean_lingering_addresses_and_paths(repr(obj)))
        except Exception:
            return _truncate(f"<unprintable '{type(obj).__name__}'>")


def format_object_changes(
    past_obj: typing.Any,
    current_obj: typing.Any,
    max_count: int = 10,
) -> list[str] | None:
    """Tracks changes between past and current state of an object.

    Formats those changes for JSON export if the number of changes is below max_count.

    Supported object types:
    - numpy arrays
    - pandas DataFrames and Series
    - dictionaries
    - lists
    - tuples

    Args:
        past_obj: The previous state of the object
        current_obj: The current state of the object
        max_count: Maximum number of changes to track before giving up

    Returns:
        A list of formatted strings representing changes, or None if too many changes
        or objects are incompatible

    Format:
        "type:metadata:value_idx:value"

    Examples:
        >>> past_list = [1, 2, 3]
        >>> current_list = [1, 5, 3]
        >>> format_object_changes(past_list, current_list)
        ['list:len=3:1:5']
    """
    if type(past_obj) is not type(current_obj):
        return None  # different objects, nothing to do here
    changes = []
    if isinstance(current_obj, np.ndarray):
        if past_obj.shape != current_obj.shape:
            return None  # different shapes, too complex to track
        diff_indices = np.where(past_obj != current_obj)
        change_count = len(diff_indices[0])
        if change_count > max_count:
            return None  # too many changes
        for index_tuple in zip(*diff_indices):
            index_str = ",".join(str(index_component) for index_component in index_tuple)
            value = current_obj[index_tuple]
            metadata = f"shape={current_obj.shape},dtype={current_obj.dtype}"
            changes.append(f"numpy.ndarray:{metadata}:{index_str}:{value}")
    elif isinstance(current_obj, pd.DataFrame):
        if past_obj.shape != current_obj.shape:
            return None  # different shapes, too complex to track
        diff_df = ~past_obj.eq(current_obj)
        if diff_df.sum().sum() > max_count:
            return None  # too many changes
        for column in diff_df.columns:
            for row in diff_df.index[diff_df[column]]:
                value = current_obj.loc[row, column]
                metadata = f"shape={current_obj.shape}"
                changes.append(f"dataframe:{metadata}:({row},{column}):{value}")
    elif isinstance(current_obj, pd.Series):
        if len(past_obj) != len(current_obj):
            return None  # different lengths, too complex to track
        diff_series = ~past_obj.eq(current_obj)
        if diff_series.sum() > max_count:
            return None  # too many changes
        for index in diff_series.index[diff_series]:
            value = current_obj[index]
            metadata = f"len={len(current_obj)}"
            changes.append(f"series:{metadata}:{index}:{value}")
    elif isinstance(current_obj, dict):
        # check for added, removed, or changed keys
        past_keys = set(past_obj.keys())
        current_keys = set(current_obj.keys())
        added = current_keys - past_keys
        removed = past_keys - current_keys
        common = past_keys & current_keys
        changed = {key for key in common if past_obj[key] != current_obj[key]}
        total_changes = len(added) + len(removed) + len(changed)
        if total_changes > max_count:
            return None  # too many changes
        metadata = f"len={len(current_obj)}"
        for key in added:
            changes.append(f"dict:{metadata}:added({key}):{current_obj[key]}")
        for key in removed:
            changes.append(f"dict:{metadata}:removed({key}):None")
        for key in changed:
            changes.append(f"dict:{metadata}:changed({key}):{current_obj[key]}")
    elif isinstance(current_obj, (list, tuple)):
        if len(past_obj) != len(current_obj):
            return None  # different lengths, too complex to track
        changed_indices = [index for index, (past, current) in enumerate(zip(past_obj, current_obj)) if past != current]
        if len(changed_indices) > max_count:
            return None  # too many changes
        metadata = f"len={len(current_obj)}"
        type_name = type(current_obj).__name__
        for index in changed_indices:
            changes.append(f"{type_name}:{metadata}:{index}:{current_obj[index]}")
    elif hasattr(current_obj, "__class__"):
        past_attrs = {
            attr: getattr(past_obj, attr)
            for attr in dir(past_obj)
            if not attr.startswith("_") and hasattr(past_obj, attr)
        }
        current_attrs = {
            attr: getattr(current_obj, attr)
            for attr in dir(current_obj)
            if not attr.startswith("_") and hasattr(current_obj, attr)
        }
        changed_attrs = {
            attr
            for attr in set(past_attrs.keys()) | set(current_attrs.keys())
            if past_attrs.get(attr, None) != current_attrs.get(attr, None)
        }
        total_changes = len(changed_attrs)
        if total_changes > max_count:
            return None  # too many changes
        class_name = current_obj.__class__.__name__
        module = current_obj.__class__.__module__
        for attr in changed_attrs:
            changes.append(f"instance:{module}.{class_name}:changed({attr}):{current_attrs.get(attr, None)}")
    else:
        # unsupported type
        return None
    return changes


def get_portable_filename(filename: str) -> str:
    """Returns a cleaned up module/framework filename for portable logging purposes."""
    import pyine.utils.filesystem
    from pyine.utils.code.execution import EXEC_TRACE_FILE_NAME

    if filename == EXEC_TRACE_FILE_NAME:
        return filename  # nothing to do (special case for code execution from strings)
    site_pkgs = site.getsitepackages() + [site.getusersitepackages()]
    for site_dir in site_pkgs:
        if filename.startswith(site_dir):
            return os.path.relpath(filename, site_dir)
    stdlib_dir = os.path.dirname(os.__file__)
    if filename.startswith(stdlib_dir):
        return os.path.relpath(filename, stdlib_dir)
    project_root = str(pyine.utils.filesystem.get_project_root_path())
    if filename.startswith(project_root):
        return os.path.relpath(filename, project_root)
    return filename


def get_portable_function_name(callabl: typing.Callable) -> str:
    """Return a stable, address-free string for a callable."""

    def _qual(module, qualname):
        if module and module != "builtins":
            return f"{module}.{qualname}"
        return qualname

    if isinstance(callabl, functools.partial):  # recursively get wrapped function names
        return f"functools.partial({get_portable_function_name(callabl.func)})"
    # noinspection PyUnreachableCode
    if isinstance(callabl, types.MethodType):  # handle bound callables
        func = callabl.__func__
        module = getattr(func, "__module__", None)
        qualname = getattr(func, "__qualname__", getattr(func, "__name__", "<?>"))
        base = _qual(module, qualname)
    elif (
        # plain functions, including nested and class/staticmethod functions
        isinstance(callable, (types.FunctionType, types.BuiltinFunctionType, types.BuiltinMethodType))
        or inspect.isbuiltin(callabl)
    ):
        module = getattr(callabl, "__module__", None)
        qualname = getattr(callabl, "__qualname__", getattr(callabl, "__name__", "<?>"))
        base = _qual(module, qualname)
    elif isinstance(callabl, type):  # handle classes (constructors are callable)
        module = getattr(callabl, "__module__", None)
        qualname = getattr(callabl, "__qualname__", getattr(callabl, "__name__", "<?>"))
        base = _qual(module, qualname)
    elif callable(callabl):  # fallback for general callables
        cls = callabl.__class__  # noqa
        module = getattr(cls, "__module__", None)
        qualname = getattr(cls, "__qualname__", getattr(cls, "__name__", "<?>"))
        base = _qual(module, f"{qualname}.__call__")
    else:
        # pragma: no cover
        raise TypeError("object is not callable")
    return base


def get_code_with_numbered_lines(
    code_string: str,
    prefixed_tabs: int = 0,
) -> str:
    """Generates a string containing code with line numbers and tabs.

    Args:
        code_string (str): the Python code string to be formatted.
        prefixed_tabs (int): the number of tabs to prefix each line with. Defaults to 0.

    Returns:
        str: The formatted code string with line numbers and tabs
    """
    tab_prefix = "\t" * prefixed_tabs
    formatted_lines = []
    for line_idx, line_content in enumerate(code_string.splitlines(), start=1):
        formatted_lines.append(f"{tab_prefix}L{line_idx:04d}:   {line_content}")
    return "\n".join(formatted_lines)


def print_code_with_numbered_lines(
    code_string: str,
    prefixed_tabs: int = 0,
    logger: typing.Any | None = None,
) -> None:
    """Prints each line of the given code string, prefixed with tabs and a fixed-width line number.

    Can output to either a logger or stdout (by default, if no logger is provided)..

    Args:
        code_string (str): the Python code string to be printed.
        prefixed_tabs (int): the number of tabs to prefix each line with. Defaults to 0.
        logger: optional logger object with a info/debug method. If None, output goes to stdout.
    """
    # pragma: no cover
    if logger is not None:
        if hasattr(logger, "info") and callable(logger.info):
            logging_func = logger.info
        elif hasattr(logger, "debug") and callable(logger.debug):
            logging_func = logger.debug
        elif hasattr(logger, "log") and callable(logger.log):
            logging_func = logger.log
        else:
            raise ValueError("could not identify how to use logger object")
    else:
        logging_func = print
    formatted_code = get_code_with_numbered_lines(code_string, prefixed_tabs)
    for line in formatted_code.splitlines():
        logging_func(line)


def estimate_tolerance(value_str: str) -> tuple[float, float]:
    """Estimate rtol and atol parameters for np.isclose based on string representation.

    The returned rtol and atol values are ARBITRARY and may not be accurate. This function will
    throw if the provided string is not a valid floating point number.

    Args:
        value_str: string representation of a floating point number.

    Returns:
        Tuple of (rtol, atol) values for np.isclose
    """
    value_str = value_str.strip()
    if "e" in value_str.lower():
        parts = value_str.lower().split("e")
        mantissa = parts[0]
        exponent = int(parts[1])
        mantissa_clean = mantissa.replace("-", "").replace("+", "").replace(".", "")
        mantissa_clean = mantissa_clean.lstrip("0") or "0"
        sig_digits = len(mantissa_clean)
        decimal_places = max(0, sig_digits - exponent - 1)
    else:
        if "." in value_str:
            decimal_part = value_str.split(".")[1]
            decimal_places = len(decimal_part.rstrip("0"))
        else:
            decimal_places = 0
    float_val = float(value_str)
    magnitude = abs(float_val)
    if decimal_places == 0:
        atol = 0.0
    else:
        atol = 0.5 * (10 ** (-decimal_places))
    if magnitude < 1e-10:
        rtol = 1e-9
    elif magnitude < 1e-6:
        rtol = 1e-8
    elif magnitude > 1e6:
        rtol = 1e-5
    else:
        rtol = 0.5 * (10 ** (-decimal_places)) / magnitude
        rtol = max(rtol, 1e-15)
        rtol = min(rtol, 1e-5)
    return rtol, atol


def import_from_dotted_path(
    dotted_path: str,
) -> typing.Any:
    """Import and return an attribute given a dotted path like 'pkg.mod.Class'.

    Args:
        dotted_path: Dotted import path.

    Returns:
        Imported attribute.
    """
    if not isinstance(dotted_path, str) or not dotted_path:
        raise ValueError("dotted_path must be a non-empty string")
    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        # assume the entire dotted path is a module name without any package path
        module_path = attr_name
        attr_name = None
    module = importlib.import_module(module_path)
    if not attr_name:
        # no need to return any attribute, just return the module
        return module
    try:
        # get the attribute from the module
        return getattr(module, attr_name)
    except AttributeError as err:
        raise ValueError(f"Attribute {attr_name!r} not found in {module_path!r}") from err


def get_fully_qualified_name(
    obj: type | types.ModuleType | typing.Callable,
) -> str:
    """Get the fully qualified name of a type, module, or callable."""
    if isinstance(obj, types.ModuleType):
        spec = getattr(obj, "__spec__", None)
        if spec and getattr(spec, "name", None):
            mod_name = spec.name
        else:
            mod_name = getattr(obj, "__name__", None)
        if mod_name is None:
            mod_name = "<module>"
        return mod_name
    if isinstance(obj, type):  # classes/types
        mod = getattr(obj, "__module__", "") or ""
        qual = getattr(obj, "__qualname__", getattr(obj, "__name__", "<unknown>"))
        return qual if mod == "builtins" or not mod else f"{mod}.{qual}"
    # fallback for functions/methods/builtins/callables
    mod = getattr(obj, "__module__", "") or ""
    qual = getattr(obj, "__qualname__", getattr(obj, "__name__", None))
    if qual is None:
        # noinspection PyUnreachableCode
        if callable(obj):
            cls = obj.__class__  # noqa
            mod = getattr(cls, "__module__", "") or ""
            qual = getattr(cls, "__qualname__", getattr(cls, "__name__", "<callable>")) + ".__call__"
        else:
            # non-callable fallback to class identity (shouldn't happen for declared types/callables)
            cls = obj.__class__  # noqa
            mod = getattr(cls, "__module__", "") or ""
            qual = getattr(cls, "__qualname__", getattr(cls, "__name__", "<object>"))
    return qual if mod == "builtins" or not mod else f"{mod}.{qual}"


_DURATION_RE = re.compile(r"(?P<value>\d+)\s*(?P<unit>[smhd])", re.IGNORECASE)
"""Regex pattern for parsing duration strings like '90m', '1h30m', '2d', '3600s'."""


def parse_duration_to_timedelta(
    spec: str | None,
) -> datetime.timedelta | None:
    """Parses a duration like '90m', '1h30m', '2d', '3600s' to a timedelta, or None if spec is falsy.

    The parser is strict: any unexpected characters (other than whitespace between tokens)
    cause a ValueError.
    """
    if not spec:
        return None
    total_seconds = 0
    idx = 0
    n = len(spec)
    matched_any = False
    while idx < n:
        # skip whitespace between tokens
        while idx < n and spec[idx].isspace():
            idx += 1
        if idx >= n:
            break
        match = _DURATION_RE.match(spec, idx)
        if not match:
            raise ValueError(f"invalid duration token starting at position {idx} in '{spec}'")
        matched_any = True
        value = int(match.group("value"))
        unit = match.group("unit").lower()
        if unit == "s":
            total_seconds += value
        elif unit == "m":
            total_seconds += value * 60
        elif unit == "h":
            total_seconds += value * 3600
        elif unit == "d":
            total_seconds += value * 86400
        else:
            # should not happen due to regex, but keep for safety
            raise ValueError(f"unsupported time unit in '{spec}': {unit}")
        idx = match.end()
    if not matched_any:
        raise ValueError(f"could not parse duration spec: '{spec}'")
    return datetime.timedelta(seconds=total_seconds)


def parse_indices_spec(
    spec: str,
) -> list[int]:
    """Parses comma-separated indices spec like '1-5,7,9-12' into a sorted list of unique ints.

    Note: when a range is specified in the provided spec, it is assumed to be inclusive on both ends.
    """
    indices: set[int] = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"invalid range '{part}': end < start")
            for idx in range(start, end + 1):
                indices.add(idx)
        else:
            indices.add(int(part))
    return sorted(indices)
