import importlib
import inspect
import re
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

    Notes:
      - This function is NOT invertible.
      - For containers, full contents are included by default with a simple "header::body" format.
      - If max_length is set and exceeded, the final string is truncated (with ellipsis when possible).

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

    def _fmt_ndarray(a: np.ndarray) -> str:
        # represent as nested lists for content, with explicit shape/dtype header
        try:
            data_str = repr(a.tolist())
        except Exception:
            data_str = "<unavailable>"
        return f"numpy.ndarray[shape={a.shape},dtype={a.dtype}]::{data_str}"

    def _fmt_series(s: pd.Series) -> str:
        name_part = f",name={s.name}" if getattr(s, "name", None) is not None else ""
        values = repr(s.tolist())
        return f"pandas.Series[len={len(s)},dtype={s.dtype}{name_part}]::{values}"

    def _fmt_dataframe(df: pd.DataFrame) -> str:
        try:
            cols = repr(list(df.columns))
            data = repr(df.values.tolist())
            body = f"columns={cols},data={data}"
        except Exception:
            body = "columns=<unavailable>,data=<unavailable>"
        return f"pandas.DataFrame[shape={df.shape}]::{{{body}}}"

    # -------- base cases and containers --------

    if isinstance(obj, (int, float, bool, str, bytes)) or obj is None:
        return _truncate(f"{repr(obj)}")
    elif isinstance(obj, np.ndarray):
        return _truncate(_fmt_ndarray(obj))
    elif isinstance(obj, pd.DataFrame):
        return _truncate(_fmt_dataframe(obj))
    elif isinstance(obj, pd.Series):
        return _truncate(_fmt_series(obj))
    elif isinstance(obj, list):
        return _truncate(repr(obj))
    elif isinstance(obj, tuple):
        return _truncate(repr(obj))
    elif isinstance(obj, set):
        return _truncate(repr(obj))
    elif isinstance(obj, dict):
        return _truncate(repr(obj))
    elif inspect.ismodule(obj):
        return _truncate(f"module({obj.__name__})")
    elif isinstance(obj, BaseException):
        return _truncate(f"{type(obj).__name__}({str(obj)})")
    elif callable(obj):
        module = getattr(obj, "__module__", None)
        if hasattr(obj, "__qualname__"):
            name = obj.__qualname__
        elif hasattr(obj, "__name__"):
            name = obj.__name__
        else:
            name = f"anonymous({obj})"
        if module is not None:
            return _truncate(f"callable({module}.{name})")
        else:
            return _truncate(f"callable({name})")
    elif hasattr(obj, "__class__") and not isinstance(obj, type):
        class_name = obj.__class__.__name__
        module = obj.__class__.__module__
        return _truncate(f"instance({module}.{class_name})")
    else:
        # fallback for any other objects; clean the default repr of memory addresses
        # noinspection PyBroadException
        try:
            default_repr = repr(obj)
            cleaned_repr = re.sub(r" at 0x[0-9a-f]+", "", default_repr)  # remove pointers
            cleaned_repr = re.sub(r" from '.*?'", "", cleaned_repr)  # remove file paths
            return _truncate(cleaned_repr)
        except Exception:
            return _truncate(f"unprintable({type(obj).__name__})")


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


def get_fully_qualified_name(type_or_func: type | typing.Callable) -> str:
    """Get the fully qualified name of a type or function."""
    mod = getattr(type_or_func, "__module__", None) or ""
    qual = getattr(type_or_func, "__qualname__", getattr(type_or_func, "__name__", repr(type_or_func)))
    if mod == "builtins":
        return qual
    return f"{mod}.{qual}"
