import collections.abc
import dataclasses
import datetime
import functools
import importlib
import inspect
import os
import pathlib
import re
import site
import types
import typing

import hydra_zen.typing
import numpy as np
import omegaconf
import pandas as pd
import pydantic
import rich
import rich.box
import rich.console
import rich.panel
import rich.segment
import rich.style
import rich.table
import rich.text
import yaml


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
        cleaned = re.sub(r" at 0x[0-9a-f]+", "", s)  # remove pointers
        return re.sub(r" from '.*?'", "", cleaned)  # remove file paths

    base_types = (int, float, bool, str, bytes, list, tuple, set, dict, BaseException)
    if isinstance(obj, base_types) or obj is None:
        return _truncate(repr(typing.cast("typing.Any", obj)))  # all these base types need no special handling
    if isinstance(obj, np.ndarray):
        array_repr = repr(typing.cast("typing.Any", obj))
        return _truncate(f"numpy.{array_repr}")  # prefix numpy package name before 'array'
    if isinstance(obj, pd.DataFrame):
        df_json = str(typing.cast("typing.Any", obj.to_json()))  # type: ignore[reportUnknownMemberType]
        return _truncate(f"pandas.DataFrame({df_json})")  # get a serializable output
    if isinstance(obj, pd.Series):
        series_json = str(typing.cast("typing.Any", obj.to_json()))  # type: ignore[reportUnknownMemberType]
        return _truncate(f"pandas.Series({series_json})")  # get a serializable output
    if inspect.ismodule(obj):
        return _truncate(f"<module '{obj.__name__}'>")
    if callable(obj):
        return _truncate(f"<callable '{get_portable_function_name(obj)}'>")
    if hasattr(obj, "__class__") and not isinstance(obj, type):
        return _truncate(f"<instance '{get_fully_qualified_name(obj)}'>")
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
    changes: list[str] = []
    if isinstance(current_obj, np.ndarray) and isinstance(past_obj, np.ndarray):
        past_array: np.ndarray = past_obj  # type: ignore[reportUnknownMemberType]
        current_array: np.ndarray = current_obj  # type: ignore[reportUnknownMemberType]
        if past_array.shape != current_array.shape:
            return None  # different shapes, too complex to track
        diff_coords = np.argwhere(past_array != current_array)
        change_count = int(diff_coords.shape[0])
        if change_count > max_count:
            return None  # too many changes
        metadata = f"shape={current_array.shape},dtype={current_array.dtype}"
        for coord_array in diff_coords:
            coord_tuple = tuple(int(component) for component in coord_array.tolist())
            index_str = ",".join(str(component) for component in coord_tuple)
            value = current_array[coord_tuple]
            changes.append(f"numpy.ndarray:{metadata}:{index_str}:{value}")

    elif isinstance(current_obj, pd.DataFrame) and isinstance(past_obj, pd.DataFrame):
        past_df: pd.DataFrame = past_obj
        current_df: pd.DataFrame = current_obj
        if past_df.shape != current_df.shape:
            return None  # different shapes, too complex to track
        diff_df = past_df.ne(current_df)  # type: ignore[reportUnknownMemberType]
        change_count = int(diff_df.to_numpy().sum())
        if change_count > max_count:
            return None  # too many changes
        metadata = f"shape={current_df.shape}"
        for row_index, row_diff in diff_df.iterrows():
            for column, changed in row_diff.items():
                if not bool(changed):
                    continue
                value: typing.Any = current_df.loc[row_index, column]  # type: ignore[reportUnknownVariableType, reportCallIssue, reportArgumentType]
                changes.append(f"dataframe:{metadata}:({row_index},{column}):{str(value)}")  # type: ignore[reportUnknownArgumentType]
    elif isinstance(current_obj, pd.Series) and isinstance(past_obj, pd.Series):
        past_series: pd.Series[typing.Any] = past_obj  # type: ignore[reportUnknownVariableType]
        current_series: pd.Series[typing.Any] = current_obj  # type: ignore[reportUnknownVariableType]
        if len(past_series) != len(current_series):
            return None  # different lengths, too complex to track
        diff_series = typing.cast("pd.Series", past_series.ne(current_series))
        change_count = int(diff_series.sum())
        if change_count > max_count:
            return None  # too many changes
        metadata = f"len={len(current_series)}"
        for index, changed in diff_series.items():
            if not bool(changed):
                continue
            value: typing.Any = current_series.loc[index]  # type: ignore[reportUnknownVariableType, reportCallIssue, reportArgumentType]
            changes.append(f"series:{metadata}:{index}:{str(value)}")  # type: ignore[reportUnknownArgumentType]
    elif isinstance(current_obj, dict) and isinstance(past_obj, dict):
        past_dict = typing.cast("dict[typing.Any, typing.Any]", past_obj)
        current_dict = typing.cast("dict[typing.Any, typing.Any]", current_obj)
        past_keys: set[typing.Any] = set(past_dict.keys())
        current_keys: set[typing.Any] = set(current_dict.keys())
        added = current_keys - past_keys
        removed = past_keys - current_keys
        common = past_keys & current_keys
        changed = {key for key in common if past_dict[key] != current_dict[key]}
        total_changes = len(added) + len(removed) + len(changed)
        if total_changes > max_count:
            return None  # too many changes
        metadata = f"len={len(current_dict)}"
        for key in added:
            changes.append(f"dict:{metadata}:added({key}):{current_dict[key]}")
        for key in removed:
            changes.append(f"dict:{metadata}:removed({key}):None")
        for key in changed:
            changes.append(f"dict:{metadata}:changed({key}):{current_dict[key]}")
    elif isinstance(current_obj, list) and isinstance(past_obj, list):
        past_list = typing.cast("list[typing.Any]", past_obj)
        current_list = typing.cast("list[typing.Any]", current_obj)
        if len(past_list) != len(current_list):
            return None  # different lengths, too complex to track
        changed_indices = [
            index
            for index, (past_value, current_value) in enumerate(zip(past_list, current_list, strict=False))
            if past_value != current_value
        ]
        if len(changed_indices) > max_count:
            return None  # too many changes
        metadata = f"len={len(current_list)}"
        for index in changed_indices:
            changes.append(f"list:{metadata}:{index}:{current_list[index]}")
    elif isinstance(current_obj, tuple) and isinstance(past_obj, tuple):
        past_tuple = typing.cast("tuple[typing.Any, ...]", past_obj)
        current_tuple = typing.cast("tuple[typing.Any, ...]", current_obj)
        if len(past_tuple) != len(current_tuple):
            return None  # different lengths, too complex to track
        changed_indices = [
            index
            for index, (past_value, current_value) in enumerate(zip(past_tuple, current_tuple, strict=False))
            if past_value != current_value
        ]
        if len(changed_indices) > max_count:
            return None  # too many changes
        metadata = f"len={len(current_tuple)}"
        for index in changed_indices:
            changes.append(f"tuple:{metadata}:{index}:{current_tuple[index]}")
    elif hasattr(current_obj, "__class__") and hasattr(past_obj, "__class__"):  # type: ignore[reportUnknownArgumentType]

        def _collect_attrs(target: typing.Any) -> dict[str, typing.Any]:
            attrs: dict[str, typing.Any] = {}
            for attr_name in dir(target):
                if attr_name.startswith("_"):
                    continue
                if hasattr(target, attr_name):
                    attrs[attr_name] = getattr(target, attr_name)
            return attrs

        past_attrs = _collect_attrs(past_obj)
        current_attrs = _collect_attrs(current_obj)
        changed_attrs = {
            attr_name
            for attr_name in set(past_attrs.keys()) | set(current_attrs.keys())
            if past_attrs.get(attr_name) != current_attrs.get(attr_name)
        }
        if len(changed_attrs) > max_count:
            return None  # too many changes
        class_name = current_obj.__class__.__name__  # type: ignore[reportUnknownMemberType]
        module = current_obj.__class__.__module__  # type: ignore[reportUnknownMemberType]
        for attr_name in changed_attrs:
            changes.append(f"instance:{module}.{class_name}:changed({attr_name}):{current_attrs.get(attr_name)}")
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


def get_portable_function_name(callabl: typing.Callable[..., typing.Any]) -> str:
    """Return a stable, address-free string for a callable."""

    def _qual(module: str | None, qualname: str) -> str:
        if module and module != "builtins":
            return f"{module}.{qualname}"
        return qualname

    if isinstance(callabl, functools.partial):  # recursively get wrapped function names
        return f"functools.partial({get_portable_function_name(callabl.func)})"
    if isinstance(callabl, types.MethodType):  # handle bound callables
        func = callabl.__func__
        module = getattr(func, "__module__", None)
        qualname = getattr(func, "__qualname__", getattr(func, "__name__", "<?>"))
        base = _qual(module, qualname)
    elif (
        # plain functions, including nested and class/staticmethod functions
        isinstance(
            callabl,
            (types.FunctionType, types.BuiltinFunctionType, types.BuiltinMethodType),
        )
        or inspect.isbuiltin(callabl)
    ) or isinstance(callabl, type):
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
    formatted_lines: list[str] = []
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
    atol = 0.0 if decimal_places == 0 else 0.5 * (10 ** (-decimal_places))
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
    if not dotted_path:
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
    obj: type | types.ModuleType | collections.abc.Callable[..., typing.Any],
) -> str:
    """Get the fully qualified name of a type, module, or callable."""
    if isinstance(obj, types.ModuleType):
        spec = getattr(obj, "__spec__", None)
        mod_name = spec.name if spec and getattr(spec, "name", None) else getattr(obj, "__name__", None)
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


def _path_representer(
    dumper: yaml.Dumper | yaml.SafeDumper,
    data: pathlib.Path,
) -> yaml.Node:
    """Helper function for yaml.SafeDumper and yaml.Dumper to represent pathlib.Path objects."""
    # keeps OS-agnostic output
    return dumper.represent_scalar("tag:yaml.org,2002:str", data.as_posix())  # type: ignore[reportUnknownMemberType]


yaml.SafeDumper.add_multi_representer(pathlib.Path, _path_representer)
yaml.Dumper.add_multi_representer(pathlib.Path, _path_representer)


def render_config(
    cfg_type: hydra_zen.typing.Builds[typing.Any] | type[typing.Any],
    cfg_values: typing.Any | None = None,
    *,
    show_field_descriptions: bool = False,
    console: rich.console.Console | None = None,
    title: str | None = None,
) -> None:
    """Pretty-prints a config for a dataclass or Pydantic model using rich.

    Args:
        cfg_type: The config generated dynamically by hydra-zen. This can be a dataclass.
        cfg_values: (Optional) An instance/dict/DictConfig with assigned values to show. If None,
            the "Value" column falls back to the default.
        show_field_descriptions: (Optional) Whether to show field descriptions. Defaults to False.
        console: (Optional) Rich Console. A new one is created if omitted.
        title: (Optional) Override panel title (defaults to class name).
    """
    cns = console or rich.console.Console(width=200)  # expect this to output in a big terminal window
    # extract basic stuff from the provided config object
    doc = (
        getattr(cfg_values, "zen_meta", {}).get("__cfg_description__", "")
        or getattr(cfg_values, "zen_meta", {}).get("__description__", "")
        or getattr(cfg_values, "zen_meta", {}).get("__doc__", "")
        or getattr(cfg_type, "__cfg_description__", "")
        or getattr(cfg_type, "__description__", "")
        or getattr(cfg_type, "__doc__", "")
        or "<no config documentation available>"
    ).strip()
    default_title = getattr(cfg_type, "__name__", "Config")
    if hasattr(cfg_type, "__cfg_name__") and hasattr(cfg_type, "__cfg_group__"):
        default_title += f" => '+{cfg_type.__cfg_group__}={cfg_type.__cfg_name__}'"
    # start building the table that will contain config fields/types/values/(descriptions)
    table = rich.table.Table(expand=True)
    table.box = rich.box.SIMPLE_HEAD
    table.pad_edge = False
    table.padding = (0, 1)
    if show_field_descriptions:
        table.add_column("FIELD", style="bold", ratio=10, min_width=10, no_wrap=True)
        table.add_column("TYPE", ratio=20, min_width=10, no_wrap=True)
        table.add_column("VALUE", ratio=35, min_width=40, no_wrap=True)
        table.add_column("DESCRIPTION", ratio=35, min_width=40, no_wrap=True)
    else:
        table.add_column("FIELD", style="bold", ratio=10, min_width=10, no_wrap=True)
        table.add_column("TYPE", ratio=20, min_width=10, no_wrap=True)
        table.add_column("VALUE", ratio=70, min_width=40, no_wrap=True)

    # --------------- helper functions ---------------

    def _table_add_row(*args: typing.Any) -> None:
        indicators = tuple(_RichFoldIndicator(arg) for arg in args)
        table.add_row(*indicators)

    def _format_type_and_val(
        tp: typing.Any,
        cfg_vals: typing.Any | None = None,
        cfg_name: str | None = None,
        default: typing.Any = "<MISSING>",
    ) -> tuple[str, typing.Any]:  # type name, value
        # first, get the value itself
        if cfg_vals is None:
            val: typing.Any = default
            target: str | None = None
        else:
            assert cfg_name is not None, "cfg_name must be provided if cfg_vals is not None"
            if isinstance(cfg_vals, (omegaconf.DictConfig, typing.Mapping)):
                val: typing.Any = cfg_vals.get(cfg_name, default)  # type: ignore[reportUnknownVariableType]
                target: str | None = cfg_vals.get("_zen_target", None) or cfg_vals.get("_target_", None)  # type: ignore[reportUnknownVariableType]
            else:
                val: typing.Any = getattr(cfg_vals, cfg_name, default)  # type: ignore[reportUnknownVariableType]
                target: str | None = cfg_vals.get("_zen_target", None) or getattr(cfg_vals, "_target_", None)  # type: ignore[reportUnknownVariableType]
        # for the type, try to be more specific than 'any' (if e.g. _target_ is specified)
        if tp is typing.Any:
            if target is not None:
                return target, val  # type: ignore[reportUnknownVariableType]
            if val is not ... and val != "<MISSING>":
                return type(val).__name__, val  # type: ignore[reportUnknownVariableType]
        origin = typing.get_origin(tp)
        if origin is None:
            return getattr(tp, "__name__", str(tp)), val  # type: ignore[reportUnknownVariableType]
        args = ", ".join(_format_type_and_val(a)[0] for a in typing.get_args(tp))
        return f"{getattr(origin, '__name__', str(origin))}[{args}]", val  # type: ignore[reportUnknownVariableType]

    def _format_val(val: typing.Any) -> str:
        if isinstance(val, omegaconf.DictConfig):
            try:
                return omegaconf.OmegaConf.to_yaml(val, resolve=True).strip()
            except omegaconf.errors.InterpolationResolutionError:
                return omegaconf.OmegaConf.to_yaml(val, resolve=False).strip()
        if isinstance(val, (dict, list, tuple, set)):
            return yaml.safe_dump(val).strip()
        return str(val)

    skipped_field_names = ["_zen_exclude", "__cfg_description__", "__description__", "zen_meta"]

    # if the provided config is a dataclass...
    if dataclasses.is_dataclass(cfg_type):
        for f in dataclasses.fields(cfg_type):  # noqa
            if f.name in skipped_field_names:
                continue
            default_factory = getattr(f, "default_factory", dataclasses.MISSING)
            if default_factory is not dataclasses.MISSING:
                factory_callable = typing.cast("typing.Callable[[], typing.Any]", default_factory)
                default = factory_callable()
            else:
                default = f.default if f.default is not dataclasses.MISSING else "<MISSING>"
            ftype, val = _format_type_and_val(f.type, cfg_values, f.name, default)
            if show_field_descriptions:
                desc = ""
                if f.metadata:
                    desc = f.metadata.get("help", "") or f.metadata.get("description", "")
                _table_add_row(f.name, ftype, _format_val(val), desc)
            else:
                _table_add_row(f.name, ftype, _format_val(val))
    # else, if the provided config is a pydantic model...
    elif isinstance(cfg_type, type) and issubclass(cfg_type, pydantic.BaseModel):
        model_fields = typing.cast(
            "dict[str, typing.Any]",
            getattr(cfg_type, "model_fields", {}) or {},
        )
        for name, field in model_fields.items():
            if name in skipped_field_names:
                continue
            ann = getattr(field, "annotation", None)
            default = getattr(field, "default", ...)
            ftype, val = _format_type_and_val(ann, cfg_values, name, default if default is not ... else "<MISSING>")
            if show_field_descriptions:
                desc = getattr(field, "description", None)
                _table_add_row(name, ftype, _format_val(val), desc or "")
            else:
                _table_add_row(name, ftype, _format_val(val))
    # else, fallback for unknown config types
    else:
        if show_field_descriptions:
            _table_add_row("<unknown>", "—", "—", "Unsupported config type")
        else:
            _table_add_row("<unknown>", "—", "—")
    # print the panel that groups the config description and table
    body_parts = [
        rich.text.Text(f"\n{doc}\n\nConfig contents listed below:\n"),
        table,
    ]
    cns.print(
        rich.panel.Panel(
            rich.console.Group(*body_parts),
            title=title or default_title,
            expand=True,
            box=rich.box.ROUNDED,
        )
    )
    return


class _RichFoldIndicator:
    """Wraps a renderable to show markers on folded lines inside a rich.table.Table cell.

    The `prefix` string will be shown at the start of continuation lines (line > 0). The `suffix`
    string will be shown at the end of all but the last line.
    """

    def __init__(
        self,
        renderable: typing.Any,
        *,
        prefix: str = " ↳ ",
        suffix: str = "",
        prefix_style: str | rich.style.Style = "dim",
        suffix_style: str | rich.style.Style = "dim",
        reserve_suffix: bool = True,  # reserve width for suffix to avoid re-wrapping
    ) -> None:
        """Initializes the internal attributes."""
        self.renderable = renderable
        self.prefix = prefix
        self.suffix = suffix
        self.prefix_style = rich.style.Style.parse(prefix_style) if isinstance(prefix_style, str) else prefix_style
        self.suffix_style = rich.style.Style.parse(suffix_style) if isinstance(suffix_style, str) else suffix_style
        self.reserve_suffix = reserve_suffix

    def __rich_console__(
        self,
        console: rich.console.Console,
        options: rich.console.ConsoleOptions,
    ) -> rich.console.RenderResult:
        """Renders the rich.table.Table cell with proper folding markers."""
        # reserve space so markers don't force rewrap
        suffix_w = len(self.suffix) if (self.suffix and self.reserve_suffix) else 0
        prefix_w = len(self.prefix) if self.prefix else 0
        wrap_width = max(1, options.max_width - suffix_w - prefix_w)
        # try to split on hard newlines to distinguish from soft wraps
        hard_chunks: list[typing.Any]
        if isinstance(self.renderable, str):
            hard_chunks = self.renderable.split("\n")
        elif isinstance(self.renderable, rich.text.Text):
            # rich's Text split preserves styling per chunk
            try:
                hard_chunks = self.renderable.split("\n")  # type: ignore[arg-type]
            except AttributeError:
                # fallback: split via str() then rebuild Text chunks (styling lost across hard breaks)
                hard_chunks = [rich.text.Text(part) for part in str(self.renderable).split("\n")]
        else:
            # unknown renderable: we can't reliably find hard breaks; fallback to old behavior
            hard_chunks = [self.renderable]
        last_chunk_idx = len(hard_chunks) - 1
        for ci, chunk in enumerate(hard_chunks):
            lines = console.render_lines(
                chunk,
                options=console.options.update_width(wrap_width),
                new_lines=False,
                pad=False,
            )
            last_line_idx = len(lines) - 1
            for li, segs in enumerate(lines):
                # only add prefix for soft wraps (continuations within same hard line)
                if li > 0 and self.prefix:
                    yield rich.segment.Segment(self.prefix, self.prefix_style)
                yield from segs
                # add suffix on all but the last overall visual line
                is_last_overall = (ci == last_chunk_idx) and (li == last_line_idx)
                if (not is_last_overall) and self.suffix:
                    yield rich.segment.Segment(self.suffix, self.suffix_style)
                yield rich.segment.Segment.line()
