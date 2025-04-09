import inspect
import re
import typing

import numpy as np
import pandas as pd


def get_portable_representation(
    obj: typing.Any,
) -> str:
    """Convert any Python object to a portable, environment-independent representation.

    Creates a structured dictionary representation of Python objects that is suitable
    for tracing reproducibility across different environments/runs.

    Args:
        obj: Any Python object encountered during code tracing.

    Returns:
        Dict containing structured information about the object with stable identifiers.
    """
    if obj is None:
        return "None"
    elif isinstance(obj, (int, float, bool, str, bytes)):
        type_name = type(obj).__name__
        if isinstance(obj, bytes):
            try:
                # try to decode as utf-8 for readability
                value_str = obj.decode("utf-8")
            except UnicodeDecodeError:
                # fallback to hex representation
                value_str = obj.hex()
        else:
            value_str = str(obj)
        return f"{type_name}:{value_str}"
    elif isinstance(obj, np.ndarray):
        return f"numpy.ndarray:shape={obj.shape},dtype={obj.dtype}"
    elif isinstance(obj, pd.DataFrame):
        return f"pandas.DataFrame:shape={obj.shape}"
    elif isinstance(obj, pd.Series):
        return f"pandas.Series:len={len(obj)},dtype={obj.dtype},name={obj.name}"
    elif isinstance(obj, (list, tuple, set)):
        container_type = type(obj).__name__
        return f"{container_type}:len={len(obj)}"
    elif isinstance(obj, dict):
        return f"dict:len={len(obj)}"
    elif inspect.ismodule(obj):
        return f"module:{obj.__name__}"
    elif callable(obj):
        module = getattr(obj, "__module__", None)
        if hasattr(obj, "__qualname__"):
            name = obj.__qualname__
        elif hasattr(obj, "__name__"):
            name = obj.__name__
        else:
            name = f"anonymous:{obj}"
        if module is not None:
            return f"callable:{module}.{name}"
        else:
            return f"callable:{name}"
    elif hasattr(obj, "__class__") and not isinstance(obj, type):
        class_name = obj.__class__.__name__
        module = obj.__class__.__module__
        return f"instance:{module}.{class_name}"
    else:
        # fallback for any other objects
        # clean the default repr of memory addresses
        try:
            default_repr = repr(obj)
            cleaned_repr = re.sub(r" at 0x[0-9a-f]+", "", default_repr)  # remove pointers
            cleaned_repr = re.sub(r" from '.*?'", "", cleaned_repr)  # remove file paths
            return cleaned_repr
        except Exception:
            return f"<unprintable-{type(obj).__name__}>"


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


def print_code_with_numbered_lines(
    code_string: str,
) -> None:
    """
    Prints each line of the given code string, prefixed with a fixed-width line number.

    Args:
        code_string (str): The Python code string to be printed.
    """
    for line_idx, line_content in enumerate(code_string.splitlines(), start=1):
        formatted_line = f"L{line_idx:04d}:   {line_content}"
        print(formatted_line)
