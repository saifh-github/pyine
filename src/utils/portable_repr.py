import hashlib
import inspect
import re
import typing

import numpy as np
import pandas as pd


def get_portable_representation(
    obj: typing.Any,
) -> typing.Dict[str, typing.Any]:
    """Convert any Python object to a portable, environment-independent representation.

    Creates a structured dictionary representation of Python objects that is suitable
    for tracing reproducibility across different environments/runs.

    Args:
        obj: Any Python object encountered during code tracing.

    Returns:
        Dict containing structured information about the object with stable identifiers.
    """
    result = {
        "type": type(obj).__name__,
        "portable_repr": None,
        "metadata": {}
    }

    # none and basic types that are already portable
    if obj is None or isinstance(obj, (int, float, bool, str, bytes)):
        result["portable_repr"] = repr(obj)
        return result

    # handle modules
    if inspect.ismodule(obj):
        result["portable_repr"] = f"module:{obj.__name__}"
        result["metadata"]["name"] = obj.__name__

        # include version info if available
        if hasattr(obj, "__version__"):
            result["metadata"]["version"] = obj.__version__
        return result

    # handle functions and methods
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        module = obj.__module__
        name = obj.__qualname__ if hasattr(obj, "__qualname__") else obj.__name__
        result["portable_repr"] = f"function:{module}.{name}"
        result["metadata"]["module"] = module
        result["metadata"]["name"] = name

        # add signature information if possible
        try:
            result["metadata"]["signature"] = str(inspect.signature(obj))
        except (ValueError, TypeError):
            pass

        # add source code hash for reproducibility
        try:
            source = inspect.getsource(obj)
            source_hash = hashlib.md5(
                source.encode()).hexdigest()  # create a hash of the source code for fingerprinting
            result["metadata"]["source_hash"] = source_hash
        except (OSError, TypeError):
            pass
        return result

    # handle classes
    if inspect.isclass(obj):
        module = obj.__module__
        name = obj.__name__
        result["portable_repr"] = f"class:{module}.{name}"
        result["metadata"]["module"] = module
        result["metadata"]["name"] = name

        # add source code hash for class definition
        try:
            source = inspect.getsource(obj)
            source_hash = hashlib.md5(source.encode()).hexdigest()
            result["metadata"]["source_hash"] = source_hash
        except (OSError, TypeError):
            pass
        return result

    # handle common scientific libraries
    # numpy arrays
    if isinstance(obj, np.ndarray):
        result["portable_repr"] = f"ndarray:shape={obj.shape},dtype={obj.dtype}"
        result["metadata"]["shape"] = obj.shape
        result["metadata"]["dtype"] = str(obj.dtype)

        # include hash of small arrays, stats of larger ones
        if obj.size < 1000:
            array_hash = hashlib.md5(obj.tobytes()).hexdigest()
            result["metadata"]["data_hash"] = array_hash
        else:
            try:
                result["metadata"]["stats"] = {
                    "min": float(np.min(obj)),
                    "max": float(np.max(obj)),
                    "mean": float(np.mean(obj)),
                    "std": float(np.std(obj))
                }
            except (TypeError, ValueError):
                pass
        return result

    # pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        result["portable_repr"] = f"dataframe:shape={obj.shape}"
        result["metadata"]["shape"] = obj.shape
        result["metadata"]["columns"] = list(obj.columns)
        result["metadata"]["dtypes"] = {str(col): str(dtype) for col, dtype in obj.dtypes.items()}
        return result

    # pandas Series
    if isinstance(obj, pd.Series):
        result["portable_repr"] = f"series:length={len(obj)}"
        result["metadata"]["length"] = len(obj)
        result["metadata"]["dtype"] = str(obj.dtype)
        result["metadata"]["name"] = obj.name
        return result

    # handle Python containers by summarizing their content
    if isinstance(obj, (list, tuple, set)):
        container_type = type(obj).__name__
        result["portable_repr"] = f"{container_type}:length={len(obj)}"
        result["metadata"]["length"] = len(obj)

        # include type info about elements
        if len(obj) > 0:
            sample_types = {type(item).__name__ for item in list(obj)[:5]}
            result["metadata"]["element_types"] = list(sample_types)

        # hash small containers for reproducibility
        if len(obj) < 100:
            try:
                # create a consistent representation of container elements
                elements_repr = str([get_portable_representation(item)["portable_repr"]
                                     for item in obj])
                container_hash = hashlib.md5(elements_repr.encode()).hexdigest()
                result["metadata"]["content_hash"] = container_hash
            except Exception:
                pass
        return result

    # handle dictionaries
    if isinstance(obj, dict):
        result["portable_repr"] = f"dict:size={len(obj)}"
        result["metadata"]["size"] = len(obj)

        # include info about keys
        if len(obj) > 0:
            key_types = {type(key).__name__ for key in list(obj.keys())[:5]}
            value_types = {type(value).__name__ for value in list(obj.values())[:5]}
            result["metadata"]["key_types"] = list(key_types)
            result["metadata"]["value_types"] = list(value_types)

            # for small dicts, include portable key representations
            if len(obj) < 20:
                try:
                    portable_keys = [get_portable_representation(key)["portable_repr"]
                                     for key in list(obj.keys())[:10]]
                    result["metadata"]["sample_keys"] = portable_keys
                except Exception:
                    pass
        return result

    # handle general class instances
    if hasattr(obj, "__class__") and not isinstance(obj, type):
        class_name = obj.__class__.__name__
        module = obj.__class__.__module__
        result["portable_repr"] = f"instance:{module}.{class_name}"
        result["metadata"]["class_module"] = module
        result["metadata"]["class_name"] = class_name

        # try to extract important attributes without using __dict__ directly
        try:
            # get public attributes that aren't methods or properties
            attributes = {}
            for attr_name in dir(obj):
                if not attr_name.startswith('_') and not callable(getattr(obj, attr_name)):
                    try:
                        attr_value = getattr(obj, attr_name)
                        attributes[attr_name] = type(attr_value).__name__
                    except Exception:
                        pass
            if attributes:
                result["metadata"]["attributes"] = attributes
        except Exception:
            pass

        # try common special methods for additional information
        if hasattr(obj, "__len__"):
            try:
                result["metadata"]["length"] = len(obj)
            except Exception:
                pass
        return result

    # fallback for any other objects
    # clean the default repr of memory addresses
    try:
        default_repr = repr(obj)
        cleaned_repr = re.sub(r" at 0x[0-9a-f]+", "", default_repr) # remove pointers
        cleaned_repr = re.sub(r" from '.*?'", "", cleaned_repr)  # remove file paths
        result["portable_repr"] = cleaned_repr
    except Exception:
        result["portable_repr"] = f"<unprintable-{type(obj).__name__}>"

    return result
