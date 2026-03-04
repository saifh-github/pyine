"""Pickle-based persistence for evaluation results.

Provides save/load utilities for ``EvalResult`` objects, enabling offline analysis in notebooks and
scripts without requiring W&B. This is distinct from the LMDB-based ``DiskEvalLogger`` which stores
flattened per-record data for re-evaluation workflows.

Warning: pickle files should only be loaded from trusted sources. ``pickle.load`` can execute
arbitrary code on untrusted input. We chose pickle files here because we might need to store
arbitrary metadata to conduct result analyses, and we do not expect to share the pickles directly.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pickle
import re
import tempfile
import typing

import pyine.evals.common

if typing.TYPE_CHECKING:
    import pathlib

logger = logging.getLogger(__name__)

_UNSAFE_CHARS_RE = re.compile(r"[/\\:\s*?\"<>|]+")
"""Regex matching filesystem-unsafe characters to sanitize in dump filenames."""


def save_eval_result(
    result: pyine.evals.common.EvalResult,
    path: pathlib.Path,
    *,
    overwrite: bool = False,
) -> None:
    """Pickle an EvalResult (or subclass) to disk.

    Creates parent directories if they don't exist. Raises FileExistsError if path exists and
    overwrite is False. When overwrite is True and the path already exists, logs a warning with the
    path being overwritten, so accidental name collisions from sanitization are visible in logs.

    Uses atomic write (write to temp file in same directory, then os.replace) to avoid leaving
    partial/corrupt pickle files on interruption.

    Args:
        result: The EvalResult (or subclass) to persist.
        path: Destination file path.
        overwrite: If True, replace existing files (with a logged warning).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"result dump already exists at '{path}'; set overwrite=True to replace")
        logger.warning(f"overwriting existing result dump at '{path}'")
    fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            pickle.dump(result, tmp_file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path_str, path)
    except BaseException:
        # clean up temp file on any failure (including KeyboardInterrupt)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path_str)
        raise


def load_eval_result[T: pyine.evals.common.EvalResult](
    path: pathlib.Path,
    expected_type: type[T] | None = None,
) -> T:
    """Load a pickled EvalResult from disk.

    Warning: only load pickle files you produced yourself or from trusted sources; pickle.load can
    execute arbitrary code on untrusted input.

    When expected_type is given, raises TypeError if the loaded object is not an instance of that
    type. This lets notebooks do the following to get proper type narrowing:

        result = load_eval_result(path, expected_type=CorrectnessEvalResult)

    Args:
        path: Path to the pickle file.
        expected_type: Optional type to validate the loaded object against.

    Returns:
        The loaded EvalResult (or subclass).
    """
    with open(path, "rb") as fh:
        result = pickle.load(fh)  # noqa: S301
    if not isinstance(result, pyine.evals.common.EvalResult):
        raise TypeError(f"loaded object is {type(result).__qualname__}, expected an EvalResult subclass")
    if expected_type is not None and not isinstance(result, expected_type):
        raise TypeError(f"loaded object is {type(result).__qualname__}, expected {expected_type.__qualname__}")
    return result  # type: ignore[return-value]


def _sanitize_name(name: str) -> str:
    """Replace filesystem-unsafe characters with underscores."""
    return _UNSAFE_CHARS_RE.sub("_", name).strip("_")


def build_result_dump_path(
    dump_dir: pathlib.Path,
    eval_subset_name: str,
    type_name: str | None = None,
) -> pathlib.Path:
    """Build a deterministic dump filename: ``{dump_dir}/{eval_subset_name}[__{type_name}].pkl``.

    Includes type_name when given, avoiding collisions in multi-type correctness evals.
    Sanitizes both eval_subset_name and type_name by replacing unsafe filesystem characters
    (``/``, ``\\``, spaces, etc.) with underscores.

    Note: sanitization can collapse different raw names to the same output (e.g. ``"a/b"`` and
    ``"a b"`` both become ``"a_b"``). With ``overwrite=False`` (default), this raises
    ``FileExistsError`` at write time. With ``overwrite=True``, a collision would silently
    replace the wrong file -- use distinct, descriptive names when overwrite is enabled.

    Args:
        dump_dir: Directory where the file will be written.
        eval_subset_name: Evaluation subset identifier.
        type_name: Optional guardrail type name (for multi-type correctness evals).

    Returns:
        Full path with ``.pkl`` extension.
    """
    safe_subset = _sanitize_name(eval_subset_name)
    if not safe_subset:
        raise ValueError(f"eval_subset_name {eval_subset_name!r} sanitizes to an empty string")
    if type_name is not None:
        safe_type = _sanitize_name(type_name)
        if not safe_type:
            raise ValueError(f"type_name {type_name!r} sanitizes to an empty string")
        filename = f"{safe_subset}__{safe_type}.pkl"
    else:
        filename = f"{safe_subset}.pkl"
    return dump_dir / filename


def maybe_dump_eval_result(
    result: pyine.evals.common.EvalResult,
    dump_dir: pathlib.Path | None,
    eval_subset_name: str | None,
    *,
    type_name: str | None = None,
    overwrite: bool = False,
) -> pathlib.Path | None:
    """Dump an EvalResult to disk if dump_dir is set, returning the path used (or None).

    Centralizes the build-path + validate + save pattern used across all eval entry points. Raises
    ValueError if dump_dir is set but eval_subset_name is empty/None.

    Args:
        result: The EvalResult to persist.
        dump_dir: Target directory, or None to skip dumping.
        eval_subset_name: Subset name for filename construction.
        type_name: Optional guardrail type name (forwarded to build_result_dump_path).
        overwrite: Whether to overwrite existing files.

    Returns:
        The path written to, or None if dump_dir was None.
    """
    if dump_dir is None:
        return None
    if not eval_subset_name or not eval_subset_name.strip():
        raise ValueError("eval_subset_name must be set when result_dump_dir is enabled")
    path = build_result_dump_path(dump_dir, eval_subset_name, type_name=type_name)
    save_eval_result(result, path, overwrite=overwrite)
    return path
