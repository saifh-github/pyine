import datetime
import fnmatch
import pathlib
import re
import typing

import pyine.utils.filesystem

_DATE_SUFFIX_RE = re.compile(r"\.(?P<date>\d{4}-\d{2}-\d{2})\.lmdb$")


def _parse_date_from_dirname(name: str) -> datetime.date | None:
    """Extract YYYY-MM-DD date from a dataset directory name like 'something.2025-09-10.lmdb'."""
    m = _DATE_SUFFIX_RE.search(name)
    if not m:
        return None
    try:
        return datetime.date.fromisoformat(m.group("date"))
    except ValueError:
        return None


def resolve_latest_dataset_path(
    kind: typing.Literal["traces", "deltas"],
    source_dataset_name: str,
    filter_rule: str | None = None,
) -> pathlib.Path:
    """Resolve the latest dataset directory for a given kind/source, optionally applying a regex filter.

    The directory names are expected to follow '<tag>.<YYYY-MM-DD>.lmdb' and we select the latest
    strictly by the date suffix (YYYY-MM-DD), ignoring the '<tag>' prefix when ordering.

    Args:
        kind: Either 'traces' or 'deltas'.
        source_dataset_name: Name of the source dataset (e.g., 'TACO').
        filter_rule: Optional regular expression applied to the directory name; only names that match
                     are kept. If provided, it must be a valid Python regex.

    Returns:
        pathlib.Path to the latest dataset directory.

    Raises:
        FileNotFoundError: If the base directory is invalid, no datasets exist, or all are filtered out.
        ValueError: If no directory matches the expected naming convention with a date suffix.
    """
    if kind not in ("traces", "deltas"):
        raise ValueError(f"invalid dataset kind: {kind}")
    root = pyine.utils.filesystem.get_data_root_path() / kind / source_dataset_name
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"invalid {kind} dataset path: {root}")
    dataset_paths = sorted(root.glob("*.[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].lmdb/"))
    if not dataset_paths:
        raise FileNotFoundError(f"no {kind} datasets found in {root}")
    if filter_rule:
        try:
            regex = re.compile(filter_rule)
        except re.error as e:
            raise ValueError(f"invalid regex for filter_rule: {filter_rule!r} ({e})")
        dataset_paths = [p for p in dataset_paths if regex.search(p.name)]
        if not dataset_paths:
            raise FileNotFoundError(
                f"no {kind} datasets left after regex filtering in {root} (pattern={filter_rule!r})"
            )
    dated_paths: list[tuple[datetime.date, pathlib.Path]] = []
    for p in dataset_paths:
        d = _parse_date_from_dirname(p.name)
        if d is not None:
            dated_paths.append((d, p))
    if not dated_paths:
        raise ValueError(f"no {kind} dataset directories with a date suffix found under {root}.")
    dated_paths.sort(key=lambda t: (t[0], t[1].name))  # tie-break by name for determinism
    return dated_paths[-1][1]


def resolve_matching_dataset_paths(
    kind: typing.Literal["traces", "deltas"],
    source_dataset_name: str,
    pattern: str,
    pattern_is_regex: bool = False,
) -> list[pathlib.Path]:
    """Return dataset directories under the given source that match the provided pattern.

    Supports two matching modes:
      - Glob/fnmatch (default): shell-style patterns (e.g., 'taco*.lmdb', '*-08-*.lmdb').
        If the pattern contains a path separator ('/' or '\\'), it is matched against the
        relative path from the source root; otherwise it's matched against the basename.
      - Regex: if pattern_is_regex is True or 're:'/'regex:' prefix is used, standard Python
        regex is applied using regex.search on the relative path from the source root.

    The search is performed recursively under '<data_root>/<kind>/<source_dataset_name>/' and only
    directories are considered.

    Results are deterministically sorted by:
      1) date suffix parsed from the top-level dataset directory name (YYYY-MM-DD) if present,
         with undated entries first; then
      2) the relative path string from the source root (POSIX style).

    Args:
        kind: Either 'traces' or 'deltas'.
        source_dataset_name: Name of the source dataset (e.g., 'TACO').
        pattern: Glob or regex pattern to match directory names/paths against.
        pattern_is_regex: If True, treat 'pattern' as a regular expression. If False, treat it as
            a glob/fnmatch pattern. If 'pattern' starts with 're:' or 'regex:' the function will
            force regex mode; if it starts with 'glob:' or 'fnmatch:' it will force glob mode.

    Returns:
        A list of pathlib.Path objects pointing to matching directories (top-level or subdirectories).
    """
    if kind not in ("traces", "deltas"):
        raise ValueError(f"invalid dataset kind: {kind}")
    if not pattern:
        raise ValueError("pattern cannot be empty")
    root = pyine.utils.filesystem.get_data_root_path() / kind / source_dataset_name
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"invalid {kind} dataset path: {root}")

    # allow explicit prefixes to select mode regardless of pattern_is_regex value
    normalized_pattern = pattern
    if pattern.startswith(("re:", "regex:")):
        pattern_is_regex = True
        normalized_pattern = pattern.split(":", 1)[1]
    elif pattern.startswith(("glob:", "fnmatch:")):
        pattern_is_regex = False
        normalized_pattern = pattern.split(":", 1)[1]

    all_dirs = sorted(
        (p for p in root.rglob("*") if p.is_dir()),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    if not all_dirs:
        raise FileNotFoundError(f"no {kind} dataset directories found in {root}")

    def rel(p: pathlib.Path) -> str:
        return p.relative_to(root).as_posix()

    if pattern_is_regex:
        try:
            regex = re.compile(normalized_pattern)
        except re.error as e:
            raise ValueError(f"invalid regex pattern: {pattern!r} ({e})")
        matched = [p for p in all_dirs if regex.search(rel(p))]
    else:
        # if the pattern includes a path separator, match against the relative path; else match basename.
        has_sep = ("/" in normalized_pattern) or ("\\" in normalized_pattern)
        if has_sep:
            matched = [p for p in all_dirs if fnmatch.fnmatchcase(rel(p), normalized_pattern)]
        else:
            matched = [p for p in all_dirs if fnmatch.fnmatchcase(p.name, normalized_pattern)]
    matched.sort()
    return matched
