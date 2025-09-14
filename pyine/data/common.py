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
    """Return dataset directories whose names match the given pattern.

    Supports two matching modes:
      - Glob/fnmatch (default): shell-style patterns (e.g., 'taco*.lmdb', '*-08-*.lmdb').
      - Regex: if pattern_is_regex is True or 're:'/'regex:' prefix is used, standard Python regex
        is applied using regex.search on the directory name.

    The search is performed under '<data_root>/<kind>/<source_dataset_name>/' and only directories
    are considered. The output is deterministically sorted by name.

    Args:
        kind: Either 'traces' or 'deltas'.
        source_dataset_name: Name of the source dataset (e.g., 'TACO').
        pattern: Glob or regex pattern to match directory names against.
        pattern_is_regex: If True, treat 'pattern' as a regular expression. If False, treat it as
            a glob/fnmatch pattern. If 'pattern' starts with 're:' or 'regex:' the function will
            force regex mode; if it starts with 'glob:' or 'fnmatch:' it will force glob mode.

    Returns:
        A list of pathlib.Path objects pointing to matching dataset directories.
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
    all_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not all_dirs:
        raise FileNotFoundError(f"no {kind} dataset directories found in {root}")
    if pattern_is_regex:
        try:
            regex = re.compile(normalized_pattern)
        except re.error as e:
            raise ValueError(f"invalid regex pattern: {pattern!r} ({e})")
        matched = [p for p in all_dirs if regex.search(p.name)]
    else:
        matched = [p for p in all_dirs if fnmatch.fnmatchcase(p.name, normalized_pattern)]
    matched.sort()
    return matched
