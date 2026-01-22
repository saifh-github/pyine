"""Centralized utilities for interacting with Weights & Biases (W&B).

This module provides functions for:
- Fetching and querying W&B runs
- Downloading run history and metrics
- Fetching logged tables with their true step information
- Discovering available metric keys in a run

These utilities are designed to be lightweight and reusable across notebooks, analysis scripts,
and other tools that need to interact with W&B data.
"""

import json
import pathlib
import re
import tempfile
import time
import typing

import pandas as pd
import wandb
import wandb.apis.public

# exceptions that indicate transient network issues worth retrying
_retryable_exceptions: list[type[Exception]] = [ConnectionError, TimeoutError]
wandb_errors = getattr(wandb, "errors", None)
wandb_comm_error = getattr(wandb_errors, "CommError", None) if wandb_errors is not None else None
if isinstance(wandb_comm_error, type) and issubclass(wandb_comm_error, Exception):
    _retryable_exceptions.append(wandb_comm_error)
_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = tuple(_retryable_exceptions)


def _retry_on_failure(
    func: typing.Callable[[], typing.Any],
    max_retries: int = 3,
    base_delay: float = 10.0,
    max_delay: float = 60.0,
    verbose: bool = True,
) -> typing.Any:
    """Retry a function with exponential backoff on transient failures.

    Args:
        func: Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts (0 = no retries).
        base_delay: Initial delay in seconds between retries.
        max_delay: Maximum delay in seconds between retries.
        verbose: Whether to print retry messages.

    Returns:
        The result of the function call.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exception = exc
            if attempt < max_retries:
                delay = min(base_delay * (2**attempt), max_delay)
                if verbose:
                    print(f"  Attempt {attempt + 1} failed ({type(exc).__name__}), retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                if verbose:
                    print(f"  All {max_retries + 1} attempts failed.")
                raise
    assert last_exception is not None  # should not reach here with last_exception=None
    raise last_exception


# =============================================================================
# URL and Run Helpers
# =============================================================================


def parse_wandb_url(url: str) -> tuple[str, str, str]:
    """Parse a W&B URL to extract entity, project, and run_id.

    Handles URLs in the format: https://wandb.ai/entity/project/runs/run_id

    Args:
        url: The W&B run URL.

    Returns:
        Tuple of (entity, project, run_id).

    Raises:
        ValueError: If the URL format is not recognized.

    Example:
        >>> entity, project, run_id = parse_wandb_url("https://wandb.ai/my-team/my-project/runs/abc123")
        >>> print(f"{entity}/{project}/{run_id}")
        my-team/my-project/abc123
    """
    pattern = r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?]+)"
    match = re.search(pattern, url)
    if not match:
        raise ValueError(f"could not parse wandb URL: {url}")
    return match.group(1), match.group(2), match.group(3)


@typing.no_type_check  # wandb has poor typing
def get_wandb_run(
    run_url: str = "",
    run_id: str = "",
    project: str = "",
    entity: str | None = None,
    timeout: int = 60,
) -> wandb.apis.public.Run:
    """Fetch a W&B run by URL or ID.

    Provide either `run_url` (which will be parsed to extract entity/project/run_id) or provide
    `run_id` along with `project` (and optionally `entity`).

    Args:
        run_url: Full W&B run URL (e.g., "https://wandb.ai/entity/project/runs/abc123").
        run_id: Run ID (e.g., "abc123"). Required if run_url is not provided.
        project: W&B project name. Used when run_url is not provided.
        entity: W&B entity (team or user). Optional.
        timeout: Timeout in seconds for W&B API requests. Default is 60.

    Returns:
        The wandb Run object.

    Raises:
        ValueError: If neither run_url nor run_id is provided.

    Example:
        >>> run = get_wandb_run(run_url="https://wandb.ai/my-team/my-project/runs/abc123")
        >>> print(f"Run: {run.name}")
    """
    api = wandb.Api(timeout=timeout)
    if run_url:
        entity, project, run_id = parse_wandb_url(run_url)
    if not run_id:
        raise ValueError("must provide either run_url or run_id")
    path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    return api.run(path)


@typing.no_type_check  # wandb has poor typing
def fetch_runs(
    project: str,
    entity: str | None = None,
    filters: dict[str, typing.Any] | None = None,
    order: str = "-created_at",
    per_page: int = 50,
    timeout: int = 120,
) -> list[wandb.apis.public.Run]:
    """Fetch runs from a W&B project matching the given filters.

    Args:
        project: The W&B project name.
        entity: The W&B entity (team or user). If None, uses the default entity.
        filters: Optional filters dict (e.g., {"config.model_name": "gpt-4o"}).
        order: Sort order for runs. Use "-field" for descending, "+field" or "field" for
            ascending. Common fields: "created_at", "updated_at", "name".
            Default is "-created_at" (newest first).
        per_page: Number of runs to fetch per page.
        timeout: Timeout in seconds for W&B API requests. Default is 120.

    Returns:
        List of wandb Run objects, sorted according to `order`.

    Example:
        >>> runs = fetch_runs("my-project", filters={"state": "finished"})
        >>> print(f"Found {len(runs)} finished runs")
    """
    api = wandb.Api(timeout=timeout)
    path = f"{entity}/{project}" if entity else project
    return list(api.runs(path=path, filters=filters, order=order, per_page=per_page))


# =============================================================================
# History and Metrics Helpers
# =============================================================================


def resolve_step_key(df: pd.DataFrame) -> str:
    """Find the best step key in a W&B history DataFrame.

    Prefers explicit global_step variants over W&B's internal _step.

    Args:
        df: DataFrame from W&B history.

    Returns:
        The name of the step column to use.

    Raises:
        ValueError: If no step key is found.

    Example:
        >>> history_df = run.history(samples=100)
        >>> step_key = resolve_step_key(history_df)
        >>> print(f"Using step key: {step_key}")
    """
    for key in ["global_step", "train/global_step", "_step"]:
        if key in df.columns:
            return key
    raise ValueError("no step key found in history DataFrame")


def resolve_time_key(df: pd.DataFrame) -> str | None:
    """Find a wall-clock time key in a W&B history DataFrame.

    Args:
        df: DataFrame from W&B history.

    Returns:
        The name of the time column, or None if not found.

    Example:
        >>> history_df = run.history(samples=100)
        >>> time_key = resolve_time_key(history_df)
        >>> if time_key:
        ...     print(f"Time key available: {time_key}")
    """
    for key in ["_timestamp", "_runtime"]:
        if key in df.columns:
            return key
    return None


@typing.no_type_check  # wandb has poor typing
def fetch_history_df(
    run: wandb.apis.public.Run,
    keys: list[str] | None = None,
    cache_path: str | pathlib.Path | None = None,
    samples: int = 500,
    verbose: bool = True,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch W&B run history as a DataFrame with optional caching.

    For specific keys, uses `scan_history` for full fidelity (no downsampling), then filters to
    requested columns. For discovery mode (keys=None), uses `history(samples=N)` which may
    downsample. W&B internal keys (`_step`, `_timestamp`, `_runtime`) are always included.

    Note:
        When keys are specified, we fetch all history first then filter columns. This is because
        W&B's `scan_history(keys=[...])` only returns rows where ALL keys are present, which
        fails when metrics are logged at different steps/intervals.

    Args:
        run: The W&B Run object.
        keys: Specific keys to fetch. If None, fetches all with sampling.
        cache_path: Optional path to cache results as parquet (useful for large runs).
        samples: Number of samples when keys=None (discovery mode).
        verbose: Whether to print progress messages.
        max_retries: Maximum number of retry attempts on network failures (default: 3).

    Returns:
        DataFrame with the run history. Internal keys (`_step`, `_timestamp`, `_runtime`) are
        always included when available.

    Example:
        >>> run = get_wandb_run(run_id="abc123", project="my-project")
        >>> df = fetch_history_df(run, keys=["reward/total"])
        >>> print(f"Fetched {len(df)} rows")
    """
    if cache_path is not None:
        cache_path = pathlib.Path(cache_path)
        if cache_path.exists():
            if verbose:
                print(f"Loading history from cache: {cache_path}")
            return pd.read_parquet(cache_path)
    if keys:
        if verbose:
            print(f"Fetching full history with scan_history (will filter to {len(keys)} keys)...")
        history = _retry_on_failure(
            lambda: list(run.scan_history()),
            max_retries=max_retries,
            verbose=verbose,
        )
        df = pd.DataFrame(history)
        internal_keys = ["_step", "_timestamp", "_runtime"]
        keys_with_internal = internal_keys + [k for k in keys if k not in internal_keys]
        existing_keys = [k for k in keys_with_internal if k in df.columns]
        df = df[existing_keys]
    else:
        if verbose:
            print(f"Fetching sampled history ({samples} samples)...")
        df = _retry_on_failure(
            lambda: run.history(samples=samples),
            max_retries=max_retries,
            verbose=verbose,
        )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        if verbose:
            print(f"Cached history to: {cache_path}")
    return df


@typing.no_type_check  # wandb has poor typing
def discover_metric_keys(
    run: wandb.apis.public.Run,
    samples: int = 10,
    max_retries: int = 3,
) -> dict[str, list[str]]:
    """Discover available metric keys in a W&B run, grouped by category.

    Samples a small portion of the run history to discover what metrics are logged.

    Args:
        run: The W&B Run object.
        samples: Number of history samples to fetch for discovery.
        max_retries: Maximum number of retry attempts on network failures (default: 3).

    Returns:
        Dict mapping category names to lists of metric keys. Categories include:
        "reward_total", "reward_terms", "reward_metrics", "parsing", "trl",
        "step_keys", and "other".

    Example:
        >>> run = get_wandb_run(run_id="abc123", project="my-project")
        >>> keys = discover_metric_keys(run)
        >>> print(f"Found {len(keys['reward_terms'])} reward term keys")
    """
    sample_df = _retry_on_failure(
        lambda: run.history(samples=samples),
        max_retries=max_retries,
        verbose=False,
    )
    all_keys: list[str] = list(sample_df.columns)

    def _list_by_patterns(
        patterns: list[str],
        exact: list[str] | None = None,
    ) -> list[str]:
        result: list[str] = []
        for k in all_keys:
            if any(p in k for p in patterns) or (exact and k in exact):
                result.append(k)
        return result

    known_reward_total_key_parts = ["reward/total", "reward/run/total"]
    reward_total_keys = _list_by_patterns(patterns=known_reward_total_key_parts)
    known_reward_term_key_parts = ["reward/terms", "reward/run/terms"]
    reward_term_keys = _list_by_patterns(patterns=known_reward_term_key_parts)
    known_reward_metric_key_parts = ["reward/metrics/", "reward/run/metrics/"]
    reward_metric_keys = _list_by_patterns(patterns=known_reward_metric_key_parts)
    known_reward_categories_key_parts = ["reward/categories/", "reward/run/categories/"]
    reward_categories_keys = _list_by_patterns(patterns=known_reward_categories_key_parts)
    parsing_key_parts = ["/parsing/", "/failures/"]
    parsing_keys = _list_by_patterns(patterns=parsing_key_parts)
    trl_keys = _list_by_patterns(
        patterns=[
            "/completions/",
            "/clip_ratio/",
            "/rewards/",
            "/sampling/",
            "/kl",
            "/entropy",
            "/loss",
            "/grad_norm",
            "/learning_rate",
            "/num_tokens",
            "/reward_std",
            "/frac_reward_zero_std",
            "/step_time",
            "/runtime",
            "/samples_per_second",
            "/steps_per_second",
            "profiling/",
        ],
        exact=["train/reward", "eval/reward", "epoch"],
    )
    known_step_key_suffixes = ["/epoch", "/batch_count", "/generation_count", "/global_step"]
    known_exact_step_keys = ["_step", "_timestamp", "_runtime"]
    step_keys = [
        k for k in all_keys if k in known_exact_step_keys or any(k.endswith(s) for s in known_step_key_suffixes)
    ]
    # build set of all categorized keys to exclude from "other"
    categorized_keys = set(
        reward_total_keys + reward_term_keys + reward_metric_keys + parsing_keys + trl_keys + step_keys
    )
    excluded_prefixes = ("completions/", "_")
    other_keys = [
        k for k in all_keys if k not in categorized_keys and not any(k.startswith(p) for p in excluded_prefixes)
    ]
    return {
        "reward_total": sorted(reward_total_keys),
        "reward_terms": sorted(reward_term_keys),
        "reward_metrics": sorted(reward_metric_keys),
        "reward_categories_keys": sorted(reward_categories_keys),
        "parsing": sorted(parsing_keys),
        "trl": sorted(trl_keys),
        "step_keys": sorted(step_keys),
        "other": sorted(other_keys),
    }


# =============================================================================
# Table Helpers
# =============================================================================


@typing.no_type_check  # wandb has poor typing
def _download_file(
    file: typing.Any,
    *,
    root: str | pathlib.Path,
) -> pathlib.Path:
    """Download a W&B file and return the local path.

    Properly handles closing the file handle to avoid ResourceWarning.
    """
    downloaded = file.download(root=str(root), replace=True)
    if hasattr(downloaded, "close"):
        downloaded.close()
    if isinstance(downloaded, pathlib.Path):
        return downloaded
    if hasattr(downloaded, "name"):
        return pathlib.Path(str(downloaded.name))
    return pathlib.Path(str(downloaded))


@typing.no_type_check  # wandb has poor typing
def fetch_table(
    run: wandb.apis.public.Run,
    table_key: str,
    *,
    verbose: bool = False,
    max_retries: int = 3,
) -> pd.DataFrame | None:
    """Fetch a W&B table artifact as a DataFrame.

    Tables logged via `wandb.log({key: wandb.Table(...)})` are stored as JSON files in the run's
    media/table directory. This function finds and parses them.

    Args:
        run: The W&B Run object.
        table_key: The key used when logging the table (e.g., "train/generation_details").
        verbose: Whether to print errors.
        max_retries: Maximum number of retry attempts on network failures (default: 3).

    Returns:
        DataFrame with the table data, or None if not found.

    Example:
        >>> run = get_wandb_run(run_id="abc123", project="my-project")
        >>> details_df = fetch_table(run, "train/generation_details")
        >>> if details_df is not None:
        ...     print(f"Loaded {len(details_df)} rows")
    """
    try:
        files = _retry_on_failure(lambda: list(run.files()), max_retries=max_retries, verbose=verbose)
        for file in files:
            file_name = file.name
            if not file_name.endswith(".table.json"):
                continue
            # table files use underscores instead of slashes in the key
            if table_key.replace("/", "_") not in file_name and table_key not in file_name:
                continue
            with tempfile.TemporaryDirectory() as tmpdir:
                downloaded_path = _retry_on_failure(
                    lambda f=file, t=tmpdir: _download_file(f, root=t),
                    max_retries=max_retries,
                    verbose=verbose,
                )
                with open(downloaded_path, encoding="utf-8") as f:
                    table_data = json.load(f)
            columns = table_data.get("columns", [])
            data = table_data.get("data", [])
            if columns and data:
                return pd.DataFrame(data=data, columns=columns)
    except Exception as exc:
        if verbose:
            print(f"Error fetching table '{table_key}': {exc}")
    return None


@typing.no_type_check  # wandb has poor typing
def fetch_tables_with_steps(
    run: wandb.apis.public.Run,
    table_key: str,
    *,
    verbose: bool = True,
    max_retries: int = 3,
) -> pd.DataFrame | None:
    """Fetch W&B tables with their true logged steps from the run history.

    When tables are logged via `wandb.log({key: table}, step=X)`, the step is stored in the
    history, not in the table data itself. This function scans the history to find all log
    entries for the given table key and returns a combined DataFrame with a `_logged_step`
    column indicating the true step at which each table was logged.

    Args:
        run: The W&B Run object.
        table_key: The key used when logging the table (e.g., "train/generation_details").
        verbose: Whether to print progress information.
        max_retries: Maximum number of retry attempts on network failures (default: 3).

    Returns:
        DataFrame with combined table data and `_logged_step` column, or None if not found.

    Example:
        >>> run = get_wandb_run(run_id="abc123", project="my-project")
        >>> details_df = fetch_tables_with_steps(run, "train/generation_details")
        >>> if details_df is not None:
        ...     print(f"Loaded {len(details_df)} rows from {details_df['_logged_step'].nunique()} steps")
    """
    table_entries: list[tuple[int, str]] = []  # (step, artifact_path)
    if verbose:
        print(f"Scanning history for '{table_key}' table entries...")
    history_rows = _retry_on_failure(
        lambda: list(run.scan_history(keys=[table_key, "_step"])),
        max_retries=max_retries,
        verbose=verbose,
    )
    for row in history_rows:
        table_ref = row.get(table_key)
        step = row.get("_step")
        if table_ref is None or step is None:
            continue
        if isinstance(table_ref, dict):
            artifact_path = table_ref.get("path") or table_ref.get("artifact_path")
            if artifact_path:
                table_entries.append((step, artifact_path))
    if not table_entries:
        if verbose:
            print(f"No table entries found for '{table_key}' in history")
        return fetch_table(run, table_key, verbose=verbose, max_retries=max_retries)
    if verbose:
        print(f"Found {len(table_entries)} table log entries")
    path_to_step: dict[str, int] = {}
    for step, path in table_entries:
        filename = pathlib.Path(path).name
        path_to_step[filename] = step
    all_tables: list[pd.DataFrame] = []
    try:
        files = _retry_on_failure(lambda: list(run.files()), max_retries=max_retries, verbose=verbose)
        for file in files:
            file_name = file.name
            if not file_name.endswith(".table.json"):
                continue
            key_pattern = table_key.replace("/", "_")
            if key_pattern not in file_name and table_key not in file_name:
                continue
            filename = pathlib.Path(file_name).name
            logged_step = path_to_step.get(filename)
            with tempfile.TemporaryDirectory() as tmpdir:
                downloaded_path = _retry_on_failure(
                    lambda f=file, t=tmpdir: _download_file(f, root=t),
                    max_retries=max_retries,
                    verbose=verbose,
                )
                with open(downloaded_path, encoding="utf-8") as f:
                    table_data = json.load(f)
            columns = table_data.get("columns", [])
            data = table_data.get("data", [])
            if columns and data:
                df = pd.DataFrame(data=data, columns=columns)
                df["_logged_step"] = logged_step
                all_tables.append(df)
                if verbose:
                    print(f"  Loaded table from step {logged_step}: {len(df)} rows")
    except Exception as exc:
        if verbose:
            print(f"Error fetching tables for '{table_key}': {exc}")
        return None
    if not all_tables:
        if verbose:
            print(f"No table files found for '{table_key}'")
        return None
    combined = pd.concat(all_tables, ignore_index=True)
    if verbose:
        print(f"Combined {len(all_tables)} tables: {len(combined)} total rows")
    return combined
