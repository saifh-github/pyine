"""Utility functions for Jupyter notebooks in this project.

This module provides helper functions that are commonly useful across multiple notebooks,
particularly for loading configurations, instantiating datamodules, formatting output, setting up
visualizations, and fetching W&B data.
"""

import ast
import contextlib
import json
import os
import re
import tempfile
import typing

import hydra
import hydra_zen
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import wandb
import wandb.apis.public

import pyine.apps.trainers.hf_rl_trainer_configs
import pyine.apps.trainers.hf_sft_trainer_configs
import pyine.configs.base
import pyine.data.datamodule
import pyine.evals.common
import pyine.organisms.datamodules.samples


def load_datamodule_from_hf_trainer_config(
    experiment_name: str,
    overrides: list[str] | None = None,
    verbose: bool = True,
) -> pyine.data.datamodule.BaseDataModule[typing.Any]:
    """Loads and instantiates a datamodule from an hf_trainer experiment config.

    This function composes a Hydra config from an experiment YAML file used by the hf_trainer app
    (both SFT and RL variants), extracts the datamodule config, and instantiates the datamodule
    exactly as it would be in an actual training run.

    Note:
        This is specific to hf_trainer configs. Other apps may have different config
        structures and would require separate loading functions.

    Args:
        experiment_name: The experiment config name (e.g., "original/v0_rl").
        overrides: Optional list of Hydra overrides to apply (e.g., ["runtime.seed=42"]).
        verbose: Whether to print verbose information during datamodule setup.

    Returns:
        The instantiated and prepared datamodule, ready for data access.

    Example:
        >>> dm = load_datamodule_from_hf_trainer_config("original/v0_rl")
        >>> train_parser = dm.get_parser("train")
        >>> print(f"Training samples: {len(train_parser)}")
    """
    # register configs for both SFT and RL trainers (so either config type can be loaded)
    pyine.configs.base.register_searchpath_plugin()
    pyine.apps.trainers.hf_sft_trainer_configs.register_hydra_configs(
        eval_type=pyine.evals.common.EvalType.CODE_EXEC,
    )
    pyine.apps.trainers.hf_rl_trainer_configs.register_hydra_configs(
        eval_type=pyine.evals.common.EvalType.CODE_EXEC,
    )
    # compose the full config using hydra
    all_overrides = [f"+experiment={experiment_name}"]
    if overrides:
        all_overrides.extend(overrides)
    with hydra.initialize(config_path=None, version_base=pyine.configs.base.target_hydra_version):
        config_dict = hydra.compose(config_name="entrypoint", overrides=all_overrides)
    # instantiate just the datamodule config (not the full app config)
    dm_config = hydra_zen.instantiate(config_dict.config.datamodule_config)
    dm = dm_config.instantiate_datamodule(verbose=verbose)
    dm.prepare_data()
    dm.setup()
    return dm


def format_long_string(
    value: str,
    max_length: int,
) -> str:
    """Format a potentially long string with optional truncation.

    Args:
        value: The string to format.
        max_length: Maximum length before truncation. Use 0 for no limit.

    Returns:
        The original string if within limits, or truncated with a marker.

    Example:
        >>> format_long_string("hello world", 5)
            'hello\\n\\n... [truncated, showing 5/11 chars]'
    """
    if 0 < max_length < len(value):
        truncated = value[:max_length]
        return f"{truncated}\n\n... [truncated, showing {max_length}/{len(value)} chars]"
    return value


def format_dict_string(
    value: str,
    max_length: int,
) -> tuple[str, bool]:
    """Try to parse and pretty-print JSON or Python dict literal.

    Attempts to parse the input as JSON first, then as a Python literal. If successful, returns a
    formatted JSON representation.

    Args:
        value: The string to parse and format.
        max_length: Maximum length before truncation. Use 0 for no limit.

    Returns:
        A tuple of (formatted_string, was_parsed_as_dict).

    Example:
        >>> text, is_dict = format_dict_string('{"key": "value"}', 1000)
        >>> print(is_dict)
            True
    """
    parsed = None
    with contextlib.suppress(json.JSONDecodeError, TypeError):
        parsed = json.loads(value)
    if parsed is None:
        with contextlib.suppress(SyntaxError, ValueError):
            parsed = ast.literal_eval(value)
    if parsed is not None and isinstance(parsed, (dict, list)):
        formatted = json.dumps(parsed, indent=2)
        return format_long_string(formatted, max_length), True
    return format_long_string(value, max_length), False


def collect_sample_metadata_rows(
    builder: pyine.organisms.datamodules.samples.SampleBuilder,
    subset_name: str,
    max_samples: int | None = None,
    keyword_detector: typing.Callable[[str], bool | None] | None = None,
    extra_fields: (
        dict[str, typing.Callable[[pyine.organisms.datamodules.samples.SampleData], typing.Any]] | None
    ) = None,
) -> list[dict[str, typing.Any]]:
    """Collect tabular rows describing generated samples from a SampleBuilder.

    This function iterates through samples from a builder and extracts common metadata fields into
    dictionaries suitable for creating a pandas DataFrame.

    Args:
        builder: SampleBuilder to draw samples from.
        subset_name: Subset name to include in each row.
        max_samples: Optional cap on the number of samples to collect.
        keyword_detector: Optional callable that takes code and returns whether a keyword is
            present. If None, the "has_keyword" field will be None.
        extra_fields: Optional dict mapping field names to callables that extract additional fields
            from each sample.

    Returns:
        List of dictionaries containing metadata for each sampled entry.

    Example:
        >>> rows = collect_sample_metadata_rows(builder, "train", max_samples=100)
        >>> df = pd.DataFrame(rows)
    """
    sample_rows: list[dict[str, typing.Any]] = []
    sample_cap = len(builder) if max_samples is None else min(len(builder), max_samples)
    sample_idx_iter = tqdm.tqdm(range(sample_cap), desc=f"collecting {subset_name} samples", smoothing=0.1)
    for sample_idx in sample_idx_iter:
        sample = builder[sample_idx]
        tag_list = [tag for tag in sample.comma_separated_tags.split(",") if tag]
        row: dict[str, typing.Any] = {
            "subset": subset_name,
            "identifier": sample.identifier,
            "code_line_count": len(sample.code.splitlines()),
            "code_target_line_span": sample.last_line - sample.first_line,
            "description_word_count": len(sample.description.split()),
            "inputs_char_length": len(sample.inputs),
            "expected_output_char_length": len(sample.expected_output),
            "expected_output_falsy": not bool(sample.expected_output),
            "predict_type": sample.predict_type,
            "code_type": sample.code_type,
            "trace_step_count": sample.trace_step_count,
            "has_code_override": sample.has_code_override,
            "has_keyword": keyword_detector(sample.code) if keyword_detector else None,
            "tags": tag_list,
        }
        if extra_fields:
            for field_name, extractor in extra_fields.items():
                row[field_name] = extractor(sample)
        sample_rows.append(row)
    return sample_rows


def setup_notebook_plotting(
    style: str = "ggplot",
    use_seaborn: bool = False,
    seaborn_style: typing.Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "whitegrid",
    figure_size: tuple[float, float] = (10, 6),
    title_size: int = 14,
    label_size: int = 12,
) -> None:
    """Configure matplotlib and optionally seaborn for notebook visualizations.

    This function sets up common plotting defaults used across notebooks, providing a consistent
    look and feel for visualizations.

    Args:
        style: Matplotlib style to use (e.g., "ggplot", "seaborn").
        use_seaborn: Whether to import and configure seaborn.
        seaborn_style: Seaborn theme style (only used if use_seaborn=True).
        figure_size: Default figure size as (width, height).
        title_size: Font size for plot titles.
        label_size: Font size for axis labels.

    Example:
        >>> setup_notebook_plotting()  # basic ggplot setup
        >>> setup_notebook_plotting(use_seaborn=True)  # with seaborn
    """
    plt.style.use(style)
    plt.rcParams.update(
        {
            "figure.figsize": figure_size,
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
        }
    )
    if use_seaborn:
        import seaborn as sns

        sns.set_theme(style=seaborn_style)


def truncate_text(
    text: str | None,
    max_length: int = 500,
) -> tuple[str, bool]:
    """Truncate text and return whether it was truncated.

    Unlike `format_long_string`, this returns a tuple indicating truncation status, which is useful
    for conditional display (e.g., showing "[truncated]" labels).

    Args:
        text: The text to truncate, or None.
        max_length: Maximum length before truncation.

    Returns:
        Tuple of (truncated_text, was_truncated). If text is None, returns ("(None)", False).

    Example:
        >>> text, truncated = truncate_text("hello world", 5)
        >>> print(f"{text}{'[truncated]' if truncated else ''}")
            hello...[truncated]
    """
    if text is None:
        return "(None)", False
    text = str(text)
    if len(text) <= max_length:
        return text, False
    return text[:max_length] + "...", True


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
        raise ValueError(f"Could not parse wandb URL: {url}")
    return match.group(1), match.group(2), match.group(3)


@typing.no_type_check  # wandb has poor typing
def get_wandb_run(
    run_url: str = "",
    run_id: str = "",
    project: str = "",
    entity: str | None = None,
) -> wandb.apis.public.Run:
    """Fetch a W&B run by URL or ID.

    Provide either `run_url` (which will be parsed to extract entity/project/run_id) or provide
    `run_id` along with `project` (and optionally `entity`).

    Args:
        run_url: Full W&B run URL (e.g., "https://wandb.ai/entity/project/runs/abc123").
        run_id: Run ID (e.g., "abc123"). Required if run_url is not provided.
        project: W&B project name. Used when run_url is not provided.
        entity: W&B entity (team or user). Optional.

    Returns:
        The wandb Run object.

    Raises:
        ValueError: If neither run_url nor run_id is provided.

    Example:
        >>> run = get_wandb_run(run_url="https://wandb.ai/my-team/my-project/runs/abc123")
        >>> print(f"Run: {run.name}")
    """
    api = wandb.Api()
    if run_url:
        entity, project, run_id = parse_wandb_url(run_url)
    if not run_id:
        raise ValueError("Must provide either run_url or run_id")
    path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    return api.run(path)


def resolve_wandb_step_key(df: pd.DataFrame) -> str:
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
        >>> step_key = resolve_wandb_step_key(history_df)
        >>> print(f"Using step key: {step_key}")
    """
    for key in ["global_step", "train/global_step", "_step"]:
        if key in df.columns:
            return key
    raise ValueError("No step key found in history DataFrame")


def resolve_wandb_time_key(df: pd.DataFrame) -> str | None:
    """Find a wall-clock time key in a W&B history DataFrame.

    Args:
        df: DataFrame from W&B history.

    Returns:
        The name of the time column, or None if not found.

    Example:
        >>> history_df = run.history(samples=100)
        >>> time_key = resolve_wandb_time_key(history_df)
        >>> if time_key:
        ...     print(f"Time key available: {time_key}")
    """
    for key in ["_timestamp", "_runtime"]:
        if key in df.columns:
            return key
    return None


@typing.no_type_check  # wandb has poor typing
def fetch_wandb_history_df(
    run: wandb.apis.public.Run,
    keys: list[str] | None = None,
    cache_path: str | None = None,
    samples: int = 500,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch W&B run history as a DataFrame with optional caching.

    For specific keys, uses `scan_history` for full fidelity (no downsampling). For discovery mode
    (keys=None), uses `history(samples=N)` which may downsample.

    Args:
        run: The W&B Run object.
        keys: Specific keys to fetch. If None, fetches all with sampling.
        cache_path: Optional path to cache results as parquet (useful for large runs).
        samples: Number of samples when keys=None (discovery mode).
        verbose: Whether to print progress messages.

    Returns:
        DataFrame with the run history.

    Example:
        >>> run = get_wandb_run(run_id="abc123", project="my-project")
        >>> df = fetch_wandb_history_df(run, keys=["reward/total", "_step"])
        >>> print(f"Fetched {len(df)} rows")
    """
    if cache_path and os.path.exists(cache_path):
        if verbose:
            print(f"Loading history from cache: {cache_path}")
        return pd.read_parquet(cache_path)
    if keys:
        if verbose:
            print(f"Fetching history with scan_history for {len(keys)} keys...")
        history = list(run.scan_history(keys=keys))
        df = pd.DataFrame(history)
    else:
        if verbose:
            print(f"Fetching sampled history ({samples} samples)...")
        df = run.history(samples=samples)
    if cache_path:
        df.to_parquet(cache_path)
        if verbose:
            print(f"Cached history to: {cache_path}")
    return df


@typing.no_type_check  # wandb has poor typing
def fetch_wandb_table(
    run: wandb.apis.public.Run,
    table_key: str,
) -> pd.DataFrame | None:
    """Fetch a W&B table artifact as a DataFrame.

    Tables logged via `wandb.log({key: wandb.Table(...)})` are stored as JSON files in the run's
    media/table directory. This function finds and parses them.

    Args:
        run: The W&B Run object.
        table_key: The key used when logging the table (e.g., "reward/rewards_table").

    Returns:
        DataFrame with the table data, or None if not found.

    Example:
        >>> run = get_wandb_run(run_id="abc123", project="my-project")
        >>> rewards_df = fetch_wandb_table(run, "reward/rewards_table")
        >>> if rewards_df is not None:
        ...     print(f"Loaded {len(rewards_df)} rows")
    """
    try:
        for file in run.files():
            file_name = file.name
            if not file_name.endswith(".table.json"):
                continue
            # table files use underscores instead of slashes in the key
            if table_key.replace("/", "_") not in file_name and table_key not in file_name:
                continue
            with tempfile.TemporaryDirectory() as tmpdir:
                downloaded = file.download(root=tmpdir, replace=True)
                with open(downloaded.name, encoding="utf-8") as f:
                    table_data = json.load(f)
                columns = table_data.get("columns", [])
                data = table_data.get("data", [])
                if columns and data:
                    return pd.DataFrame(data=data, columns=columns)
    except Exception as e:
        print(f"Error fetching table '{table_key}': {e}")
    return None


@typing.no_type_check  # wandb has poor typing
def discover_wandb_metric_keys(
    run: wandb.apis.public.Run,
    samples: int = 10,
) -> dict[str, list[str]]:
    """Discover available metric keys in a W&B run, grouped by category.

    Samples a small portion of the run history to discover what metrics are logged.

    Args:
        run: The W&B Run object.
        samples: Number of history samples to fetch for discovery.

    Returns:
        Dict mapping category names to lists of metric keys.

    Example:
        >>> run = get_wandb_run(run_id="abc123", project="my-project")
        >>> keys = discover_wandb_metric_keys(run)
        >>> print(f"Found {len(keys['reward'])} reward keys")
    """
    sample_df = run.history(samples=samples)
    all_keys: list[str] = list(sample_df.columns)

    def _list_by_prefix(prefix: str) -> list[str]:
        return [k for k in all_keys if k.startswith(prefix)]

    def _list_by_patterns(patterns: list[str], exact: list[str] | None = None) -> list[str]:
        result: list[str] = []
        for k in all_keys:
            if any(p in k for p in patterns) or exact and k in exact:
                result.append(k)
        return result

    # handle unprefixed, train/-, and eval/-prefixed metrics
    reward_total_keys = [k for k in all_keys if k in ("reward/total", "train/reward/total", "eval/reward/total")]
    reward_term_keys = (
        _list_by_prefix("reward/terms/")
        + _list_by_prefix("train/reward/terms/")
        + _list_by_prefix("eval/reward/terms/")
    )
    reward_metric_keys = (
        _list_by_prefix("reward/metrics/")
        + _list_by_prefix("train/reward/metrics/")
        + _list_by_prefix("eval/reward/metrics/")
    )
    parsing_prefixes = (
        "parsing/",
        "train/parsing/",
        "eval/parsing/",
        "reward/metrics/parsing/",
    )
    parsing_keys = [k for k in all_keys if any(k.startswith(p) for p in parsing_prefixes)]
    excluded_prefixes = (
        "reward/",
        "train/reward/",
        "eval/reward/",
        "parsing/",
        "train/parsing/",
        "eval/parsing/",
        "completions/",
        "_",
    )
    return {
        "reward_total": sorted(reward_total_keys),
        "reward_terms": sorted(reward_term_keys),
        "reward_metrics": sorted(reward_metric_keys),
        "parsing": sorted(parsing_keys),
        "trl": sorted(
            _list_by_patterns(
                ["completions/", "rewards/", "clip_ratio"],
                exact=["kl", "entropy", "reward", "reward_std"],
            )
        ),
        "step_keys": sorted(k for k in all_keys if k in ["_step", "global_step", "train/global_step"]),
        "other": sorted(
            k
            for k in all_keys
            if not any(k.startswith(p) for p in excluded_prefixes)
            and k not in ["kl", "entropy", "reward", "reward_std"]
        ),
    }
