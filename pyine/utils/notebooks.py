"""Utility functions for Jupyter notebooks in this project.

This module provides helper functions that are commonly useful across multiple notebooks,
particularly for loading configurations, instantiating datamodules, formatting output, and
setting up visualizations.

For W&B-related utilities, see `pyine.utils.wandb_utils`.
"""

import ast
import contextlib
import json
import typing

import hydra
import hydra_zen
import matplotlib.pyplot as plt
import pydantic
import rich.console
import rich.syntax
import tqdm

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


def display_pydantic_config(
    config: pydantic.BaseModel,
    title: str | None = "Configuration",
    theme: str = "monokai",
) -> None:
    """Display a pydantic model config with syntax-highlighted JSON.

    Uses rich to render the config as indented, colorized JSON output that's easy to read
    in Jupyter notebooks or terminal environments.

    Args:
        config: The pydantic model to display.
        title: Optional title to show above the config. Set to None to skip.
        theme: Syntax highlighting theme (e.g., "monokai", "dracula", "github-dark").

    Example:
        >>> display_pydantic_config(datamodule.config, title="Datamodule Config")
    """
    config_json = config.model_dump_json(indent=2)
    console = rich.console.Console()
    if title:
        console.print(f"[bold]{title}[/bold]")
    syntax = rich.syntax.Syntax(config_json, "json", theme=theme, line_numbers=False)
    console.print(syntax)


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
        base_id = sample.identifier.split("::")[0]
        family_id = base_id.split("/a:")[0] if "/a:" in base_id else base_id
        row: dict[str, typing.Any] = {
            "subset": subset_name,
            "identifier": sample.identifier,
            "base_id": base_id,
            "family_id": family_id,
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
