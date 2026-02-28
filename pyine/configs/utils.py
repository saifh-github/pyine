import collections.abc
import pathlib
import typing

import hydra
import hydra_zen
import hydra_zen.typing
import rich.console
import rich.table

import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.utils.portability

CallableReturningAny = collections.abc.Callable[..., typing.Any]
HydraZenBuild = hydra_zen.typing.Builds[typing.Any]

_MAX_DESC_LEN = 80


def discover_yaml_experiment_configs() -> list[tuple[str, pathlib.Path]]:
    """Discover YAML experiment configs from Hydra search paths.

    Reuses ``SearchPathPlugin._paths_to_try()`` to find ``experiment/**/*.yaml`` and
    ``experiment/**/*.yml`` files. Only includes files that look like Hydra experiment configs
    (i.e. that contain ``@package _global_`` or ``defaults:`` in the first ~15 lines). Deduplicates
    by config name (first root wins, matching Hydra's search path priority).

    Returns:
        Sorted list of (config_name, file_path) tuples.
    """
    seen_names: dict[str, pathlib.Path] = {}
    for _label, root in pyine.configs.searchpath.SearchPathPlugin._paths_to_try():  # pyright: ignore[reportPrivateUsage]
        if not (root.exists() and root.is_dir()):
            continue
        experiment_dir = root / "experiment"
        if not experiment_dir.is_dir():
            continue
        for pattern in ("**/*.yaml", "**/*.yml"):
            for yaml_path in sorted(experiment_dir.glob(pattern)):
                if not yaml_path.is_file():
                    continue
                config_name = str(yaml_path.relative_to(experiment_dir)).rsplit(".", maxsplit=1)[0]
                if config_name in seen_names:
                    continue  # first root wins (matches Hydra search path priority)
                if not _is_hydra_experiment_yaml(yaml_path):
                    continue
                seen_names[config_name] = yaml_path
    return sorted(seen_names.items())


def _is_hydra_experiment_yaml(
    yaml_path: pathlib.Path,
) -> bool:
    """Check if a YAML file looks like a Hydra experiment config.

    Scans the first ~15 lines for ``@package _global_`` or ``defaults:`` markers that indicate a
    Hydra config file.
    """
    try:
        with open(yaml_path) as fh:
            for line_idx, line in enumerate(fh):
                if line_idx >= 15:
                    break
                stripped = line.strip()
                if "@package _global_" in stripped:
                    return True
                if stripped.startswith("defaults:"):
                    return True
    except OSError:
        return False
    return False


def _extract_yaml_description(
    yaml_path: pathlib.Path,
) -> str:
    """Extract a short description from a YAML experiment config file.

    Reads the first ~10 lines and returns the first comment line that is not the ``@package``
    directive, stripped of the ``#`` prefix. Falls back to the filename if no suitable comment is
    found.
    """
    try:
        with open(yaml_path) as fh:
            for line_idx, line in enumerate(fh):
                if line_idx >= 10:
                    break
                stripped = line.strip()
                if not stripped.startswith("#"):
                    continue
                comment_text = stripped.lstrip("#").strip()
                if not comment_text or "@package" in comment_text:
                    continue
                return comment_text
    except OSError:
        pass
    return yaml_path.name


def _find_entrypoint_config(
    config_descriptions: list[pyine.configs.schemas.ConfigDescription],
) -> pyine.configs.schemas.ConfigDescription:
    """Find the unique entrypoint config (name='entrypoint', group=None).

    Raises:
        ValueError: If zero or multiple entrypoint configs are found.
    """
    entrypoints = [desc for desc in config_descriptions if desc.name == "entrypoint" and desc.group is None]
    if len(entrypoints) == 0:
        raise ValueError("no entrypoint config found (expected name='entrypoint', group=None)")
    if len(entrypoints) > 1:
        raise ValueError(f"multiple entrypoint configs found: {len(entrypoints)}")
    return entrypoints[0]


def _truncate_description(
    desc: str | None,
    max_len: int = _MAX_DESC_LEN,
) -> str:
    """Truncate a description string to a maximum length."""
    if not desc:
        return ""
    first_line = desc.split("\n", maxsplit=1)[0].strip()
    if len(first_line) <= max_len:
        return first_line
    return first_line[: max_len - 3] + "..."


def _print_config_listing(
    config_descriptions: list[pyine.configs.schemas.ConfigDescription],
    app_name: str,
    console: rich.console.Console,
) -> None:
    """Print a compact config listing grouped by config group.

    Displays hydra-zen registered configs grouped by their ``group`` field, then appends YAML
    experiment configs from search paths in a separate section.
    """
    console.print()
    console.print(f"AVAILABLE CONFIGS FOR '{app_name}'", style="bold")
    console.print("=" * (len(f"AVAILABLE CONFIGS FOR '{app_name}'")))
    console.print()
    # group configs by their group field
    grouped: dict[str | None, list[pyine.configs.schemas.ConfigDescription]] = {}
    for desc in config_descriptions:
        grouped.setdefault(desc.group, []).append(desc)
    # display each group as a table
    group_order = sorted(grouped.keys(), key=lambda g: (g is not None, g or ""))
    for group_key in group_order:
        descs = grouped[group_key]
        label = "[entrypoint]" if group_key is None else f"[{group_key}]"
        table = rich.table.Table(
            show_header=False,
            box=None,
            padding=(0, 2),
            title=label,
            title_style="bold cyan",
            title_justify="left",
        )
        table.add_column("override", style="green", no_wrap=True)
        table.add_column("description", style="dim")
        for desc in descs:
            if group_key is None:
                override_str = f"{desc.name}"
            elif group_key == "experiment":
                override_str = f"+experiment={desc.name}"
            else:
                override_str = f"{group_key}={desc.name}"
            table.add_row(override_str, _truncate_description(desc.description))
        console.print(table)
        console.print()
    # discover and display YAML experiment configs
    yaml_configs = discover_yaml_experiment_configs()
    if yaml_configs:
        table = rich.table.Table(
            show_header=False,
            box=None,
            padding=(0, 2),
            title="[experiment]  (YAML, from search path)",
            title_style="bold cyan",
            title_justify="left",
        )
        table.add_column("override", style="green", no_wrap=True)
        table.add_column("description", style="dim")
        for config_name, yaml_path in yaml_configs:
            override_str = f"+experiment={config_name}"
            desc_str = _extract_yaml_description(yaml_path)
            table.add_row(override_str, _truncate_description(desc_str))
        console.print(table)
        console.print(
            "  Note: YAML experiment configs are shown globally; not all may be compatible with this app.",
            style="dim italic",
        )
        console.print()


def _print_usage(
    app_name: str,
    console: rich.console.Console,
) -> None:
    """Print usage information for the config printing app."""
    console.print("Usage:", style="bold")
    module_path = f"python -m pyine.apps.trainers.{app_name}_configs"
    console.print(f"  {module_path:<70s} List all configs", style="dim")
    console.print(f"  {module_path} +experiment=<name>{'':<16s} Show config in detail", style="dim")
    console.print(f"  {module_path} +experiment=<name> key=val  Show config with overrides", style="dim")
    console.print(f"  {module_path} --help{'':<28s} Show this help", style="dim")
    console.print()


def print_experiment_configs(
    config_descriptions: list[pyine.configs.schemas.ConfigDescription],
    app_name: str,
    cli_args: list[str] | None = None,
) -> None:
    """Print available configs or render a specific config with overrides.

    Supports three modes based on ``cli_args``:

    - **List mode** (default, no args or ``--help``/``-h``): Print a compact summary of all
      available configs, including both hydra-zen registered configs and YAML experiment configs
      from search paths;
    - **Show mode** (args containing ``=``): Compose via Hydra with the provided overrides and
      render the resolved config;
    - **Error** (args present but none contain ``=``): Print error + usage info.

    Args:
        config_descriptions: List of configuration descriptions to display.
        app_name: Name of the application.
        cli_args: CLI arguments (typically ``sys.argv[1:]``). If None or empty, defaults to list mode.
    """
    from pyine.configs.base import target_hydra_version

    console = rich.console.Console(width=160)
    args = cli_args or []
    # mode 3: help
    if "--help" in args or "-h" in args:
        _print_config_listing(config_descriptions, app_name, console)
        _print_usage(app_name, console)
        return
    # mode 1: list all (no args)
    if not args:
        _print_config_listing(config_descriptions, app_name, console)
        _print_usage(app_name, console)
        return
    # check if args look like hydra overrides (contain '=')
    has_overrides = any("=" in arg for arg in args)
    if not has_overrides:
        console.print(f"\n[bold red]Error:[/bold red] unrecognized arguments: {' '.join(args)}")
        console.print("Arguments must be Hydra overrides containing '=' (e.g. +experiment=name, key=val).")
        console.print()
        _print_usage(app_name, console)
        return
    # mode 2: show config with hydra overrides
    entrypoint = _find_entrypoint_config(config_descriptions)
    with hydra.initialize(config_path=None, version_base=target_hydra_version):
        config_dict = hydra.compose(config_name="entrypoint", overrides=args)
    pyine.utils.portability.render_config(entrypoint.config, config_dict, console=console)


def hydra_zen_builds(
    callable_obj: CallableReturningAny,
    **kwargs: typing.Any,
) -> HydraZenBuild:
    """Wrapper for `hydra_zen.builds` that casts output to `typing.Any` for type checks compat.

    This function wraps `hydra_zen.builds()` and casts the result to `typing.Any` to avoid
    type checker issues with the dynamic configuration objects returned by hydra-zen.

    Args:
        callable_obj: The callable object to build a configuration for.
        **kwargs: Additional keyword arguments to pass to `hydra_zen.builds()`.

    Returns:
        A hydra-zen configuration object cast to `typing.Any`.

    Raises:
        AssertionError: If callable_obj is not callable.
    """
    assert isinstance(callable_obj, collections.abc.Callable), (
        f"callable_obj must be callable, got: {type(callable_obj)}"
    )
    return typing.cast("HydraZenBuild", hydra_zen.builds(callable_obj, **kwargs))


def hydra_zen_make_config(
    **kwargs: typing.Any,
) -> HydraZenBuild:
    """Wrapper for `hydra_zen.make_config` that casts output to `typing.Any` for type checks compat.

    This function wraps `hydra_zen.make_config()` and casts the result to `typing.Any` to avoid
    type checker issues with the dynamic configuration objects returned by hydra-zen.

    Args:
        **kwargs: Keyword arguments to pass to `hydra_zen.make_config()`.

    Returns:
        A hydra-zen configuration object cast to `typing.Any`.
    """
    return typing.cast("HydraZenBuild", hydra_zen.make_config(**kwargs))


def make_config_description(
    *callable_obj: CallableReturningAny,
    name: str,
    config: dict[str, typing.Any],
    group: str | None = None,
    package: str | None = None,
    description: str | None = None,
) -> pyine.configs.schemas.ConfigDescription:
    """Create a ConfigDescription object with hydra-zen configuration.

    This helper function creates a ConfigDescription by either building a configuration
    around a callable object or creating a standalone configuration from parameters.

    Args:
        *callable_obj: Optional callable object(s) to build configuration around.
            Only one callable is allowed.
        name: Name identifier for the configuration.
        config: Dictionary of configuration parameters.
        group: Optional group name for organizing configurations.
        package: Optional package name for the configuration.
        description: Optional human-readable description of the configuration. Providing a
            description is STRONGLY encouraged to document your configuration decisions.

    Returns:
        A ConfigDescription object containing the hydra-zen configuration.

    Raises:
        ValueError: If more than one callable object is specified.
    """
    if len(callable_obj) > 1:
        raise ValueError("only one callable can be specified for a ConfigDescription")
    zen_config: HydraZenBuild
    if len(callable_obj) == 1:
        zen_config = hydra_zen_builds(*callable_obj, **config)  # trust hydra-zen's typing guarantees
    else:
        zen_config = hydra_zen_make_config(**config)  # trust hydra-zen's typing guarantees
    return pyine.configs.schemas.ConfigDescription(
        name=name,
        group=group,
        package=package,
        description=description,
        config=zen_config,
    )
