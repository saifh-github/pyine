import collections.abc
import typing

import hydra
import hydra_zen
import hydra_zen.typing

import pyine.configs.schemas
import pyine.utils.portability
from pyine.configs.base import target_hydra_version

CallableReturningAny = collections.abc.Callable[..., typing.Any]
HydraZenBuild = hydra_zen.typing.Builds[typing.Any]


def print_experiment_configs(
    config_descriptions: list[pyine.configs.schemas.ConfigDescription],
    app_name: str,
) -> None:
    """Print available experiment configurations for a given application.

    This function filters configuration descriptions for experiment configs,
    renders each configuration using Hydra composition, and displays a summary
    of available experiment overrides.

    Args:
        config_descriptions: List of configuration descriptions to filter and display.
        app_name: Name of the application for which to display experiment configs.

    Note:
        Only configurations with group="experiment" are displayed. All experiment
        configurations are expected to be fully specified and instantiable.
    """
    splash_msg = f"AVAILABLE EXPERIMENT CONFIGURATIONS FOR THE '{app_name}' APPLICATION:"
    bar_str = "=" * len(splash_msg)
    print(f"\n\n{splash_msg}\n{bar_str}\n\n")
    exp_config_names: list[str] = []
    for config_desc in config_descriptions:
        if config_desc.group != "experiment":
            continue
        # all experiment configs should be fully specified (and thus instantiable as-is)
        with hydra.initialize(config_path=None, version_base=target_hydra_version):
            config_dict = hydra.compose(config_name="entrypoint", overrides=[f"+experiment={config_desc.name}"])
        pyine.utils.portability.render_config(config_desc.config, config_dict)
        assert config_desc.name is not None, "experiment config name must be defined before registration"
        exp_config_names.append(config_desc.name)
    if not exp_config_names:
        print(">>> No experiment configs found.")
    else:
        print("\n>>> Summary of available experiment overrides:")
        for exp_config_name in exp_config_names:
            print(f"\t+experiment={exp_config_name}")
    print("\nAll done.")


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
