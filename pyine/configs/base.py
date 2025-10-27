import os
import typing

import hydra
import hydra.conf
import hydra.core.plugins
import hydra_zen

import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.configs.utils
import pyine.utils.filesystem
import pyine.utils.logging
import pyine.utils.openai

target_hydra_version = "1.3"
"""Target Hydra version for this project.

Version 1.3 adds support for `hydra_convert="object"`, absolute config path specification (using
`--config-path`), and maintains previous changes including e.g. `hydra.job.chdir=False` by default.
"""


def get_hydra_runtime_configs(
    group: str = "runtime",
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns runtime configs for hydra zen storage."""
    default_runtime_config = pyine.configs.utils.make_config_description(
        pyine.configs.schemas.RuntimeConfig,
        name="default",
        group=group,
        description=(
            "Provides default runtime settings for all apps; should only be missing a "
            "`exp_name` definition by an override or by the user, and optional notes/tags."
        ),
        config={
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    dry_run_runtime_config = pyine.configs.utils.make_config_description(
        name="dry_run",
        group=group,
        description=(
            "Provides default runtime settings for dry-run apps; overrides the default "
            "settings, but still requires `exp_name` to be set by an override or by the "
            "user, and optional notes/tags."
        ),
        config={
            "dry_run": True,
            # -------------
            "bases": (default_runtime_config.config,),
        },
    )
    return [default_runtime_config, dry_run_runtime_config]


def get_openai_client_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns OpenAI client configs for hydra zen storage."""
    default_openai_client_config = pyine.configs.utils.make_config_description(
        pyine.utils.openai.OpenAIClientConfig,
        name="default",
        group=group,
        description="Default OpenAI client settings with all default (no timeout).",
        config={
            "params": {"timeout": None},  # override the default unserializable 'NOT_GIVEN' field
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    timeout300s_openai_client_config = pyine.configs.utils.make_config_description(
        pyine.utils.openai.OpenAIClientConfig,
        name="timeout300s",
        group=group,
        description="Default OpenAI client settings with a 300-second timeout.",
        config={
            "params": {"timeout": 300},
            # -------------
            "builds_bases": (default_openai_client_config.config,),
        },
    )
    return [default_openai_client_config, timeout300s_openai_client_config]


def get_sweeper_configs(
    group: str = "hydra/sweeper",
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns sweeper configs for hydra zen storage."""
    import hydra_plugins.hydra_wandb_sweeper.config as sweeper_config

    base_wandb_sweep_config = pyine.configs.utils.make_config_description(
        sweeper_config.WandbConfig,
        name="wandb_sweep_base",
        group=f"{group}/wandb_sweep_config",
        description="Base wandb sweep config combining plugin defaults with runtime args.",
        config={
            "name": "${runtime.run_name}-sweep",
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    base_wandb_sweeper_config = pyine.configs.utils.make_config_description(
        sweeper_config.WandbSweeperConf,
        name="wandb_sweeper_base",
        group=group,
        description="Base wandb sweeper configuration combining plugin defaults with runtime args.",
        config={
            # -------------
            "populate_full_signature": True,
            "hydra_convert": "object",
            "hydra_defaults": [
                "_self_",
                {"wandb_sweep_config": "wandb_sweep_base"},
            ],
        },
    )
    return [base_wandb_sweep_config, base_wandb_sweeper_config]


def get_base_store_and_configs(
    app_name: str,
) -> tuple[hydra_zen.ZenStore, list[pyine.configs.schemas.ConfigDescription]]:
    """Stores project-wide configs in a new zen store and returns it, along with config descriptions."""
    store = hydra_zen.ZenStore()
    output_dir_root = str(pyine.utils.filesystem.get_logs_root_path())
    store(
        hydra.conf.HydraConf(
            run=hydra.conf.RunDir(
                dir=f"{output_dir_root}/runs/{app_name}/${{runtime.exp_name}}/${{runtime.run_name}}",
            ),
            sweep=hydra.conf.SweepDir(
                dir=f"{output_dir_root}/sweeps/{app_name}/${{runtime.exp_name}}/${{runtime.run_name}}",
                subdir="${hydra:job.num}_${hydra.job.override_dirname}",
            ),
            job=hydra.conf.JobConf(name=app_name),
            job_logging={
                "handlers": {
                    "file": {
                        "class": "logging.FileHandler",
                        "formatter": "simple",
                        "filename": "${hydra:runtime.output_dir}/output.log",
                    },
                },
                "loggers": {
                    pyine.utils.logging.PROJECT_LOGGER_NAME: {
                        "level": os.getenv("LOGLEVEL", "INFO"),
                    },
                },
                "root": {
                    "level": "INFO",
                },
            },
        )
    )
    output_configs: list[pyine.configs.schemas.ConfigDescription] = [
        *get_hydra_runtime_configs(),
        *get_sweeper_configs(),
        # ...add more here if needed (job callbacks? loggers? profilers?)
    ]
    for config in output_configs:
        if config.name is None:
            raise ValueError("config name must be defined before registration")
        store(config.config, name=config.name, group=config.group, package=config.package)
    return store, output_configs


def register_searchpath_plugin() -> None:
    """Registers the SearchPathPlugin for Hydra."""
    hydra.core.plugins.Plugins.instance().register(pyine.configs.searchpath.SearchPathPlugin)


def get_base_hydra_default_overrides() -> list[dict[str, typing.Any]]:
    """Returns the base default overrides to use for most apps across the project."""
    return [
        # provide overrides for logging that rely on the hydra-color package
        {"override /hydra/hydra_logging": "colorlog"},
        {"override /hydra/job_logging": "colorlog"},
    ]
