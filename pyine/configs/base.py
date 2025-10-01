import typing

import hydra
import hydra.conf
import hydra.core.plugins
import hydra_zen
import hydra_zen.typing

import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.utils.filesystem
import pyine.utils.openai
import pyine.utils.portability
import pyine.utils.reprod

target_hydra_version = "1.3"
"""Target Hydra version for this project.

Version 1.3 adds support for `hydra_convert="object"`, absolute config path specification (using
`--config-path`), and maintains previous changes including e.g. `hydra.job.chdir=False` by default.
"""


def get_hydra_runtime_configs(
    group: str = "runtime",
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns runtime configs for hydra zen storage."""
    default_runtime_config = pyine.configs.schemas.ConfigDescription(
        name="default",
        group=group,
        config=hydra_zen.builds(
            pyine.configs.schemas.RuntimeConfig,
            # -------------
            populate_full_signature=True,
            hydra_convert="object",
            zen_meta={
                "__description__": (
                    "Provides default runtime settings for all apps; should only be missing a "
                    "`exp_name` definition by an override or by the user, and optional notes/tags."
                ),
            },
        ),
    )
    dry_run_runtime_config = pyine.configs.schemas.ConfigDescription(
        name="dry_run",
        group=group,
        config=hydra_zen.make_config(
            dry_run=True,
            # -------------
            bases=(default_runtime_config.config,),
            zen_meta={
                "__description__": (
                    "Provides default runtime settings for dry-run apps; overrides the default "
                    "settings, but still requires `exp_name` to be set by an override or by the "
                    "user, and optional notes/tags."
                ),
            },
        ),
    )
    return [default_runtime_config, dry_run_runtime_config]


def get_openai_client_configs(
    group: str,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Generates and returns OpenAI client configs for hydra zen storage."""
    default_openai_client_config = pyine.configs.schemas.ConfigDescription(
        name="default",
        group=group,
        config=hydra_zen.builds(
            pyine.utils.openai.OpenAIClientConfig,
            params=dict(timeout=None),  # override the default unserializable 'NOT_GIVEN' field
            # -------------
            populate_full_signature=True,
            hydra_convert="object",
            zen_meta={
                "__description__": "Default OpenAI client settings with all default (no timeout).",
            },
        ),
    )
    timeout300s_openai_client_config = pyine.configs.schemas.ConfigDescription(
        name="timeout300s",
        group=group,
        config=hydra_zen.builds(
            pyine.utils.openai.OpenAIClientConfig,
            params=dict(timeout=300),
            # -------------
            builds_bases=(default_openai_client_config.config,),
            zen_meta={
                "__description__": "Default OpenAI client settings with a 300-second timeout.",
            },
        ),
    )
    return [default_openai_client_config, timeout300s_openai_client_config]


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
            },
        )
    )
    runtime_configs = get_hydra_runtime_configs()
    for config in runtime_configs:
        store(config.config, name=config.name, group=config.group, package=config.package)

    # ...add more here if needed (job callbacks? loggers? profilers?)

    output_configs = runtime_configs  # add more here too
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


def print_experiment_configs(
    config_descriptions: list[pyine.configs.schemas.ConfigDescription],
    app_name: str,
) -> None:
    """Prints the available experiment configs for the app."""
    splash_msg = f"AVAILABLE EXPERIMENT CONFIGURATIONS FOR THE '{app_name}' APPLICATION:"
    bar_str = "=" * len(splash_msg)
    print(f"\n\n{splash_msg}\n{bar_str}\n\n")
    exp_config_names = []
    for config_desc in config_descriptions:
        if config_desc.group != "experiment":
            continue
        # all experiment configs should be fully specified (and thus instantiable as-is)
        with hydra.initialize(config_path=None, version_base=pyine.configs.base.target_hydra_version):
            config_dict = hydra.compose(config_name="entrypoint", overrides=[f"+experiment={config_desc.name}"])
        pyine.utils.portability.render_config(config_desc.config, config_dict)
        exp_config_names.append(config_desc.name)
    if not exp_config_names:
        print(">>> No experiment configs found.")
    else:
        print("\n>>> Summary of available experiment overrides:")
        for exp_config_name in exp_config_names:
            print(f"\t+experiment={exp_config_name}")
    print("\nAll done.")
