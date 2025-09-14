import typing

import hydra
import hydra.conf
import hydra_zen
import hydra_zen.typing

import pyine.configs.schemas
import pyine.utils.filesystem
import pyine.utils.openai
import pyine.utils.reprod

target_hydra_version = "1.3"
"""Target Hydra version for this project.

Version 1.3 adds support for `hydra_convert="object"`, absolute config path specification (using
`--config-path`), and maintains previous changes including e.g. `hydra.job.chdir=False` by default.
"""


def store_hydra_runtime_configs(store: hydra_zen.ZenStore) -> None:
    """Stores runtime configs in the provided hydra zen store."""
    runtime_config = hydra_zen.builds(
        pyine.configs.schemas.RuntimeConfig,
        # -------------
        populate_full_signature=True,
        hydra_convert="object",
    )
    store(runtime_config, name="default")
    store(
        hydra_zen.make_config(
            dry_run=True,
            bases=(runtime_config,),
        ),
        name="dry_run",
    )


def get_openai_client_configs() -> dict[str, hydra_zen.typing.Builds]:
    """Returns OpenAI client configs to be stored in a zen store."""
    default_openai_client_config = hydra_zen.builds(
        pyine.utils.openai.OpenAIClientConfig,
        params=dict(timeout=None),  # override the default unserializable 'NOT_GIVEN' field
        # -------------
        populate_full_signature=True,
        hydra_convert="object",
    )
    timeout300s_openai_client_config = hydra_zen.builds(
        pyine.utils.openai.OpenAIClientConfig,
        params=dict(timeout=300),
        builds_bases=(default_openai_client_config,),
    )
    outputs = dict(
        default=default_openai_client_config,
        timeout300s=timeout300s_openai_client_config,
    )
    return outputs


def get_base_store() -> hydra_zen.ZenStore:
    """Stores project-wide configs in a new zen store and returns it.

    The default configs are the following:
    - 'runtime' group: ['default', 'dry_run']
    - 'openai_client_config' group: ['default', 'timeout300s']
    - 'hydra' group: ['config']
    """
    store = hydra_zen.ZenStore()

    runtime_store = store(group="runtime")
    store_hydra_runtime_configs(runtime_store)

    output_dir_root = str(pyine.utils.filesystem.get_logs_root_path())
    store(
        hydra.conf.HydraConf(
            run=hydra.conf.RunDir(
                dir=f"{output_dir_root}/runs/${{runtime.app_name}}/${{runtime.exp_name}}/${{runtime.run_name}}",
            ),
            sweep=hydra.conf.SweepDir(
                dir=f"{output_dir_root}/sweeps/${{runtime.app_name}}/${{runtime.exp_name}}/${{runtime.run_name}}",
                subdir="${hydra:job.num}_${hydra.job.override_dirname}",
            ),
            job_logging={
                "handlers": {
                    "file": {
                        "class": "logging.FileHandler",
                        "formatter": "simple",
                        "filename": "${hydra:runtime.output_dir}/output.log",
                    },
                },
            },
        ),
    )
    # ...add more here if needed (job callbacks? loggers? profilers?)
    return store


def get_base_hydra_default_overrides() -> list[dict[str, typing.Any]]:
    """Returns the base default overrides to use for most apps across the project."""
    return [
        # provide overrides for logging that rely on the hydra-color package
        {"override /hydra/hydra_logging": "colorlog"},
        {"override /hydra/job_logging": "colorlog"},
    ]
