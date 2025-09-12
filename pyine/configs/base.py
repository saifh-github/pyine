import typing

import hydra
import hydra.conf
import hydra_zen
import hydra_zen.typing

import pyine.configs.schemas
import pyine.configs.utils
import pyine.utils.filesystem
import pyine.utils.reprod

target_hydra_version = "1.3"
"""Target Hydra version for this project.

Version 1.3 adds support for `hydra_convert="object"`, absolute config path specification (using
`--config-path`), and maintains previous changes including e.g. `hydra.job.chdir=False` by default.
"""


def register_hydra_configs(store: hydra_zen.ZenStore | None = None) -> hydra_zen.ZenStore:
    """Registers default, project-wide, generic configs in the hydra store."""
    if store is None:
        store = hydra_zen.ZenStore()

    store(
        pyine.configs.schemas.RuntimeConfig,
        **pyine.configs.utils.get_defaults(pyine.configs.schemas.RuntimeConfig),
        hydra_convert="object",
        group="runtime",
        name="default",
    )

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
