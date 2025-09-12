# import logging
import pathlib

import hydra.conf
import pydantic

import pyine.utils.reprod


class RuntimeConfig(pydantic.BaseModel):
    """Global hydra runtime configuration settings."""

    exp_name: str = hydra.conf.MISSING  # should be set by users (or automatically in experiments)
    """Name of the experiment; used for output artifact naming and logging."""
    run_name: str = "${now:%Y%m%d_%H%M%S}"  # provides a unique default for each execution by default
    """Name of the run; used for output artifact naming and logging."""
    app_name: str = pydantic.Field(default="${hydra:job.name}", frozen=True)  # don't override!
    """Name of the application/script/launcher used for this runtime."""
    output_dir: str = pydantic.Field(default="${hydra:runtime.output_dir}", frozen=True)  # don't override!
    """Output directory for logging and artifacts."""
    seed: int = 0
    """Seed to use for random number generation."""
    seed_workers: bool = False
    """Whether to seed the workers for parallel execution."""
    metadata: dict[str, str] = pyine.utils.reprod.get_reprod_metadata(include_installed_packages=False)
    """Reproducibility metadata."""

    @property
    def console_log_path(self) -> str:
        """Path to the console log file. If None, will use framework default."""
        log_extension = pyine.utils.reprod.get_log_extension_slug(self)
        output_log_path = pathlib.Path(self.output_dir) / f"console{log_extension}"
        return str(output_log_path)
