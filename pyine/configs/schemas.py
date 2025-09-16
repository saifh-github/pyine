import logging
import pathlib

import hydra.conf
import pydantic

import pyine.utils.reprod
import wandb

logger = logging.getLogger(__name__)


class RuntimeConfig(pydantic.BaseModel):
    """Global application runtime configuration settings."""

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,  # for e.g. wandb run object; it'll be excluded from dumps
        extra="allow",
        validate_assignment=True,
    )
    """Pydantic model configuration (allow extra fields and validate assignments)."""

    exp_name: str = pydantic.Field(hydra.conf.MISSING, frozen=True)
    """Name of the experiment; used for output artifact naming and logging."""
    run_name: str = pydantic.Field("${now:%Y%m%d_%H%M%S}", frozen=True)
    """Name of the run; used for output artifact naming and logging."""
    run_group: str | None = pydantic.Field(None, frozen=True)
    """Group name for the run; used for grouping/filtering in wandb e.g. for distributed/K-fold runs."""
    app_name: str = pydantic.Field(default="${hydra:job.name}", frozen=True)  # don't override!
    """Name of the application/script/launcher used for this runtime."""
    notes: str | None = pydantic.Field(None, frozen=True)
    """Additional notes about the experiment/run; should be e.g. a git-commit-like description."""
    tags: list[str] | None = pydantic.Field(None, frozen=True)
    """Tags associated with this experiment/run; used for grouping/filtering in wandb."""
    output_dir: str = pydantic.Field(default="${hydra:runtime.output_dir}", frozen=True)  # don't override!
    """Output directory for logging and artifacts."""
    seed: int = pydantic.Field(0, frozen=True)
    """Seed to use for random number generation."""
    seed_workers: bool = pydantic.Field(False, frozen=True)
    """Whether to seed the workers for parallel execution."""
    metadata: dict[str, str] = pyine.utils.reprod.get_reprod_metadata(include_installed_packages=False)
    """Reproducibility metadata."""
    dry_run: bool = pydantic.Field(False, frozen=True)
    """Whether to skip the actual execution of the application (dry run for hydra setup)."""
    wandb_run: wandb.Run | None = pydantic.Field(default=None, exclude=True)
    """W&B run object created for this runtime (may be None if wandb logging is disabled).

    This attribute is not instantiated when the runtime config is created: instead, it is
    instantiated when the `init_wandb` method is called (which is application-specific).
    """

    @pydantic.computed_field
    @property
    def console_log_path(self) -> str:
        """Path to the console log file, located inside the output directory."""
        log_extension = pyine.utils.reprod.get_log_extension_slug(self)
        output_log_path = pathlib.Path(self.output_dir) / f"console{log_extension}"
        return str(output_log_path)

    @pydantic.computed_field
    @property
    def wandb_run_id(self) -> str | None:
        """ID of the associated W&B run (if wandb logging is enabled)."""
        if self.wandb_run is None:
            return None
        return self.wandb_run.id

    def init_wandb(self, **extra_kwargs) -> str:
        """Initializes W&B logging for this run and returns the run ID.

        Note: for distributed setups, you might want to only initialize wandb once (on the 'rank 0'
        node) if you wish to aggregate results across all nodes yourself.
        """
        if self.dry_run:
            raise RuntimeError("wandb logging should not happen in dry run mode?")
        default_kwargs = dict(
            name=f"{self.exp_name}-{self.run_name}",
            notes=self.notes,
            tags=tuple(sorted(set(self.tags))) if self.tags else None,
            group=self.run_group,
            job_type=self.app_name,
            # TODO: could set run id based on e.g. slurm id here if needed
        )
        default_kwargs.update(extra_kwargs)
        self.wandb_run = wandb.init(**default_kwargs)
        assert self.wandb_run.id == self.wandb_run_id
        offline = "offline " if self.wandb_run.offline else ""
        run_info_str = f"initialized {offline} wandb run '{self.wandb_run.name}'\n\tid: {self.wandb_run.id}"
        if self.wandb_run.url is not None:
            run_info_str += f"\n\turl: {self.wandb_run.url}"
        if self.wandb_run.tags is not None:
            run_info_str += f"\n\ttags: {self.wandb_run.tags}"
        if self.wandb_run.notes is not None:
            run_info_str += f"\n\tnotes: {self.wandb_run.notes}"
        logger.info(run_info_str)
        return self.wandb_run.id

    @property
    def wandb_run_initialized(self) -> bool:
        """Whether W&B logging is initialized, i.e. a run object has been created."""
        return self.wandb_run is not None

    def add_wandb_tag(self, tag: str) -> None:
        """Adds a new tag to the wandb run; raises if wandb is not initialized.

        Note: this does NOT add the tag to the `tags` attribute of the runtime config.
        """
        if not self.wandb_run_initialized:
            raise RuntimeError("wandb run is not initialized, cannot add tag")
        tags_list = list(self.wandb_run.tags) if self.wandb_run.tags else []
        tags_list.append(tag)
        self.wandb_run.tags = tuple(sorted(set(tags_list)))
        logger.info(f"added tag '{tag}' to wandb run id: {self.wandb_run_id}")
