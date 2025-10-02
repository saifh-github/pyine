import logging
import pathlib
import typing

import hydra.conf
import hydra_zen.typing
import pydantic

import pyine.utils.reprod
import wandb

logger = logging.getLogger(__name__)


class RuntimeConfig(pydantic.BaseModel):
    """Global application runtime configuration settings."""

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,  # for e.g. wandb run object; it'll be excluded from dumps
        extra="allow",
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
    def output_dir_path(self) -> pathlib.Path:
        """Path to the run's output directory."""
        output_dir_path = pathlib.Path(self.output_dir).expanduser()
        assert output_dir_path.is_dir(), "output run directory should have been auto-created?"
        return output_dir_path

    @pydantic.computed_field
    @property
    def console_log_path(self) -> pathlib.Path:
        """Path to the console log file, located inside the output directory."""
        log_extension = pyine.utils.reprod.get_log_extension_slug(self)
        return self.output_dir_path / f"console{log_extension}"

    @pydantic.computed_field
    @property
    def wandb_run_id(self) -> str | None:
        """ID of the associated W&B run (if wandb logging is enabled)."""
        if self.wandb_run is None:
            return None
        return self.wandb_run.id

    def init_wandb(
        self,
        **extra_kwargs: typing.Any,
    ) -> str:
        """Initializes W&B logging for this run and returns the run ID.

        Note: for distributed setups, you might want to only initialize wandb once (on the 'rank 0'
        node) if you wish to aggregate results across all nodes yourself.
        """
        if self.dry_run:
            raise RuntimeError("wandb logging should not happen in dry run mode?")
        default_kwargs = {
            "name": f"{self.exp_name}-{self.run_name}",
            "notes": self.notes,
            "tags": tuple(sorted(set(self.tags))) if self.tags else None,
            "group": self.run_group,
            "job_type": self.app_name,
            # TODO: could set run id based on e.g. slurm id here if needed
        }
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


class ConfigDescription(pydantic.BaseModel):
    """Provides a description of a hydra-zen configuration object."""

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)
    """Pydantic model configuration (frozen, allow arbitrary types)."""

    name: str | None
    """Name of the configuration; used for registration in the hydra store.

    Note: although this field defaults to `None`, it is required for validation; when it is `None`,
    we will look for a "__cfg_name__" field in the config to use as the config name.
    """
    group: str | None
    """Optional group name for grouping similar configurations together.

    Note: if None, we will look for a "__cfg_group__" field in the config to use as the group.
    """
    package: str | None = None
    """Optional package name used when storing this config in the hydra store."""
    config: hydra_zen.typing.Builds | type[typing.Protocol]
    """Hydra-zen config object (can be used for instantiations and as a base in new definitions)."""
    description: str | None = None
    """Description of the config; should be informative to potential users.

    Note: although this field defaults to `None`, it is required for validation; when it is `None`,
    we will look for a "__description__" or "__doc__" field in the config to use as the description.
    """

    @pydantic.model_validator(mode="before")
    @classmethod
    def _fill_attribs(cls, data: typing.Any) -> typing.Any:
        """Fills the attributes that may be missing."""
        if not isinstance(data, dict):
            return data
        cfg = data.get("config")

        name = data.get("name")
        if name in (None, "") and cfg is not None:
            name = getattr(cfg, "__cfg_name__", None)
        if not name:
            raise ValueError("name must be provided or found in the config")
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name)}")
        data["name"] = name

        group = data.get("group")
        if group in (None, "") and cfg is not None:
            group = getattr(cfg, "__cfg_group__", None)
        if group is not None:
            if not isinstance(group, str):
                raise TypeError(f"group must be a string, got {type(group)}")
            data["group"] = group

        desc = data.get("description")
        if desc in (None, "") and cfg is not None:
            desc = getattr(cfg, "__description__", "") or getattr(cfg, "__doc__", "")
        if not desc:
            raise ValueError("description must be provided or found in the config")
        if not isinstance(desc, str):
            raise TypeError(f"description must be a string, got {type(desc)}")
        data["description"] = desc

        return data

    @pydantic.model_validator(mode="after")
    def _attach_config_attributes(self) -> "ConfigDescription":
        """Attaches description/name/group to the config."""
        # @@@@ might need to update zen exclude?
        self.config.__description__ = self.description
        self.config.__cfg_name__ = self.name
        self.config.__cfg_group__ = self.group
        return self
