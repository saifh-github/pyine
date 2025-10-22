import logging
import pathlib
import typing

import hydra_zen.typing
import omegaconf
import pydantic
import wandb

import pyine.utils.reprod

logger = logging.getLogger(__name__)


class RuntimeConfig(pydantic.BaseModel):
    """Global application runtime configuration settings."""

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,  # for e.g. wandb run object; it'll be excluded from dumps
        extra="allow",
    )
    """Pydantic model configuration (allow extra fields and validate assignments)."""

    exp_name: str = pydantic.Field(omegaconf.MISSING, frozen=True)
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
    metadata: dict[str, str] = pydantic.Field(
        default_factory=lambda: pyine.utils.reprod.get_reprod_metadata(include_installed_packages=False)
    )
    """Reproducibility metadata."""
    dry_run: bool = pydantic.Field(False, frozen=True)
    """Whether to skip the actual execution of the application (dry run for hydra setup)."""
    wandb_run: wandb.Run | None = pydantic.Field(default=None, exclude=True)
    """W&B run object created for this runtime (may be None if wandb logging is disabled).

    This attribute is not instantiated when the runtime config is created: instead, it is
    instantiated when the `init_wandb` method is called (which is application-specific).
    """

    @property
    def output_dir_path(self) -> pathlib.Path:
        """Path to the run's output directory."""
        output_dir_path = pathlib.Path(self.output_dir).expanduser().resolve()
        assert output_dir_path.is_dir(), "output run directory should have been auto-created?"
        return output_dir_path

    @pydantic.computed_field
    @property
    def wandb_run_project(self) -> str | None:
        """Project of the associated W&B run (if wandb logging is enabled)."""
        if self.wandb_run is None:
            return None
        return self.wandb_run.project

    @pydantic.computed_field
    @property
    def wandb_run_entity(self) -> str | None:
        """Entity of the associated W&B run (if wandb logging is enabled)."""
        if self.wandb_run is None:
            return None
        return self.wandb_run.entity

    @pydantic.computed_field
    @property
    def wandb_run_id(self) -> str | None:
        """ID of the associated W&B run (if wandb logging is enabled)."""
        if self.wandb_run is None:
            return None
        return self.wandb_run.id

    @pydantic.computed_field
    @property
    def wandb_run_url(self) -> str | None:
        """URL of the associated W&B run (if wandb logging is enabled and not in offline mode)."""
        if self.wandb_run is None:
            return None
        return self.wandb_run.url

    def init_wandb(
        self,
        **init_kwargs: typing.Any,
    ) -> str:
        """Initializes W&B logging for this run and returns the run ID.

        Note: for distributed setups, you might want to only initialize wandb once (on the 'rank 0'
        node) if you wish to aggregate results across all nodes yourself.
        """
        if self.dry_run:
            raise RuntimeError("wandb logging should not happen in dry run mode?")
        default_kwargs: dict[str, typing.Any] = {
            "name": f"{self.exp_name}-{self.run_name}",
            "notes": self.notes,
            "tags": sorted(set(self.tags)) if self.tags else None,
            "group": self.run_group,
            "job_type": self.app_name,
            # TODO: could set run id based on e.g. slurm id here if needed
        }
        default_kwargs.update(init_kwargs)
        self.wandb_run = wandb.init(**default_kwargs)
        assert self.wandb_run.id == self.wandb_run_id
        wandb_run_obj = typing.cast("typing.Any", self.wandb_run)
        offline = "offline " if getattr(wandb_run_obj, "offline", False) else ""
        run_info_str = f"initialized {offline} wandb run '{wandb_run_obj.name}'\n\tid: {wandb_run_obj.id}"
        run_url = getattr(wandb_run_obj, "url", None)
        if run_url is not None:
            run_info_str += f"\n\turl: {run_url}"
        run_tags = getattr(wandb_run_obj, "tags", None)
        if run_tags:
            run_info_str += f"\n\ttags: {run_tags}"
        run_notes = getattr(wandb_run_obj, "notes", None)
        if run_notes is not None:
            run_info_str += f"\n\tnotes: {run_notes}"
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
        wandb_run_obj = typing.cast("typing.Any", self.wandb_run)
        raw_tags = getattr(wandb_run_obj, "tags", None)
        tags_tuple = tuple(typing.cast("typing.Iterable[str]", raw_tags)) if raw_tags is not None else ()
        existing_tags = list(tags_tuple)
        existing_tags.append(tag)
        wandb_run_obj.tags = tuple(sorted(set(existing_tags)))
        logger.info(f"added tag '{tag}' to wandb run id: {self.wandb_run_id}")

    def finalize(self) -> None:
        """Finalizes the run by e.g. closing the wandb run if one exists."""
        if self.wandb_run is not None:
            self.wandb_run.finish()  # type: ignore[reportUnknownMemberType]


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
    """Optional package name used when storing this config in the hydra store.

    Note: if None, we will look for a "__cfg_package__" field in the config to use as the package.
    """
    description: str | None = None
    """Description of the config; should be informative to potential users.

    Providing descriptions is STRONGLY encouraged to document your configuration decisions; these
    will likely be printed alongside your config when registered in the hydra store.

    Note: although this field defaults to `None`, it is required for validation; when it is `None`,
    we will look for a `__cfg_description__`, `__description__`, or `__doc__` field in the config
    to use as the description.
    """
    config: hydra_zen.typing.Builds[typing.Any] | type[typing.Any]
    """Hydra-zen config object (can be used for instantiations and as a base in new definitions)."""

    @staticmethod
    def _check_and_validate_attrib(
        data: dict[str, typing.Any],
        cfg: dict[str, typing.Any],
        attrib_name: str,
        is_mandatory: bool,
    ) -> None:
        """Checks whether the attribute is present in the config and validates its type."""
        zen_meta: dict[str, typing.Any] = getattr(cfg, "zen_meta", {})
        attrib_val = data.get(attrib_name)
        if attrib_val in (None, ""):
            attrib_val = getattr(zen_meta, f"__cfg_{attrib_name}__", None)
        if attrib_val in (None, ""):
            attrib_val = getattr(cfg, f"__cfg_{attrib_name}__", None)
        if attrib_val is not None and not isinstance(attrib_val, str):
            raise TypeError(f"{attrib_name} must be a string, got {type(attrib_val)}")
        if not attrib_val and is_mandatory:
            raise ValueError(f"{attrib_name} must be provided directly or in the config")
        if attrib_val:
            data[attrib_name] = attrib_val

    @pydantic.model_validator(mode="before")
    @classmethod
    def _fill_attribs(cls, data: typing.Any) -> typing.Any:
        """Fills the attributes that may be missing."""
        if not isinstance(data, dict) or "config" not in data or not isinstance(data["config"], dict):
            return data  # type: ignore[reportUnknownMemberType] --- let pydantic deal with the mess
        data = typing.cast("dict[str, typing.Any]", data)
        cfg = typing.cast("dict[str, typing.Any]", data["config"])
        cls._check_and_validate_attrib(data=data, cfg=cfg, attrib_name="name", is_mandatory=True)
        cls._check_and_validate_attrib(data=data, cfg=cfg, attrib_name="group", is_mandatory=False)
        cls._check_and_validate_attrib(data=data, cfg=cfg, attrib_name="package", is_mandatory=False)
        cls._check_and_validate_attrib(data=data, cfg=cfg, attrib_name="description", is_mandatory=True)
        return data

    @pydantic.model_validator(mode="after")
    def _attach_config_attributes(self) -> "ConfigDescription":
        """Attaches description/name/group to the config."""
        self.config.__cfg_name__ = self.name
        self.config.__cfg_group__ = self.group
        self.config.__cfg_package__ = self.package
        self.config.__cfg_description__ = self.description
        if hasattr(self.config, "zen_meta"):
            # @@@@ TODO might need to update zen exclude?
            self.config.zen_meta["__cfg_name__"] = self.name
            self.config.zen_meta["__cfg_group__"] = self.group
            self.config.zen_meta["__cfg_package__"] = self.package
            self.config.zen_meta["__cfg_description__"] = self.description
        return self
