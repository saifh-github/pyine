import collections.abc
import logging
import os
import pathlib
import typing

import hydra.core.hydra_config
import hydra.errors
import hydra.types
import hydra_zen.typing
import omegaconf
import pydantic
import wandb

import pyine.utils.distrib
import pyine.utils.reprod

logger = logging.getLogger(__name__)


class RuntimeConfig(pydantic.BaseModel):
    """Global application runtime configuration settings, resolved by hydra apps on launch."""

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,  # for e.g. wandb run object; it'll be excluded from dumps
        extra="allow",
    )
    """Pydantic model configuration (allow extra fields and validate assignments)."""

    exp_name: str = pydantic.Field(omegaconf.MISSING, frozen=True)
    """Name of the experiment; used for output artifact naming and logging."""
    run_name: str = pydantic.Field("${now:%Y%m%d-%H%M%S}", frozen=True)
    """Name of the run; used for output artifact naming and logging."""
    run_group: str | None = pydantic.Field(default="${runtime.exp_name}", frozen=True)
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
        default_factory=lambda: pyine.utils.reprod.get_reprod_metadata(
            include_installed_packages=False,  # would be too verbose in dumped dicts
        )
    )
    """Reproducibility metadata."""
    dry_run: bool = pydantic.Field(False, frozen=True)
    """Whether to skip the actual execution of the application (dry run for hydra setup)."""
    wandb_run: wandb.Run | None = pydantic.Field(default=None, exclude=True)
    """W&B run object created for this runtime (may be None if wandb logging is disabled).

    This attribute is not instantiated when the runtime config is created: instead, it is
    instantiated when the `init_wandb` method is called (which is application-specific). It may
    also have been created prior to the launch of the application itself, e.g. by a sweep manager,
    in which case this is just a reference to the existing run object.
    """

    @property
    def output_dir_path(self) -> pathlib.Path:
        """Path to the run's output directory."""
        output_dir_path = pathlib.Path(self.output_dir).expanduser().resolve()
        assert output_dir_path.is_dir(), "output run directory should have been auto-created?"
        return output_dir_path

    @pydantic.computed_field
    @property
    def hydra_runtime_config(self) -> omegaconf.DictConfig | None:
        """Returns the active Hydra runtime config if available, otherwise None."""
        hydra_config_cls = hydra.core.hydra_config.HydraConfig
        if not hydra_config_cls.initialized():
            return None
        try:
            config = hydra_config_cls.get()
        except (hydra.errors.HydraException, ValueError):
            return None
        assert isinstance(config, omegaconf.DictConfig)
        return config.copy()

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

    @pydantic.computed_field
    @property
    def wandb_run_dir(self) -> pathlib.Path | None:
        """Directory of the associated W&B run (if wandb logging is enabled)."""
        if self.wandb_run is None:
            return None
        return pathlib.Path(self.wandb_run.dir)

    def _is_using_wandb_sweeper(self) -> bool:
        """Returns whether the run was launched by a wandb sweeper."""
        if self.hydra_runtime_config is None:
            return False
        if self.hydra_runtime_config.mode != hydra.types.RunMode.MULTIRUN:
            return False
        sweeper_target = self.hydra_runtime_config.sweeper._target_
        return "WandbSweeper" in sweeper_target

    def init_wandb(
        self,
        **init_kwargs: typing.Any,
    ) -> str:
        """Initializes W&B logging for this run (if needed) and returns the run ID.

        Notes:
            - For distributed setups, you might want to only initialize wandb once (on the 'rank 0'
              node) if you wish to aggregate results across all nodes yourself.
            - When conducting sweeps with the hydra wandb sweeper, we will NOT be instantiating a
              wandb run object here as the sweeper will do that for us. Instead, we will simply
              fetch the run object from wandb.
            - The 'use_shared_mode' kwarg controls whether wandb runs in shared mode (multi-rank
              coordination). When False (default), avoids wandb.log step argument limitations.
        """
        if self.dry_run:
            raise RuntimeError("wandb logging should not happen in dry run mode?")
        if self._is_using_wandb_sweeper():
            init_kwargs.pop("config", None)
            if init_kwargs:
                logger.warning("wandb init kwargs are ignored when using wandb sweeper")
            assert wandb.run is not None, "wandb run should have been created by sweeper"
            self.wandb_run = wandb.run
        else:
            curr_rank = pyine.utils.distrib.get_global_rank()
            is_main_process = pyine.utils.distrib.is_main_process(curr_rank)
            os.environ.pop("WANDB_SERVICE", None)  # as of Dec. 2025, fixes shared mode worker inits
            # extract and remove our custom kwarg before passing to wandb.init
            use_shared_mode = init_kwargs.pop("use_shared_mode", False)
            wandb_settings: dict[str, typing.Any] = {"x_label": f"rank_{curr_rank}"}
            if use_shared_mode:
                wandb_settings.update(
                    {
                        "mode": "shared",
                        "x_primary": is_main_process,
                        "x_update_finish_state": is_main_process,
                    }
                )
            default_kwargs: dict[str, typing.Any] = {
                "name": self.run_name,
                "notes": self.notes,
                "tags": sorted(set(self.tags)) if self.tags else None,
                "group": self.run_group,
                "job_type": self.app_name,
                "dir": self.output_dir_path,
                "mode": os.environ.get("WANDB_MODE", None),
                "settings": wandb.Settings(**wandb_settings),
                # TODO: could set run id based on e.g. slurm id here if needed
            }
            default_kwargs.update(init_kwargs)
            self.wandb_run = wandb.init(**default_kwargs)
        assert self.wandb_run.id == self.wandb_run_id
        wandb_run_obj = typing.cast("typing.Any", self.wandb_run)
        offline = "offline " if getattr(wandb_run_obj, "offline", False) else " "
        run_info_str = f"initialized {offline}wandb run '{wandb_run_obj.name}'\n\tid: {wandb_run_obj.id}"
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

    def is_wandb_run_finished(self) -> bool:
        """Returns whether the wandb run is finished."""
        if not self.wandb_run_initialized:
            raise RuntimeError("wandb run is not initialized, cannot check for status")
        # wandb does not expose a nice/clean way to check if a run is finalized; this one is brittle...
        return getattr(self.wandb_run, "_is_finished", True)

    def resume_wandb_run_if_needed(self, resume_kwargs: dict[str, typing.Any] | None = None) -> None:
        """Resumes a previously initialized and 'finished' wandb run if needed."""
        if not self.is_wandb_run_finished():
            return
        base_reinit_args: dict[str, typing.Any] = {
            "project": self.wandb_run_project,
            "entity": self.wandb_run_entity,
            "id": self.wandb_run_id,
            "dir": self.wandb_run_dir,
            "resume": "must",
        }
        base_reinit_args.update(resume_kwargs or {})
        self.wandb_run = wandb.init(**base_reinit_args)

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
    config: (
        hydra_zen.typing.Builds[typing.Any]
        | type[typing.Any]
        | collections.abc.MutableMapping[str, typing.Any]
        | omegaconf.DictConfig
    )
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
        if isinstance(self.config, collections.abc.MutableMapping):
            config_mapping = typing.cast("collections.abc.MutableMapping[str, typing.Any]", self.config)
            config_mapping["__cfg_name__"] = self.name
            config_mapping["__cfg_group__"] = self.group
            config_mapping["__cfg_package__"] = self.package
            config_mapping["__cfg_description__"] = self.description
        else:
            self.config.__cfg_name__ = self.name
            self.config.__cfg_group__ = self.group
            self.config.__cfg_package__ = self.package
            self.config.__cfg_description__ = self.description
        zen_meta = getattr(self.config, "zen_meta", None)
        if isinstance(zen_meta, collections.abc.MutableMapping):
            # TODO might need to update zen exclude?
            zen_meta["__cfg_name__"] = self.name
            zen_meta["__cfg_group__"] = self.group
            zen_meta["__cfg_package__"] = self.package
            zen_meta["__cfg_description__"] = self.description
        return self
