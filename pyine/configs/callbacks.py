import logging
import pathlib
import shutil
import tempfile
import typing

import hydra.experimental.callback
import omegaconf

import pyine.utils.distrib


class NonPrimaryRankCleanupCallback(hydra.experimental.callback.Callback):
    """Hydra callback that prevents non-primary distributed ranks from writing artifacts to disk.

    The callback re-routes Hydra's run directory to a temporary location, optionally disables
    file-based logging, and may clean up the temporary directory once the job finishes. It
    activates only when a non-zero distributed rank is detected by `pyine.utils.distrib`.
    """

    def __init__(
        self,
        remove_temp_dir_on_exit: bool = False,
        disable_disk_logging: bool = False,
    ) -> None:
        """Initializes the callback class."""
        self._cleanup_path: pathlib.Path | None = None
        self._detected_rank: int | None = None
        self._logger = logging.getLogger(__name__)
        self._remove_temp_dir_on_exit = remove_temp_dir_on_exit
        self._disable_disk_logging = disable_disk_logging

    def on_run_start(
        self,
        config: omegaconf.DictConfig,
        **kwargs: typing.Any,
    ) -> None:
        """Adjusts Hydra's configuration before the job starts."""
        rank_value = pyine.utils.distrib.get_global_rank(default=None)
        self._detected_rank = rank_value
        if pyine.utils.distrib.is_main_process(rank_value):
            return  # this is the main process, keep the config as-is
        temp_directory = pathlib.Path(tempfile.mkdtemp(prefix="pyine-hydra-rank-", dir=tempfile.gettempdir()))
        self._cleanup_path = temp_directory
        self._logger.debug(f"redirecting hydra outputs for rank {rank_value} to {temp_directory}")
        hydra_section = self._get_hydra_section(config)
        # if there's a run section, move its dir to a temp directory
        run_section = typing.cast("omegaconf.DictConfig | None", hydra_section.get("run"))
        if not isinstance(run_section, omegaconf.DictConfig):
            run_section = omegaconf.OmegaConf.create({})
            with omegaconf.open_dict(hydra_section):
                hydra_section["run"] = run_section
        with omegaconf.open_dict(run_section):
            run_section["dir"] = str(temp_directory)
        # if there's a sweep section, move its dir to a temp directory
        sweep_section = typing.cast("omegaconf.DictConfig | None", hydra_section.get("sweep"))
        if isinstance(sweep_section, omegaconf.DictConfig):
            with omegaconf.open_dict(sweep_section):
                sweep_section["dir"] = str(temp_directory / "sweeps")
        # disable output subdirs and logging
        if self._disable_disk_logging:
            with omegaconf.open_dict(hydra_section):
                hydra_section["output_subdir"] = None
                hydra_section["job_logging"] = None
                if "hydra_logging" in hydra_section:
                    hydra_section["hydra_logging"] = None

    def on_job_start(
        self,
        config: omegaconf.DictConfig,
        *,
        task_function: typing.Callable[..., typing.Any],
        **kwargs: typing.Any,
    ) -> None:
        """Drops any file handlers that might have been attached before task execution."""
        if self._is_main_process():
            return
        runtime_output_dir = self._get_runtime_output_dir(config)
        if runtime_output_dir is not None:
            self._cleanup_path = runtime_output_dir
        if self._disable_disk_logging:
            self._remove_file_handlers()

    def on_job_end(
        self,
        config: omegaconf.DictConfig,
        job_return: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        """Removes the temporary Hydra directory after the job completes."""
        if self._is_main_process():
            return
        cleanup_path = self._cleanup_path or self._get_runtime_output_dir(config)
        if cleanup_path is None:
            return
        if self._remove_temp_dir_on_exit:
            try:
                shutil.rmtree(cleanup_path, ignore_errors=True)
            except Exception as exception:
                self._logger.debug(f"failed to remove hydra directory {cleanup_path}: {exception}")
        else:
            self._logger.debug(f"preserving hydra outputs for rank {self._detected_rank} at {cleanup_path}")
        self._cleanup_path = None
        self._detected_rank = None

    def _is_main_process(self) -> bool:
        """Returns whether the callback is running on the primary process (rank 0)."""
        return pyine.utils.distrib.is_main_process(rank=self._detected_rank)

    def _get_runtime_output_dir(
        self,
        config: omegaconf.DictConfig,
    ) -> pathlib.Path | None:
        """Extracts Hydra's runtime output directory as a path, if available."""
        runtime_output_dir = omegaconf.OmegaConf.select(config, "hydra.runtime.output_dir")
        if runtime_output_dir is None:
            return None
        return pathlib.Path(str(runtime_output_dir)).expanduser().resolve()

    def _remove_file_handlers(self) -> None:
        """Removes file handlers from all active loggers."""
        loggers_to_check: list[logging.Logger] = [logging.getLogger()]
        for _logger_name, logger_candidate in logging.Logger.manager.loggerDict.items():
            if not isinstance(logger_candidate, logging.Logger):
                continue
            loggers_to_check.append(logger_candidate)
        for logger_instance in loggers_to_check:
            handlers_snapshot = list(logger_instance.handlers)
            for handler in handlers_snapshot:
                if isinstance(handler, logging.FileHandler):
                    logger_instance.removeHandler(handler)
                    handler.close()

    def _get_hydra_section(
        self,
        config: omegaconf.DictConfig,
    ) -> omegaconf.DictConfig:
        """Returns the `hydra` section of the composed config, raising if missing."""
        with omegaconf.open_dict(config):
            hydra_section = config.get("hydra")
        if not isinstance(hydra_section, omegaconf.DictConfig):
            raise RuntimeError("Hydra configuration is missing the mandatory 'hydra' section.")
        return hydra_section
