import logging
import pathlib
import shutil
import typing

import hydra.core.utils
import omegaconf
import pytest

import pyine.configs.callbacks


def build_sample_hydra_config(
    base_directory: pathlib.Path,
) -> omegaconf.DictConfig:
    run_directory = base_directory / "runs" / "app" / "exp" / "run"
    job_log_file = run_directory / "output.log"
    sweep_directory = base_directory / "sweeps"
    config_dict: dict[str, typing.Any] = {
        "hydra": {
            "run": {
                "dir": str(run_directory),
            },
            "sweep": {
                "dir": str(sweep_directory),
                "subdir": "${hydra:job.num}_${hydra.job.override_dirname}",
            },
            "job_logging": {
                "version": 1,
                "handlers": {
                    "file": {
                        "class": "logging.FileHandler",
                        "formatter": "simple",
                        "filename": str(job_log_file),
                    },
                },
                "root": {
                    "handlers": ["file"],
                    "level": "INFO",
                },
            },
            "hydra_logging": {
                "version": 1,
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                    },
                },
                "root": {
                    "handlers": ["console"],
                    "level": "INFO",
                },
            },
            "output_subdir": ".hydra",
            "runtime": {
                "output_dir": str(run_directory),
                "cwd": str(base_directory),
            },
        },
    }
    return omegaconf.OmegaConf.create(config_dict)


class TestNonPrimaryRankCleanupCallback:
    def test_noop_without_rank(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        config = build_sample_hydra_config(tmp_path)
        original_run_dir = omegaconf.OmegaConf.select(config, "hydra.run.dir")
        original_output_subdir = omegaconf.OmegaConf.select(config, "hydra.output_subdir")
        original_job_logging = omegaconf.OmegaConf.select(config, "hydra.job_logging")
        callback = pyine.configs.callbacks.NonPrimaryRankCleanupCallback()
        callback.on_run_start(config=config)
        assert omegaconf.OmegaConf.select(config, "hydra.run.dir") == original_run_dir
        assert omegaconf.OmegaConf.select(config, "hydra.output_subdir") == original_output_subdir
        assert omegaconf.OmegaConf.select(config, "hydra.job_logging") == original_job_logging

    def test_non_primary_rank_redirects_and_cleans_outputs(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RANK", "1")
        config = build_sample_hydra_config(tmp_path)
        callback = pyine.configs.callbacks.NonPrimaryRankCleanupCallback(
            remove_temp_dir_on_exit=True,
            disable_disk_logging=True,
        )
        callback.on_run_start(config=config)
        run_dir_value = typing.cast("str", omegaconf.OmegaConf.select(config, "hydra.run.dir"))
        assert run_dir_value is not None
        run_dir_path = pathlib.Path(run_dir_value)
        assert run_dir_path.exists()
        assert omegaconf.OmegaConf.select(config, "hydra.output_subdir") is None
        assert omegaconf.OmegaConf.select(config, "hydra.job_logging") is None
        assert omegaconf.OmegaConf.select(config, "hydra.hydra_logging") is None
        with omegaconf.open_dict(config["hydra"]["runtime"]):
            config["hydra"]["runtime"]["output_dir"] = str(run_dir_path)
        run_dir_path.mkdir(parents=True, exist_ok=True)
        (run_dir_path / "dummy.txt").write_text("hello", encoding="utf-8")
        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        try:
            log_file_path = run_dir_path / "preexisting.log"
            preexisting_handler = logging.FileHandler(log_file_path)
            root_logger.addHandler(preexisting_handler)
            callback.on_job_start(
                config=config,
                task_function=lambda *args, **kwargs: None,
            )
            assert all(not isinstance(handler, logging.FileHandler) for handler in root_logger.handlers)
            job_return = hydra.core.utils.JobReturn()
            callback.on_job_end(config=config, job_return=job_return)
        finally:
            root_logger.handlers = original_handlers
        assert not run_dir_path.exists()

    def test_local_rank_triggers_cleanup(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "2")
        config = build_sample_hydra_config(tmp_path)
        callback = pyine.configs.callbacks.NonPrimaryRankCleanupCallback(
            remove_temp_dir_on_exit=True,
            disable_disk_logging=True,
        )
        callback.on_run_start(config=config)
        assert omegaconf.OmegaConf.select(config, "hydra.job_logging") is None

    def test_non_primary_rank_preserves_outputs_when_configured(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RANK", "3")
        config = build_sample_hydra_config(tmp_path)
        original_output_subdir = omegaconf.OmegaConf.select(config, "hydra.output_subdir")
        original_job_logging = omegaconf.OmegaConf.select(config, "hydra.job_logging")
        callback = pyine.configs.callbacks.NonPrimaryRankCleanupCallback(
            remove_temp_dir_on_exit=False,
            disable_disk_logging=False,
        )
        callback.on_run_start(config=config)
        run_dir_value = typing.cast("str", omegaconf.OmegaConf.select(config, "hydra.run.dir"))
        run_dir_path = pathlib.Path(run_dir_value)
        assert run_dir_path.exists()
        assert omegaconf.OmegaConf.select(config, "hydra.output_subdir") == original_output_subdir
        assert omegaconf.OmegaConf.select(config, "hydra.job_logging") == original_job_logging
        with omegaconf.open_dict(config["hydra"]["runtime"]):
            config["hydra"]["runtime"]["output_dir"] = str(run_dir_path)
        run_dir_path.mkdir(parents=True, exist_ok=True)
        (run_dir_path / "debug.txt").write_text("debug", encoding="utf-8")
        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        try:
            preexisting_handler = logging.FileHandler(run_dir_path / "kept.log")
            root_logger.addHandler(preexisting_handler)
            callback.on_job_start(
                config=config,
                task_function=lambda *args, **kwargs: None,
            )
            assert any(isinstance(handler, logging.FileHandler) for handler in root_logger.handlers)
            job_return = hydra.core.utils.JobReturn()
            callback.on_job_end(config=config, job_return=job_return)
        finally:
            root_logger.handlers = original_handlers
        assert run_dir_path.exists()
        shutil.rmtree(run_dir_path)

    def test_local_rank0_on_local_fs_keeps_real_output_dir(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Local-rank-0 on node-local storage keeps the real Hydra output dir."""
        monkeypatch.setenv("RANK", "4")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(
            "pyine.utils.filesystem.is_path_on_shared_filesystem",
            lambda _path: False,
        )
        config = build_sample_hydra_config(tmp_path)
        original_run_dir = omegaconf.OmegaConf.select(config, "hydra.run.dir")
        original_output_subdir = omegaconf.OmegaConf.select(config, "hydra.output_subdir")
        original_job_logging = omegaconf.OmegaConf.select(config, "hydra.job_logging")
        callback = pyine.configs.callbacks.NonPrimaryRankCleanupCallback(
            remove_temp_dir_on_exit=True,
            disable_disk_logging=True,
        )
        callback.on_run_start(config=config)
        assert not callback._redirected
        assert omegaconf.OmegaConf.select(config, "hydra.run.dir") == original_run_dir
        assert omegaconf.OmegaConf.select(config, "hydra.output_subdir") == original_output_subdir
        assert omegaconf.OmegaConf.select(config, "hydra.job_logging") == original_job_logging

    def test_local_rank0_on_local_fs_full_lifecycle(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Full lifecycle: local-rank-0 on local FS keeps real dir, no cleanup on job end."""
        monkeypatch.setenv("RANK", "4")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(
            "pyine.utils.filesystem.is_path_on_shared_filesystem",
            lambda _path: False,
        )
        config = build_sample_hydra_config(tmp_path)
        run_dir = pathlib.Path(typing.cast("str", omegaconf.OmegaConf.select(config, "hydra.run.dir")))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "important.txt").write_text("data", encoding="utf-8")
        callback = pyine.configs.callbacks.NonPrimaryRankCleanupCallback(
            remove_temp_dir_on_exit=True,
            disable_disk_logging=True,
        )
        callback.on_run_start(config=config)
        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        try:
            preexisting_handler = logging.FileHandler(run_dir / "test.log")
            root_logger.addHandler(preexisting_handler)
            callback.on_job_start(
                config=config,
                task_function=lambda *args, **kwargs: None,
            )
            # file handlers should NOT be removed (not redirected)
            assert any(isinstance(handler, logging.FileHandler) for handler in root_logger.handlers)
            job_return = hydra.core.utils.JobReturn()
            callback.on_job_end(config=config, job_return=job_return)
        finally:
            root_logger.handlers = original_handlers
        # real output dir should be untouched
        assert run_dir.exists()
        assert (run_dir / "important.txt").read_text(encoding="utf-8") == "data"

    def test_local_rank0_on_shared_fs_still_redirects(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Local-rank-0 on shared storage still gets redirected (would collide with global-rank-0)."""
        monkeypatch.setenv("RANK", "4")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(
            "pyine.utils.filesystem.is_path_on_shared_filesystem",
            lambda _path: True,
        )
        config = build_sample_hydra_config(tmp_path)
        original_run_dir = omegaconf.OmegaConf.select(config, "hydra.run.dir")
        callback = pyine.configs.callbacks.NonPrimaryRankCleanupCallback()
        callback.on_run_start(config=config)
        assert callback._redirected
        assert omegaconf.OmegaConf.select(config, "hydra.run.dir") != original_run_dir

    def test_non_local_rank0_on_local_fs_still_redirects(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-local-rank-0 on local storage still gets redirected."""
        monkeypatch.setenv("RANK", "5")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setattr(
            "pyine.utils.filesystem.is_path_on_shared_filesystem",
            lambda _path: False,
        )
        config = build_sample_hydra_config(tmp_path)
        original_run_dir = omegaconf.OmegaConf.select(config, "hydra.run.dir")
        callback = pyine.configs.callbacks.NonPrimaryRankCleanupCallback()
        callback.on_run_start(config=config)
        assert callback._redirected
        assert omegaconf.OmegaConf.select(config, "hydra.run.dir") != original_run_dir

    def test_local_rank0_no_explicit_rank_shared_fs_redirects(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """LOCAL_RANK=0 without RANK on shared FS must redirect (inferred global rank is ambiguous)."""
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(
            "pyine.utils.distrib.has_explicit_global_rank",
            lambda: False,
        )
        monkeypatch.setattr(
            "pyine.utils.filesystem.is_path_on_shared_filesystem",
            lambda _path: True,
        )
        config = build_sample_hydra_config(tmp_path)
        original_run_dir = omegaconf.OmegaConf.select(config, "hydra.run.dir")
        callback = pyine.configs.callbacks.NonPrimaryRankCleanupCallback()
        callback.on_run_start(config=config)
        assert callback._redirected
        assert omegaconf.OmegaConf.select(config, "hydra.run.dir") != original_run_dir

    def test_local_rank0_no_explicit_rank_local_fs_keeps_real_dir(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """LOCAL_RANK=0 without RANK on local FS keeps the real dir (each node has own storage)."""
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(
            "pyine.utils.distrib.has_explicit_global_rank",
            lambda: False,
        )
        monkeypatch.setattr(
            "pyine.utils.filesystem.is_path_on_shared_filesystem",
            lambda _path: False,
        )
        config = build_sample_hydra_config(tmp_path)
        original_run_dir = omegaconf.OmegaConf.select(config, "hydra.run.dir")
        callback = pyine.configs.callbacks.NonPrimaryRankCleanupCallback()
        callback.on_run_start(config=config)
        assert not callback._redirected
        assert omegaconf.OmegaConf.select(config, "hydra.run.dir") == original_run_dir
