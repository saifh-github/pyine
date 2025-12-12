import pathlib

import hydra_zen
import pytest

import pyine.apps.trainers.hf_trainer
import pyine.apps.trainers.hf_trainer_configs
import pyine.configs.base
import pyine.evals.common
import pyine.utils.filesystem
import pyine.utils.reprod
import tests.env_checks


@pytest.mark.integration
@pytest.mark.dataset
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check sample generation",
)
def test_code_exec_experiment_config_taco_latest_20s_eval_only(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry-run launch of the HuggingFace trainer using the TACO_latest_20s_eval_only preset."""
    monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
    app_configs = pyine.apps.trainers.hf_trainer_configs.register_hydra_configs(pyine.evals.common.EvalType.CODE_EXEC)
    assert len(app_configs) > 1
    entrypoint_config = next(
        (cfg for cfg in app_configs if cfg.name == "entrypoint" and cfg.group is None),
        None,
    )
    assert entrypoint_config is not None
    _ = hydra_zen.launch(
        entrypoint_config.config,
        hydra_zen.zen(pyine.apps.trainers.hf_trainer_configs._async_main_wrapper),
        overrides={
            "+experiment": "shortcuts_TACO_latest_20s_base",
            "runtime": "dry_run",
            "runtime.seed": 123,
            "config.training_args_config.do_train": False,
            "config.training_args_config.do_predict": True,
        },
        config_name="hf_trainer",
        version_base=pyine.configs.base.target_hydra_version,
    )


def test_register_hydra_configs_registers_sweepers(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
    configs = pyine.apps.trainers.hf_trainer_configs.register_hydra_configs(pyine.evals.common.EvalType.CODE_EXEC)
    sweeper_configs = {(cfg.group, cfg.name) for cfg in configs if cfg.group == "hydra/sweeper"}
    assert ("hydra/sweeper", "wandb_sweeper_base") in sweeper_configs
