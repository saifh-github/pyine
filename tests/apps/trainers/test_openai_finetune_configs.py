import pathlib

import hydra_zen
import pytest

import pyine.apps.trainers.openai_finetune
import pyine.apps.trainers.openai_finetune_configs
import pyine.configs.base
import pyine.utils.filesystem
import pyine.utils.reprod
import tests.env_checks as env_checks


@pytest.mark.skipif(
    env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check sample generation",
)
def test_experiment_config_TACO_latest_20s_eval_only(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    clear_hydra_config_store,
) -> None:
    """End-to-end exercise of main() using the TACO dataset and a max sample count of 5."""
    monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
    app_configs = pyine.apps.trainers.openai_finetune_configs.register_hydra_configs()
    assert len(app_configs) > 1
    entrypoint_config = next((cfg for cfg in app_configs if cfg.name == "entrypoint" and cfg.group is None), None)
    assert entrypoint_config is not None
    with pytest.raises(pyine.utils.reprod.DryRunExit):
        _ = hydra_zen.launch(
            entrypoint_config.config,
            hydra_zen.zen(pyine.apps.trainers.openai_finetune_configs._async_main_wrapper),
            overrides={
                "+experiment": "TACO_latest_20s_eval_only",
                "runtime": "dry_run",
                "runtime.seed": 123,
            },
            config_name="openai_finetune",
            version_base=pyine.configs.base.target_hydra_version,
        )


@pytest.mark.skipif(
    env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check sample generation",
)
def test_experiment_config_TACO_latest(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    clear_hydra_config_store,
) -> None:
    """End-to-end exercise of main() using the TACO dataset and a max sample count of 5."""
    monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
    app_configs = pyine.apps.trainers.openai_finetune_configs.register_hydra_configs()
    assert len(app_configs) > 1
    entrypoint_config = next((cfg for cfg in app_configs if cfg.name == "entrypoint" and cfg.group is None), None)
    assert entrypoint_config is not None
    with pytest.raises(pyine.utils.reprod.DryRunExit):
        _ = hydra_zen.launch(
            entrypoint_config.config,
            hydra_zen.zen(pyine.apps.trainers.openai_finetune_configs._async_main_wrapper),
            overrides={
                "+experiment": "TACO_latest",
                "runtime": "dry_run",
                "runtime.seed": 123,
            },
            config_name="openai_finetune",
            version_base=pyine.configs.base.target_hydra_version,
        )
