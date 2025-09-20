import pathlib

import hydra_zen
import pytest

import pyine.apps.trainers.hf_trainer
import pyine.apps.trainers.hf_trainer_configs
import pyine.configs.base
import pyine.utils.filesystem
import pyine.utils.reprod
import tests.env_checks


@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check sample generation",
)
def test_experiment_config_TACO_latest_20s_eval_only(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry-run launch of the HuggingFace trainer using the TACO_latest_20s_eval_only preset."""
    monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
    main_config = pyine.apps.trainers.hf_trainer_configs.register_hydra_configs()
    with pytest.raises(pyine.utils.reprod.DryRunExit):
        _ = hydra_zen.launch(
            main_config,
            hydra_zen.zen(pyine.apps.trainers.hf_trainer.main),
            overrides={
                "+experiment": "TACO_latest_20s_eval_only",
                "runtime": "dry_run",
                "runtime.seed": 123,
            },
            config_name="hf_trainer",
            version_base=pyine.configs.base.target_hydra_version,
        )
