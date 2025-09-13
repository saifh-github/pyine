import pathlib

import hydra_zen
import pytest

import pyine.apps.trainers.openai_finetune
import pyine.apps.trainers.openai_finetune_configs
import pyine.configs.base
import pyine.utils.filesystem
import pyine.utils.reprod
import tests.data.utils.env_checks as env_checks


@pytest.mark.skipif(
    env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    env_checks.OPENAI_API_KEY_MISSING,
    reason="OpenAI API key missing, cannot run OpenAI-backed evaluation.",
)
def test_experiment_config_TACO_latest_200t_eval_only(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end exercise of main() using the TACO dataset and a max sample count of 5."""
    monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
    main_config = pyine.apps.trainers.openai_finetune_configs.register_hydra_configs()
    with pytest.raises(pyine.utils.reprod.DryRunExit):
        _ = hydra_zen.launch(
            main_config,
            hydra_zen.zen(pyine.apps.trainers.openai_finetune_configs._async_main_wrapper),
            overrides={
                "+experiment": "TACO_latest_200t_eval_only",
                "runtime": "dry_run",
            },
            config_name="openai_finetune_main",
            version_base=pyine.configs.base.target_hydra_version,
        )
