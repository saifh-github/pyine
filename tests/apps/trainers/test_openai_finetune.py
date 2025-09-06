import pathlib

import pytest

import pyine.apps.trainers.openai_finetune as finetune
import pyine.data.traces.dataset_utils as traces_dataset_utils
import pyine.data.utils.splits as split_utils
import pyine.organisms.datamodules.shortcuts as shortcuts_dm
import pyine.organisms.models.utils.openai as openai_utils
import tests.data.utils.env_checks as env_checks


@pytest.mark.slow
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
def test_main_evaluates_base_model_with_skip_fine_tuning(
    tmp_path: pathlib.Path,
) -> None:
    """End-to-end exercise of main() using the real OpenAI API but skipping fine-tuning."""

    real_dm_config = shortcuts_dm.ShortcutBiasDataModuleConfig(
        lmdb_paths=[
            traces_dataset_utils.get_latest_dataset_path("TACO"),
        ],
        max_trace_count=20,
        split_file_path=split_utils.get_dataset_split_file_path("TACO"),
    )
    finetuner_params_config = openai_utils.OpenAIFineTunerParamsConfig(
        base_model="gpt-4o-mini",
        method={"type": "supervised"},
        seed=0,
        suffix="pyine-pytest-openai-ft",
    )
    finetuner_cfg = openai_utils.OpenAIFineTunerConfig(params=finetuner_params_config)
    cfg = finetune.MainConfig(
        datamodule_config=real_dm_config,
        openai_finetuner=finetuner_cfg,
        seed=0,
    )
    finetune.main(cfg, skip_fine_tuning=True)
