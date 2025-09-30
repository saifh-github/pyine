import pathlib

import pytest

import pyine.apps.trainers.openai_finetune
import pyine.apps.trainers.openai_finetune_configs
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.evals.code_exec.configs
import pyine.organisms.datamodules.shortcuts_configs
import pyine.utils.openai
import tests.env_checks


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_MISSING,
    reason="TACO traces dataset is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.TACO_TRACES_DATASET_SPLIT_MISSING,
    reason="TACO traces dataset split is missing, cannot check sample generation",
)
@pytest.mark.skipif(
    tests.env_checks.OPENAI_API_KEY_MISSING or tests.env_checks.NETWORK_UNAVAILABLE,
    reason="OpenAI API key or network not available, cannot run OpenAI-backed evaluation.",
)
async def test_code_exec_eval_base_model_with_skip_fine_tuning(
    tmp_path: pathlib.Path,
) -> None:
    """End-to-end exercise of main() using the real OpenAI API but skipping fine-tuning."""
    latest_dataset_path = pyine.data.traces.dataset_utils.get_latest_dataset_path("TACO")
    assert latest_dataset_path is not None and latest_dataset_path.exists()
    real_dm_config = pyine.organisms.datamodules.shortcuts_configs.ShortcutBiasDataModuleConfig(
        lmdb_paths=[latest_dataset_path],
        max_solution_count=2,
        split_file_path=pyine.data.utils.splits.get_dataset_split_file_path("TACO"),
    )
    openai_client_config = pyine.utils.openai.OpenAIClientConfig()
    finetuner_params_config = pyine.utils.openai.OpenAIFineTunerParamsConfig(
        base_model="gpt-4o-mini",
        method={"type": "supervised"},
        seed=0,
        suffix="pyine-pytest-openai-ft",
    )
    finetuner_cfg = pyine.utils.openai.OpenAIFineTunerConfig(params=finetuner_params_config)
    cfg = pyine.apps.trainers.openai_finetune_configs.OpenAIFineTuneAppMainConfig(
        datamodule_config=real_dm_config,
        openai_client_config=openai_client_config,
        openai_finetuner_config=finetuner_cfg,
        evals_config=pyine.evals.code_exec.configs.CodeExecEvalsConfig(),
    )
    await pyine.apps.trainers.openai_finetune.main(cfg, skip_fine_tuning=True)
