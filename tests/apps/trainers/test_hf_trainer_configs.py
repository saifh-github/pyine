import pathlib

import hydra_zen
import omegaconf
import pytest

import pyine.apps.trainers.common
import pyine.apps.trainers.hf_rl_trainer_configs
import pyine.apps.trainers.hf_sft_trainer_configs
import pyine.apps.trainers.hf_trainer
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
@pytest.mark.parametrize(
    ("config_module", "experiment_name", "do_train_override", "do_predict_override"),
    [
        pytest.param(
            pyine.apps.trainers.hf_sft_trainer_configs,
            "hf_sft_shortcuts_TACO_latest_20s_base",
            "config.training_args_config.do_train",
            "config.training_args_config.do_predict",
            id="sft",
        ),
        pytest.param(
            pyine.apps.trainers.hf_rl_trainer_configs,
            "hf_rl_shortcuts_TACO_latest_20s_base_hard_matching",
            "config.grpo_config.do_train",
            "config.grpo_config.do_predict",
            id="rl",
        ),
    ],
)
def test_code_exec_experiment_config_taco_latest_20s_eval_only_dry_run(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    config_module: type,
    experiment_name: str,
    do_train_override: str,
    do_predict_override: str,
) -> None:
    """Dry-run launch of the HuggingFace trainer using the TACO_latest_20s_eval_only preset."""
    monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
    app_configs = config_module.register_hydra_configs(pyine.evals.common.EvalType.CODE_EXEC)
    assert len(app_configs) > 1
    entrypoint_config = next(
        (cfg for cfg in app_configs if cfg.name == "entrypoint" and cfg.group is None),
        None,
    )
    assert entrypoint_config is not None
    _ = hydra_zen.launch(
        entrypoint_config.config,
        hydra_zen.zen(pyine.apps.trainers.common.async_hf_trainer_main_wrapper),
        overrides={
            "+experiment": experiment_name,
            "runtime": "dry_run",
            "runtime.seed": 123,
            do_train_override: False,
            do_predict_override: True,
        },
        config_name="entrypoint",
        version_base=pyine.configs.base.target_hydra_version,
    )


@pytest.mark.slow
def test_register_hydra_configs_registers_sweepers(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
    sft_configs = pyine.apps.trainers.hf_sft_trainer_configs.register_hydra_configs(
        pyine.evals.common.EvalType.CODE_EXEC
    )
    sweeper_configs = {(cfg.group, cfg.name) for cfg in sft_configs if cfg.group == "hydra/sweeper"}
    assert ("hydra/sweeper", "wandb_sweeper_base") in sweeper_configs
    rl_configs = pyine.apps.trainers.hf_rl_trainer_configs.register_hydra_configs(pyine.evals.common.EvalType.CODE_EXEC)
    sweeper_configs = {(cfg.group, cfg.name) for cfg in rl_configs if cfg.group == "hydra/sweeper"}
    assert ("hydra/sweeper", "wandb_sweeper_base") in sweeper_configs


def test_v0_rl_eval_base_uses_vllm_and_respects_runtime_dry_run(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyine.configs.base.register_searchpath_plugin()
    monkeypatch.setattr(pyine.utils.filesystem, "get_logs_root_path", lambda: tmp_path)
    app_configs = pyine.apps.trainers.hf_rl_trainer_configs.register_hydra_configs(
        pyine.evals.common.EvalType.CODE_EXEC
    )
    entrypoint_config = next(
        (cfg for cfg in app_configs if cfg.name == "entrypoint" and cfg.group is None),
        None,
    )
    assert entrypoint_config is not None

    def _extract_payload(cfg: omegaconf.DictConfig) -> dict[str, object]:
        lmdb_paths = omegaconf.OmegaConf.select(cfg, "config.datamodule_config.lmdb_paths")
        if lmdb_paths is None:
            raise AssertionError("missing datamodule lmdb_paths")
        lmdb_paths_list = list(lmdb_paths)
        return {
            "runtime_dry_run": omegaconf.OmegaConf.select(cfg, "runtime.dry_run"),
            "lmdb_paths_count": len(lmdb_paths_list),
            "vllm_provider": omegaconf.OmegaConf.select(
                cfg,
                "config.evals_config.vllm_provider_config.provider",
            ),
            "vllm_model": omegaconf.OmegaConf.select(
                cfg,
                "config.evals_config.vllm_provider_config.model_kwargs.model",
            ),
            "base_model": omegaconf.OmegaConf.select(cfg, "config.base_model"),
        }

    job = hydra_zen.launch(
        entrypoint_config.config,
        _extract_payload,
        overrides=[
            "+experiment=original/v0_rl_eval_base",
            "runtime=dry_run",
            "config.use_wandb_logging=false",
        ],
        config_name="entrypoint",
        version_base=pyine.configs.base.target_hydra_version,
        with_log_configuration=False,
    )
    payload = job.return_value
    assert payload["runtime_dry_run"] is True
    assert payload["lmdb_paths_count"] == 4
    assert payload["vllm_provider"] == "vllm"
    assert payload["vllm_model"] == payload["base_model"]
