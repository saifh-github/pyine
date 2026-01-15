"""Integration test for RewardManager with real WandB logging.

This test verifies the end-to-end reward logging pipeline with WandB, including:
- WandBRewardLogger creation and metric definition
- Per-sample reward logging with step tracking
- Run-level summary logging
- Table flushing
- Key prefix switching (train/eval phases)

All outputs are written to temporary directories to avoid polluting real logs.
"""

import pathlib
import typing

import pytest
import wandb

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.logging as reward_logging
import pyine.organisms.models.rewards.core.manager as reward_manager
import tests.organisms.models.rewards.conftest as rewards_conftest


@pytest.fixture
def isolate_wandb_env(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set environment variables for isolated WandB test execution."""
    monkeypatch.setenv("WANDB_PROJECT", "pyine-tests")
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WANDB_DIR", str(wandb_dir))
    # set all wandb storage dirs to temp to avoid permission errors in sandboxed environments
    data_dir = tmp_path / "wandb_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WANDB_DATA_DIR", str(data_dir))
    cache_dir = tmp_path / "wandb_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WANDB_CACHE_DIR", str(cache_dir))
    config_dir = tmp_path / "wandb_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(config_dir))


@pytest.fixture
def wandb_run(
    tmp_path: pathlib.Path,
    isolate_wandb_env: None,
) -> typing.Generator[wandb.sdk.wandb_run.Run, None, None]:
    """Create a real WandB run for integration testing."""
    del isolate_wandb_env  # used for setup
    run = wandb.init(
        project="pyine-tests",
        name="reward-manager-integration-test",
        dir=str(tmp_path),
        mode="offline",  # don't actually sync to cloud
        reinit=True,
    )
    yield run
    run.finish()


@pytest.mark.integration
class TestRewardManagerWandBIntegration:
    """Integration tests for RewardManager with real WandB logging."""

    def test_reward_manager_logs_to_wandb_with_step_tracking(
        self,
        wandb_run: wandb.sdk.wandb_run.Run,
    ) -> None:
        """Verify that RewardManager logs per-sample and run-level metrics to WandB with correct steps."""
        logging_config = reward_configs.LoggingConfig(
            enabled=True,
            scalar_log_every_n_generations=1,
            log_terms=True,
            log_metrics=True,
            log_tables=True,
            table_flush_every_n_generations=5,
        )
        term_spec = reward_configs.RewardTermSpec(
            name="parseable",
            type="format/parseable_answer",
            weight=1.0,
        )
        config = reward_configs.RewardManagerConfig(
            terms=[term_spec],
            logging=logging_config,
        )
        logger = reward_logging.make_wandb_reward_logger(wandb_run, logging_config)
        manager = reward_manager.RewardManager(config=config, logger=logger)
        manager.set_key_prefix("train")
        # simulate training steps
        for step_idx in range(3):
            manager.set_step(step_idx * 10)
            sample_ctx = rewards_conftest.make_sample_context(
                identifier=f"train_sample_{step_idx}",
                model_output=f"<final>output for step {step_idx}</final>",
                parsed=rewards_conftest.make_parsed_output(
                    f"<final>output for step {step_idx}</final>",
                    final_answer=f"output for step {step_idx}",
                    reasoning="thinking...",
                ),
            )
            manager.compute(sample_ctx)
        # flush stats
        manager.flush_stats()
        # verify wandb received logs (check summary for logged data)
        summary = dict(wandb_run.summary)  # type: ignore[reportUnknownArgumentType]
        assert "train/global_step" in summary, f"expected train/global_step in summary, got: {sorted(summary)}"
        # verify reward metrics were logged with the train/ prefix
        reward_keys = [k for k in summary if k.startswith("train/reward/")]
        assert len(reward_keys) > 0, f"expected train/reward/ prefixed keys, got: {sorted(summary.keys())}"

    def test_reward_manager_handles_train_eval_prefix_switching(
        self,
        wandb_run: wandb.sdk.wandb_run.Run,
    ) -> None:
        """Verify that key prefix switching works correctly for train/eval phases."""
        logging_config = reward_configs.LoggingConfig(
            enabled=True,
            scalar_log_every_n_generations=1,
            log_terms=True,
            log_metrics=True,
        )
        term_spec = reward_configs.RewardTermSpec(
            name="parseable",
            type="format/parseable_answer",
            weight=1.0,
        )
        config = reward_configs.RewardManagerConfig(terms=[term_spec], logging=logging_config)
        logger = reward_logging.make_wandb_reward_logger(wandb_run, logging_config)
        manager = reward_manager.RewardManager(config=config, logger=logger)
        # train phase
        manager.set_key_prefix("train")
        manager.set_step(100)
        train_ctx = rewards_conftest.make_sample_context(
            identifier="train_sample",
            model_output="<final>train result</final>",
            parsed=rewards_conftest.make_parsed_output(
                "<final>train result</final>",
                final_answer="train result",
            ),
        )
        manager.compute(train_ctx)
        manager.flush_stats()
        # switch to eval phase
        manager.set_key_prefix("eval")
        manager.set_step(100)
        eval_ctx = rewards_conftest.make_sample_context(
            identifier="eval_sample",
            model_output="<final>eval result</final>",
            parsed=rewards_conftest.make_parsed_output(
                "<final>eval result</final>",
                final_answer="eval result",
            ),
        )
        manager.compute(eval_ctx)
        manager.flush_stats()
        # verify both prefixes were used (summary contains all keys ever logged)
        summary = dict(wandb_run.summary)  # type: ignore[reportUnknownArgumentType]
        train_keys = [k for k in summary if k.startswith("train/")]
        eval_keys = [k for k in summary if k.startswith("eval/")]
        assert len(train_keys) > 0, f"expected train-prefixed keys, got: {sorted(summary.keys())}"
        assert len(eval_keys) > 0, f"expected eval-prefixed keys, got: {sorted(summary.keys())}"

    def test_reward_manager_tables_are_flushed_to_wandb(
        self,
        wandb_run: wandb.sdk.wandb_run.Run,
    ) -> None:
        """Verify that reward tables are flushed to WandB."""
        logging_config = reward_configs.LoggingConfig(
            enabled=True,
            scalar_log_every_n_generations=1,
            log_tables=True,
            table_row_every_n_generations=1,  # add row every sample for test
            table_flush_every_n_generations=2,  # flush after every 2 samples
        )
        term_spec = reward_configs.RewardTermSpec(
            name="parseable",
            type="format/parseable_answer",
            weight=1.0,
        )
        config = reward_configs.RewardManagerConfig(terms=[term_spec], logging=logging_config)
        logger = reward_logging.make_wandb_reward_logger(wandb_run, logging_config)
        manager = reward_manager.RewardManager(config=config, logger=logger)
        # log enough samples to trigger table flush
        for sample_idx in range(3):
            manager.set_step(sample_idx)
            sample_ctx = rewards_conftest.make_sample_context(
                identifier=f"sample_{sample_idx}",
                model_output=f"<final>answer {sample_idx}</final>",
                parsed=rewards_conftest.make_parsed_output(
                    f"<final>answer {sample_idx}</final>",
                    final_answer=f"answer {sample_idx}",
                ),
            )
            manager.compute(sample_ctx)
        manager.flush_stats()
        # verify table was logged (check summary for table entries)
        summary = dict(wandb_run.summary)  # type: ignore[reportUnknownArgumentType]
        table_keys = [k for k in summary if "generation_details" in k]
        assert len(table_keys) > 0, f"expected table key in logged data, got keys: {sorted(summary.keys())}"
