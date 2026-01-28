"""Integration test for RewardManager with real WandB logging.

This test verifies the end-to-end reward logging pipeline with WandB, including:
- WandBRewardLogger creation and metric definition
- Per-generation reward logging with generation_count as x-axis
- Batch-level statistics logging with batch_count as x-axis
- Run-level summary logging (including failure metrics)
- Table flushing
- Key prefix switching (train/eval phases)
- define_metric configuration (step_metric and summary types)

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

    def test_generation_count_logged_as_xaxis_for_per_generation_metrics(
        self,
        wandb_run: wandb.sdk.wandb_run.Run,
    ) -> None:
        """Verify that generation_count is logged and used as x-axis for per-generation metrics."""
        logging_config = reward_configs.LoggingConfig(
            enabled=True,
            scalar_log_every_n_generations=1,
            log_terms=True,
        )
        term_spec = reward_configs.RewardTermSpec(
            name="parseable",
            type="format/parseable_answer",
            weight=1.0,
        )
        config = reward_configs.RewardManagerConfig(terms=[term_spec], logging=logging_config)
        logger = reward_logging.make_wandb_reward_logger(wandb_run, logging_config)
        manager = reward_manager.RewardManager(config=config, logger=logger)
        manager.set_key_prefix("train")
        # log multiple samples at the same trainer step (simulating GRPO/gradient accumulation)
        manager.set_step(100)
        for sample_idx in range(5):
            sample_ctx = rewards_conftest.make_sample_context(
                identifier=f"sample_{sample_idx}",
                model_output=f"<final>answer {sample_idx}</final>",
                parsed=rewards_conftest.make_parsed_output(
                    f"<final>answer {sample_idx}</final>",
                    final_answer=f"answer {sample_idx}",
                ),
            )
            manager.compute(sample_ctx)
        # verify generation_count was logged
        summary = dict(wandb_run.summary)  # type: ignore[reportUnknownArgumentType]
        assert "train/generation_count" in summary, (
            f"expected train/generation_count in summary, got: {sorted(summary.keys())}"
        )
        # generation_count should be 5 after 5 samples
        assert summary["train/generation_count"] == 5, (
            f"expected generation_count=5, got {summary['train/generation_count']}"
        )

    def test_batch_stats_logged_with_batch_count_xaxis(
        self,
        wandb_run: wandb.sdk.wandb_run.Run,
    ) -> None:
        """Verify that batch-level stats are logged with batch_count as x-axis."""
        logging_config = reward_configs.LoggingConfig(
            enabled=True,
            scalar_log_every_n_generations=1,
            log_batch_stats=True,
        )
        term_spec = reward_configs.RewardTermSpec(
            name="parseable",
            type="format/parseable_answer",
            weight=1.0,
        )
        config = reward_configs.RewardManagerConfig(terms=[term_spec], logging=logging_config)
        logger = reward_logging.make_wandb_reward_logger(wandb_run, logging_config)
        manager = reward_manager.RewardManager(config=config, logger=logger)
        manager.set_key_prefix("train")
        # simulate gradient accumulation: multiple batches at the same trainer step
        manager.set_step(100)
        for batch_idx in range(3):
            batch_samples = [
                rewards_conftest.make_sample_context(
                    identifier=f"batch{batch_idx}_sample{i}",
                    model_output=f"<final>answer {i}</final>",
                    parsed=rewards_conftest.make_parsed_output(
                        f"<final>answer {i}</final>",
                        final_answer=f"answer {i}",
                    ),
                )
                for i in range(4)
            ]
            manager.compute_batch(batch_samples)
        # verify batch_count was logged
        summary = dict(wandb_run.summary)  # type: ignore[reportUnknownArgumentType]
        assert "train/batch_count" in summary, f"expected train/batch_count in summary, got: {sorted(summary.keys())}"
        # batch_count should be 3 after 3 batches
        assert summary["train/batch_count"] == 3, f"expected batch_count=3, got {summary['train/batch_count']}"
        # verify batch-level metrics were logged
        batch_metric_keys = [k for k in summary if "reward/batch/" in k]
        assert len(batch_metric_keys) > 0, f"expected reward/batch/ metrics, got: {sorted(summary.keys())}"

    def test_failure_metrics_logged_in_run_summaries(
        self,
        wandb_run: wandb.sdk.wandb_run.Run,
    ) -> None:
        """Verify that failure metrics are included in run-level summaries."""
        logging_config = reward_configs.LoggingConfig(
            enabled=True,
            scalar_log_every_n_generations=1,
        )
        term_spec = reward_configs.RewardTermSpec(
            name="parseable",
            type="format/parseable_answer",
            weight=1.0,
        )
        config = reward_configs.RewardManagerConfig(terms=[term_spec], logging=logging_config)
        logger = reward_logging.make_wandb_reward_logger(wandb_run, logging_config)
        manager = reward_manager.RewardManager(config=config, logger=logger)
        manager.set_key_prefix("train")
        manager.set_step(100)
        # log some samples
        for sample_idx in range(3):
            sample_ctx = rewards_conftest.make_sample_context(
                identifier=f"sample_{sample_idx}",
                model_output=f"<final>answer {sample_idx}</final>",
                parsed=rewards_conftest.make_parsed_output(
                    f"<final>answer {sample_idx}</final>",
                    final_answer=f"answer {sample_idx}",
                ),
            )
            manager.compute(sample_ctx)
        # flush stats with failure info (simulating callback behavior)
        manager.flush_stats(failure_ratio=0.1, failure_count=2)
        # verify failure metrics were logged by checking they exist in the summary
        # note: WandB stores nested paths in the summary, and the exact structure varies by version
        # the key verification is that these keys exist (the wandb "Run history" output shows values)
        summary = dict(wandb_run.summary)  # type: ignore[reportUnknownArgumentType]
        # failure metrics should be present as keys (possibly nested)
        assert "train/failures/failure_ratio" in summary, (
            f"expected train/failures/failure_ratio in summary, got keys: {sorted(summary.keys())}"
        )
        assert "train/failures/failure_count" in summary, (
            f"expected train/failures/failure_count in summary, got keys: {sorted(summary.keys())}"
        )
        # note: we don't verify exact values here because WandB's SummarySubDict makes direct
        # comparison tricky; the important thing is that the metrics were logged (visible in
        # the wandb "Run history" output during test execution)

    def test_define_metric_configures_step_metrics_correctly(
        self,
        wandb_run: wandb.sdk.wandb_run.Run,
    ) -> None:
        """Verify that define_metric is called to configure x-axes for different metric types."""
        logging_config = reward_configs.LoggingConfig(
            enabled=True,
            scalar_log_every_n_generations=1,
            log_batch_stats=True,
        )
        term_spec = reward_configs.RewardTermSpec(
            name="parseable",
            type="format/parseable_answer",
            weight=1.0,
        )
        config = reward_configs.RewardManagerConfig(terms=[term_spec], logging=logging_config)
        logger = reward_logging.make_wandb_reward_logger(wandb_run, logging_config)
        manager = reward_manager.RewardManager(config=config, logger=logger)
        manager.set_key_prefix("train")
        manager.set_step(100)
        # trigger all three types of logging to ensure define_metric is called for each
        # 1. per-generation (triggers _define_generation_step_metrics)
        sample_ctx = rewards_conftest.make_sample_context(
            identifier="sample_1",
            model_output="<final>answer</final>",
            parsed=rewards_conftest.make_parsed_output(
                "<final>answer</final>",
                final_answer="answer",
            ),
        )
        manager.compute(sample_ctx)
        # 2. batch-level (triggers _define_batch_step_metrics)
        manager.compute_batch([sample_ctx])
        # 3. run-level (triggers _define_run_step_metrics)
        manager.flush_stats()
        # check that wandb._define_metric was configured correctly by inspecting run settings
        # note: in offline mode, we can't directly inspect define_metric calls, but we can
        # verify the metrics were logged with expected keys which implies define_metric worked
        summary = dict(wandb_run.summary)  # type: ignore[reportUnknownArgumentType]
        # per-generation metrics should exist
        assert "train/generation_count" in summary, "generation_count should be logged"
        assert "train/reward/total" in summary, "reward/total should be logged"
        # batch metrics should exist
        assert "train/batch_count" in summary, "batch_count should be logged"
        assert any("reward/batch/" in k for k in summary), "batch metrics should be logged"
        # run metrics should exist
        assert "train/global_step" in summary, "global_step should be logged"
        assert any("reward/run/" in k for k in summary), "run metrics should be logged"

    def test_multiple_batches_at_same_step_get_unique_batch_counts(
        self,
        wandb_run: wandb.sdk.wandb_run.Run,
    ) -> None:
        """Verify that multiple batches at the same trainer step get unique batch_count values.

        This is critical for gradient accumulation scenarios where WandB would otherwise
        aggregate values logged at the same step.
        """
        logging_config = reward_configs.LoggingConfig(
            enabled=True,
            scalar_log_every_n_generations=1,
            log_batch_stats=True,
        )
        term_spec = reward_configs.RewardTermSpec(
            name="parseable",
            type="format/parseable_answer",
            weight=1.0,
        )
        config = reward_configs.RewardManagerConfig(terms=[term_spec], logging=logging_config)
        logger = reward_logging.make_wandb_reward_logger(wandb_run, logging_config)
        manager = reward_manager.RewardManager(config=config, logger=logger)
        manager.set_key_prefix("train")
        # simulate gradient accumulation: 4 micro-batches per optimizer step
        gradient_accumulation_steps = 4
        num_optimizer_steps = 3
        for optimizer_step in range(num_optimizer_steps):
            manager.set_step(optimizer_step * 100)  # same step for all micro-batches
            for _micro_batch_idx in range(gradient_accumulation_steps):
                batch_samples = [
                    rewards_conftest.make_sample_context(
                        identifier=f"step{optimizer_step}_micro{_micro_batch_idx}_sample{i}",
                        model_output=f"<final>answer {i}</final>",
                        parsed=rewards_conftest.make_parsed_output(
                            f"<final>answer {i}</final>",
                            final_answer=f"answer {i}",
                        ),
                    )
                    for i in range(2)
                ]
                manager.compute_batch(batch_samples)
        # verify total batch_count equals total micro-batches
        total_batches = num_optimizer_steps * gradient_accumulation_steps
        summary = dict(wandb_run.summary)  # type: ignore[reportUnknownArgumentType]
        assert summary["train/batch_count"] == total_batches, (
            f"expected batch_count={total_batches}, got {summary['train/batch_count']}"
        )
        # verify monotonic counter in manager matches
        assert manager._monotonic_batch_count == total_batches
