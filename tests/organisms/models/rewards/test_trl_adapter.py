"""Tests for the TRL reward adapter integration with logging.

This test verifies the end-to-end flow of the TRL adapter calling the reward manager,
specifically focusing on ensuring that logged values are unique and correct (no cyclic
repetition bug).
"""

import pytest

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.logging as reward_logging
import pyine.organisms.models.rewards.core.manager as reward_manager
import pyine.organisms.models.rewards.core.registry as reward_registry
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.organisms.models.rewards.trl as rewards_trl
import tests.organisms.models.rewards.conftest as rewards_conftest


class _SampleIndexRewardTerm:
    """Test reward term that returns a unique value based on the sample's identifier.

    Expects identifier format like "sample_123" and returns 123.0 as the reward.
    This allows us to verify that the correct sample is being logged.
    """

    def reset(self, run_init_ctx: reward_types.RunInitContext) -> None:
        del run_init_ctx

    def __call__(self, sample_ctx: reward_types.SampleContext) -> reward_types.TermResult:
        # extract numeric suffix from identifier "sample_123" -> 123.0
        identifier = sample_ctx.sample_data.identifier
        parts = identifier.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            value = float(int(parts[1]))
        else:
            value = -1.0  # sentinel for unexpected format
        return reward_types.TermResult(value=value)


class TestTRLAdapterLogging:
    """Tests for TRL adapter integration with reward logging."""

    def test_trl_adapter_logs_unique_values_across_batches(self) -> None:
        """Verify that the TRL adapter logs unique values without cyclic repetition.

        This test simulates the pattern used in GRPO training:
        - Multiple calls to the reward function (like TRL would do)
        - Each call has a batch of completions with associated sample_data
        - Frequency gating should log samples at specific generation counts
        - Logged values should be UNIQUE (no cyclic repetition)

        This test was created to diagnose an issue where users reported seeing
        the same values repeat periodically in reward logs during GRPO training.
        """
        # register custom term that returns unique values based on sample index
        registry = reward_registry.RewardRegistry()

        def factory(
            spec: reward_configs.RewardTermSpec,
            *,
            parser: reward_types.OutputParser | None,
        ) -> reward_types.RewardTerm:
            del spec, parser
            return _SampleIndexRewardTerm()

        registry.register_term("sample_index_term", factory)

        # config similar to real GRPO setup
        scalar_log_freq = 10
        logger_obj = reward_logging.InMemoryRewardLogger(
            log_every_n_generations=scalar_log_freq,
        )
        config = reward_configs.RewardManagerConfig(
            terms=[
                reward_configs.RewardTermSpec(
                    name="index",
                    type="sample_index_term",
                    require_parsed=False,  # don't need parsing for this test
                )
            ],
            logging=reward_configs.LoggingConfig(
                enabled=True,
                log_every_n_generations=scalar_log_freq,
                log_tables=False,
            ),
        )
        manager = reward_manager.RewardManager(config, logger=logger_obj, registry=registry)

        # create TRL adapter
        adapter = rewards_trl.TRLRewardAdapter(
            manager=manager,
            sample_data_key="sample_data",
            skip_on_error=False,
        )

        # simulate GRPO-style batching: multiple calls, each with num_generations completions
        num_calls = 40  # number of times TRL would call the reward function
        samples_per_call = 8  # like num_generations in GRPO
        total_samples = num_calls * samples_per_call  # 320

        for call_idx in range(num_calls):
            manager.set_step(call_idx)  # like RewardLoggingCallback would do

            # build TRL-style inputs
            completions: list[list[dict[str, str]]] = []
            sample_data_list: list[dict[str, object]] = []

            for local_idx in range(samples_per_call):
                global_idx = call_idx * samples_per_call + local_idx

                # TRL-style completion (list of message dicts)
                completions.append([{"role": "assistant", "content": f"output_{global_idx}"}])

                # sample_data as dict (like HuggingFace datasets would serialize it)
                sample_data = rewards_conftest.make_sample_data(f"sample_{global_idx}")
                sample_data_list.append(sample_data._asdict())

            # call the adapter like TRL would
            rewards = adapter(
                completions,
                prompts=[f"prompt_{call_idx}_{i}" for i in range(samples_per_call)],
                sample_data=sample_data_list,
            )

            # verify rewards were computed
            assert len(rewards) == samples_per_call
            assert all(r is not None for r in rewards)

        # verify generation count is correct
        assert manager._total_global_generation_count == total_samples

        # verify we logged the expected number of samples
        expected_logged = total_samples // scalar_log_freq
        assert len(logger_obj.samples) == expected_logged, (
            f"expected {expected_logged} logged samples, got {len(logger_obj.samples)}"
        )

        # verify generation counts are correct (10, 20, 30, ...)
        logged_gen_counts = [s["generation_count"] for s in logger_obj.samples]
        expected_gen_counts = list(range(scalar_log_freq, total_samples + 1, scalar_log_freq))
        assert logged_gen_counts == expected_gen_counts, (
            f"generation counts mismatch: {logged_gen_counts[:10]}... vs {expected_gen_counts[:10]}..."
        )

        # CRITICAL: verify logged values are NOT repeating in a cyclic pattern
        logged_rewards = [s["reward_total"] for s in logger_obj.samples]

        def has_cyclic_repetition(values: list[float], min_cycle_len: int = 3) -> tuple[bool, int]:
            """Check if a list has cyclic repetition of a subsequence."""
            n = len(values)
            for cycle_len in range(min_cycle_len, n // 2 + 1):
                is_cyclic = True
                for i in range(cycle_len, n):
                    if values[i] != values[i % cycle_len]:
                        is_cyclic = False
                        break
                if is_cyclic:
                    return True, cycle_len
            return False, 0

        has_cycle, cycle_len = has_cyclic_repetition(logged_rewards)
        assert not has_cycle, (
            f"CRITICAL: logged rewards show cyclic repetition with period {cycle_len}! "
            f"First {min(20, len(logged_rewards))} values: {logged_rewards[:20]}"
        )

        # verify that all logged rewards are unique
        assert len(set(logged_rewards)) == len(logged_rewards), (
            f"logged rewards should all be unique, but found duplicates: {logged_rewards[:20]}..."
        )

        # verify that rewards match expected values based on sample index
        # sample at generation_count G has index G-1, so reward = G-1
        for entry in logger_obj.samples:
            gen_count = entry["generation_count"]
            sample_index = gen_count - 1  # 0-indexed (generation_count is 1-indexed)
            expected_reward = float(sample_index)
            actual_reward = entry["reward_total"]
            assert actual_reward == pytest.approx(expected_reward), (
                f"at gen_count={gen_count}, expected reward {expected_reward}, got {actual_reward}"
            )

    def test_trl_adapter_accepts_sample_data_objects(self) -> None:
        registry = reward_registry.RewardRegistry()

        def factory(
            spec: reward_configs.RewardTermSpec,
            *,
            parser: reward_types.OutputParser | None,
        ) -> reward_types.RewardTerm:
            del spec, parser
            return _SampleIndexRewardTerm()

        registry.register_term("sample_index_term_objects", factory)
        config = reward_configs.RewardManagerConfig(
            terms=[
                reward_configs.RewardTermSpec(
                    name="index",
                    type="sample_index_term_objects",
                    require_parsed=False,
                )
            ],
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = reward_manager.RewardManager(config, registry=registry)
        adapter = rewards_trl.TRLRewardAdapter(
            manager=manager,
            sample_data_key="sample_data",
            skip_on_error=False,
        )

        completions = [
            [{"role": "assistant", "content": "output_0"}],
            [{"role": "assistant", "content": "output_1"}],
            [{"role": "assistant", "content": "output_2"}],
        ]
        sample_data_list = [
            rewards_conftest.make_sample_data("sample_0"),
            rewards_conftest.make_sample_data("sample_1"),
            rewards_conftest.make_sample_data("sample_2"),
        ]
        prompts = ["prompt_0", "prompt_1", "prompt_2"]

        rewards = adapter(completions, prompts=prompts, sample_data=sample_data_list)
        assert rewards == [0.0, 1.0, 2.0]

    def test_trl_adapter_with_train_eval_prefix_switching(self) -> None:
        """Verify that prefix switching during train/eval doesn't cause value repetition.

        This simulates the RewardLoggingCallback behavior of switching prefixes
        between train and eval phases.
        """
        registry = reward_registry.RewardRegistry()

        def factory(
            spec: reward_configs.RewardTermSpec,
            *,
            parser: reward_types.OutputParser | None,
        ) -> reward_types.RewardTerm:
            del spec, parser
            return _SampleIndexRewardTerm()

        registry.register_term("sample_index_term_v2", factory)

        scalar_log_freq = 5
        logger_obj = reward_logging.InMemoryRewardLogger(
            log_every_n_generations=scalar_log_freq,
        )
        config = reward_configs.RewardManagerConfig(
            terms=[
                reward_configs.RewardTermSpec(
                    name="index",
                    type="sample_index_term_v2",
                    require_parsed=False,
                )
            ],
            logging=reward_configs.LoggingConfig(
                enabled=True,
                log_every_n_generations=scalar_log_freq,
                log_tables=False,
            ),
        )
        manager = reward_manager.RewardManager(config, logger=logger_obj, registry=registry)
        adapter = rewards_trl.TRLRewardAdapter(
            manager=manager,
            sample_data_key="sample_data",
            skip_on_error=False,
        )

        sample_counter = 0

        def make_batch(size: int) -> tuple[list[list[dict[str, str]]], list[dict[str, object]], list[str]]:
            nonlocal sample_counter
            completions = []
            sample_data_list = []
            prompts = []
            for _ in range(size):
                completions.append([{"role": "assistant", "content": f"output_{sample_counter}"}])
                sample_data = rewards_conftest.make_sample_data(f"sample_{sample_counter}")
                sample_data_list.append(sample_data._asdict())
                prompts.append(f"prompt_{sample_counter}")
                sample_counter += 1
            return completions, sample_data_list, prompts

        # simulate alternating train/eval phases (like RewardLoggingCallback does)
        for step in range(10):
            # train phase
            manager.set_key_prefix("train")
            manager.set_step(step)
            completions, sample_data_list, prompts = make_batch(8)
            adapter(completions, prompts=prompts, sample_data=sample_data_list)

            # eval phase (every 3 steps)
            if step % 3 == 0:
                manager.set_key_prefix("eval")
                manager.set_step(step)
                completions, sample_data_list, prompts = make_batch(4)
                adapter(completions, prompts=prompts, sample_data=sample_data_list)

        # verify no cyclic repetition in logged rewards
        logged_rewards = [s["reward_total"] for s in logger_obj.samples]

        # all rewards should be unique (since each sample has a unique index)
        assert len(set(logged_rewards)) == len(logged_rewards), (
            f"logged rewards should all be unique after train/eval switching, "
            f"but found duplicates. First 20 values: {logged_rewards[:20]}"
        )

        # verify per-phase counters are correct
        # with the new per-phase counter design, generation counts are independent per phase
        # train: 10 batches * 8 samples = 80 samples
        # eval: 4 batches * 4 samples = 16 samples (at steps 0, 3, 6, 9)
        assert manager._global_generation_counts["train/"] == 80
        assert manager._global_generation_counts["eval/"] == 16
        assert manager._total_global_generation_count == 96  # 80 + 16

    def test_batch_count_increments_per_compute_batch_call(self) -> None:
        """Verify that batch_count increments once per compute_batch call, not per sample."""
        registry = reward_registry.RewardRegistry()

        def factory(
            spec: reward_configs.RewardTermSpec,
            *,
            parser: reward_types.OutputParser | None,
        ) -> reward_types.RewardTerm:
            del spec, parser
            return _SampleIndexRewardTerm()

        registry.register_term("sample_index_term_batch", factory)
        logger_obj = reward_logging.InMemoryRewardLogger(log_every_n_generations=1)
        config = reward_configs.RewardManagerConfig(
            terms=[
                reward_configs.RewardTermSpec(
                    name="index",
                    type="sample_index_term_batch",
                    require_parsed=False,
                )
            ],
            logging=reward_configs.LoggingConfig(
                enabled=True,
                log_batch_stats=True,
                log_every_n_generations=1,
                log_tables=False,
            ),
        )
        manager = reward_manager.RewardManager(config, logger=logger_obj, registry=registry)
        adapter = rewards_trl.TRLRewardAdapter(
            manager=manager,
            sample_data_key="sample_data",
            skip_on_error=False,
        )
        num_calls = 10
        samples_per_call = 4
        for call_idx in range(num_calls):
            manager.set_step(call_idx)
            completions = []
            sample_data_list = []
            for local_idx in range(samples_per_call):
                global_idx = call_idx * samples_per_call + local_idx
                completions.append([{"role": "assistant", "content": f"output_{global_idx}"}])
                sample_data = rewards_conftest.make_sample_data(f"sample_{global_idx}")
                sample_data_list.append(sample_data._asdict())
            adapter(
                completions,
                prompts=[f"prompt_{call_idx}_{idx}" for idx in range(samples_per_call)],
                sample_data=sample_data_list,
            )
        # batch_count should equal number of compute_batch calls
        assert manager._total_global_batch_count == num_calls, (
            f"expected batch_count={num_calls}, got {manager._total_global_batch_count}"
        )
        # generation_count should equal total samples processed
        total_samples = num_calls * samples_per_call
        assert manager._total_global_generation_count == total_samples, (
            f"expected generation_count={total_samples}, got {manager._total_global_generation_count}"
        )
        # verify batch_stats were logged with correct batch_count values
        logged_batch_counts = [s["batch_count"] for s in logger_obj.batch_stats]
        expected_batch_counts = list(range(1, num_calls + 1))
        assert logged_batch_counts == expected_batch_counts, (
            f"batch_counts mismatch: {logged_batch_counts} vs {expected_batch_counts}"
        )

    def test_multiple_batches_at_same_step_get_unique_batch_counts(self) -> None:
        """Simulate gradient accumulation: multiple compute_batch calls at the same trainer step.

        Each call should get a unique batch_count even though step is the same.
        This prevents WandB aggregation issues.
        """
        registry = reward_registry.RewardRegistry()

        def factory(
            spec: reward_configs.RewardTermSpec,
            *,
            parser: reward_types.OutputParser | None,
        ) -> reward_types.RewardTerm:
            del spec, parser
            return _SampleIndexRewardTerm()

        registry.register_term("sample_index_term_accum", factory)
        logger_obj = reward_logging.InMemoryRewardLogger(log_every_n_generations=1)
        config = reward_configs.RewardManagerConfig(
            terms=[
                reward_configs.RewardTermSpec(
                    name="index",
                    type="sample_index_term_accum",
                    require_parsed=False,
                )
            ],
            logging=reward_configs.LoggingConfig(
                enabled=True,
                log_batch_stats=True,
                log_every_n_generations=1,
                log_tables=False,
            ),
        )
        manager = reward_manager.RewardManager(config, logger=logger_obj, registry=registry)
        adapter = rewards_trl.TRLRewardAdapter(
            manager=manager,
            sample_data_key="sample_data",
            skip_on_error=False,
        )
        # simulate gradient accumulation: 4 batches per optimizer step
        gradient_accumulation_steps = 4
        num_optimizer_steps = 5
        samples_per_micro_batch = 2
        sample_counter = 0
        for optimizer_step in range(num_optimizer_steps):
            manager.set_step(optimizer_step)  # same step for all batches
            for _micro_batch_idx in range(gradient_accumulation_steps):
                completions = []
                sample_data_list = []
                for _ in range(samples_per_micro_batch):
                    completions.append([{"role": "assistant", "content": f"output_{sample_counter}"}])
                    sample_data = rewards_conftest.make_sample_data(f"sample_{sample_counter}")
                    sample_data_list.append(sample_data._asdict())
                    sample_counter += 1
                adapter(
                    completions,
                    prompts=[f"prompt_{sample_counter}"] * samples_per_micro_batch,
                    sample_data=sample_data_list,
                )
        total_batches = num_optimizer_steps * gradient_accumulation_steps
        assert manager._total_global_batch_count == total_batches
        # verify all batch_counts are unique (no duplicates from same step)
        logged_batch_counts = [s["batch_count"] for s in logger_obj.batch_stats]
        assert len(set(logged_batch_counts)) == len(logged_batch_counts), (
            f"batch_counts should all be unique, but found duplicates: {logged_batch_counts}"
        )
        # verify batch_counts are strictly increasing
        for idx in range(1, len(logged_batch_counts)):
            assert logged_batch_counts[idx] > logged_batch_counts[idx - 1], "batch_counts should be strictly increasing"
