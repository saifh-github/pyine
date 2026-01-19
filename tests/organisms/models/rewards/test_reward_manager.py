import pytest

import pyine.evals.utils
import pyine.organisms.models.rewards.core.configs
import pyine.organisms.models.rewards.core.logging
import pyine.organisms.models.rewards.core.manager
import pyine.organisms.models.rewards.core.registry
import pyine.organisms.models.rewards.core.types
import tests.organisms.models.rewards.conftest as rewards_conftest


class _NeedsParsedTerm:
    """Test helper term that asserts `parsed` is available and returns 1.0."""

    def reset(
        self,
        run_init_ctx: pyine.organisms.models.rewards.core.types.RunInitContext,
    ) -> None:
        del run_init_ctx

    def __call__(
        self,
        sample_ctx: pyine.organisms.models.rewards.core.types.SampleContext,
    ) -> pyine.organisms.models.rewards.core.types.TermResult:
        assert sample_ctx.parsed is not None, "expected parsed to be populated"
        return pyine.organisms.models.rewards.core.types.TermResult(value=1.0)


class _SampleIdSuffixAsFloatTerm:
    """Test helper term that returns the last digit of the sample id as a float."""

    def reset(
        self,
        run_init_ctx: pyine.organisms.models.rewards.core.types.RunInitContext,
    ) -> None:
        del run_init_ctx

    def __call__(
        self,
        sample_ctx: pyine.organisms.models.rewards.core.types.SampleContext,
    ) -> pyine.organisms.models.rewards.core.types.TermResult:
        suffix = sample_ctx.sample_id[-1]
        return pyine.organisms.models.rewards.core.types.TermResult(value=float(int(suffix)))


class _SampleIndexAsFloatTerm:
    """Test helper term that extracts the numeric index from sample_id and returns it as reward.

    Expects sample_id format like "sample_123" and returns 123.0.
    """

    def reset(
        self,
        run_init_ctx: pyine.organisms.models.rewards.core.types.RunInitContext,
    ) -> None:
        del run_init_ctx

    def __call__(
        self,
        sample_ctx: pyine.organisms.models.rewards.core.types.SampleContext,
    ) -> pyine.organisms.models.rewards.core.types.TermResult:
        # extract numeric suffix from "sample_123" -> 123
        parts = sample_ctx.sample_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            value = float(int(parts[1]))
        else:
            value = 0.0
        return pyine.organisms.models.rewards.core.types.TermResult(value=value)


class TestRewardManager:
    def test_weighting_and_breakdown(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    weight=2.0,
                    params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        model_output = "<final>ok</final>"
        sample_ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output=model_output,
            sample_data=rewards_conftest.make_sample_data("s1"),
            parsed=rewards_conftest.make_parsed_output(model_output, final_answer="ok"),
        )
        output = manager.compute(sample_ctx)
        assert output.total == 2.0
        assert output.weighted_terms == {"parseable": 2.0}

    def test_parseable_answer_qol_clean_stop_and_single_block(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    params={
                        "final_tag": "final",
                        "reward_if_present": 1.0,
                        "reward_if_missing": 0.0,
                        "bonus_if_stops_after_final_tag": 0.5,
                        "bonus_if_single_final_block": 0.25,
                    },
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        model_output = "<final>ok</final>\n"
        sample_ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output=model_output,
            sample_data=rewards_conftest.make_sample_data("s1"),
            parsed=rewards_conftest.make_parsed_output(model_output, final_answer="ok"),
        )
        output = manager.compute(sample_ctx, log=False)
        assert output.total == 1.75
        assert output.metrics["parseable/has_final_answer"] == 1
        assert output.metrics["parseable/has_single_final_block"] == 1
        assert output.metrics["parseable/stops_after_final_tag"] == 1

    def test_parseable_answer_qol_trailing_text_no_penalty(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    params={
                        "final_tag": "final",
                        "reward_if_present": 1.0,
                        "reward_if_missing": 0.0,
                        "bonus_if_stops_after_final_tag": 0.5,
                        "bonus_if_single_final_block": 0.25,
                    },
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        model_output = "<final>ok</final> trailing"
        sample_ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output=model_output,
            sample_data=rewards_conftest.make_sample_data("s1"),
            parsed=rewards_conftest.make_parsed_output(model_output, final_answer="ok"),
        )
        output = manager.compute(sample_ctx, log=False)
        assert output.total == 1.25
        assert output.metrics["parseable/stops_after_final_tag"] == 0

    def test_prepopulated_parsed_is_shared_across_terms(self) -> None:
        """Verify that pre-populated parsed data is accessible to all terms."""
        registry = pyine.organisms.models.rewards.core.registry.RewardRegistry()

        def factory(
            spec: pyine.organisms.models.rewards.core.configs.RewardTermSpec,
            *,
            parser: pyine.organisms.models.rewards.core.types.OutputParser | None,
        ) -> pyine.organisms.models.rewards.core.types.RewardTerm:
            del spec
            del parser
            return _NeedsParsedTerm()

        registry.register_term("test_needs_parsed", factory)
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t1", type="test_needs_parsed"),
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t2", type="test_needs_parsed"),
            ],
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, registry=registry)
        model_output = "raw"
        sample_ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output=model_output,
            sample_data=rewards_conftest.make_sample_data("s2"),
            parsed=rewards_conftest.make_parsed_output(model_output, final_answer="x"),
        )
        output = manager.compute(sample_ctx, log=False)
        assert output.total == 2.0

    def test_logging_frequency_and_scoping(self) -> None:
        logging_config = pyine.organisms.models.rewards.core.configs.LoggingConfig(
            enabled=True,
            scalar_log_every_n_generations=2,
        )
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger(
            scalar_log_every_n_generations=2,
        )
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=logging_config,
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger_obj)
        sample1 = manager.build_sample_context(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        sample2 = manager.build_sample_context(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s2"),
        )
        manager.compute(sample1)
        manager.compute(sample2)
        assert len(logger_obj.samples) == 1
        entry = logger_obj.samples[0]
        terms = entry["reward_terms"]
        assert isinstance(terms, dict)
        assert "reward/terms/parseable" in terms
        manager.finalize_run()
        assert len(logger_obj.runs) == 1

    def test_generation_idx_is_computed_per_identifier_group(self) -> None:
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                scalar_log_every_n_generations=1,
                log_tables=False,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger_obj)
        sample_ctxs = [
            rewards_conftest.make_sample_context(
                identifier="prompt_a",
                model_output="<final>a0</final>",
                parsed=rewards_conftest.make_parsed_output("<final>a0</final>", final_answer="a0"),
            ),
            rewards_conftest.make_sample_context(
                identifier="prompt_a",
                model_output="<final>a1</final>",
                parsed=rewards_conftest.make_parsed_output("<final>a1</final>", final_answer="a1"),
            ),
            rewards_conftest.make_sample_context(
                identifier="prompt_b",
                model_output="<final>b0</final>",
                parsed=rewards_conftest.make_parsed_output("<final>b0</final>", final_answer="b0"),
            ),
            rewards_conftest.make_sample_context(
                identifier="prompt_a",
                model_output="<final>a2</final>",
                parsed=rewards_conftest.make_parsed_output("<final>a2</final>", final_answer="a2"),
            ),
        ]
        manager.compute_batch(sample_ctxs)
        assert [e["generation_idx"] for e in logger_obj.samples] == [0, 1, 0, 2]

    def test_batch_stats_are_computed_and_logged(self) -> None:
        registry = pyine.organisms.models.rewards.core.registry.RewardRegistry()

        def factory(
            spec: pyine.organisms.models.rewards.core.configs.RewardTermSpec,
            *,
            parser: pyine.organisms.models.rewards.core.types.OutputParser | None,
        ) -> pyine.organisms.models.rewards.core.types.RewardTerm:
            del spec
            del parser
            return _SampleIdSuffixAsFloatTerm()

        registry.register_term("test_sample_id_suffix_for_batch_stats", factory)
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="id",
                    type="test_sample_id_suffix_for_batch_stats",
                )
            ],
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                scalar_log_every_n_generations=1,
                log_tables=False,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(
            config,
            logger=logger_obj,
            registry=registry,
        )
        manager.set_step(10)
        manager.compute_batch(
            [
                rewards_conftest.make_sample_context(identifier="s1"),
                rewards_conftest.make_sample_context(identifier="s2"),
            ]
        )
        manager.compute_batch(
            [
                rewards_conftest.make_sample_context(identifier="s3"),
                rewards_conftest.make_sample_context(identifier="s4"),
            ]
        )
        assert len(logger_obj.batch_stats) == 2
        first = logger_obj.batch_stats[0]
        assert first["batch_mean"] == pytest.approx(1.5)
        assert first["batch_std"] == pytest.approx(0.5)
        assert first["batch_mean_rolling_mean"] == pytest.approx(1.5)
        assert first["batch_mean_rolling_std"] == pytest.approx(0.0)
        assert first["batch_std_rolling_mean"] == pytest.approx(0.5)
        assert first["batch_std_rolling_std"] == pytest.approx(0.0)

        second = logger_obj.batch_stats[1]
        assert second["batch_mean"] == pytest.approx(3.5)
        assert second["batch_std"] == pytest.approx(0.5)
        assert second["batch_mean_rolling_mean"] == pytest.approx(2.5)
        assert second["batch_mean_rolling_std"] == pytest.approx(1.0)
        assert second["batch_std_rolling_mean"] == pytest.approx(0.5)
        assert second["batch_std_rolling_std"] == pytest.approx(0.0)

    def test_state_includes_monotonic_and_batch_stats(self) -> None:
        registry = pyine.organisms.models.rewards.core.registry.RewardRegistry()

        def factory(
            spec: pyine.organisms.models.rewards.core.configs.RewardTermSpec,
            *,
            parser: pyine.organisms.models.rewards.core.types.OutputParser | None,
        ) -> pyine.organisms.models.rewards.core.types.RewardTerm:
            del spec
            del parser
            return _SampleIdSuffixAsFloatTerm()

        registry.register_term("test_sample_id_suffix_for_state", factory)
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="id",
                    type="test_sample_id_suffix_for_state",
                )
            ],
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                scalar_log_every_n_generations=1,
                log_tables=False,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(
            config,
            logger=logger_obj,
            registry=registry,
        )
        manager.compute_batch(
            [
                rewards_conftest.make_sample_context(identifier="s1"),
                rewards_conftest.make_sample_context(identifier="s2"),
            ]
        )
        state = manager.get_state()
        assert state["monotonic_generation_count"] == 2
        assert state["batch_reward_mean_stats"]["count"] == 1
        assert state["batch_reward_std_stats"]["count"] == 1

    def test_table_rows_include_reward_breakdown_when_scalars_gated_off(self) -> None:
        logging_config = pyine.organisms.models.rewards.core.configs.LoggingConfig(
            enabled=True,
            scalar_log_every_n_generations=9999,  # scalars should be skipped
            log_tables=True,
            table_row_every_n_generations=1,  # row should be added for the sample
        )
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger(
            scalar_log_every_n_generations=9999,
            log_tables=True,
            table_row_every_n_generations=1,
        )
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=logging_config,
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger_obj)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        manager.compute(ctx)
        assert logger_obj.samples == []
        assert len(logger_obj.table_rows) == 1
        row = logger_obj.table_rows[0]
        assert row["reward_total"] == pytest.approx(1.0)
        terms = row["reward_terms"]
        assert isinstance(terms, dict)
        assert "reward/terms/parseable" in terms

    def test_step_is_propagated_to_logger(self) -> None:
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                scalar_log_every_n_generations=1,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger_obj)
        manager.set_step(123)
        sample = manager.build_sample_context(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        manager.compute(sample)
        assert logger_obj.samples[0]["step"] == 123
        manager.compute(sample, step=7)
        assert logger_obj.samples[1]["step"] == 7

    def test_grpo_style_batching_logs_unique_values_without_repetition(self) -> None:
        """Verify that logging frequency gating produces unique values across GRPO-style batches.

        This test simulates the batch pattern used in GRPO training:
        - Multiple "training steps", each with multiple batches
        - Each batch has num_generations samples (like GRPO)
        - Frequency gating should log samples at specific generation counts
        - Logged values should be UNIQUE (no cyclic repetition)

        This test was added to diagnose an issue where users reported seeing
        the same values repeat periodically in reward logs.
        """
        # use a term that returns unique values based on sample index (sample_123 -> 123.0)
        registry = pyine.organisms.models.rewards.core.registry.RewardRegistry()

        def factory(
            spec: pyine.organisms.models.rewards.core.configs.RewardTermSpec,
            *,
            parser: pyine.organisms.models.rewards.core.types.OutputParser | None,
        ) -> pyine.organisms.models.rewards.core.types.RewardTerm:
            del spec, parser
            return _SampleIndexAsFloatTerm()

        registry.register_term("sample_index_term", factory)

        # config similar to real GRPO setup: log every 10 samples for faster testing
        scalar_log_freq = 10
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger(
            scalar_log_every_n_generations=scalar_log_freq,
        )
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="index",
                    type="sample_index_term",
                )
            ],
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                scalar_log_every_n_generations=scalar_log_freq,
                log_tables=False,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(
            config, logger=logger_obj, registry=registry
        )

        # simulate GRPO-style batching: 10 training steps, 4 batches per step, 8 samples per batch
        num_training_steps = 10
        micro_batches_per_step = 4
        samples_per_micro_batch = 8  # like num_generations in GRPO
        total_samples = num_training_steps * micro_batches_per_step * samples_per_micro_batch  # 320

        sample_idx = 0
        for step in range(num_training_steps):
            manager.set_step(step)
            for _micro_batch in range(micro_batches_per_step):
                batch_ctxs = []
                for _sample in range(samples_per_micro_batch):
                    # use sample_idx as identifier suffix so reward = sample_idx
                    ctx = rewards_conftest.make_sample_context(identifier=f"sample_{sample_idx}")
                    batch_ctxs.append(ctx)
                    sample_idx += 1
                manager.compute_batch(batch_ctxs)

        # verify generation count is correct
        assert manager._monotonic_generation_count == total_samples

        # verify we logged the expected number of samples (every scalar_log_freq-th sample)
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

        # with _SampleIndexAsFloatTerm, each logged sample should have a unique reward
        # (since each sample_idx is unique and reward = sample_idx)
        # if there's cyclic repetition, it indicates stale values being logged
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

        # verify that all logged rewards are unique (no duplicates)
        assert len(set(logged_rewards)) == len(logged_rewards), (
            f"logged rewards should all be unique, but found duplicates: {logged_rewards[:20]}..."
        )

        # verify that rewards match expected values based on sample index
        # sample at generation_count G has index G-1, so reward = G-1
        for entry in logger_obj.samples:
            gen_count = entry["generation_count"]
            sample_index = gen_count - 1  # 0-indexed (generation_count is 1-indexed)
            expected_reward = float(sample_index)  # _SampleIndexAsFloatTerm returns the full index
            actual_reward = entry["reward_total"]
            assert actual_reward == pytest.approx(expected_reward), (
                f"at gen_count={gen_count}, expected reward {expected_reward}, got {actual_reward}"
            )

    def test_introspection_term_names_and_get_term(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t1", type="parseable_answer"),
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="t2",
                    type="parseable_answer",
                    enabled=False,
                ),
            ],
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        assert manager.term_names == ("t1", "t2")
        assert manager.enabled_term_names == ("t1",)
        assert manager.get_term("t1") is not None
        with pytest.raises(KeyError, match="disabled"):
            manager.get_term("t2")

    def test_require_parsed_enforced(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    require_parsed=True,
                )
            ],
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        with pytest.raises(ValueError, match="require parsed"):
            pyine.organisms.models.rewards.core.manager.RewardManager(config)

    def test_require_parsed_enforced_at_runtime(self) -> None:
        """Verify that compute raises if parsed=None but require_parsed=True."""
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    require_parsed=True,
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        sample_ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
            parsed=None,  # explicitly None, bypassing build_sample_context
        )
        with pytest.raises(ValueError, match="parsed is None.*require parsed.*build_sample_context"):
            manager.compute(sample_ctx)

    def test_registry_snapshot_supports_aliases(self) -> None:
        import pyine.organisms.models.rewards.terms

        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        snapshot = pyine.organisms.models.rewards.core.registry.get_global_registry().snapshot()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="format/parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, registry=snapshot)
        model_output = "<final>ok</final>"
        sample_ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output=model_output,
            sample_data=rewards_conftest.make_sample_data("s1"),
            parsed=rewards_conftest.make_parsed_output(model_output, final_answer="ok"),
        )
        assert manager.compute(sample_ctx).total == 1.0

    def test_no_enabled_terms_raises(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="x",
                    type="parseable_answer",
                    enabled=False,
                )
            ],
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        sample_ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        with pytest.raises(ValueError, match="no enabled"):
            manager.compute(sample_ctx)

    def test_unknown_term_type_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown key"):
            pyine.organisms.models.rewards.core.manager.RewardManager(
                pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
                    terms=[pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="x", type="does_not_exist")],
                    logging=rewards_conftest.make_disabled_logging_config(),
                )
            )

    def test_logging_enabled_without_logger_raises(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="x", type="parseable_answer")],
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(enabled=True),
        )
        with pytest.raises(ValueError, match="logging is enabled.*but no logger"):
            pyine.organisms.models.rewards.core.manager.RewardManager(config)

    def test_tag_inconsistency_warning(self) -> None:
        import warnings

        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    params={"final_tag": "answer"},  # different from parser's "final"
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(final_tag="final"),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            pyine.organisms.models.rewards.core.manager.RewardManager(config)
        assert len(recorded) == 1
        assert "final_tag='answer'" in str(recorded[0].message)
        assert "parser uses final_tag='final'" in str(recorded[0].message)

    def test_compute_batch_preserves_input_order(self) -> None:
        registry = pyine.organisms.models.rewards.core.registry.RewardRegistry()

        def factory(
            spec: pyine.organisms.models.rewards.core.configs.RewardTermSpec,
            *,
            parser: pyine.organisms.models.rewards.core.types.OutputParser | None,
        ) -> pyine.organisms.models.rewards.core.types.RewardTerm:
            del spec
            del parser
            return _SampleIdSuffixAsFloatTerm()

        registry.register_term("test_sample_id_suffix", factory)
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="id",
                    type="test_sample_id_suffix",
                )
            ],
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, registry=registry)
        sample1 = rewards_conftest.make_sample_context(identifier="s1")
        sample2 = rewards_conftest.make_sample_context(identifier="s2")
        outputs = manager.compute_batch([sample2, sample1], log=False)
        assert [out.total for out in outputs] == [2.0, 1.0]

    def test_logging_toggles_control_logged_payload(self) -> None:
        class _MetricsTerm:
            def reset(
                self,
                run_init_ctx: pyine.organisms.models.rewards.core.types.RunInitContext,
            ) -> None:
                del run_init_ctx

            def __call__(
                self,
                sample_ctx: pyine.organisms.models.rewards.core.types.SampleContext,
            ) -> pyine.organisms.models.rewards.core.types.TermResult:
                del sample_ctx
                return pyine.organisms.models.rewards.core.types.TermResult(
                    value=1.0,
                    metrics={"flag": True, "count": 3},
                )

        registry = pyine.organisms.models.rewards.core.registry.RewardRegistry()

        def factory(
            spec: pyine.organisms.models.rewards.core.configs.RewardTermSpec,
            *,
            parser: pyine.organisms.models.rewards.core.types.OutputParser | None,
        ) -> pyine.organisms.models.rewards.core.types.RewardTerm:
            del spec
            del parser
            return _MetricsTerm()

        registry.register_term("test_metrics", factory)
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t", type="test_metrics")],
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                scalar_log_every_n_generations=1,
                log_total=False,
                log_terms=False,
                log_metrics=False,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(
            config, logger=logger_obj, registry=registry
        )
        ctx = rewards_conftest.make_sample_context(identifier="s1")
        manager.compute(ctx)
        assert len(logger_obj.samples) == 1
        entry = logger_obj.samples[0]
        assert entry["reward_total"] is None
        assert entry["reward_terms"] == {}
        assert entry["reward_metrics"] == {}

    def test_finalize_run_scopes_run_summaries(self) -> None:
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="parseable", type="parseable_answer")
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                scalar_log_every_n_generations=9999,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger_obj)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        manager.compute(ctx, log=False)
        manager.finalize_run()
        assert len(logger_obj.runs) == 1
        run_entry = logger_obj.runs[0]
        reward_totals = run_entry["reward_totals"]
        reward_term_summaries = run_entry["reward_term_summaries"]
        assert isinstance(reward_totals, dict)
        assert isinstance(reward_term_summaries, dict)
        # totals: reward/run/total/{metric_key}
        assert "reward/run/total/mean" in reward_totals
        # terms: reward/run/terms/{metric_key}
        assert "reward/run/terms/parseable/mean" in reward_term_summaries

    def test_invalid_term_factory_signature_raises(self) -> None:
        registry = pyine.organisms.models.rewards.core.registry.RewardRegistry()

        def factory(
            spec: pyine.organisms.models.rewards.core.configs.RewardTermSpec,
        ) -> pyine.organisms.models.rewards.core.types.RewardTerm:
            del spec
            return _NeedsParsedTerm()

        registry.register_term("bad_signature", factory)  # type: ignore[arg-type]
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t", type="bad_signature")],
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        with pytest.raises(TypeError, match="invalid term factory signature"):
            pyine.organisms.models.rewards.core.manager.RewardManager(config, registry=registry)

    def test_term_result_validation_rejects_invalid_outputs(self) -> None:
        registry = pyine.organisms.models.rewards.core.registry.RewardRegistry()

        class _BadValueTerm:
            def reset(
                self,
                run_init_ctx: pyine.organisms.models.rewards.core.types.RunInitContext,
            ) -> None:
                del run_init_ctx

            def __call__(
                self,
                sample_ctx: pyine.organisms.models.rewards.core.types.SampleContext,
            ) -> pyine.organisms.models.rewards.core.types.TermResult:
                del sample_ctx
                return pyine.organisms.models.rewards.core.types.TermResult(value=True)

        def factory(
            spec: pyine.organisms.models.rewards.core.configs.RewardTermSpec,
            *,
            parser: pyine.organisms.models.rewards.core.types.OutputParser | None,
        ) -> pyine.organisms.models.rewards.core.types.RewardTerm:
            del spec
            del parser
            return _BadValueTerm()

        registry.register_term("bad_value", factory)
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t", type="bad_value")],
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, registry=registry)
        ctx = rewards_conftest.make_sample_context(identifier="s1")
        with pytest.raises(TypeError, match="non-numeric"):
            manager.compute(ctx, log=False)

    def test_term_result_validation_rejects_non_finite_values_and_metrics(self) -> None:
        registry = pyine.organisms.models.rewards.core.registry.RewardRegistry()

        class _NonFiniteTerm:
            def __init__(self, *, emit_bad_metric: bool) -> None:
                self._emit_bad_metric = emit_bad_metric

            def reset(
                self,
                run_init_ctx: pyine.organisms.models.rewards.core.types.RunInitContext,
            ) -> None:
                del run_init_ctx

            def __call__(
                self,
                sample_ctx: pyine.organisms.models.rewards.core.types.SampleContext,
            ) -> pyine.organisms.models.rewards.core.types.TermResult:
                del sample_ctx
                if self._emit_bad_metric:
                    return pyine.organisms.models.rewards.core.types.TermResult(
                        value=1.0,
                        metrics={"m": float("nan")},
                    )
                return pyine.organisms.models.rewards.core.types.TermResult(value=float("nan"))

        def factory_value(
            spec: pyine.organisms.models.rewards.core.configs.RewardTermSpec,
            *,
            parser: pyine.organisms.models.rewards.core.types.OutputParser | None,
        ) -> pyine.organisms.models.rewards.core.types.RewardTerm:
            del spec
            del parser
            return _NonFiniteTerm(emit_bad_metric=False)

        def factory_metric(
            spec: pyine.organisms.models.rewards.core.configs.RewardTermSpec,
            *,
            parser: pyine.organisms.models.rewards.core.types.OutputParser | None,
        ) -> pyine.organisms.models.rewards.core.types.RewardTerm:
            del spec
            del parser
            return _NonFiniteTerm(emit_bad_metric=True)

        registry.register_term("nan_value", factory_value)
        registry.register_term("nan_metric", factory_metric)
        ctx = rewards_conftest.make_sample_context(identifier="s1")
        manager_value = pyine.organisms.models.rewards.core.manager.RewardManager(
            pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
                terms=[pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t", type="nan_value")],
                logging=rewards_conftest.make_disabled_logging_config(),
            ),
            registry=registry,
        )
        with pytest.raises(ValueError, match="non-finite value"):
            manager_value.compute(ctx, log=False)

        manager_metric = pyine.organisms.models.rewards.core.manager.RewardManager(
            pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
                terms=[pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t", type="nan_metric")],
                logging=rewards_conftest.make_disabled_logging_config(),
            ),
            registry=registry,
        )
        with pytest.raises(ValueError, match="non-finite"):
            manager_metric.compute(ctx, log=False)

    def test_term_result_validation_rejects_invalid_metric_keys_and_types(self) -> None:
        registry = pyine.organisms.models.rewards.core.registry.RewardRegistry()

        class _BadMetricTerm:
            def __init__(
                self,
                metrics: dict[object, object],
            ) -> None:
                self._metrics = metrics

            def reset(
                self,
                run_init_ctx: pyine.organisms.models.rewards.core.types.RunInitContext,
            ) -> None:
                del run_init_ctx

            def __call__(
                self,
                sample_ctx: pyine.organisms.models.rewards.core.types.SampleContext,
            ) -> pyine.organisms.models.rewards.core.types.TermResult:
                del sample_ctx
                return pyine.organisms.models.rewards.core.types.TermResult(
                    value=1.0,
                    metrics=self._metrics,  # type: ignore[arg-type]
                )

        def factory(
            metrics: dict[object, object],
        ) -> pyine.organisms.models.rewards.core.types.RewardTermFactory:
            def inner(
                spec: pyine.organisms.models.rewards.core.configs.RewardTermSpec,
                *,
                parser: pyine.organisms.models.rewards.core.types.OutputParser | None,
            ) -> pyine.organisms.models.rewards.core.types.RewardTerm:
                del spec
                del parser
                return _BadMetricTerm(metrics)

            return inner

        registry.register_term("bad_key", factory({"": 1}))
        registry.register_term("bad_type", factory({"ok": []}))

        ctx = rewards_conftest.make_sample_context(identifier="s1")
        manager_key = pyine.organisms.models.rewards.core.manager.RewardManager(
            pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
                terms=[pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t", type="bad_key")],
                logging=rewards_conftest.make_disabled_logging_config(),
            ),
            registry=registry,
        )
        with pytest.raises(ValueError, match="invalid metric key"):
            manager_key.compute(ctx, log=False)

        manager_type = pyine.organisms.models.rewards.core.manager.RewardManager(
            pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
                terms=[pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t", type="bad_type")],
                logging=rewards_conftest.make_disabled_logging_config(),
            ),
            registry=registry,
        )
        with pytest.raises(TypeError, match="invalid type"):
            manager_type.compute(ctx, log=False)

    def test_long_string_metric_emits_warning_but_is_accepted(self) -> None:
        import warnings

        registry = pyine.organisms.models.rewards.core.registry.RewardRegistry()

        class _LongStringMetricTerm:
            def reset(
                self,
                run_init_ctx: pyine.organisms.models.rewards.core.types.RunInitContext,
            ) -> None:
                del run_init_ctx

            def __call__(
                self,
                sample_ctx: pyine.organisms.models.rewards.core.types.SampleContext,
            ) -> pyine.organisms.models.rewards.core.types.TermResult:
                del sample_ctx
                return pyine.organisms.models.rewards.core.types.TermResult(
                    value=1.0,
                    metrics={"reason": "x" * 501},
                )

        def factory(
            spec: pyine.organisms.models.rewards.core.configs.RewardTermSpec,
            *,
            parser: pyine.organisms.models.rewards.core.types.OutputParser | None,
        ) -> pyine.organisms.models.rewards.core.types.RewardTerm:
            del spec
            del parser
            return _LongStringMetricTerm()

        registry.register_term("long_string_metric", factory)
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[pyine.organisms.models.rewards.core.configs.RewardTermSpec(name="t", type="long_string_metric")],
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, registry=registry)
        ctx = rewards_conftest.make_sample_context(identifier="s1")
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            output = manager.compute(ctx, log=False)
        assert output.total == 1.0
        assert len(recorded) == 1
        assert "exceeds" in str(recorded[0].message)


class TestCategoryWiseRewardTracking:
    def test_category_metrics_accumulate_by_code_type(self) -> None:
        category_config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.code_type}),
        )
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=False,
                category_extraction_config=category_config,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        # sample with code_type="original"
        ctx1 = manager.build_sample_context(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1", code_type="original"),
        )
        # sample with code_type="bugfix"
        ctx2 = manager.build_sample_context(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s2", code_type="bugfix"),
        )
        # sample with code_type="original" again but no final answer (reward=0)
        ctx3 = manager.build_sample_context(
            prompt="p",
            model_output="no final tag here",
            sample_data=rewards_conftest.make_sample_data("s3", code_type="original"),
        )
        manager.compute(ctx1, log=False)
        manager.compute(ctx2, log=False)
        manager.compute(ctx3, log=False)
        metrics = manager.get_reward_category_metrics()
        # original: mean=(1.0 + 0.0) / 2 = 0.5, std=0.5, min=0.0, max=1.0
        # note: getters return bare keys (callers add prefixes)
        assert "code_type/original/mean" in metrics
        assert metrics["code_type/original/mean"] == pytest.approx(0.5)
        assert metrics["code_type/original/std"] == pytest.approx(0.5)
        assert metrics["code_type/original/min"] == pytest.approx(0.0)
        assert metrics["code_type/original/max"] == pytest.approx(1.0)
        assert metrics["code_type/original/sample_count"] == 2.0
        # bugfix: mean=1.0, std=0.0, min=max=1.0
        assert "code_type/bugfix/mean" in metrics
        assert metrics["code_type/bugfix/mean"] == pytest.approx(1.0)
        assert metrics["code_type/bugfix/std"] == pytest.approx(0.0)
        assert metrics["code_type/bugfix/sample_count"] == 1.0

    def test_reset_accumulators_clears_all_stats(self) -> None:
        category_config = pyine.evals.utils.SampleCategoryExtractionConfig(
            enabled_fields=frozenset({pyine.evals.utils.SampleCategoryField.code_type}),
        )
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=False,
                category_extraction_config=category_config,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1", code_type="original"),
        )
        manager.compute(ctx, log=False)
        assert len(manager.get_reward_category_metrics()) > 0
        manager.reset_accumulators()
        assert len(manager.get_reward_category_metrics()) == 0

    def test_category_tracking_disabled_without_config(self) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=False,
                category_extraction_config=None,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1", code_type="original"),
        )
        manager.compute(ctx, log=False)
        # no categories accumulated when config is None
        assert len(manager.get_reward_category_metrics()) == 0

    def test_set_key_prefix_delegates_to_logger(self) -> None:
        logger_obj = pyine.organisms.models.rewards.core.logging.InMemoryRewardLogger()
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(
                enabled=True,
                scalar_log_every_n_generations=1,
            ),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config, logger=logger_obj)
        # set_key_prefix should not raise even if logger doesn't support it
        manager.set_key_prefix("train")
        manager.set_key_prefix("valid")
        manager.set_key_prefix("")


def _make_test_manager(parsing: bool = True) -> pyine.organisms.models.rewards.core.manager.RewardManager:
    """Create a simple manager for tests."""
    config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
        terms=[
            pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                name="format",
                type="parseable_answer",
                weight=1.0,
                require_parsed=parsing,
            )
        ],
        parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(final_tag="final") if parsing else None,
        logging=pyine.organisms.models.rewards.core.configs.LoggingConfig(enabled=False),
    )
    return pyine.organisms.models.rewards.core.manager.RewardManager(config)


class TestBuildSampleContext:
    def test_builds_context_with_parsing(self) -> None:
        manager = _make_test_manager(parsing=True)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        assert ctx.prompt == "p"
        assert ctx.model_output == "<final>ok</final>"
        assert ctx.parsed is not None
        assert ctx.parsed.final_answer == "ok"

    def test_builds_context_without_parsing(self) -> None:
        manager = _make_test_manager(parsing=False)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        assert ctx.parsed is None

    def test_builds_context_with_code_exec_eval(self) -> None:
        manager = _make_test_manager(parsing=True)
        code_exec_eval = pyine.organisms.models.rewards.core.types.CodeExecEvalData(
            expected="42",
            predicted="42",
        )
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<final>42</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
            code_exec_eval=code_exec_eval,
        )
        assert ctx.code_exec_eval is code_exec_eval

    def test_builds_context_with_extras(self) -> None:
        manager = _make_test_manager(parsing=True)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
            extras={"key": "value"},
        )
        assert ctx.extras == {"key": "value"}

    def test_compute_with_build_sample_context(self) -> None:
        manager = _make_test_manager(parsing=True)
        ctx = manager.build_sample_context(
            prompt="p",
            model_output="<final>ok</final>",
            sample_data=rewards_conftest.make_sample_data("s1"),
        )
        output = manager.compute(ctx)
        assert output.total == 1.0


class TestListAvailableTerms:
    def test_returns_term_info_list(self) -> None:
        terms = pyine.organisms.models.rewards.core.registry.list_available_terms()
        assert isinstance(terms, list)
        assert len(terms) >= 2  # at least parseable_answer and text_length

    def test_includes_parseable_answer(self) -> None:
        terms = pyine.organisms.models.rewards.core.registry.list_available_terms()
        canonical_types = [t.canonical_type for t in terms]
        assert "parseable_answer" in canonical_types

    def test_includes_text_length(self) -> None:
        terms = pyine.organisms.models.rewards.core.registry.list_available_terms()
        canonical_types = [t.canonical_type for t in terms]
        assert "text_length" in canonical_types

    def test_term_info_has_aliases(self) -> None:
        terms = pyine.organisms.models.rewards.core.registry.list_available_terms()
        parseable = next(t for t in terms if t.canonical_type == "parseable_answer")
        assert "format/parseable_answer" in parseable.aliases

    def test_term_info_has_docstring(self) -> None:
        terms = pyine.organisms.models.rewards.core.registry.list_available_terms()
        parseable = next(t for t in terms if t.canonical_type == "parseable_answer")
        assert parseable.docstring is not None
        assert "ParseableAnswerTerm" in parseable.docstring


class TestPackageLevelExports:
    def test_rewards_package_exports_rewardmanager(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.RewardManager is pyine.organisms.models.rewards.core.manager.RewardManager

    def test_rewards_package_exports_list_available_terms(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.list_available_terms is pyine.organisms.models.rewards.core.registry.list_available_terms

    def test_rewards_package_exports_samplecontext(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.SampleContext is pyine.organisms.models.rewards.core.types.SampleContext

    def test_rewards_package_exports_terminfo(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.TermInfo is pyine.organisms.models.rewards.core.registry.TermInfo

    def test_rewards_package_exports_runinitcontext(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.RunInitContext is pyine.organisms.models.rewards.core.types.RunInitContext

    def test_rewards_package_exports_parsedoutput(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.ParsedOutput is pyine.organisms.models.rewards.core.types.ParsedOutput

    def test_rewards_package_exports_codeexecevaldata(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.CodeExecEvalData is pyine.organisms.models.rewards.core.types.CodeExecEvalData

    def test_rewards_package_exports_rewardoutput(self) -> None:
        import pyine.organisms.models.rewards as rewards

        assert rewards.RewardOutput is pyine.organisms.models.rewards.core.types.RewardOutput


def _deep_equal_with_nan(
    val1: object,
    val2: object,
) -> bool:
    """Compare two values, treating NaN == NaN as True."""
    import math

    if isinstance(val1, float) and isinstance(val2, float):
        if math.isnan(val1) and math.isnan(val2):
            return True
        return val1 == val2
    if isinstance(val1, dict) and isinstance(val2, dict):
        if val1.keys() != val2.keys():
            return False
        return all(_deep_equal_with_nan(val1[k], val2[k]) for k in val1)
    if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
        if len(val1) != len(val2):
            return False
        return all(_deep_equal_with_nan(v1, v2) for v1, v2 in zip(val1, val2, strict=True))
    return val1 == val2


class TestRewardManagerState:
    def test_state_roundtrip(
        self,
    ) -> None:
        config = pyine.organisms.models.rewards.core.configs.RewardManagerConfig(
            terms=[
                pyine.organisms.models.rewards.core.configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    weight=2.0,
                    params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
                )
            ],
            parsing=pyine.organisms.models.rewards.core.configs.ParsingConfig(fallback_policy="none"),
            logging=rewards_conftest.make_disabled_logging_config(),
        )
        manager = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        manager.set_step(5)
        model_output = "<final>ok</final>"
        sample_ctx = pyine.organisms.models.rewards.core.types.SampleContext(
            prompt="p",
            model_output=model_output,
            sample_data=rewards_conftest.make_sample_data("s1"),
            parsed=rewards_conftest.make_parsed_output(model_output, final_answer="ok"),
        )
        manager.compute(sample_ctx, log=False)
        state = manager.get_state()
        restored = pyine.organisms.models.rewards.core.manager.RewardManager(config)
        restored.load_state(state)
        # use custom comparison to handle NaN values in parsing stats
        assert _deep_equal_with_nan(restored.get_state(), state)
