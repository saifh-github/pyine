"""Tests for verbosity-based reward scaling."""

import math
import warnings

import pytest

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.manager as reward_manager
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.organisms.models.rewards.core.verbosity_scaling as verbosity_scaling
import pyine.organisms.models.rewards.terms
import tests.organisms.models.rewards.conftest as rewards_conftest


def _make_cache(
    token_count: int,
    source: reward_types.LengthSource = reward_types.LengthSource.model_output,
) -> reward_types.TokenCountCache:
    """Create a token count cache with a single source."""
    cache = reward_types.TokenCountCache()
    cache.set(source, token_count)
    return cache


class TestAbsoluteMode:
    """Tests for absolute (threshold-based) scaling mode."""

    def test_no_penalty_below_threshold(self) -> None:
        """Factor should equal max_factor when tokens < threshold."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="absolute",
            decay_type="linear",
            threshold_tokens=100,
            end_tokens=200,
            max_factor=1.0,
            min_factor=0.0,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        assert scaler.compute_absolute_factor(50) == 1.0
        assert scaler.compute_absolute_factor(100) == 1.0  # at threshold, no penalty

    def test_linear_decay_starts_from_max_factor(self) -> None:
        """Linear decay should start at max_factor, not 1.0."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="absolute",
            decay_type="linear",
            threshold_tokens=100,
            end_tokens=200,
            max_factor=0.8,  # not 1.0
            min_factor=0.2,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        factor_at_threshold = scaler.compute_absolute_factor(100)
        assert factor_at_threshold == 0.8

    def test_linear_decay_reaches_min_factor_at_end(self) -> None:
        """Factor should reach min_factor at end_tokens."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="absolute",
            decay_type="linear",
            threshold_tokens=100,
            end_tokens=200,
            max_factor=1.0,
            min_factor=0.1,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        factor_at_end = scaler.compute_absolute_factor(200)
        assert factor_at_end == pytest.approx(0.1)
        # beyond end_tokens should stay at min_factor
        assert scaler.compute_absolute_factor(300) == pytest.approx(0.1)

    def test_linear_interpolation_midpoint(self) -> None:
        """Factor at midpoint should be (max_factor + min_factor) / 2."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="absolute",
            decay_type="linear",
            threshold_tokens=100,
            end_tokens=200,
            max_factor=1.0,
            min_factor=0.0,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        midpoint = (100 + 200) // 2  # 150
        factor = scaler.compute_absolute_factor(midpoint)
        assert factor == pytest.approx(0.5)

    def test_exponential_decay_starts_from_max_factor(self) -> None:
        """Exponential: factor = max_factor * exp(-k * excess)."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="absolute",
            decay_type="exponential",
            threshold_tokens=100,
            end_tokens=1000,  # not used for exponential
            decay_rate=0.01,
            max_factor=0.9,
            min_factor=0.1,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        factor_at_threshold = scaler.compute_absolute_factor(100)
        assert factor_at_threshold == 0.9

    def test_exponential_decay_decreases(self) -> None:
        """Exponential decay should decrease as tokens increase."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="absolute",
            decay_type="exponential",
            threshold_tokens=100,
            decay_rate=0.01,
            max_factor=1.0,
            min_factor=0.0,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        factor_100 = scaler.compute_absolute_factor(100)
        factor_200 = scaler.compute_absolute_factor(200)
        factor_300 = scaler.compute_absolute_factor(300)
        assert factor_100 > factor_200 > factor_300

    def test_factor_clamped_to_range(self) -> None:
        """Factor should be clamped to [min_factor, max_factor]."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="absolute",
            decay_type="exponential",
            threshold_tokens=100,
            decay_rate=1.0,  # very aggressive decay
            max_factor=0.8,
            min_factor=0.2,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        # very large token count should still give min_factor, not lower
        factor = scaler.compute_absolute_factor(10000)
        assert factor >= 0.2


class TestRelativeMode:
    """Tests for relative (group-normalized) scaling mode."""

    def test_at_mean_gets_factor_near_one(self) -> None:
        """Sample exactly at group mean should get factor ~ max_factor (not 0.5)."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="relative",
            temperature=1.0,
            max_factor=1.0,
            min_factor=0.0,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        # z_score = 0 at mean
        factor = scaler.compute_relative_factor(token_count=100, group_mean=100.0, group_std=20.0)
        # sigmoid(0) = 0.5, then 2*0.5 = 1.0, so factor should be ~1.0
        assert factor == pytest.approx(1.0, abs=0.01)

    def test_below_mean_gets_factor_near_one(self) -> None:
        """Samples below mean should get factor ~ max_factor (no penalty)."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="relative",
            temperature=1.0,
            max_factor=1.0,
            min_factor=0.0,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        # below mean (z_score < 0)
        factor = scaler.compute_relative_factor(token_count=60, group_mean=100.0, group_std=20.0)
        # sigmoid(positive) > 0.5, then 2*that > 1.0, clamped to 1.0
        assert factor == pytest.approx(1.0, abs=0.01)

    def test_above_mean_gets_penalty(self) -> None:
        """Samples above group mean should get factor < max_factor."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="relative",
            temperature=1.0,
            max_factor=1.0,
            min_factor=0.0,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        # well above mean (z_score >> 0)
        factor = scaler.compute_relative_factor(token_count=160, group_mean=100.0, group_std=20.0)
        # z_score = 3, sigmoid(-3) ~= 0.05, 2*0.05 = 0.1
        assert factor < 0.5

    def test_temperature_affects_steepness(self) -> None:
        """Higher temperature should produce gentler slope."""
        base_config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="relative",
            temperature=1.0,
            max_factor=1.0,
            min_factor=0.0,
        )
        gentle_config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="relative",
            temperature=3.0,  # higher temperature
            max_factor=1.0,
            min_factor=0.0,
        )
        base_scaler = verbosity_scaling.VerbosityScaler(base_config)
        gentle_scaler = verbosity_scaling.VerbosityScaler(gentle_config)
        # same above-mean token count
        base_factor = base_scaler.compute_relative_factor(token_count=140, group_mean=100.0, group_std=20.0)
        gentle_factor = gentle_scaler.compute_relative_factor(token_count=140, group_mean=100.0, group_std=20.0)
        # gentle (higher temp) should have higher factor (less penalty)
        assert gentle_factor > base_factor

    def test_std_zero_skips_with_warning(self) -> None:
        """Group with std=0 should return factor=1.0 and emit warning once."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="relative",
            max_factor=1.0,
            min_factor=0.0,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        # create caches with identical token counts
        caches = [_make_cache(100) for _ in range(3)]
        rewards = [1.0, 1.0, 1.0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scaled_totals, _ = scaler.apply_relative(rewards, caches)
            # warning should be emitted
            assert len(w) == 1
            assert "std=0" in str(w[0].message)
        # all factors should be 1.0 (no scaling)
        assert scaled_totals == [1.0, 1.0, 1.0]
        # second call should not emit warning again
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scaler.apply_relative(rewards, caches)
            assert len(w) == 0  # no new warning

    def test_single_sample_warns_and_skips(self) -> None:
        """Single sample group should warn and return factor=1.0."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="relative",
            max_factor=1.0,
            min_factor=0.0,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        cache = _make_cache(100)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scaled_totals, _ = scaler.apply_relative([1.0], [cache])
            assert len(w) == 1
            assert "< 2 samples" in str(w[0].message)
        assert scaled_totals == [1.0]


class TestConfigValidation:
    """Tests for config validation."""

    def test_min_factor_must_be_lte_max_factor(self) -> None:
        """min_factor > max_factor should raise."""
        with pytest.raises(ValueError, match="min_factor .* must be <= max_factor"):
            reward_configs.VerbosityScalingConfig(
                enabled=True,
                min_factor=0.9,
                max_factor=0.5,
            )

    def test_max_factor_must_be_lte_one(self) -> None:
        """max_factor > 1.0 should raise for penalty behavior."""
        with pytest.raises(ValueError, match="max_factor .* must be <= 1.0"):
            reward_configs.VerbosityScalingConfig(
                enabled=True,
                max_factor=1.5,
            )

    def test_end_tokens_must_exceed_threshold_for_linear(self) -> None:
        """end_tokens <= threshold_tokens should raise for linear decay."""
        with pytest.raises(ValueError, match="end_tokens .* must be > threshold_tokens"):
            reward_configs.VerbosityScalingConfig(
                enabled=True,
                mode="absolute",
                decay_type="linear",
                threshold_tokens=100,
                end_tokens=100,  # must be > 100
            )

    def test_exponential_mode_allows_any_end_tokens(self) -> None:
        """end_tokens validation only applies to linear mode."""
        # should not raise
        reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="absolute",
            decay_type="exponential",
            threshold_tokens=100,
            end_tokens=50,  # allowed for exponential
        )

    def test_unsupported_length_source_raises(self) -> None:
        """Unsupported length sources should raise at config validation time."""
        with pytest.raises(ValueError, match="length_source.*not supported"):
            reward_configs.VerbosityScalingConfig(
                enabled=True,
                length_source=reward_types.LengthSource.prompt,  # not supported
            )
        with pytest.raises(ValueError, match="length_source.*not supported"):
            reward_configs.VerbosityScalingConfig(
                enabled=True,
                length_source=reward_types.LengthSource.sample_code,  # not supported
            )


class TestScalingConsistency:
    """Tests for scaling behavior consistency."""

    def test_absolute_mode_scales_total_only(self) -> None:
        """Absolute mode should only scale the total reward."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="absolute",
            decay_type="linear",
            threshold_tokens=10,
            end_tokens=100,
            max_factor=1.0,
            min_factor=0.5,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        cache = _make_cache(50)  # 50 tokens, in decay range
        aggregated_reward = 2.0
        scaled_total, metrics = scaler.apply_absolute(aggregated_reward, cache)
        # total should be scaled (factor < 1.0 at 50 tokens)
        assert scaled_total < aggregated_reward
        # metrics should contain the factor
        assert "verbosity/factor" in metrics

    def test_relative_mode_scales_total_only(self) -> None:
        """Relative mode should only scale the total rewards."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="relative",
            max_factor=1.0,
            min_factor=0.5,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        caches = [_make_cache(50), _make_cache(100), _make_cache(150)]
        rewards = [1.0, 1.0, 1.0]
        scaled_totals, all_metrics = scaler.apply_relative(rewards, caches)
        # above-mean samples should be scaled down
        assert scaled_totals[2] < rewards[2]
        # metrics should contain group stats
        assert "verbosity/group_mean" in all_metrics[0]


class TestNumericalStability:
    """Tests for numerical stability of sigmoid computation."""

    def test_stable_sigmoid_large_positive(self) -> None:
        """Large positive x should not overflow."""
        result = verbosity_scaling._stable_sigmoid(1000.0)
        assert math.isfinite(result)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_stable_sigmoid_large_negative(self) -> None:
        """Large negative x should not overflow."""
        result = verbosity_scaling._stable_sigmoid(-1000.0)
        assert math.isfinite(result)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_stable_sigmoid_zero(self) -> None:
        """sigmoid(0) should be 0.5."""
        result = verbosity_scaling._stable_sigmoid(0.0)
        assert result == pytest.approx(0.5)


class TestMetricsEmission:
    """Tests for metrics emission."""

    def test_absolute_mode_emits_metrics(self) -> None:
        """Absolute mode should emit verbosity metrics."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="absolute",
            decay_type="linear",
            threshold_tokens=10,
            end_tokens=100,
            emit_metrics=True,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        cache = _make_cache(50)
        _, metrics = scaler.apply_absolute(1.0, cache)
        assert "verbosity/factor" in metrics
        assert "verbosity/token_count" in metrics
        assert "verbosity/pre_scaling_reward" in metrics

    def test_relative_mode_emits_group_metrics(self) -> None:
        """Relative mode should emit group statistics metrics."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="relative",
            emit_metrics=True,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        caches = [_make_cache(50), _make_cache(100), _make_cache(150)]
        _, all_metrics = scaler.apply_relative([1.0, 1.0, 1.0], caches)
        for metrics in all_metrics:
            assert "verbosity/group_mean" in metrics
            assert "verbosity/group_std" in metrics
            assert "verbosity/factor" in metrics

    def test_metrics_disabled(self) -> None:
        """No metrics emitted when emit_metrics=False."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="absolute",
            emit_metrics=False,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        cache = _make_cache(50)
        _, metrics = scaler.apply_absolute(1.0, cache)
        assert len(metrics) == 0


class TestSkipNegativeRewards:
    """Tests for skip_negative_rewards option."""

    def test_absolute_mode_skips_negative_reward(self) -> None:
        """Negative rewards should skip scaling when skip_negative_rewards=True."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="absolute",
            decay_type="linear",
            threshold_tokens=10,
            end_tokens=100,
            max_factor=1.0,
            min_factor=0.5,
            skip_negative_rewards=True,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        cache = _make_cache(50)  # would normally get factor < 1.0
        # positive reward gets scaled
        scaled_pos, metrics_pos = scaler.apply_absolute(1.0, cache)
        assert scaled_pos < 1.0  # factor was applied
        assert metrics_pos["verbosity/skipped_negative"] == 0
        # negative reward skips scaling (factor=1.0)
        scaled_neg, metrics_neg = scaler.apply_absolute(-1.0, cache)
        assert scaled_neg == -1.0  # unchanged
        assert metrics_neg["verbosity/skipped_negative"] == 1
        assert metrics_neg["verbosity/factor"] == 1.0

    def test_relative_mode_skips_negative_reward(self) -> None:
        """Negative rewards should skip scaling in relative mode when skip_negative_rewards=True."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="relative",
            max_factor=1.0,
            min_factor=0.5,
            skip_negative_rewards=True,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        # mix of positive and negative rewards
        caches = [_make_cache(50), _make_cache(100), _make_cache(150)]
        rewards = [1.0, -0.5, 0.8]  # second one is negative
        scaled_totals, all_metrics = scaler.apply_relative(rewards, caches)
        # first and third get scaled normally
        assert scaled_totals[0] != 1.0 or scaled_totals[2] != 0.8
        # second (negative) should be unchanged
        assert scaled_totals[1] == -0.5
        assert all_metrics[1]["verbosity/skipped_negative"] == 1
        assert all_metrics[1]["verbosity/factor"] == 1.0

    def test_skip_negative_can_be_disabled(self) -> None:
        """When skip_negative_rewards=False, negative rewards are scaled."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="absolute",
            decay_type="linear",
            threshold_tokens=10,
            end_tokens=100,
            max_factor=1.0,
            min_factor=0.5,
            skip_negative_rewards=False,  # explicitly disabled
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        cache = _make_cache(50)
        scaled, metrics = scaler.apply_absolute(-1.0, cache)
        # negative reward gets scaled (moves toward 0)
        assert scaled > -1.0
        assert "verbosity/skipped_negative" not in metrics


class TestMissingSource:
    """Tests for handling missing source in cache."""

    def test_missing_source_raises(self) -> None:
        """When source is missing from cache, raise ValueError."""
        config = reward_configs.VerbosityScalingConfig(
            enabled=True,
            mode="absolute",
            length_source=reward_types.LengthSource.parsed_reasoning,  # different from cache
            emit_metrics=True,
        )
        scaler = verbosity_scaling.VerbosityScaler(config)
        # cache only has MODEL_OUTPUT, not PARSED_REASONING
        cache = _make_cache(100, source=reward_types.LengthSource.model_output)
        with pytest.raises(ValueError, match="requires token count for source"):
            scaler.apply_absolute(1.0, cache)


class TestManagerIntegration:
    """Integration tests with RewardManager."""

    def test_absolute_mode_scales_total_not_weighted_terms(self) -> None:
        """Absolute mode should scale total but preserve weighted_terms and raw_terms."""
        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        config = reward_configs.RewardManagerConfig(
            terms=[
                reward_configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    weight=1.0,
                    params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
                ),
            ],
            parsing=reward_configs.ParsingConfig(
                final_tag="final",
                track_token_lengths=True,  # enables parsing stats token tracking
                openai_tokenizer_model="gpt-4o",
            ),
            verbosity_scaling=reward_configs.VerbosityScalingConfig(
                enabled=True,
                mode="absolute",
                decay_type="linear",
                threshold_tokens=10,
                end_tokens=100,
                max_factor=1.0,
                min_factor=0.5,
            ),
            aggregation=reward_configs.AggregationConfig(return_raw_breakdown=True),
            logging=reward_configs.LoggingConfig(enabled=False),
        )
        manager = reward_manager.RewardManager(config)
        # ~21 chars in final block, ~500 extra = ~521 total chars
        # gpt-4o tokenizer: roughly 1 token per 4 chars -> ~130 tokens (well above end_tokens=100)
        ctx = rewards_conftest.make_sample_context(
            prompt="test",
            model_output="<final>answer</final>" + "x" * 500,
            identifier="s1",
            parsed=rewards_conftest.make_parsed_output(
                raw="<final>answer</final>" + "x" * 500,
                final_answer="answer",
            ),
        )
        output = manager.compute(ctx)
        # raw_terms should be unscaled (original term value)
        assert output.raw_terms is not None
        assert output.raw_terms["parseable"] == 1.0  # original unscaled value
        # weighted_terms should also be unscaled (verbosity only scales total)
        assert output.weighted_terms["parseable"] == output.raw_terms["parseable"]
        # total should be scaled (less than sum of weighted_terms)
        assert output.total < output.weighted_terms["parseable"]

    def test_relative_mode_warns_on_compute(self) -> None:
        """Relative mode should warn when using compute (no group context)."""
        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        config = reward_configs.RewardManagerConfig(
            terms=[
                reward_configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    weight=1.0,
                    params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
                ),
            ],
            parsing=reward_configs.ParsingConfig(
                final_tag="final",
                track_token_lengths=True,
                openai_tokenizer_model="gpt-4o",
            ),
            verbosity_scaling=reward_configs.VerbosityScalingConfig(
                enabled=True,
                mode="relative",
                temperature=1.0,
            ),
            logging=reward_configs.LoggingConfig(enabled=False),
        )
        manager = reward_manager.RewardManager(config)
        ctx = rewards_conftest.make_sample_context(
            prompt="test",
            model_output="<final>answer</final>",
            identifier="s1",
            parsed=rewards_conftest.make_parsed_output(
                raw="<final>answer</final>",
                final_answer="answer",
            ),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = manager.compute(ctx)
            # should emit warning about relative mode without group
            assert any("relative" in str(warning.message).lower() for warning in w)
        # reward should be unscaled (factor=1.0 when warning)
        assert output.total == 1.0

    def test_relative_mode_works_with_compute_batch(self) -> None:
        """Relative mode should work correctly with compute_batch (auto-groups by identifier)."""
        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        config = reward_configs.RewardManagerConfig(
            terms=[
                reward_configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    weight=1.0,
                    params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
                ),
            ],
            parsing=reward_configs.ParsingConfig(
                final_tag="final",
                track_token_lengths=True,
                openai_tokenizer_model="gpt-4o",
            ),
            verbosity_scaling=reward_configs.VerbosityScalingConfig(
                enabled=True,
                mode="relative",
                temperature=1.0,
                min_factor=0.1,
                max_factor=1.0,
            ),
            logging=reward_configs.LoggingConfig(enabled=False),
        )
        manager = reward_manager.RewardManager(config)
        # create 3 contexts with different lengths, same identifier (same group)
        # in GRPO, multiple completions for the same prompt share the same identifier/trace_id
        short_ctx = rewards_conftest.make_sample_context(
            prompt="test",
            model_output="<final>a</final>",  # short
            identifier="same_group",  # same identifier = same group
            parsed=rewards_conftest.make_parsed_output(raw="<final>a</final>", final_answer="a"),
        )
        medium_ctx = rewards_conftest.make_sample_context(
            prompt="test",
            model_output="<final>answer</final>" + "x" * 50,  # medium
            identifier="same_group",  # same identifier = same group
            parsed=rewards_conftest.make_parsed_output(raw="<final>answer</final>" + "x" * 50, final_answer="answer"),
        )
        long_ctx = rewards_conftest.make_sample_context(
            prompt="test",
            model_output="<final>answer</final>" + "x" * 200,  # long
            identifier="same_group",  # same identifier = same group
            parsed=rewards_conftest.make_parsed_output(raw="<final>answer</final>" + "x" * 200, final_answer="answer"),
        )
        # compute_batch auto-groups by identifier (falls back from trace_id)
        outputs = manager.compute_batch([short_ctx, medium_ctx, long_ctx])
        # short should have highest reward (below mean -> factor ~1.0)
        # long should have lowest reward (above mean -> factor < 1.0)
        assert outputs[0].total >= outputs[1].total
        assert outputs[1].total >= outputs[2].total
        # the shortest should be close to 1.0 (no penalty)
        assert outputs[0].total >= 0.9

    def test_verbosity_scaling_requires_tokenizer(self) -> None:
        """Verbosity scaling should require a tokenizer source (HF or tiktoken)."""
        pyine.organisms.models.rewards.terms.ensure_builtin_terms_registered()
        config = reward_configs.RewardManagerConfig(
            terms=[
                reward_configs.RewardTermSpec(
                    name="parseable",
                    type="parseable_answer",
                    weight=1.0,
                    params={"reward_if_present": 1.0, "reward_if_missing": 0.0},
                ),
            ],
            parsing=reward_configs.ParsingConfig(
                final_tag="final",
                # no openai_tokenizer_model set, and no tokenizer passed to RewardManager
            ),
            verbosity_scaling=reward_configs.VerbosityScalingConfig(
                enabled=True,
                mode="absolute",
            ),
            logging=reward_configs.LoggingConfig(enabled=False),
        )
        with pytest.raises(ValueError, match="token counting requires"):
            reward_manager.RewardManager(config)
