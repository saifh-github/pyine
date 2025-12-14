"""Tests for WeightedSumAggregator."""

import pytest

import pyine.organisms.models.rewards.core.aggregator as reward_aggregator
import pyine.organisms.models.rewards.core.configs as reward_configs


class TestWeightedSumAggregator:
    def test_basic_weighted_sum(self) -> None:
        config = reward_configs.AggregationConfig()
        aggregator = reward_aggregator.WeightedSumAggregator(config)
        total, weighted, unweighted = aggregator.aggregate(
            values={"a": 1.0, "b": 2.0},
            weights={"a": 1.0, "b": 0.5},
        )
        assert total == pytest.approx(2.0)  # 1.0*1.0 + 2.0*0.5
        assert weighted == {"a": 1.0, "b": 1.0}
        assert unweighted == {"a": 1.0, "b": 2.0}

    def test_missing_weight_defaults_to_one(self) -> None:
        config = reward_configs.AggregationConfig()
        aggregator = reward_aggregator.WeightedSumAggregator(config)
        total, weighted, _ = aggregator.aggregate(
            values={"a": 3.0},
            weights={},  # no explicit weights
        )
        assert total == pytest.approx(3.0)
        assert weighted == {"a": 3.0}

    def test_per_term_clipping(self) -> None:
        config = reward_configs.AggregationConfig(clip_term_min=0.0, clip_term_max=1.0)
        aggregator = reward_aggregator.WeightedSumAggregator(config)
        total, weighted, unweighted = aggregator.aggregate(
            values={"a": -0.5, "b": 2.0},
            weights={"a": 1.0, "b": 1.0},
        )
        assert total == pytest.approx(1.0)  # 0.0 + 1.0 (both clipped)
        assert weighted == {"a": 0.0, "b": 1.0}
        assert unweighted == {"a": -0.5, "b": 2.0}  # unweighted preserves original

    def test_total_clipping(self) -> None:
        config = reward_configs.AggregationConfig(clip_total_min=-1.0, clip_total_max=1.0)
        aggregator = reward_aggregator.WeightedSumAggregator(config)
        total, _, _ = aggregator.aggregate(
            values={"a": 5.0, "b": 5.0},
            weights={"a": 1.0, "b": 1.0},
        )
        assert total == pytest.approx(1.0)  # clipped to max

    def test_combined_term_and_total_clipping(self) -> None:
        config = reward_configs.AggregationConfig(
            clip_term_min=0.0,
            clip_term_max=2.0,
            clip_total_min=0.0,
            clip_total_max=3.0,
        )
        aggregator = reward_aggregator.WeightedSumAggregator(config)
        total, weighted, _ = aggregator.aggregate(
            values={"a": 5.0, "b": 5.0},
            weights={"a": 1.0, "b": 1.0},
        )
        assert weighted == {"a": 2.0, "b": 2.0}  # term clipped to 2.0
        assert total == pytest.approx(3.0)  # total clipped (2+2=4 -> 3)

    def test_empty_values(self) -> None:
        config = reward_configs.AggregationConfig()
        aggregator = reward_aggregator.WeightedSumAggregator(config)
        total, weighted, unweighted = aggregator.aggregate(
            values={},
            weights={},
        )
        assert total == pytest.approx(0.0)
        assert weighted == {}
        assert unweighted == {}

    def test_negative_weights(self) -> None:
        config = reward_configs.AggregationConfig()
        aggregator = reward_aggregator.WeightedSumAggregator(config)
        total, weighted, _ = aggregator.aggregate(
            values={"a": 2.0},
            weights={"a": -1.0},
        )
        assert total == pytest.approx(-2.0)
        assert weighted == {"a": -2.0}

    def test_unsupported_strategy_raises(self) -> None:
        config = reward_configs.AggregationConfig()
        object.__setattr__(config, "strategy", "unsupported")  # bypass frozen
        with pytest.raises(ValueError, match="unsupported aggregation strategy"):
            reward_aggregator.WeightedSumAggregator(config)

    def test_zero_weights(self) -> None:
        config = reward_configs.AggregationConfig()
        aggregator = reward_aggregator.WeightedSumAggregator(config)
        total, weighted, unweighted = aggregator.aggregate(
            values={"a": 5.0, "b": 3.0},
            weights={"a": 0.0, "b": 0.0},
        )
        assert total == pytest.approx(0.0)
        assert weighted == {"a": 0.0, "b": 0.0}
        assert unweighted == {"a": 5.0, "b": 3.0}

    def test_mixed_zero_and_nonzero_weights(self) -> None:
        config = reward_configs.AggregationConfig()
        aggregator = reward_aggregator.WeightedSumAggregator(config)
        total, weighted, _ = aggregator.aggregate(
            values={"a": 2.0, "b": 3.0, "c": 4.0},
            weights={"a": 1.0, "b": 0.0, "c": 2.0},
        )
        assert total == pytest.approx(10.0)  # 2*1 + 3*0 + 4*2
        assert weighted == {"a": 2.0, "b": 0.0, "c": 8.0}

    def test_fractional_weights(self) -> None:
        config = reward_configs.AggregationConfig()
        aggregator = reward_aggregator.WeightedSumAggregator(config)
        total, weighted, _ = aggregator.aggregate(
            values={"a": 4.0, "b": 6.0},
            weights={"a": 0.25, "b": 0.5},
        )
        assert total == pytest.approx(4.0)  # 4*0.25 + 6*0.5 = 1 + 3
        assert weighted["a"] == pytest.approx(1.0)
        assert weighted["b"] == pytest.approx(3.0)

    def test_only_min_clipping(self) -> None:
        config = reward_configs.AggregationConfig(clip_term_min=-1.0)
        aggregator = reward_aggregator.WeightedSumAggregator(config)
        total, weighted, _ = aggregator.aggregate(
            values={"a": -5.0, "b": 10.0},
            weights={"a": 1.0, "b": 1.0},
        )
        assert total == pytest.approx(9.0)  # -1 + 10
        assert weighted["a"] == pytest.approx(-1.0)
        assert weighted["b"] == pytest.approx(10.0)

    def test_only_max_clipping(self) -> None:
        config = reward_configs.AggregationConfig(clip_term_max=2.0)
        aggregator = reward_aggregator.WeightedSumAggregator(config)
        total, weighted, _ = aggregator.aggregate(
            values={"a": -5.0, "b": 10.0},
            weights={"a": 1.0, "b": 1.0},
        )
        assert total == pytest.approx(-3.0)  # -5 + 2
        assert weighted["a"] == pytest.approx(-5.0)
        assert weighted["b"] == pytest.approx(2.0)

    def test_single_term(self) -> None:
        config = reward_configs.AggregationConfig()
        aggregator = reward_aggregator.WeightedSumAggregator(config)
        total, weighted, unweighted = aggregator.aggregate(
            values={"only": 7.5},
            weights={"only": 2.0},
        )
        assert total == pytest.approx(15.0)
        assert weighted == {"only": 15.0}
        assert unweighted == {"only": 7.5}
