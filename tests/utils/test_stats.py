import math

import pytest

import pyine.utils.stats


class TestRunningStats:
    def test_empty_defaults(self) -> None:
        stats = pyine.utils.stats.RunningStats()
        assert stats.count == 0
        assert stats.mean() == 0.0
        assert stats.std() == 0.0
        assert stats.min is None
        assert stats.max is None

    def test_update_tracks_mean_min_max(self) -> None:
        stats = pyine.utils.stats.RunningStats()
        stats.update(1.0)
        stats.update(3.0)
        stats.update(2.0)
        assert stats.count == 3
        assert stats.mean() == pytest.approx(2.0)
        assert stats.min == 1.0
        assert stats.max == 3.0

    def test_std_is_population_std(self) -> None:
        stats = pyine.utils.stats.RunningStats()
        stats.update(1.0)
        stats.update(3.0)
        mean = 2.0
        expected_variance = ((1.0 - mean) ** 2 + (3.0 - mean) ** 2) / 2.0
        assert stats.std() == pytest.approx(math.sqrt(expected_variance))

    def test_merge_combines_stats(self) -> None:
        left = pyine.utils.stats.RunningStats()
        right = pyine.utils.stats.RunningStats()
        left.update(1.0)
        left.update(2.0)
        right.update(10.0)
        right.update(20.0)
        left.merge(right)
        assert left.count == 4
        assert left.min == 1.0
        assert left.max == 20.0
        assert left.mean() == pytest.approx((1.0 + 2.0 + 10.0 + 20.0) / 4.0)

    def test_as_state_roundtrip_preserves_none_min_max(self) -> None:
        stats = pyine.utils.stats.RunningStats()
        state = stats.as_state()
        assert isinstance(state["count"], int)
        assert math.isnan(state["min"])
        assert math.isnan(state["max"])
        restored = pyine.utils.stats.RunningStats.from_state(state)
        assert restored.count == 0
        assert restored.min is None
        assert restored.max is None
