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


class TestRunningCorrStats:
    """Tests for RunningCorrStats correlation and regression computation."""

    def test_empty_defaults(self) -> None:
        stats = pyine.utils.stats.RunningCorrStats()
        assert stats.count == 0
        assert stats.correlation() is None
        assert stats.slope() is None
        assert stats.intercept() is None

    def test_single_observation_returns_none(self) -> None:
        stats = pyine.utils.stats.RunningCorrStats()
        stats.update(1.0, 2.0)
        assert stats.count == 1
        assert stats.correlation() is None
        assert stats.slope() is None

    def test_perfect_positive_correlation(self) -> None:
        stats = pyine.utils.stats.RunningCorrStats()
        for x in range(1, 11):
            stats.update(float(x), float(x * 2))  # y = 2x
        assert stats.correlation() == pytest.approx(1.0)
        assert stats.slope() == pytest.approx(2.0)
        assert stats.intercept() == pytest.approx(0.0)

    def test_perfect_negative_correlation(self) -> None:
        stats = pyine.utils.stats.RunningCorrStats()
        for x in range(1, 11):
            stats.update(float(x), float(-x))  # y = -x
        assert stats.correlation() == pytest.approx(-1.0)
        assert stats.slope() == pytest.approx(-1.0)

    def test_zero_variance_returns_none(self) -> None:
        stats = pyine.utils.stats.RunningCorrStats()
        stats.update(1.0, 5.0)
        stats.update(1.0, 5.0)  # same x, same y
        stats.update(1.0, 5.0)
        assert stats.correlation() is None  # zero variance in both x and y
        assert stats.slope() is None

    def test_zero_x_variance_returns_none(self) -> None:
        stats = pyine.utils.stats.RunningCorrStats()
        stats.update(1.0, 1.0)
        stats.update(1.0, 2.0)  # same x, different y
        stats.update(1.0, 3.0)
        assert stats.correlation() is None  # zero variance in x
        assert stats.slope() is None

    def test_merge_combines_stats(self) -> None:
        left = pyine.utils.stats.RunningCorrStats()
        right = pyine.utils.stats.RunningCorrStats()
        for x in range(1, 6):
            left.update(float(x), float(x * 2))
        for x in range(6, 11):
            right.update(float(x), float(x * 2))
        left.merge(right)
        assert left.count == 10
        assert left.correlation() == pytest.approx(1.0)
        assert left.slope() == pytest.approx(2.0)

    def test_as_state_roundtrip(self) -> None:
        stats = pyine.utils.stats.RunningCorrStats()
        for x in range(1, 6):
            stats.update(float(x), float(x * 3 + 1))
        state = stats.as_state()
        restored = pyine.utils.stats.RunningCorrStats.from_state(state)
        assert restored.count == stats.count
        assert restored.correlation() == pytest.approx(stats.correlation())  # type: ignore[arg-type]
        assert restored.slope() == pytest.approx(stats.slope())  # type: ignore[arg-type]
        assert restored.intercept() == pytest.approx(stats.intercept())  # type: ignore[arg-type]

    def test_intercept_computation(self) -> None:
        stats = pyine.utils.stats.RunningCorrStats()
        for x in range(1, 6):
            stats.update(float(x), float(x * 2 + 5))  # y = 2x + 5
        assert stats.slope() == pytest.approx(2.0)
        assert stats.intercept() == pytest.approx(5.0)
