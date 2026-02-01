import math

import pytest

import pyine.organisms.datamodules.samples.common as samples_common
import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.difficulty as difficulty_module


def _make_sample_data(
    *,
    identifier: str = "test",
    trace_step_count: int = 50,
    has_code_override: bool = False,
    complexity_metrics: dict[str, float | int] | None = None,
    code: str = "def foo():\n    return 1",
    inputs: str = "",
    expected_output: str = "1",
    first_line: int = 0,
    last_line: int = 2,
    first_step_idx: int = 0,
    last_step_idx: int = 0,
) -> samples_common.SampleData:
    """Build a SampleData instance for difficulty tests."""
    return samples_common.SampleData(
        identifier=identifier,
        code=code,
        description="test",
        entrypoint="foo",
        first_line=first_line,
        last_line=last_line,
        inputs=inputs,
        expected_output=expected_output,
        predict_type=samples_common.SamplePredictType.program_output,
        code_type="original",
        trace_step_count=trace_step_count,
        comma_separated_tags="",
        has_code_override=has_code_override,
        complexity_metrics=complexity_metrics or {},
        first_step_idx=first_step_idx,
        last_step_idx=last_step_idx,
    )


class TestBinEdges:
    """Tests for bin edge computation."""

    def test_bin_edges_fixed_range(self) -> None:
        """Fixed range mode should produce uniform edges in [0, 1]."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="maintainability_index",
            normalization_mode="fixed_range",
            num_difficulty_bins=5,
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        # should be [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        assert len(estimator._bin_edges) == 6
        assert estimator._bin_edges[0] == pytest.approx(0.0)
        assert estimator._bin_edges[-1] == pytest.approx(1.0)
        # check intermediate edges
        assert estimator._bin_edges[1] == pytest.approx(0.2)
        assert estimator._bin_edges[2] == pytest.approx(0.4)
        assert estimator._bin_edges[3] == pytest.approx(0.6)
        assert estimator._bin_edges[4] == pytest.approx(0.8)

    def test_bin_edges_log_with_overflow(self) -> None:
        """Log mode should produce uniform edges plus overflow bin."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            normalization_mode="log",
            num_difficulty_bins=5,
            bin_max_score=5.0,
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        # should be [0, 1, 2, 3, 4, 5, inf]
        assert len(estimator._bin_edges) == 7  # 5 bins + overflow
        assert estimator._bin_edges[-1] == float("inf")
        assert estimator._bin_edges[0] == pytest.approx(0.0)
        assert estimator._bin_edges[5] == pytest.approx(5.0)

    def test_bin_edges_manual(self) -> None:
        """Manual bin edges should be used when provided."""
        manual_edges = [0.0, 1.0, 2.0, 3.0, float("inf")]
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            normalization_mode="log",
            bin_edges=manual_edges,
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        assert estimator._bin_edges == manual_edges


class TestNormalization:
    """Tests for normalization modes."""

    def test_log_normalization(self) -> None:
        """Log normalization should apply log1p transform."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            normalization_mode="log",
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        raw_value = 100.0
        normalized = estimator.normalize(raw_value, "trace_step_count")
        assert normalized == pytest.approx(math.log1p(100.0))

    def test_score_clip_max(self) -> None:
        """score_clip_max should cap normalized score."""
        sample_data = _make_sample_data(trace_step_count=50)
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            normalization_mode="log",
            score_clip_max=3.0,
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        # log1p(50) is approximately 3.93, should be clipped to 3.0
        metrics = estimator.compute(sample_data, reward_total=1.0)
        assert metrics["difficulty/score"] == pytest.approx(3.0)

    def test_fixed_range_with_invert(self) -> None:
        """Fixed range with invert=True should invert the score (higher raw = lower difficulty)."""
        sample_data = _make_sample_data(complexity_metrics={"maintainability_index": 80.0})
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="maintainability_index",
            normalization_mode="fixed_range",
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        # maintainability_index: (0, 100, True) -> higher is easier, so invert
        # score = (100 - 80) / (100 - 0) = 0.2
        metrics = estimator.compute(sample_data, reward_total=1.0)
        assert metrics["difficulty/score"] == pytest.approx(0.2)


class TestCodeOverrideHandling:
    """Tests for has_code_override handling."""

    def test_code_override_skip(self) -> None:
        """Skip mode should return limited metrics when has_code_override=True."""
        sample_data = _make_sample_data(has_code_override=True, trace_step_count=50)
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            code_override_mode=difficulty_module.CodeOverrideMode.skip,
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        metrics = estimator.compute(sample_data, reward_total=1.0)
        assert metrics.get("difficulty/execution_skipped") == 1
        assert metrics.get("difficulty/primary_missing") == 1
        assert "difficulty/score" not in metrics

    def test_code_override_use_original(self) -> None:
        """use_original mode should use original trace_step_count."""
        sample_data = _make_sample_data(has_code_override=True, trace_step_count=50)
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            code_override_mode=difficulty_module.CodeOverrideMode.use_original,
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        metrics = estimator.compute(sample_data, reward_total=1.0)
        assert "difficulty/score" in metrics
        assert metrics["difficulty/raw_primary"] == 50.0

    def test_code_override_recompute(self) -> None:
        """recompute_step_count mode should use segment indices."""
        sample_data = _make_sample_data(
            has_code_override=True,
            trace_step_count=50,
            first_step_idx=10,
            last_step_idx=30,
        )
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            code_override_mode=difficulty_module.CodeOverrideMode.recompute_step_count,
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        raw = estimator.get_raw_value(sample_data, "trace_step_count")
        assert raw == 21.0  # 30 - 10 + 1


class TestRunSummaries:
    """Tests for run-level summary computation."""

    def test_run_summaries_basic_metrics(self) -> None:
        """Run summaries should include score stats and per-bin reward stats."""
        sample_data = _make_sample_data(trace_step_count=50)
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            num_difficulty_bins=3,
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        estimator.compute(sample_data, reward_total=0.5)
        summaries = estimator.get_run_summaries()
        assert "score/count" in summaries
        assert summaries["score/count"] == 1

    def test_run_summaries_correlation(self) -> None:
        """Run summaries should include correlation between difficulty and reward."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        # add multiple samples with different difficulties and rewards
        for step_count, reward in [(10, 0.9), (50, 0.5), (100, 0.2), (200, 0.1)]:
            sample_data = _make_sample_data(trace_step_count=step_count)
            estimator.compute(sample_data, reward_total=reward)
        summaries = estimator.get_run_summaries()
        assert "reward/correlation" in summaries
        assert "reward/slope" in summaries
        # higher difficulty should correlate with lower reward
        assert summaries["reward/correlation"] < 0

    def test_run_summaries_return_minimal_metrics_when_no_scores(self) -> None:
        """Run summaries should still include diagnostic ratios even when no scores were computed."""
        sample_data = _make_sample_data(has_code_override=True)
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            code_override_mode=difficulty_module.CodeOverrideMode.skip,
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        metrics = estimator.compute(sample_data, reward_total=0.5)
        assert metrics["difficulty/execution_skipped"] == 1
        assert metrics["difficulty/primary_missing"] == 1
        summaries = estimator.get_run_summaries()
        assert summaries["score/count"] == 0
        assert summaries["override_skip_ratio"] == pytest.approx(1.0)

    def test_run_summaries_percentiles(self) -> None:
        """Run summaries should include percentiles when track_percentiles is enabled."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            track_percentiles="always",
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        for step_count in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            sample_data = _make_sample_data(trace_step_count=step_count)
            estimator.compute(sample_data, reward_total=0.5)
        summaries = estimator.get_run_summaries()
        assert "score/median" in summaries
        assert "score/p90" in summaries
        assert "score/p99" in summaries


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_config_rejects_none_without_bin_edges(self) -> None:
        """normalization_mode='none' requires explicit bin_edges."""
        with pytest.raises(ValueError, match="normalization_mode='none' requires explicit bin_edges"):
            reward_configs.DifficultyConfig(
                enabled=True,
                primary_source="trace_step_count",
                normalization_mode="none",
            )

    def test_config_allows_none_with_explicit_bin_edges(self) -> None:
        """normalization_mode='none' with explicit bin_edges should be valid."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            normalization_mode="none",
            bin_edges=[0.0, 10.0, 50.0, 100.0, float("inf")],
        )
        assert config.normalization_mode == "none"

    def test_config_rejects_fixed_range_with_unknown_source(self) -> None:
        """fixed_range with auto-binning requires known-range source."""
        with pytest.raises(ValueError, match="normalization_mode='fixed_range'"):
            reward_configs.DifficultyConfig(
                enabled=True,
                primary_source="trace_step_count",  # not a known-range metric
                normalization_mode="fixed_range",
            )

    def test_config_allows_fixed_range_with_known_source(self) -> None:
        """fixed_range with known-range source should be valid."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="maintainability_index",
            normalization_mode="fixed_range",
        )
        assert config.normalization_mode == "fixed_range"

    def test_config_allows_fixed_range_with_custom_source_range(self) -> None:
        """fixed_range with custom source_ranges should be valid."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="custom_metric",
            normalization_mode="fixed_range",
            source_ranges={"custom_metric": (0.0, 1000.0, False)},
        )
        assert config.normalization_mode == "fixed_range"

    def test_config_rejects_unsorted_bin_edges(self) -> None:
        """bin_edges must be sorted."""
        with pytest.raises(ValueError, match="sorted in ascending order"):
            reward_configs.DifficultyConfig(
                enabled=True,
                primary_source="trace_step_count",
                bin_edges=[0.0, 3.0, 1.0, 5.0],  # not sorted
            )


class TestTokenSources:
    """Tests for token-based difficulty sources."""

    def test_token_sources_require_token_counter(self) -> None:
        """Token sources should require a token_counter."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="code_tokens",
        )
        with pytest.raises(ValueError, match="require a token_counter"):
            difficulty_module.DifficultyEstimator(config)

    def test_token_sources_work_with_token_counter(self) -> None:
        """Token sources should work when token_counter is provided."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="code_tokens",
        )

        def token_counter(s: str) -> int:
            return len(s.split())

        estimator = difficulty_module.DifficultyEstimator(config, token_counter=token_counter)
        sample_data = _make_sample_data(code="def foo():\n    return 1")
        metrics = estimator.compute(sample_data, reward_total=0.5)
        assert "difficulty/raw_primary" in metrics
        # "def foo():\n    return 1" has 4 words: "def", "foo():", "return", "1"
        assert metrics["difficulty/raw_primary"] == 4.0


class TestSecondarySources:
    """Tests for secondary difficulty sources."""

    def test_secondary_sources_logged(self) -> None:
        """Secondary sources should be logged as raw values."""
        sample_data = _make_sample_data(
            trace_step_count=50,
            complexity_metrics={"halstead_effort": 100.0},
        )
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            secondary_sources=frozenset({"halstead_effort"}),
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        metrics = estimator.compute(sample_data, reward_total=0.5)
        assert "difficulty/raw/halstead_effort" in metrics
        assert metrics["difficulty/raw/halstead_effort"] == 100.0

    def test_secondary_sources_correlation_tracked(self) -> None:
        """Secondary sources should have correlation tracked via getters (for table logging)."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            secondary_sources=frozenset({"code_length"}),
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        for step_count, code_len, reward in [(10, 20, 0.9), (50, 100, 0.5), (100, 200, 0.2)]:
            sample_data = _make_sample_data(trace_step_count=step_count, code="x" * code_len)
            estimator.compute(sample_data, reward_total=reward)
        # secondary stats are now accessed via getters (for table logging), not run summaries
        secondary_stats = estimator.get_secondary_stats()
        secondary_corr_stats = estimator.get_secondary_corr_stats()
        assert "code_length" in secondary_stats
        assert "code_length" in secondary_corr_stats
        assert secondary_stats["code_length"].count == 3
        assert secondary_corr_stats["code_length"].correlation() is not None


class TestStateSerialization:
    """Tests for state serialization and deserialization."""

    def test_round_trip_state(self) -> None:
        """Estimator state should survive serialization round trip."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            track_percentiles="always",
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        for step_count in [10, 50, 100]:
            sample_data = _make_sample_data(trace_step_count=step_count)
            estimator.compute(sample_data, reward_total=0.5)
        original_summaries = estimator.get_run_summaries()
        # serialize and restore
        state = estimator.as_state()
        restored = difficulty_module.DifficultyEstimator.from_state(config, state)
        restored_summaries = restored.get_run_summaries()
        # check key metrics match
        assert original_summaries["score/mean"] == pytest.approx(restored_summaries["score/mean"])
        assert original_summaries["score/count"] == restored_summaries["score/count"]

    def test_from_state_rejects_mismatched_bin_count(self) -> None:
        """from_state should raise ValueError if bin counts don't match config."""
        config_5_bins = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            num_difficulty_bins=5,
        )
        estimator = difficulty_module.DifficultyEstimator(config_5_bins)
        sample_data = _make_sample_data(trace_step_count=50)
        estimator.compute(sample_data, reward_total=0.5)
        state = estimator.as_state()
        # try to restore with different bin count
        config_10_bins = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            num_difficulty_bins=10,
        )
        with pytest.raises(ValueError, match="state has .* bins.*config defines .* bins"):
            difficulty_module.DifficultyEstimator.from_state(config_10_bins, state)

    def test_merge_rejects_mismatched_bin_count(self) -> None:
        """merge should raise ValueError if bin counts don't match."""
        config_5_bins = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            num_difficulty_bins=5,
        )
        config_10_bins = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            num_difficulty_bins=10,
        )
        estimator1 = difficulty_module.DifficultyEstimator(config_5_bins)
        estimator2 = difficulty_module.DifficultyEstimator(config_10_bins)
        with pytest.raises(ValueError, match="cannot merge.*different bin counts"):
            estimator1.merge(estimator2)


class TestTokenSourcesWithOverride:
    """Tests for token sources with code override handling."""

    def test_code_tokens_with_override_skip_still_logged(self) -> None:
        """Token sources should still work when has_code_override=True and mode=skip."""

        def token_counter(s: str) -> int:
            return len(s.split())

        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="code_tokens",  # token source, not execution source
            code_override_mode=difficulty_module.CodeOverrideMode.skip,
        )
        estimator = difficulty_module.DifficultyEstimator(config, token_counter=token_counter)
        # code is "def foo(): pass" which splits into 3 tokens
        sample_data = _make_sample_data(has_code_override=True, trace_step_count=50, code="def foo(): pass")
        # code_tokens is a context source, not an execution source - should still compute
        metrics = estimator.compute(sample_data, reward_total=0.5)
        # execution_skipped should be True (because has_code_override=True and mode=skip)
        assert metrics.get("difficulty/execution_skipped") == 1
        # BUT since primary_source is a token source (context, not execution),
        # we should still get the difficulty score
        assert "difficulty/score" in metrics
        assert "difficulty/raw_primary" in metrics
        # "def foo(): pass" = 3 tokens (word-based split)
        assert metrics["difficulty/raw_primary"] == 3.0

    def test_context_secondary_sources_logged_when_execution_skipped(self) -> None:
        """Context secondary sources should still be logged when execution is skipped."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",  # execution source
            secondary_sources=frozenset({"code_length", "inputs_length"}),  # context sources
            code_override_mode=difficulty_module.CodeOverrideMode.skip,
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        sample_data = _make_sample_data(
            has_code_override=True,
            trace_step_count=50,
            code="x" * 100,
            inputs="abc",
        )
        metrics = estimator.compute(sample_data, reward_total=0.5)
        # execution skipped, primary missing (trace_step_count is execution)
        assert metrics.get("difficulty/execution_skipped") == 1
        assert metrics.get("difficulty/primary_missing") == 1
        assert "difficulty/score" not in metrics
        # BUT context secondary sources should still be logged
        assert metrics.get("difficulty/raw/code_length") == 100.0
        assert metrics.get("difficulty/raw/inputs_length") == 3.0


class TestOverflowBinBehavior:
    """Tests for overflow bin behavior with score_clip_max."""

    def test_no_overflow_bin_when_score_clip_max_bounds_to_bin_max(self) -> None:
        """No overflow bin when score_clip_max <= bin_max_score (all scores within bin range)."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            normalization_mode="log",
            num_difficulty_bins=5,
            score_clip_max=5.0,  # clips scores, bin_max derived from this
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        # should have 6 edges for 5 bins, no overflow (inf)
        assert len(estimator._bin_edges) == 6
        assert estimator._bin_edges[-1] != float("inf")
        assert estimator._bin_edges[-1] == pytest.approx(5.0)

    def test_overflow_bin_when_no_score_clip_max(self) -> None:
        """When score_clip_max is not set, overflow bin should be added."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            normalization_mode="log",
            num_difficulty_bins=5,
            bin_max_score=5.0,  # no clip, just bin_max
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        # should have 7 edges for 5 bins + overflow
        assert len(estimator._bin_edges) == 7
        assert estimator._bin_edges[-1] == float("inf")

    def test_overflow_bin_when_score_clip_max_exceeds_bin_max(self) -> None:
        """Overflow bin needed when score_clip_max > bin_max_score (scores can exceed last edge)."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            normalization_mode="log",
            num_difficulty_bins=5,
            bin_max_score=5.0,
            score_clip_max=10.0,  # clip > bin_max, so overflow needed for scores 5-10
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        # should have 7 edges: 5 bins + overflow (scores can be 5-10)
        assert len(estimator._bin_edges) == 7
        assert estimator._bin_edges[-1] == float("inf")
        assert estimator._bin_edges[-2] == pytest.approx(5.0)

    def test_no_overflow_bin_when_score_clip_max_equals_bin_max(self) -> None:
        """No overflow bin when score_clip_max == bin_max_score (boundary case)."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            normalization_mode="log",
            num_difficulty_bins=5,
            bin_max_score=5.0,
            score_clip_max=5.0,  # equal, so no overflow needed
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        # should have 6 edges for 5 bins, no overflow
        assert len(estimator._bin_edges) == 6
        assert estimator._bin_edges[-1] != float("inf")
        assert estimator._bin_edges[-1] == pytest.approx(5.0)


class TestRewardManagerConfigValidation:
    """Tests for RewardManagerConfig validation related to reserved term names."""

    def test_rejects_term_named_difficulty(self) -> None:
        """RewardManagerConfig should reject a term named 'difficulty'."""
        with pytest.raises(ValueError, match="term name 'difficulty' is reserved"):
            reward_configs.RewardManagerConfig(
                terms=[
                    reward_configs.RewardTermSpec(
                        name="difficulty",  # reserved name
                        type="dummy_term",
                    )
                ]
            )

    @pytest.mark.parametrize("reserved_name", ["total", "batch", "run", "parsing", "verbosity"])
    def test_rejects_other_reserved_names(self, reserved_name: str) -> None:
        """RewardManagerConfig should reject other reserved term names."""
        with pytest.raises(ValueError, match=f"term name '{reserved_name}' is reserved"):
            reward_configs.RewardManagerConfig(
                terms=[
                    reward_configs.RewardTermSpec(
                        name=reserved_name,
                        type="dummy_term",
                    )
                ]
            )

    def test_allows_term_with_similar_but_not_reserved_name(self) -> None:
        """RewardManagerConfig should allow terms with names that contain reserved words."""
        # "difficulty_bonus" is NOT reserved, only exactly "difficulty" is
        config = reward_configs.RewardManagerConfig(
            terms=[
                reward_configs.RewardTermSpec(
                    name="difficulty_bonus",  # not exactly "difficulty"
                    type="dummy_term",
                ),
                reward_configs.RewardTermSpec(
                    name="total_score",  # not exactly "total"
                    type="dummy_term",
                ),
            ]
        )
        assert len(config.terms) == 2

    def test_rejects_term_name_with_slash(self) -> None:
        """RewardManagerConfig should reject term names containing '/'."""
        with pytest.raises(ValueError, match="slashes are forbidden"):
            reward_configs.RewardManagerConfig(
                terms=[
                    reward_configs.RewardTermSpec(
                        name="my/term",  # slashes forbidden
                        type="dummy_term",
                    )
                ]
            )


class TestTrackingModes:
    """Tests for tracking mode options (disabled, eval_only, always)."""

    def test_always_mode_tracks_all(self) -> None:
        """Always mode should track every generation."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            track_percentiles="always",
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        for idx in range(10):
            sample_data = _make_sample_data(trace_step_count=(idx + 1) * 10)
            estimator.compute(sample_data, reward_total=0.5)
        # should track all 10 samples
        assert len(estimator._score_values) == 10

    def test_eval_only_mode_respects_phase(self) -> None:
        """Eval-only mode should only track during eval phase."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            track_percentiles="eval_only",
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        # process in train phase (default)
        for idx in range(5):
            sample_data = _make_sample_data(trace_step_count=(idx + 1) * 10)
            estimator.compute(sample_data, reward_total=0.5)
        assert len(estimator._score_values) == 0  # no tracking in train
        # switch to eval phase
        estimator.set_key_prefix("eval/")
        for idx in range(5):
            sample_data = _make_sample_data(trace_step_count=(idx + 6) * 10)
            estimator.compute(sample_data, reward_total=0.5)
        assert len(estimator._score_values) == 5  # tracked during eval

    def test_disabled_mode_never_tracks(self) -> None:
        """Disabled mode should never track percentiles."""
        config = reward_configs.DifficultyConfig(
            enabled=True,
            primary_source="trace_step_count",
            track_percentiles="disabled",
        )
        estimator = difficulty_module.DifficultyEstimator(config)
        # process in train phase
        for idx in range(5):
            sample_data = _make_sample_data(trace_step_count=(idx + 1) * 10)
            estimator.compute(sample_data, reward_total=0.5)
        assert len(estimator._score_values) == 0  # not tracked
        # switch to eval phase
        estimator.set_key_prefix("eval/")
        for idx in range(5):
            sample_data = _make_sample_data(trace_step_count=(idx + 6) * 10)
            estimator.compute(sample_data, reward_total=0.5)
        assert len(estimator._score_values) == 0  # still not tracked
