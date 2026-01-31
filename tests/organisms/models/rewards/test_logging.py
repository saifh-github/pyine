"""Tests for reward logging implementations."""

import wandb

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.logging as reward_logging
import pyine.utils.stats as stats_utils


class TestInMemoryRewardLogger:
    def test_log_records_sample_event(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        logger.log_sample(
            "sample_1",
            total=1.5,
            terms={"term_a": 1.0, "term_b": 0.5},
            metrics={"term_a/hit": True, "term_b/count": 3},
            step=10,
        )
        assert len(logger.samples) == 1
        entry = logger.samples[0]
        assert entry["sample_id"] == "sample_1"
        assert entry["reward_total"] == 1.5
        assert entry["reward_terms"] == {"term_a": 1.0, "term_b": 0.5}
        assert entry["reward_metrics"] == {"term_a/hit": True, "term_b/count": 3}
        assert entry["step"] == 10

    def test_log_without_step(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        logger.log_sample("sample_1", total=1.0, terms={}, metrics={})
        assert logger.samples[0]["step"] is None

    def test_log_multiple_samples(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        logger.log_sample("s1", total=1.0, terms={}, metrics={}, step=1)
        logger.log_sample("s2", total=2.0, terms={}, metrics={}, step=2)
        logger.log_sample("s3", total=3.0, terms={}, metrics={}, step=3)
        assert len(logger.samples) == 3
        assert [e["sample_id"] for e in logger.samples] == ["s1", "s2", "s3"]
        assert [e["reward_total"] for e in logger.samples] == [1.0, 2.0, 3.0]

    def test_log_phase_summaries_records_run_event(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        logger.log_phase_summaries(
            reward_totals={"count": 100.0, "mean_total": 0.75},
            reward_term_summaries={"term_a": 0.5, "term_b": 0.25},
            reward_category_summaries={"cat_a": 0.8},
            step=50,
        )
        assert len(logger.runs) == 1
        entry = logger.runs[0]
        assert entry["reward_totals"] == {"count": 100.0, "mean_total": 0.75}
        assert entry["reward_term_summaries"] == {"term_a": 0.5, "term_b": 0.25}
        assert entry["reward_category_summaries"] == {"cat_a": 0.8}
        assert entry["step"] == 50

    def test_log_phase_summaries_without_step(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        logger.log_phase_summaries(reward_totals={}, reward_term_summaries={}, reward_category_summaries={})
        assert logger.runs[0]["step"] is None

    def test_samples_and_runs_are_independent(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        logger.log_sample("s1", total=1.0, terms={}, metrics={})
        logger.log_phase_summaries(reward_totals={"count": 1.0}, reward_term_summaries={}, reward_category_summaries={})
        logger.log_sample("s2", total=2.0, terms={}, metrics={})
        assert len(logger.samples) == 2
        assert len(logger.runs) == 1

    def test_empty_logger_has_empty_lists(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        assert logger.samples == []
        assert logger.runs == []

    def test_terms_and_metrics_are_copied(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        terms = {"a": 1.0}
        metrics = {"b": 2}
        logger.log_sample("s1", total=1.0, terms=terms, metrics=metrics)
        terms["a"] = 999.0  # mutate original
        metrics["b"] = 999
        assert logger.samples[0]["reward_terms"] == {"a": 1.0}  # logger has copy
        assert logger.samples[0]["reward_metrics"] == {"b": 2}

    def test_log_phase_summaries_with_failure_stats(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
            failure_ratio=0.25,
            failure_count=5,
            step=10,
        )
        assert len(logger.runs) == 1
        entry = logger.runs[0]
        assert entry["failure_ratio"] == 0.25
        assert entry["failure_count"] == 5
        assert entry["step"] == 10

    def test_log_phase_summaries_without_failure_stats(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
        )
        entry = logger.runs[0]
        assert "failure_ratio" not in entry
        assert "failure_count" not in entry


class TestMakeWandBRewardLogger:
    def test_creates_logger_from_config(self) -> None:
        class MockWandBRun:
            pass

        mock_run = MockWandBRun()
        logging_config = reward_configs.LoggingConfig(
            log_tables=True,
            table_max_rows=500,
        )
        logger = reward_logging.make_wandb_reward_logger(mock_run, logging_config, step=100)
        assert isinstance(logger, reward_logging.WandBRewardLogger)
        assert logger._step == 100
        assert logger._log_generation_table is True
        assert logger._generation_table_max_rows == 500
        assert logger._generation_table_key == "generation_details"


class TestLoggingConfigValidation:
    def test_histogram_num_bins_must_be_positive(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="histogram_num_bins must be > 0"):
            reward_configs.LoggingConfig(histogram_num_bins=0)
        with pytest.raises(ValueError, match="histogram_num_bins must be > 0"):
            reward_configs.LoggingConfig(histogram_num_bins=-1)

    def test_histogram_max_samples_must_be_non_negative(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="histogram_max_samples must be >= 0"):
            reward_configs.LoggingConfig(histogram_max_samples=-1)
        # 0 is allowed (disables limit)
        config = reward_configs.LoggingConfig(histogram_max_samples=0)
        assert config.histogram_max_samples == 0


class TestWandBRewardLogger:
    def test_log_calls_wandb_run_log(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        logger.log_sample(
            "s1",
            total=1.5,
            terms={"t1": 1.0},
            metrics={"m1": True},
            generation_count=1,
        )
        assert len(logged_payloads) == 1
        payload = logged_payloads[0]
        # step is NOT included in per-generation payloads (uses generation_count as x-axis)
        assert "train/global_step" not in payload
        assert payload["generation_count"] == 1
        assert payload["reward/total"] == 1.5
        assert payload["t1"] == 1.0
        assert payload["m1"] is True

    def test_log_with_key_prefix(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        logger.set_key_prefix("exp/")
        logger.log_sample("s1", total=1.0, terms={"t1": 0.5}, metrics={}, step=1)
        payload = logged_payloads[0]
        assert "exp/reward/total" in payload
        assert "exp/t1" in payload

    def test_log_phase_summaries_calls_wandb_run_log(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        logger.log_phase_summaries(
            reward_totals={"count": 100.0, "mean": 0.5},
            reward_term_summaries={"t1": 0.3},
            reward_category_summaries={"cat1": 0.8},
            step=50,
        )
        assert len(logged_payloads) == 1
        payload = logged_payloads[0]
        assert payload["train/global_step"] == 50
        assert payload["count"] == 100.0
        assert payload["mean"] == 0.5
        assert payload["t1"] == 0.3
        assert payload["cat1"] == 0.8

    def test_set_step_updates_default_step_for_log_phase_summaries(self) -> None:
        """Verify set_step affects log_phase_summaries (only used for run-level summaries, not per-generation)."""
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        logger.set_step(42)
        logger.log_phase_summaries(reward_totals={"mean": 0.5}, reward_term_summaries={}, reward_category_summaries={})
        assert logged_payloads[0]["train/global_step"] == 42

    def test_explicit_step_overrides_default_for_log_phase_summaries(self) -> None:
        """Verify explicit step parameter overrides set_step for log_phase_summaries."""
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run, step=100)
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5}, reward_term_summaries={}, reward_category_summaries={}, step=200
        )
        assert logged_payloads[0]["train/global_step"] == 200  # explicit step overrides default

    def test_per_generation_log_does_not_include_step(self) -> None:
        """Verify per-generation log() does not include step (uses generation_count as x-axis instead)."""
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        logger.set_step(42)
        logger.log_sample("s1", total=1.0, terms={}, metrics={}, generation_count=1)
        assert "train/global_step" not in logged_payloads[0]
        assert logged_payloads[0]["generation_count"] == 1

    def test_log_phase_summaries_with_failure_stats_calls_wandb_run_log(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
            failure_ratio=0.25,
            failure_count=5,
            step=10,
        )
        assert len(logged_payloads) == 1
        payload = logged_payloads[0]
        assert payload["train/global_step"] == 10
        assert payload["failures/failure_ratio"] == 0.25
        assert payload["failures/failure_count"] == 5.0
        assert payload["mean"] == 0.5

    def test_log_phase_summaries_failure_stats_with_key_prefix(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        logger.set_key_prefix("train/")
        logger.log_phase_summaries(
            reward_totals={},
            reward_term_summaries={},
            reward_category_summaries={},
            failure_ratio=0.1,
            failure_count=2,
            step=5,
        )
        payload = logged_payloads[0]
        assert "train/failures/failure_ratio" in payload
        assert "train/failures/failure_count" in payload


class TestTableConstructionHelpers:
    """Tests for table construction helper functions."""

    def test_build_stats_table_from_running_stats(self) -> None:
        stats_items = [
            ("term_a", stats_utils.RunningStats()),
            ("term_b", stats_utils.RunningStats()),
        ]
        stats_items[0][1].update(1.0)
        stats_items[0][1].update(2.0)
        stats_items[1][1].update(3.0)
        table = reward_logging._build_stats_table_from_running_stats(stats_items, "term")
        assert table.columns == ["term", "mean", "std", "min", "max", "count"]
        assert len(table.data) == 2
        assert table.data[0][0] == "term_a"  # first row, first column
        assert table.data[1][0] == "term_b"  # second row, first column

    def test_build_stats_table_skips_empty_stats(self) -> None:
        stats_items = [
            ("term_a", stats_utils.RunningStats()),  # empty, count=0
            ("term_b", stats_utils.RunningStats()),
        ]
        stats_items[1][1].update(1.0)  # only term_b has data
        table = reward_logging._build_stats_table_from_running_stats(stats_items, "term")
        assert len(table.data) == 1
        assert table.data[0][0] == "term_b"

    def test_build_difficulty_bin_table(self) -> None:
        bin_stats = [stats_utils.RunningStats() for _ in range(3)]
        bin_stats[0].update(0.5)
        bin_stats[0].update(0.6)
        bin_stats[1].update(0.7)
        bin_edges = [0.0, 0.33, 0.67, 1.0]
        table = reward_logging._build_difficulty_bin_table(bin_stats, bin_edges)
        assert "bin_idx" in table.columns
        assert "reward_mean" in table.columns
        assert "reward_p50" in table.columns
        assert len(table.data) == 2  # bin_stats[2] is empty

    def test_build_difficulty_bin_table_with_quantiles(self) -> None:
        bin_stats = [stats_utils.RunningStats() for _ in range(2)]
        for val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            bin_stats[0].update(val)
        bin_edges = [0.0, 0.5, 1.0]
        bin_reward_values = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], []]
        table = reward_logging._build_difficulty_bin_table(bin_stats, bin_edges, bin_reward_values)
        assert len(table.data) == 1
        p10_col_idx = table.columns.index("reward_p10")
        p50_col_idx = table.columns.index("reward_p50")
        p90_col_idx = table.columns.index("reward_p90")
        assert table.data[0][p10_col_idx] is not None  # p10
        assert table.data[0][p50_col_idx] is not None  # p50
        assert table.data[0][p90_col_idx] is not None  # p90

    def test_build_difficulty_bin_term_table(self) -> None:
        bin_term_stats = {
            "term_a": [stats_utils.RunningStats(), stats_utils.RunningStats()],
            "term_b": [stats_utils.RunningStats(), stats_utils.RunningStats()],
        }
        bin_term_stats["term_a"][0].update(0.5)
        bin_term_stats["term_b"][1].update(0.7)
        bin_edges = [0.0, 0.5, 1.0]
        term_order = ["term_a", "term_b"]
        table = reward_logging._build_difficulty_bin_term_table(bin_term_stats, bin_edges, term_order)
        assert "term" in table.columns
        assert len(table.data) == 2  # two non-empty entries

    def test_build_parsing_category_table(self) -> None:
        category_stats = [
            ("cat_a", {"count": 10, "output_length_chars_mean": 100.0, "missing_reasoning_ratio": 0.1}),
            ("cat_b", {"count": 20, "output_length_chars_mean": 200.0}),
        ]
        table = reward_logging._build_parsing_category_table(category_stats)
        assert table.columns[0] == "category"
        assert table.columns[1] == "count"
        assert len(table.data) == 2
        # columns are dynamically built from union of keys
        assert "output_length_chars_mean" in table.columns
        assert "missing_reasoning_ratio" in table.columns

    def test_build_difficulty_bin_table_validates_bin_edges(self) -> None:
        """Verify validation catches mismatched bin_edges and bin_stats lengths."""
        import pytest

        bin_stats = [stats_utils.RunningStats() for _ in range(3)]
        # wrong: should be len(bin_stats) + 1 = 4
        bin_edges = [0.0, 0.5, 1.0]  # only 3 elements
        with pytest.raises(ValueError, match="bin_edges length"):
            reward_logging._build_difficulty_bin_table(bin_stats, bin_edges)

    def test_build_difficulty_bin_term_table_validates_bin_edges(self) -> None:
        """Verify validation catches bin_edges with fewer than 2 elements."""
        import pytest

        bin_term_stats = {"term_a": [stats_utils.RunningStats()]}
        bin_edges = [0.0]  # only 1 element
        with pytest.raises(ValueError, match="at least 2 elements"):
            reward_logging._build_difficulty_bin_term_table(bin_term_stats, bin_edges, ["term_a"])

    def test_build_difficulty_bin_term_table_validates_per_term_list_length(self) -> None:
        """Verify validation catches per-term stats lists with wrong length."""
        import pytest

        bin_edges = [0.0, 0.5, 1.0]  # 2 bins
        bin_term_stats = {
            "term_a": [stats_utils.RunningStats()],  # only 1 element, should be 2
        }
        with pytest.raises(ValueError, match="term 'term_a' has 1 bin stats, expected 2"):
            reward_logging._build_difficulty_bin_term_table(bin_term_stats, bin_edges, ["term_a"])


class TestScalarSuppressionPatterns:
    """Tests for scalar suppression pattern matching.

    Scalars are only suppressed when the corresponding table flag is True.
    This prevents silent data loss when table data is not provided.
    """

    def test_term_scalars_are_suppressed_when_flag_is_true(self) -> None:
        logger = reward_logging.WandBRewardLogger(object())
        # suppressed only when has_term_table=True
        assert logger._is_suppressed_scalar("reward/run/terms/accuracy/mean", has_term_table=True)
        assert logger._is_suppressed_scalar("reward/run/terms/accuracy/std", has_term_table=True)
        assert logger._is_suppressed_scalar("reward/run/terms/accuracy/min", has_term_table=True)
        assert logger._is_suppressed_scalar("reward/run/terms/accuracy/max", has_term_table=True)
        assert logger._is_suppressed_scalar("reward/run/terms/accuracy/count", has_term_table=True)
        # NOT suppressed when has_term_table=False (no table data provided)
        assert not logger._is_suppressed_scalar("reward/run/terms/accuracy/mean", has_term_table=False)

    def test_category_scalars_are_suppressed_when_flag_is_true(self) -> None:
        logger = reward_logging.WandBRewardLogger(object())
        # suppressed only when has_category_table=True
        assert logger._is_suppressed_scalar("reward/run/categories/code_type_python/mean", has_category_table=True)
        assert logger._is_suppressed_scalar("reward/run/categories/code_type_python/count", has_category_table=True)
        # nested category names with slashes
        assert logger._is_suppressed_scalar("reward/run/categories/code_type/original/mean", has_category_table=True)
        # NOT suppressed when has_category_table=False
        assert not logger._is_suppressed_scalar("reward/run/categories/code_type_python/mean", has_category_table=False)

    def test_parsing_category_scalars_are_suppressed_when_flag_is_true(self) -> None:
        logger = reward_logging.WandBRewardLogger(object())
        # all parsing/categories/* scalars are suppressed when flag is True
        assert logger._is_suppressed_scalar(
            "parsing/categories/code_type/output_length_chars/mean", has_parsing_category_table=True
        )
        assert logger._is_suppressed_scalar(
            "parsing/categories/code_type/output_length_chars/std", has_parsing_category_table=True
        )
        assert logger._is_suppressed_scalar(
            "parsing/categories/code_type/reasoning_length_tokens/count", has_parsing_category_table=True
        )
        assert logger._is_suppressed_scalar(
            "parsing/categories/predict_type/missing_reasoning_ratio", has_parsing_category_table=True
        )
        assert logger._is_suppressed_scalar("parsing/categories/predict_type/count", has_parsing_category_table=True)
        # nested category names with slashes
        assert logger._is_suppressed_scalar(
            "parsing/categories/code_type/original/output_length_chars/mean",
            has_parsing_category_table=True,
        )
        # any new metrics would also be suppressed
        assert logger._is_suppressed_scalar("parsing/categories/foo/some_new_metric", has_parsing_category_table=True)
        # NOT suppressed when has_parsing_category_table=False
        assert not logger._is_suppressed_scalar(
            "parsing/categories/code_type/output_length_chars/mean", has_parsing_category_table=False
        )

    def test_difficulty_bin_scalars_are_suppressed_when_flag_is_true(self) -> None:
        logger = reward_logging.WandBRewardLogger(object())
        # bin stats suppressed when has_bin_table=True
        assert logger._is_suppressed_scalar("difficulty/run/bin_0/count", has_bin_table=True)
        assert logger._is_suppressed_scalar("difficulty/run/bin_0/reward_mean", has_bin_table=True)
        assert logger._is_suppressed_scalar("difficulty/run/bin_5/reward_p50", has_bin_table=True)
        # bin term stats suppressed when has_bin_term_table=True
        assert logger._is_suppressed_scalar("difficulty/run/bin_0/term_accuracy/reward_mean", has_bin_term_table=True)
        # NOT suppressed when flags are False
        assert not logger._is_suppressed_scalar("difficulty/run/bin_0/count", has_bin_table=False)

    def test_total_scalars_are_not_suppressed(self) -> None:
        logger = reward_logging.WandBRewardLogger(object())
        # total scalars are never suppressed, regardless of flags
        assert not logger._is_suppressed_scalar("reward/run/total/mean", has_term_table=True)
        assert not logger._is_suppressed_scalar("reward/run/total/count", has_category_table=True)

    def test_difficulty_score_scalars_are_not_suppressed(self) -> None:
        logger = reward_logging.WandBRewardLogger(object())
        # score scalars are never suppressed (only bin stats are)
        assert not logger._is_suppressed_scalar("difficulty/run/score/mean", has_bin_table=True)
        assert not logger._is_suppressed_scalar("difficulty/run/score/p90", has_bin_table=True)

    def test_global_parsing_scalars_are_not_suppressed(self) -> None:
        logger = reward_logging.WandBRewardLogger(object())
        # global parsing scalars are never suppressed (only per-category are)
        assert not logger._is_suppressed_scalar("parsing/output_length_chars_mean", has_parsing_category_table=True)
        assert not logger._is_suppressed_scalar("parsing/missing_reasoning_ratio", has_parsing_category_table=True)


class TestWandBRewardLoggerTables:
    """Tests for table and histogram logging in WandBRewardLogger."""

    def test_log_phase_summaries_emits_term_table(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        term_stats = [("term_a", stats_utils.RunningStats())]
        term_stats[0][1].update(0.5)
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
            term_stats=term_stats,
        )
        payload = logged_payloads[0]
        assert "reward/run/term_summary" in payload
        assert isinstance(payload["reward/run/term_summary"], wandb.Table)

    def test_log_phase_summaries_skips_empty_tables(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        # empty term_stats (all RunningStats have count=0)
        term_stats = [("term_a", stats_utils.RunningStats())]
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
            term_stats=term_stats,
        )
        payload = logged_payloads[0]
        assert "reward/run/term_summary" not in payload  # empty table not logged

    def test_log_phase_summaries_emits_histogram(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run, histogram_num_bins=20)
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
            reward_total_values=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        payload = logged_payloads[0]
        assert "reward/run/total_histogram" in payload
        assert isinstance(payload["reward/run/total_histogram"], wandb.Histogram)

    def test_log_phase_summaries_filters_suppressed_scalars_when_table_provided(self) -> None:
        """Verify scalars are suppressed only when corresponding table data is provided."""
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        # provide term_stats so term scalars ARE suppressed
        term_stats = [("accuracy", stats_utils.RunningStats())]
        term_stats[0][1].update(0.8)
        logger.log_phase_summaries(
            reward_totals={"reward/run/total/mean": 0.5},
            reward_term_summaries={
                "reward/run/terms/accuracy/mean": 0.8,  # suppressed because term_stats provided
                "reward/run/terms/accuracy/count": 100,  # suppressed because term_stats provided
            },
            reward_category_summaries={},
            term_stats=term_stats,  # providing this causes term scalars to be suppressed
        )
        payload = logged_payloads[0]
        assert "reward/run/total/mean" in payload  # not suppressed
        assert "reward/run/terms/accuracy/mean" not in payload  # suppressed
        assert "reward/run/terms/accuracy/count" not in payload  # suppressed
        assert "reward/run/term_summary" in payload  # table is emitted

    def test_log_phase_summaries_keeps_scalars_when_table_not_provided(self) -> None:
        """Verify scalars are NOT suppressed when table data is not provided."""
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        # NO term_stats provided, so term scalars should NOT be suppressed
        logger.log_phase_summaries(
            reward_totals={"reward/run/total/mean": 0.5},
            reward_term_summaries={
                "reward/run/terms/accuracy/mean": 0.8,  # NOT suppressed (no term_stats)
                "reward/run/terms/accuracy/count": 100,  # NOT suppressed (no term_stats)
            },
            reward_category_summaries={},
            # term_stats NOT provided
        )
        payload = logged_payloads[0]
        assert "reward/run/total/mean" in payload
        assert "reward/run/terms/accuracy/mean" in payload  # NOT suppressed
        assert "reward/run/terms/accuracy/count" in payload  # NOT suppressed
        assert "reward/run/term_summary" not in payload  # no table emitted

    def test_log_phase_summaries_keeps_scalars_when_all_stats_empty(self) -> None:
        """Verify scalars are NOT suppressed when term_stats is provided but all have count=0."""
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        # term_stats provided but all have count=0, so no table will be emitted
        # and scalars should NOT be suppressed
        term_stats = [("accuracy", stats_utils.RunningStats())]  # count=0, no data
        logger.log_phase_summaries(
            reward_totals={"reward/run/total/mean": 0.5},
            reward_term_summaries={
                "reward/run/terms/accuracy/mean": 0.8,
                "reward/run/terms/accuracy/count": 100,
            },
            reward_category_summaries={},
            term_stats=term_stats,  # provided but empty
        )
        payload = logged_payloads[0]
        assert "reward/run/total/mean" in payload
        # scalars NOT suppressed because table would be empty (no non-zero stats)
        assert "reward/run/terms/accuracy/mean" in payload
        assert "reward/run/terms/accuracy/count" in payload
        # table NOT emitted because all stats have count=0
        assert "reward/run/term_summary" not in payload


class TestInMemoryRewardLoggerStructuredData:
    """Tests for InMemoryRewardLogger storing snapshots of structured kwargs."""

    def test_stores_term_stats_as_snapshot(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        running_stats = stats_utils.RunningStats()
        running_stats.update(0.5)
        term_stats = [("term_a", running_stats)]
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
            term_stats=term_stats,
        )
        assert len(logger.runs) == 1
        assert "term_stats" in logger.runs[0]
        # stored as dict snapshot, not original object
        stored = logger.runs[0]["term_stats"]
        assert stored[0][0] == "term_a"
        assert isinstance(stored[0][1], dict)  # snapshot is a dict
        assert stored[0][1]["count"] == 1
        # verify mutation of original doesn't affect stored snapshot
        running_stats.update(100.0)
        assert stored[0][1]["count"] == 1  # still 1, not 2

    def test_stores_reward_total_values_as_copy(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        values = [0.1, 0.2, 0.3]
        logger.log_phase_summaries(
            reward_totals={},
            reward_term_summaries={},
            reward_category_summaries={},
            reward_total_values=values,
        )
        stored = logger.runs[0]["reward_total_values"]
        assert stored == [0.1, 0.2, 0.3]
        # verify mutation of original doesn't affect stored copy
        values.append(0.4)
        assert len(stored) == 3  # still 3, not 4

    def test_stores_bin_stats_as_snapshot(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        bin_stats = [stats_utils.RunningStats() for _ in range(3)]
        bin_stats[0].update(0.5)
        bin_edges = [0.0, 0.33, 0.67, 1.0]
        logger.log_phase_summaries(
            reward_totals={},
            reward_term_summaries={},
            reward_category_summaries={},
            bin_stats=bin_stats,
            bin_edges=bin_edges,
        )
        stored_stats = logger.runs[0]["bin_stats"]
        stored_edges = logger.runs[0]["bin_edges"]
        # stored as dict snapshots
        assert isinstance(stored_stats[0], dict)
        assert stored_stats[0]["count"] == 1
        assert stored_edges == [0.0, 0.33, 0.67, 1.0]
        # verify mutation of original doesn't affect stored snapshot
        bin_stats[0].update(100.0)
        bin_edges.append(2.0)
        assert stored_stats[0]["count"] == 1  # still 1
        assert len(stored_edges) == 4  # still 4
