"""Tests for reward logging implementations."""

import pathlib

import pydantic
import pytest
import wandb

import pyine.data.utils.lmdb_io
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
            ("cat_a", {"count": 10, "output_length_tokens_mean": 20.0, "missing_reasoning_ratio": 0.1}),
            ("cat_b", {"count": 20, "output_length_tokens_mean": 40.0}),
        ]
        table = reward_logging._build_parsing_category_table(category_stats)
        assert table.columns[0] == "category"
        assert table.columns[1] == "count"
        assert len(table.data) == 2
        # columns are dynamically built from union of keys
        assert "output_length_tokens_mean" in table.columns
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
            "parsing/categories/code_type/output_length_tokens/mean", has_parsing_category_table=True
        )
        assert logger._is_suppressed_scalar(
            "parsing/categories/code_type/output_length_tokens/std", has_parsing_category_table=True
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
            "parsing/categories/code_type/original/output_length_tokens/mean",
            has_parsing_category_table=True,
        )
        # any new metrics would also be suppressed
        assert logger._is_suppressed_scalar("parsing/categories/foo/some_new_metric", has_parsing_category_table=True)
        # NOT suppressed when has_parsing_category_table=False
        assert not logger._is_suppressed_scalar(
            "parsing/categories/code_type/output_length_tokens/mean", has_parsing_category_table=False
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
        assert not logger._is_suppressed_scalar("parsing/output_length_tokens_mean", has_parsing_category_table=True)
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


class TestSecondarySourceTable:
    """Tests for secondary difficulty source summary table emission."""

    def test_log_phase_summaries_emits_secondary_source_table_when_stats_exist(self) -> None:
        """Verify secondary_source_summary table is emitted when secondary stats have data."""
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        # create secondary stats with data
        secondary_stats = {"source_a": stats_utils.RunningStats(), "source_b": stats_utils.RunningStats()}
        secondary_stats["source_a"].update(0.5)
        secondary_stats["source_a"].update(0.7)
        secondary_stats["source_b"].update(0.3)
        # create correlation stats
        secondary_corr_stats = {"source_a": stats_utils.RunningCorrStats()}
        secondary_corr_stats["source_a"].update(0.5, 0.8)
        secondary_corr_stats["source_a"].update(0.7, 0.9)
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
            secondary_stats=secondary_stats,
            secondary_corr_stats=secondary_corr_stats,
        )
        assert len(logged_payloads) == 1
        payload = logged_payloads[0]
        assert "difficulty/run/secondary_source_summary" in payload
        table = payload["difficulty/run/secondary_source_summary"]
        assert isinstance(table, wandb.Table)
        assert "source" in table.columns
        assert "mean" in table.columns
        assert "reward_correlation" in table.columns
        assert len(table.data) == 2  # two sources

    def test_log_phase_summaries_skips_secondary_source_table_when_empty(self) -> None:
        """Verify secondary_source_summary table is not emitted when stats are empty."""
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        # empty secondary stats
        secondary_stats = {"source_a": stats_utils.RunningStats()}  # count=0
        secondary_corr_stats = {"source_a": stats_utils.RunningCorrStats()}  # count=0
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
            secondary_stats=secondary_stats,
            secondary_corr_stats=secondary_corr_stats,
        )
        assert len(logged_payloads) == 1
        payload = logged_payloads[0]
        # table should NOT be emitted when all stats are empty
        assert "difficulty/run/secondary_source_summary" not in payload

    def test_in_memory_logger_stores_secondary_stats_as_snapshot(self) -> None:
        """Verify InMemoryRewardLogger stores secondary stats for testing parity."""
        logger = reward_logging.InMemoryRewardLogger()
        # create secondary stats with data
        secondary_stats = {"source_a": stats_utils.RunningStats()}
        secondary_stats["source_a"].update(0.5)
        secondary_corr_stats = {"source_a": stats_utils.RunningCorrStats()}
        secondary_corr_stats["source_a"].update(0.5, 0.8)
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
            secondary_stats=secondary_stats,
            secondary_corr_stats=secondary_corr_stats,
        )
        assert len(logger.runs) == 1
        run_entry = logger.runs[0]
        # verify secondary stats stored as state dicts
        assert "secondary_stats" in run_entry
        assert "source_a" in run_entry["secondary_stats"]
        assert run_entry["secondary_stats"]["source_a"]["count"] == 1
        # verify secondary corr stats stored as state dicts
        assert "secondary_corr_stats" in run_entry
        assert "source_a" in run_entry["secondary_corr_stats"]
        assert run_entry["secondary_corr_stats"]["source_a"]["count"] == 1
        # verify mutation of original doesn't affect snapshot
        secondary_stats["source_a"].update(100.0)
        assert run_entry["secondary_stats"]["source_a"]["count"] == 1  # still 1


class TestCategoryTermStatsHelpers:
    """Tests for per-category per-term stats table construction and emission."""

    def test_build_category_term_stats_table(self) -> None:
        stats_a1 = stats_utils.RunningStats()
        stats_a1.update(0.5)
        stats_a1.update(0.7)
        stats_a2 = stats_utils.RunningStats()
        stats_a2.update(1.0)
        stats_b1 = stats_utils.RunningStats()
        stats_b1.update(0.3)
        category_term_stats = {
            "code_type/original": [("term_a", stats_a1), ("term_b", stats_a2)],
            "code_type/bugfix": [("term_a", stats_b1)],
        }
        table = reward_logging._build_category_term_stats_table(category_term_stats)
        assert table.columns == ["category", "term", "mean", "std", "min", "max", "count"]
        assert len(table.data) == 3
        # check contents as set of (category, term, count) tuples for order-independence
        row_keys = {(row[0], row[1], row[6]) for row in table.data}
        assert ("code_type/original", "term_a", 2) in row_keys
        assert ("code_type/original", "term_b", 1) in row_keys
        assert ("code_type/bugfix", "term_a", 1) in row_keys

    def test_build_category_term_stats_table_empty(self) -> None:
        table = reward_logging._build_category_term_stats_table({})
        assert table.columns == ["category", "term", "mean", "std", "min", "max", "count"]
        assert len(table.data) == 0

    def test_has_nonempty_category_term_stats(self) -> None:
        assert not reward_logging._has_nonempty_category_term_stats(None)
        assert not reward_logging._has_nonempty_category_term_stats({})
        # all zero-count
        empty_stats = stats_utils.RunningStats()
        assert not reward_logging._has_nonempty_category_term_stats({"cat": [("t", empty_stats)]})
        # valid case
        valid_stats = stats_utils.RunningStats()
        valid_stats.update(1.0)
        assert reward_logging._has_nonempty_category_term_stats({"cat": [("t", valid_stats)]})

    def test_in_memory_logger_stores_category_term_stats_snapshot(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        running_stats = stats_utils.RunningStats()
        running_stats.update(0.5)
        category_term_stats = {
            "code_type/original": [("term_a", running_stats)],
        }
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
            category_term_stats=category_term_stats,
        )
        assert len(logger.runs) == 1
        stored = logger.runs[0]["category_term_stats"]
        assert "code_type/original" in stored
        assert stored["code_type/original"][0][0] == "term_a"
        assert isinstance(stored["code_type/original"][0][1], dict)  # snapshot
        assert stored["code_type/original"][0][1]["count"] == 1
        # verify mutation of original doesn't affect stored snapshot
        running_stats.update(100.0)
        assert stored["code_type/original"][0][1]["count"] == 1

    def test_wandb_logger_emits_category_term_summary_table(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        stats_obj = stats_utils.RunningStats()
        stats_obj.update(0.8)
        category_term_stats = {
            "code_type/original": [("accuracy", stats_obj)],
        }
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
            category_term_stats=category_term_stats,
        )
        assert len(logged_payloads) == 1
        payload = logged_payloads[0]
        assert "reward/run/category_term_summary" in payload
        table = payload["reward/run/category_term_summary"]
        assert isinstance(table, wandb.Table)
        assert len(table.data) == 1
        assert table.data[0][0] == "code_type/original"
        assert table.data[0][1] == "accuracy"

    def test_wandb_logger_skips_category_term_summary_when_empty(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        # all zero-count stats
        empty_stats = stats_utils.RunningStats()
        category_term_stats = {"code_type/original": [("accuracy", empty_stats)]}
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
            category_term_stats=category_term_stats,
        )
        assert len(logged_payloads) == 1
        payload = logged_payloads[0]
        assert "reward/run/category_term_summary" not in payload


class TestDiskRewardLogger:
    def test_write_and_read_back_sample(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "generations"
        disk_logger = reward_logging.DiskRewardLogger(output_path=output_dir)
        disk_logger.set_key_prefix("train/")
        disk_logger.log_sample(
            "TACO/s0001/t0001",
            generation_count=100,
            total=0.75,
            model_output="print(42)",
            prompt="What is the output?",
            expected_output="42",
            terms={"hard_match": 0.75},
            step=10,
            batch_count=5,
            completion_idx=0,
            rank=0,
            predict_type="program_output",
            code_type="original",
            tags=["subset:train"],
            categories=["code_type/original"],
        )
        disk_logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        assert len(reader.key_map) == 1
        key = next(iter(reader.key_map))
        assert key == "train/TACO/s0001/t0001/100"
        record = reader.get(key)
        assert record["model_output"] == "print(42)"
        assert record["reward_total"] == 0.75
        assert record["prompt"] == "What is the output?"
        assert record["reward_terms"] == {"hard_match": 0.75}
        assert record["tags"] == ["subset:train"]
        assert record["categories"] == ["code_type/original"]
        assert record["key_prefix"] == "train/"
        reader.close()

    def test_rejects_non_empty_output_path(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        (output_dir / "data.mdb").write_bytes(b"fake")
        with pytest.raises(ValueError, match="non-empty"):
            reward_logging.DiskRewardLogger(output_path=output_dir)

    def test_phase_prefix_normalization(self, tmp_path: pathlib.Path) -> None:
        disk_logger = reward_logging.DiskRewardLogger(output_path=tmp_path / "gen")
        disk_logger.set_key_prefix("eval")  # no trailing slash
        assert disk_logger.get_key_prefix() == "eval/"

    def test_should_log_sample_frequency_gating(self, tmp_path: pathlib.Path) -> None:
        disk_logger = reward_logging.DiskRewardLogger(
            output_path=tmp_path / "gen",
            log_every_n_generations=3,
        )
        assert not disk_logger.should_log_sample(1)
        assert not disk_logger.should_log_sample(2)
        assert disk_logger.should_log_sample(3)
        assert not disk_logger.should_log_sample(4)
        assert disk_logger.should_log_sample(6)
        disk_logger.close()

    def test_multiple_samples_round_trip(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "gen"
        disk_logger = reward_logging.DiskRewardLogger(output_path=output_dir)
        disk_logger.set_key_prefix("train/")
        for gen_idx in range(5):
            disk_logger.log_sample(
                f"sample_{gen_idx}",
                generation_count=gen_idx + 1,
                total=float(gen_idx),
                model_output=f"output_{gen_idx}",
            )
        disk_logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        assert len(reader.key_map) == 5
        record = reader.get("train/sample_2/3")
        assert record["model_output"] == "output_2"
        assert record["reward_total"] == 2.0
        reader.close()

    def test_reward_total_none_when_log_total_false(self, tmp_path: pathlib.Path) -> None:
        output_dir = tmp_path / "gen"
        disk_logger = reward_logging.DiskRewardLogger(output_path=output_dir)
        disk_logger.set_key_prefix("train/")
        disk_logger.log_sample("s1", generation_count=1, total=None, model_output="out")
        disk_logger.close()
        reader = pyine.data.utils.lmdb_io.LMDBReader(output_dir)
        record = reader.get("train/s1/1")
        assert record["reward_total"] is None
        reader.close()


class TestCompositeRewardLogger:
    def test_fan_out_log_sample(self) -> None:
        inner1 = reward_logging.InMemoryRewardLogger(log_every_n_generations=1)
        inner2 = reward_logging.InMemoryRewardLogger(log_every_n_generations=1)
        composite = reward_logging.CompositeRewardLogger([inner1, inner2])
        composite.log_sample("s1", generation_count=1, total=0.5)
        assert len(inner1.samples) == 1
        assert len(inner2.samples) == 1

    def test_per_logger_frequency_gating(self) -> None:
        inner_all = reward_logging.InMemoryRewardLogger(log_every_n_generations=1)
        inner_every3 = reward_logging.InMemoryRewardLogger(log_every_n_generations=3)
        composite = reward_logging.CompositeRewardLogger([inner_all, inner_every3])
        for gen_count in range(1, 7):
            if composite.should_log_sample(gen_count):
                composite.log_sample(f"s{gen_count}", generation_count=gen_count, total=float(gen_count))
        assert len(inner_all.samples) == 6  # logs all
        assert len(inner_every3.samples) == 2  # logs gen_count=3 and gen_count=6

    def test_should_log_sample_union_semantics(self) -> None:
        inner_every2 = reward_logging.InMemoryRewardLogger(log_every_n_generations=2)
        inner_every3 = reward_logging.InMemoryRewardLogger(log_every_n_generations=3)
        composite = reward_logging.CompositeRewardLogger([inner_every2, inner_every3])
        assert not composite.should_log_sample(1)
        assert composite.should_log_sample(2)  # inner_every2 says yes
        assert composite.should_log_sample(3)  # inner_every3 says yes
        assert composite.should_log_sample(4)  # inner_every2 says yes
        assert not composite.should_log_sample(5)  # neither
        assert composite.should_log_sample(6)  # both say yes

    def test_set_key_prefix_propagates(self) -> None:
        inner1 = reward_logging.InMemoryRewardLogger()
        inner2 = reward_logging.InMemoryRewardLogger()
        composite = reward_logging.CompositeRewardLogger([inner1, inner2])
        composite.set_key_prefix("eval/")
        assert inner1.get_key_prefix() == "eval/"
        assert inner2.get_key_prefix() == "eval/"
        assert composite.get_key_prefix() == "eval/"

    def test_close_propagates(self, tmp_path: pathlib.Path) -> None:
        disk_logger = reward_logging.DiskRewardLogger(output_path=tmp_path / "gen")
        inner = reward_logging.InMemoryRewardLogger()
        composite = reward_logging.CompositeRewardLogger([inner, disk_logger])
        composite.set_key_prefix("train/")
        composite.log_sample("s1", generation_count=1, total=1.0, model_output="out")
        composite.close()
        # verify disk logger was closed (LMDB readable)
        reader = pyine.data.utils.lmdb_io.LMDBReader(tmp_path / "gen")
        assert len(reader.key_map) == 1
        reader.close()

    def test_empty_loggers_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            reward_logging.CompositeRewardLogger([])

    def test_set_step_and_epoch_propagate(self) -> None:
        inner = reward_logging.InMemoryRewardLogger()
        composite = reward_logging.CompositeRewardLogger([inner])
        composite.set_step(42)
        composite.set_epoch(1.5)
        assert inner._step == 42
        assert inner._epoch == 1.5


class TestMakeDiskRewardLogger:
    def test_creates_logger_from_config(self, tmp_path: pathlib.Path) -> None:
        config = reward_configs.GenerationExportConfig(
            output_path=tmp_path / "gen",
            log_every_n_generations=5,
        )
        disk_logger = reward_logging.make_disk_reward_logger(config)
        assert isinstance(disk_logger, reward_logging.DiskRewardLogger)
        assert disk_logger._log_every_n_generations == 5
        disk_logger.close()


class TestGenerationExportConfig:
    def test_frozen_config(self, tmp_path: pathlib.Path) -> None:
        config = reward_configs.GenerationExportConfig(output_path=tmp_path / "gen")
        with pytest.raises(pydantic.ValidationError):
            config.output_path = pathlib.Path("/other")  # type: ignore[misc]

    def test_extra_forbid(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(pydantic.ValidationError):
            reward_configs.GenerationExportConfig(output_path=tmp_path / "gen", unknown_field="x")  # type: ignore[call-arg]

    def test_log_every_n_must_be_positive(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(pydantic.ValidationError):
            reward_configs.GenerationExportConfig(output_path=tmp_path / "gen", log_every_n_generations=0)
