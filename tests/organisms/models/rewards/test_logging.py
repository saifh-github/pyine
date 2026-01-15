"""Tests for reward logging implementations."""

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.logging as reward_logging


class TestInMemoryRewardLogger:
    def test_log_records_sample_event(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        logger.log(
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
        logger.log("sample_1", total=1.0, terms={}, metrics={})
        assert logger.samples[0]["step"] is None

    def test_log_multiple_samples(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        logger.log("s1", total=1.0, terms={}, metrics={}, step=1)
        logger.log("s2", total=2.0, terms={}, metrics={}, step=2)
        logger.log("s3", total=3.0, terms={}, metrics={}, step=3)
        assert len(logger.samples) == 3
        assert [e["sample_id"] for e in logger.samples] == ["s1", "s2", "s3"]
        assert [e["reward_total"] for e in logger.samples] == [1.0, 2.0, 3.0]

    def test_log_run_records_run_event(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        logger.log_run(
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

    def test_log_run_without_step(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        logger.log_run(reward_totals={}, reward_term_summaries={}, reward_category_summaries={})
        assert logger.runs[0]["step"] is None

    def test_samples_and_runs_are_independent(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        logger.log("s1", total=1.0, terms={}, metrics={})
        logger.log_run(reward_totals={"count": 1.0}, reward_term_summaries={}, reward_category_summaries={})
        logger.log("s2", total=2.0, terms={}, metrics={})
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
        logger.log("s1", total=1.0, terms=terms, metrics=metrics)
        terms["a"] = 999.0  # mutate original
        metrics["b"] = 999
        assert logger.samples[0]["reward_terms"] == {"a": 1.0}  # logger has copy
        assert logger.samples[0]["reward_metrics"] == {"b": 2}

    def test_log_failures_records_to_failures_list(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        logger.log_failures(failure_ratio=0.25, failure_count=5, step=10)
        assert len(logger.failures) == 1
        assert logger.runs == []  # not mixed into runs
        entry = logger.failures[0]
        assert entry["failure_ratio"] == 0.25
        assert entry["failure_count"] == 5
        assert entry["step"] == 10

    def test_log_failures_without_step(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        logger.log_failures(failure_ratio=0.1, failure_count=2)
        assert logger.failures[0]["step"] is None

    def test_failures_list_is_initially_empty(self) -> None:
        logger = reward_logging.InMemoryRewardLogger()
        assert logger.failures == []


class TestMakeWandBRewardLogger:
    def test_creates_logger_from_config(self) -> None:
        class MockWandBRun:
            pass

        mock_run = MockWandBRun()
        logging_config = reward_configs.LoggingConfig(
            log_tables=True,
            table_flush_every_n_generations=50,
            table_max_rows=500,
        )
        logger = reward_logging.make_wandb_reward_logger(mock_run, logging_config, step=100)
        assert isinstance(logger, reward_logging.WandBRewardLogger)
        assert logger._step == 100
        assert logger._log_tables is True
        assert logger._table_flush_every_n_generations == 50
        assert logger._table_max_rows == 500
        assert logger._table_key == "generation_details"


class TestWandBRewardLogger:
    def test_log_calls_wandb_run_log(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        logger.log(
            "s1",
            total=1.5,
            terms={"t1": 1.0},
            metrics={"m1": True},
            step=10,
        )
        assert len(logged_payloads) == 1
        payload = logged_payloads[0]
        assert payload["train/global_step"] == 10
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
        logger.log("s1", total=1.0, terms={"t1": 0.5}, metrics={}, step=1)
        payload = logged_payloads[0]
        assert "exp/reward/total" in payload
        assert "exp/t1" in payload

    def test_log_run_calls_wandb_run_log(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        logger.log_run(
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

    def test_set_step_updates_default_step(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        logger.set_step(42)
        logger.log("s1", total=1.0, terms={}, metrics={})
        assert logged_payloads[0]["train/global_step"] == 42

    def test_explicit_step_overrides_default(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run, step=100)
        logger.log("s1", total=1.0, terms={}, metrics={}, step=200)
        assert logged_payloads[0]["train/global_step"] == 200  # explicit step overrides default

    def test_log_failures_calls_wandb_run_log(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        logger.log_failures(failure_ratio=0.25, failure_count=5, step=10)
        assert len(logged_payloads) == 1
        payload = logged_payloads[0]
        assert payload["train/global_step"] == 10
        assert payload["failures/failure_ratio"] == 0.25
        assert payload["failures/failure_count"] == 5.0

    def test_log_failures_with_key_prefix(self) -> None:
        logged_payloads: list[dict] = []

        class MockWandBRun:
            def log(self, payload: dict) -> None:
                logged_payloads.append(dict(payload))

        mock_run = MockWandBRun()
        logger = reward_logging.WandBRewardLogger(mock_run)
        logger.set_key_prefix("train/")
        logger.log_failures(failure_ratio=0.1, failure_count=2, step=5)
        payload = logged_payloads[0]
        assert "train/failures/failure_ratio" in payload
        assert "train/failures/failure_count" in payload
