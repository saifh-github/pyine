import json
import typing

import pytest
import pytest_mock

import pyine.organisms.models.rewards.core.logging as reward_logging


class _FakeWandBRun:
    def __init__(self) -> None:
        self.logged: list[dict[str, object]] = []

    def log(self, payload: dict[str, object]) -> None:
        self.logged.append(dict(payload))


class TestWandBRewardLogger:
    def test_key_prefix_applies_to_scalars(self) -> None:
        fake_run = _FakeWandBRun()
        logger = reward_logging.WandBRewardLogger(fake_run)
        logger.set_key_prefix("train")
        logger.log_sample(
            "s1",
            total=1.0,
            terms={"reward/terms/t": 0.25},
            metrics={"reward/metrics/m": 2},
            generation_count=7,
        )
        assert len(fake_run.logged) == 1
        payload = fake_run.logged[0]
        # step is NOT included in per-generation payloads (uses generation_count as x-axis)
        assert "train/global_step" not in payload
        assert payload["train/generation_count"] == 7
        assert payload["train/reward/total"] == pytest.approx(1.0)
        assert payload["train/reward/terms/t"] == pytest.approx(0.25)
        assert payload["train/reward/metrics/m"] == 2

    def test_key_prefix_applies_to_table_key_and_json(self, mocker: pytest_mock.MockerFixture) -> None:
        # capture table rows via add_data calls
        added_rows: list[tuple] = []

        class FakeTable:
            def __init__(self, columns: list[str]) -> None:
                self.columns = columns

            def add_data(self, *args: object) -> None:
                added_rows.append(args)

        mocker.patch.object(reward_logging.wandb, "Table", FakeTable)
        fake_run = _FakeWandBRun()
        logger = reward_logging.WandBRewardLogger(
            fake_run,
            log_tables=True,
            table_max_rows=1,
        )
        logger.set_key_prefix("train/")
        logger.log_sample(
            "s1",
            generation_count=1,  # pass generation_count to trigger frequency gating/flush
            total=1.0,
            terms={"reward/terms/t": 0.25},
            metrics={"reward/metrics/m": 2},
            step=7,
            prompt="test prompt",
            model_output="test output",
            reasoning="test reasoning",
            final_answer="test answer",
            categories=["difficulty:easy", "total_steps:10"],
            tags=["tag1", "tag2:value"],
        )
        assert len(fake_run.logged) == 2
        table_payload = fake_run.logged[1]
        # verify table key is prefixed
        assert "train/generation_details" in table_payload
        # verify table was created with correct columns (includes reasoning, final_answer, reward/ prefixes)
        table_obj = table_payload["train/generation_details"]
        expected_cols = [
            "sample_id",
            "generation_count",
            "batch_count",
            "local_batch_idx",
            "completion_idx",
            "rank",
            "step",
            "prompt",
            "expected_output",
            "model_output",
            "reasoning",
            "final_answer",
            "reward_total",
            "reward_terms_json",
            "reward_terms_raw_json",
            "reward_metrics_json",
            "categories_json",
            "tags_json",
            "difficulty_source",
            "difficulty_score",
            "difficulty_bin",
            "difficulty_raw_primary",
            "difficulty_secondary_json",
            "predict_type",
            "code_type",
            "has_code_override",
        ]
        assert table_obj.columns == expected_cols
        # verify row content
        assert len(added_rows) == 1
        row = added_rows[0]
        assert row[0] == "s1"  # sample_id
        assert row[1] == 1  # generation_count
        assert row[2] is None  # batch_count (not provided)
        assert row[3] is None  # local_batch_idx (not provided)
        assert row[4] is None  # completion_idx (not provided)
        assert row[5] is None  # rank (not provided)
        assert row[6] == 7  # step
        assert row[7] == "test prompt"  # prompt
        assert row[8] is None  # expected_output (not provided)
        assert row[9] == "test output"  # model_output
        assert row[10] == "test reasoning"  # reasoning
        assert row[11] == "test answer"  # final_answer
        assert row[12] == 1.0  # reward_total
        # verify terms_json contains prefixed keys
        terms_json = json.loads(typing.cast("str", row[13]))
        assert "train/reward/terms/t" in terms_json
        assert terms_json["train/reward/terms/t"] == 0.25
        assert row[14] is None  # reward_terms_raw_json (not provided in this test)
        # verify metrics_json contains prefixed keys
        metrics_json = json.loads(typing.cast("str", row[15]))
        assert "train/reward/metrics/m" in metrics_json
        assert metrics_json["train/reward/metrics/m"] == 2
        # verify categories_json contains category labels
        categories_json = json.loads(typing.cast("str", row[16]))
        assert categories_json == ["difficulty:easy", "total_steps:10"]
        # verify tags_json contains sample tags
        tags_json = json.loads(typing.cast("str", row[17]))
        assert tags_json == ["tag1", "tag2:value"]

    def test_raw_terms_are_logged_in_scalars_and_table_json(self, mocker: pytest_mock.MockerFixture) -> None:
        added_rows: list[tuple] = []
        captured_table: typing.Any = None

        class FakeTable:
            def __init__(self, columns: list[str]) -> None:
                nonlocal captured_table
                self.columns = columns
                captured_table = self

            def add_data(self, *args: object) -> None:
                added_rows.append(args)

        mocker.patch.object(reward_logging.wandb, "Table", FakeTable)
        fake_run = _FakeWandBRun()
        logger = reward_logging.WandBRewardLogger(
            fake_run,
            log_tables=True,
            table_max_rows=1,
        )
        logger.set_key_prefix("train/")
        logger.log_sample(
            "s1",
            generation_count=1,
            total=1.0,
            terms={"reward/terms/t": 0.25},
            raw_terms={"t": 2.5},
            metrics={"reward/metrics/m": 2},
            step=7,
            prompt="test prompt",
            model_output="test output",
            reasoning="test reasoning",
            final_answer="test answer",
        )
        assert len(fake_run.logged) == 2
        scalar_payload = typing.cast("dict[str, object]", fake_run.logged[0])
        assert scalar_payload["train/reward/raw_terms/t"] == pytest.approx(2.5)

        assert len(added_rows) == 1
        row = added_rows[0]
        raw_terms_col_idx = captured_table.columns.index("reward_terms_raw_json")
        raw_terms_json = json.loads(typing.cast("str", row[raw_terms_col_idx]))
        assert raw_terms_json["train/reward/raw_terms/t"] == pytest.approx(2.5)

    def test_set_key_prefix_switches_prefix_dynamically(self) -> None:
        fake_run = _FakeWandBRun()
        logger = reward_logging.WandBRewardLogger(fake_run)
        logger.set_key_prefix("train")
        # log with initial train prefix
        logger.log_sample(
            "s1",
            total=1.0,
            terms={"reward/terms/t": 0.25},
            metrics={},
            step=1,
        )
        # switch to valid prefix
        logger.set_key_prefix("valid")
        # log with new valid prefix
        logger.log_sample(
            "s2",
            total=2.0,
            terms={"reward/terms/t": 0.5},
            metrics={},
            step=2,
        )
        assert len(fake_run.logged) == 2
        # verify first entry has train prefix
        payload1 = typing.cast("dict[str, object]", fake_run.logged[0])
        assert "train/reward/total" in payload1
        assert payload1["train/reward/total"] == pytest.approx(1.0)
        # verify second entry has valid prefix
        payload2 = typing.cast("dict[str, object]", fake_run.logged[1])
        assert "valid/reward/total" in payload2
        assert "train/reward/total" not in payload2
        assert payload2["valid/reward/total"] == pytest.approx(2.0)

    def test_set_key_prefix_normalizes_prefix(self) -> None:
        fake_run = _FakeWandBRun()
        logger = reward_logging.WandBRewardLogger(fake_run)
        # set prefix without trailing slash
        logger.set_key_prefix("eval")
        logger.log_sample(
            "s1",
            total=1.0,
            terms={},
            metrics={},
            step=1,
        )
        payload = typing.cast("dict[str, object]", fake_run.logged[0])
        # should have normalized prefix with trailing slash applied
        assert "eval/reward/total" in payload

    def test_set_key_prefix_to_empty_removes_prefix(self) -> None:
        fake_run = _FakeWandBRun()
        logger = reward_logging.WandBRewardLogger(fake_run)
        logger.set_key_prefix("train")
        # switch to empty prefix
        logger.set_key_prefix("")
        logger.log_sample(
            "s1",
            total=1.0,
            terms={},
            metrics={},
            step=1,
        )
        payload = typing.cast("dict[str, object]", fake_run.logged[0])
        # should have no key_prefix (train/eval), only the hardcoded reward/ prefix
        assert "reward/total" in payload
        assert "train/reward/total" not in payload

    def test_frequency_gating_query_method(self) -> None:
        """Test that should_log_sample returns correct values based on frequency settings."""
        fake_run = _FakeWandBRun()
        logger = reward_logging.WandBRewardLogger(
            fake_run,
            log_every_n_generations=2,
        )
        # verify query method returns correct values based on frequency settings
        assert not logger.should_log_sample(1)  # 1 % 2 != 0
        assert logger.should_log_sample(2)  # 2 % 2 == 0
        assert not logger.should_log_sample(3)  # 3 % 2 != 0
        assert logger.should_log_sample(4)  # 4 % 2 == 0
        # verify that log_sample always emits when called (gating is caller's responsibility)
        logger.log_sample("s1", generation_count=1, total=1.0, terms={}, metrics={})
        logger.log_sample("s2", generation_count=2, total=2.0, terms={}, metrics={})
        assert len(fake_run.logged) == 2  # both calls emit since gating is caller's job
        payload = typing.cast("dict[str, object]", fake_run.logged[1])
        assert payload["generation_count"] == 2
        assert payload["reward/total"] == pytest.approx(2.0)

    def test_flush_interval_triggers_on_generation_count(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that table flush triggers based on generation_count when buffer has rows."""
        added_rows: list[tuple[object, ...]] = []

        class FakeTable:
            def __init__(self, columns: list[str]) -> None:
                self.columns = columns

            def add_data(self, *args: object) -> None:
                added_rows.append(args)

        mocker.patch.object(reward_logging.wandb, "Table", FakeTable)
        fake_run = _FakeWandBRun()
        logger = reward_logging.WandBRewardLogger(
            fake_run,
            log_tables=True,
            log_every_n_generations=1,  # log every sample
            table_max_rows=5,
        )
        for generation_count in range(1, 11):
            logger.log_sample(
                f"s{generation_count}",
                generation_count=generation_count,
                total=float(generation_count),
                terms={},
                metrics={},
                step=generation_count,
            )
        table_payloads = [p for p in fake_run.logged if "generation_details" in p]
        assert len(table_payloads) == 2  # flush at generation_count 5 and 10
        assert len(added_rows) == 10  # all 10 calls add rows

    def test_unified_gating_controls_both_scalars_and_table_rows(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that log_every_n_generations controls both scalar and table row logging."""
        added_rows: list[tuple[object, ...]] = []

        class FakeTable:
            def __init__(self, columns: list[str]) -> None:
                self.columns = columns

            def add_data(self, *args: object) -> None:
                added_rows.append(args)

        mocker.patch.object(reward_logging.wandb, "Table", FakeTable)
        fake_run = _FakeWandBRun()
        logger = reward_logging.WandBRewardLogger(
            fake_run,
            log_tables=True,
            log_every_n_generations=2,
            table_max_rows=6,
        )
        # verify query method returns unified values
        for generation_count in range(1, 7):
            should_log = logger.should_log_sample(generation_count)
            expected = generation_count % 2 == 0  # 2, 4, 6
            assert should_log == expected, f"mismatch at {generation_count}"
        # log_sample always emits when called (caller is responsible for gating)
        for generation_count in range(1, 7):
            logger.log_sample(
                f"s{generation_count}",
                generation_count=generation_count,
                total=float(generation_count),
                terms={},
                metrics={},
                step=generation_count,
            )
        scalar_payloads = [p for p in fake_run.logged if "reward/total" in p]
        table_payloads = [p for p in fake_run.logged if "generation_details" in p]
        assert len(scalar_payloads) == 6  # all 6 calls emit scalars
        assert len(table_payloads) == 1  # flush at generation_count 6
        assert len(added_rows) == 6  # all 6 calls add rows


class TestWandBDefineMetric:
    """Tests for wandb.define_metric calls that configure x-axes for metrics."""

    def test_generation_step_metrics_defined_on_first_log(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Verify that define_metric for generation_count is called lazily on first log()."""
        fake_run = _FakeWandBRun()
        mock_define_metric = mocker.patch.object(reward_logging.wandb, "define_metric")
        # mock wandb.run to be non-None so define_metric is actually called
        mocker.patch.object(reward_logging.wandb, "run", fake_run)
        logger = reward_logging.WandBRewardLogger(fake_run)
        logger.set_key_prefix("train")  # metrics are defined per-prefix
        # define_metric should NOT be called yet (deferred until first log)
        mock_define_metric.assert_not_called()
        # first log call should trigger define_metric for "train" prefix
        logger.log_sample("s1", total=1.0, terms={}, metrics={}, step=1)
        # verify define_metric was called for generation_count patterns
        calls = mock_define_metric.call_args_list
        assert len(calls) > 0, "define_metric should be called on first log()"
        # check that generation_count is used as step_metric for per-generation patterns
        generation_patterns_found = []
        for call in calls:
            pattern = call[0][0]
            step_metric = call[1].get("step_metric", "")
            if "generation_count" in step_metric:
                generation_patterns_found.append(pattern)
        expected_patterns = [
            "train/reward/total",
            "train/reward/terms/*",
            "train/reward/metrics/*",
            "train/parsing/*",
        ]
        for expected in expected_patterns:
            assert any(expected in p for p in generation_patterns_found), (
                f"expected pattern '{expected}' to use generation_count as step_metric"
            )

    def test_generation_step_metrics_only_defined_once(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Verify that define_metric for generation_count is only called once per prefix."""
        fake_run = _FakeWandBRun()
        mock_define_metric = mocker.patch.object(reward_logging.wandb, "define_metric")
        mocker.patch.object(reward_logging.wandb, "run", fake_run)
        logger = reward_logging.WandBRewardLogger(fake_run)
        logger.set_key_prefix("train")  # metrics are defined per-prefix
        # multiple log calls with the same prefix
        logger.log_sample("s1", total=1.0, terms={}, metrics={}, step=1)
        first_call_count = mock_define_metric.call_count
        logger.log_sample("s2", total=2.0, terms={}, metrics={}, step=2)
        logger.log_sample("s3", total=3.0, terms={}, metrics={}, step=3)
        # call count should not increase after first log for same prefix
        assert mock_define_metric.call_count == first_call_count, (
            "define_metric should only be called once per prefix, not on every log()"
        )

    def test_batch_step_metrics_defined_on_first_log_batch_stats(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Verify that define_metric for batch_count is called lazily on first log_batch_stats()."""
        fake_run = _FakeWandBRun()
        mock_define_metric = mocker.patch.object(reward_logging.wandb, "define_metric")
        mocker.patch.object(reward_logging.wandb, "run", fake_run)
        logger = reward_logging.WandBRewardLogger(fake_run)
        logger.set_key_prefix("train")  # metrics are defined per-prefix
        # define_metric should NOT be called yet
        mock_define_metric.assert_not_called()
        # first log_batch_stats call should trigger define_metric for batch patterns
        logger.log_batch_stats(
            batch_mean=1.0,
            batch_std=0.1,
            batch_count=1,
        )
        calls = mock_define_metric.call_args_list
        assert len(calls) > 0, "define_metric should be called on first log_batch_stats()"
        # check that batch_count is used as step_metric for batch patterns
        batch_patterns_found = []
        for call in calls:
            pattern = call[0][0]
            step_metric = call[1].get("step_metric", "")
            if "batch_count" in step_metric:
                batch_patterns_found.append(pattern)
        assert any("reward/batch" in p for p in batch_patterns_found), (
            "expected reward/batch/* patterns to use batch_count as step_metric"
        )

    def test_batch_step_metrics_only_defined_once(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Verify that define_metric for batch_count is only called once per prefix."""
        fake_run = _FakeWandBRun()
        mock_define_metric = mocker.patch.object(reward_logging.wandb, "define_metric")
        mocker.patch.object(reward_logging.wandb, "run", fake_run)
        logger = reward_logging.WandBRewardLogger(fake_run)
        logger.set_key_prefix("train")  # metrics are defined per-prefix
        logger.log_batch_stats(
            batch_mean=1.0,
            batch_std=0.1,
            batch_count=1,
        )
        first_call_count = mock_define_metric.call_count
        # additional calls with same prefix should not trigger define_metric again
        logger.log_batch_stats(
            batch_mean=2.0,
            batch_std=0.2,
            batch_count=2,
        )
        assert mock_define_metric.call_count == first_call_count, (
            "define_metric should only be called once per prefix for batch metrics"
        )

    def test_generation_and_batch_metrics_defined_independently(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Verify that generation and batch define_metric calls are independent."""
        fake_run = _FakeWandBRun()
        mock_define_metric = mocker.patch.object(reward_logging.wandb, "define_metric")
        mocker.patch.object(reward_logging.wandb, "run", fake_run)
        logger = reward_logging.WandBRewardLogger(fake_run)
        logger.set_key_prefix("train")  # metrics are defined per-prefix
        # call log_batch_stats first (should only define batch metrics for "train")
        logger.log_batch_stats(
            batch_mean=1.0,
            batch_std=0.1,
            batch_count=1,
        )
        batch_call_count = mock_define_metric.call_count
        # then call log (should define generation metrics for "train")
        logger.log_sample("s1", total=1.0, terms={}, metrics={}, step=1)
        total_call_count = mock_define_metric.call_count
        # both should have triggered define_metric calls
        assert total_call_count > batch_call_count, (
            "log() should trigger additional define_metric calls for generation metrics"
        )

    def test_batch_count_logged_in_payload(self) -> None:
        """Verify that batch_count is included in the logged payload."""
        fake_run = _FakeWandBRun()
        logger = reward_logging.WandBRewardLogger(fake_run)
        logger.set_key_prefix("train")
        logger.log_batch_stats(
            batch_mean=1.5,
            batch_std=0.2,
            batch_count=42,
        )
        assert len(fake_run.logged) == 1
        payload = fake_run.logged[0]
        assert payload["train/batch_count"] == 42
        assert payload["train/reward/batch/mean"] == pytest.approx(1.5)
        assert payload["train/reward/batch/std"] == pytest.approx(0.2)

    def test_generation_count_logged_in_payload(self) -> None:
        """Verify that generation_count is included in the logged payload."""
        fake_run = _FakeWandBRun()
        logger = reward_logging.WandBRewardLogger(fake_run, log_every_n_generations=1)
        logger.set_key_prefix("train")
        logger.log_sample(
            "sample_1",
            generation_count=99,
            total=2.5,
            terms={},
            metrics={},
            step=10,
        )
        assert len(fake_run.logged) == 1
        payload = fake_run.logged[0]
        assert payload["train/generation_count"] == 99
        assert payload["train/reward/total"] == pytest.approx(2.5)

    def test_run_step_metrics_defined_on_first_log_phase_summaries(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Verify that define_metric for run-level metrics is called lazily on first log_phase_summaries()."""
        fake_run = _FakeWandBRun()
        mock_define_metric = mocker.patch.object(reward_logging.wandb, "define_metric")
        mocker.patch.object(reward_logging.wandb, "run", fake_run)
        logger = reward_logging.WandBRewardLogger(fake_run)
        logger.set_key_prefix("train")  # metrics are defined per-prefix
        mock_define_metric.assert_not_called()
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
            failure_ratio=0.1,
            failure_count=5,
            step=100,
        )
        calls = mock_define_metric.call_args_list
        assert len(calls) > 0, "define_metric should be called on first log_phase_summaries()"
        # verify run-level patterns are defined with "last" summary
        patterns_defined = [call.args[0] for call in calls]
        assert any("reward/run" in p or "failures" in p for p in patterns_defined)
        # verify summary type is "last" for run-level metrics
        for call in calls:
            assert call.kwargs.get("summary") == "last"

    def test_run_step_metrics_only_defined_once(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Verify that define_metric for run-level metrics is only called once per prefix."""
        fake_run = _FakeWandBRun()
        mock_define_metric = mocker.patch.object(reward_logging.wandb, "define_metric")
        mocker.patch.object(reward_logging.wandb, "run", fake_run)
        logger = reward_logging.WandBRewardLogger(fake_run)
        logger.set_key_prefix("train")  # metrics are defined per-prefix
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
            step=1,
        )
        first_call_count = mock_define_metric.call_count
        logger.log_phase_summaries(
            reward_totals={"mean": 0.6},
            reward_term_summaries={},
            reward_category_summaries={},
            step=2,
        )
        assert mock_define_metric.call_count == first_call_count, (
            "define_metric should only be called once per prefix for run metrics"
        )

    def test_define_metric_includes_summary_types(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Verify that define_metric calls include appropriate summary types."""
        fake_run = _FakeWandBRun()
        mock_define_metric = mocker.patch.object(reward_logging.wandb, "define_metric")
        mocker.patch.object(reward_logging.wandb, "run", fake_run)
        logger = reward_logging.WandBRewardLogger(fake_run, log_every_n_generations=1)
        logger.set_key_prefix("train")  # metrics are defined per-prefix
        # trigger all define_metric calls for "train" prefix
        logger.log_sample("s1", total=1.0, terms={}, metrics={}, generation_count=1)
        logger.log_batch_stats(
            batch_mean=1.0,
            batch_std=0.1,
            batch_count=1,
        )
        logger.log_phase_summaries(
            reward_totals={"mean": 0.5},
            reward_term_summaries={},
            reward_category_summaries={},
            step=1,
        )
        # verify all calls include a summary parameter
        for call in mock_define_metric.call_args_list:
            assert "summary" in call.kwargs, f"define_metric call missing summary: {call}"

    def test_new_prefix_triggers_define_metric(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Verify that switching to a new prefix triggers additional define_metric calls."""
        fake_run = _FakeWandBRun()
        mock_define_metric = mocker.patch.object(reward_logging.wandb, "define_metric")
        mocker.patch.object(reward_logging.wandb, "run", fake_run)
        logger = reward_logging.WandBRewardLogger(fake_run, log_every_n_generations=1)
        # first prefix: "train"
        logger.set_key_prefix("train")
        logger.log_sample("s1", total=1.0, terms={}, metrics={}, generation_count=1)
        train_call_count = mock_define_metric.call_count
        assert train_call_count > 0, "define_metric should be called for 'train' prefix"
        # verify patterns are for "train" prefix
        train_patterns = [call.args[0] for call in mock_define_metric.call_args_list]
        assert all("train/" in p for p in train_patterns), "all patterns should be for 'train' prefix"
        # switch to new prefix: "eval"
        logger.set_key_prefix("eval")
        logger.log_sample("s2", total=2.0, terms={}, metrics={}, generation_count=2)
        eval_call_count = mock_define_metric.call_count
        # new prefix should trigger additional define_metric calls
        assert eval_call_count > train_call_count, (
            "switching to new prefix should trigger additional define_metric calls"
        )
        # verify new patterns are for "eval" prefix
        new_patterns = [call.args[0] for call in mock_define_metric.call_args_list[train_call_count:]]
        assert all("eval/" in p for p in new_patterns), "new patterns should be for 'eval' prefix"
        # switching back to "train" should NOT trigger more calls (already defined)
        logger.set_key_prefix("train")
        logger.log_sample("s3", total=3.0, terms={}, metrics={}, generation_count=3)
        assert mock_define_metric.call_count == eval_call_count, (
            "switching back to already-defined prefix should not trigger more define_metric calls"
        )

    def test_custom_prefix_triggers_define_metric(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Verify that any custom prefix (not just train/eval) triggers define_metric."""
        fake_run = _FakeWandBRun()
        mock_define_metric = mocker.patch.object(reward_logging.wandb, "define_metric")
        mocker.patch.object(reward_logging.wandb, "run", fake_run)
        logger = reward_logging.WandBRewardLogger(fake_run, log_every_n_generations=1)
        # use a custom prefix
        logger.set_key_prefix("custom_phase")
        logger.log_sample("s1", total=1.0, terms={}, metrics={}, generation_count=1)
        calls = mock_define_metric.call_args_list
        assert len(calls) > 0, "define_metric should be called for custom prefix"
        # verify patterns use the custom prefix
        patterns = [call.args[0] for call in calls]
        assert all("custom_phase/" in p for p in patterns), f"all patterns should use custom prefix, got: {patterns}"
