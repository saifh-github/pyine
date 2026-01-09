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
        logger = reward_logging.WandBRewardLogger(
            fake_run,
            key_prefix="train",
        )
        logger.log(
            "s1",
            total=1.0,
            terms={"reward/terms/t": 0.25},
            metrics={"reward/metrics/m": 2},
            step=7,
        )
        assert len(fake_run.logged) == 1
        payload = fake_run.logged[0]
        assert payload["train/global_step"] == 7
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
            key_prefix="train/",
            log_tables=True,
            table_key="reward/rewards_table",
            table_flush_every_n_logs=1,
        )
        logger.log(
            "s1",
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
        assert "train/reward/rewards_table" in table_payload
        # verify table was created with correct columns (includes reasoning, final_answer, reward/ prefixes)
        table_obj = table_payload["train/reward/rewards_table"]
        expected_cols = [
            "sample_id",
            "step",
            "prompt",
            "model_output",
            "reasoning",
            "final_answer",
            "reward_total",
            "reward_terms_json",
            "reward_metrics_json",
            "categories_json",
            "tags_json",
        ]
        assert table_obj.columns == expected_cols
        # verify row content
        assert len(added_rows) == 1
        row = added_rows[0]
        assert row[0] == "s1"  # sample_id
        assert row[1] == 7  # step
        assert row[2] == "test prompt"  # prompt
        assert row[3] == "test output"  # model_output
        assert row[4] == "test reasoning"  # reasoning
        assert row[5] == "test answer"  # final_answer
        assert row[6] == 1.0  # reward/total
        # verify terms_json contains prefixed keys
        terms_json = json.loads(typing.cast("str", row[7]))
        assert "train/reward/terms/t" in terms_json
        assert terms_json["train/reward/terms/t"] == 0.25
        # verify metrics_json contains prefixed keys
        metrics_json = json.loads(typing.cast("str", row[8]))
        assert "train/reward/metrics/m" in metrics_json
        assert metrics_json["train/reward/metrics/m"] == 2
        # verify categories_json contains category labels
        categories_json = json.loads(typing.cast("str", row[9]))
        assert categories_json == ["difficulty:easy", "total_steps:10"]
        # verify tags_json contains sample tags
        tags_json = json.loads(typing.cast("str", row[10]))
        assert tags_json == ["tag1", "tag2:value"]

    def test_set_key_prefix_switches_prefix_dynamically(self) -> None:
        fake_run = _FakeWandBRun()
        logger = reward_logging.WandBRewardLogger(
            fake_run,
            key_prefix="train",
        )
        # log with initial train prefix
        logger.log(
            "s1",
            total=1.0,
            terms={"reward/terms/t": 0.25},
            metrics={},
            step=1,
        )
        # switch to valid prefix
        logger.set_key_prefix("valid")
        # log with new valid prefix
        logger.log(
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
        logger = reward_logging.WandBRewardLogger(
            fake_run,
            key_prefix="",
        )
        # set prefix without trailing slash
        logger.set_key_prefix("eval")
        logger.log(
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
        logger = reward_logging.WandBRewardLogger(
            fake_run,
            key_prefix="train",
        )
        # switch to empty prefix
        logger.set_key_prefix("")
        logger.log(
            "s1",
            total=1.0,
            terms={},
            metrics={},
            step=1,
        )
        payload = typing.cast("dict[str, object]", fake_run.logged[0])
        # should have no key_prefix, just scope_prefix
        assert "reward/total" in payload
        assert "train/reward/total" not in payload
