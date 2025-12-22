import typing

import pytest

import pyine.organisms.models.rewards.core.logging as reward_logging


class _FakeWandBRun:
    def __init__(self) -> None:
        self.logged: list[dict[str, object]] = []

    def log(
        self,
        payload: dict[str, object],
        *,
        step: int | None = None,
    ) -> None:
        self.logged.append(
            {
                "payload": dict(payload),
                "step": step,
            }
        )


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
        entry = fake_run.logged[0]
        assert entry["step"] == 7
        payload = typing.cast("dict[str, object]", entry["payload"])
        assert payload["train/reward/total"] == pytest.approx(1.0)
        assert payload["train/reward/terms/t"] == pytest.approx(0.25)
        assert payload["train/reward/metrics/m"] == 2

    def test_key_prefix_applies_to_table_key_and_json(self) -> None:
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
        )
        assert len(fake_run.logged) == 2
        table_payload = typing.cast("dict[str, object]", fake_run.logged[1]["payload"])
        assert list(table_payload.keys()) == ["train/reward/rewards_table"]
        table = typing.cast("typing.Any", table_payload["train/reward/rewards_table"])
        assert table.columns == ["sample_id", "step", "total", "terms_json", "metrics_json"]
        assert len(table.data) == 1
        assert "\"train/reward/terms/t\"" in table.data[0][3]
        assert "\"train/reward/metrics/m\"" in table.data[0][4]

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
        payload1 = typing.cast("dict[str, object]", fake_run.logged[0]["payload"])
        assert "train/reward/total" in payload1
        assert payload1["train/reward/total"] == pytest.approx(1.0)
        # verify second entry has valid prefix
        payload2 = typing.cast("dict[str, object]", fake_run.logged[1]["payload"])
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
        payload = typing.cast("dict[str, object]", fake_run.logged[0]["payload"])
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
        payload = typing.cast("dict[str, object]", fake_run.logged[0]["payload"])
        # should have no key_prefix, just scope_prefix
        assert "reward/total" in payload
        assert "train/reward/total" not in payload
