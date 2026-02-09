import logging

import pytest

import pyine.utils.logging as logging_utils


def _make_log_record(
    message: str,
) -> logging.LogRecord:
    return logging.LogRecord(
        name="pyine.tests",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )


class TestDistributedRankFilter:
    def test_filter_noop_when_world_size_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("pyine.utils.distrib.get_world_size", lambda default=None: None)
        log_filter = logging_utils.DistributedRankFilter()
        record = _make_log_record("hello world")
        assert log_filter.filter(record) is True
        assert record.msg == "hello world"
        assert record.rank_info == ""

    def test_filter_drops_info_from_secondary_rank(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(logging_utils.REDUCE_NON_PRIMARY_LOG_LEVEL_ENV, "1")
        monkeypatch.setattr("pyine.utils.distrib.get_world_size", lambda default=None: 3)
        monkeypatch.setattr("pyine.utils.distrib.get_local_rank", lambda default=None: 2)
        log_filter = logging_utils.DistributedRankFilter()
        record = _make_log_record("train start")
        assert log_filter.filter(record) is False
        assert record.msg == "train start"
        assert record.rank_info == ""

    def test_filter_falls_back_to_global_rank_when_local_rank_unknown(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When LOCAL_RANK is unavailable, suppression falls back to global rank."""
        monkeypatch.setenv(logging_utils.REDUCE_NON_PRIMARY_LOG_LEVEL_ENV, "1")
        monkeypatch.setattr("pyine.utils.distrib.get_world_size", lambda default=None: 4)
        monkeypatch.setattr("pyine.utils.distrib.get_local_rank", lambda default=None: None)
        monkeypatch.setattr("pyine.utils.distrib.get_global_rank", lambda default=None: 2)
        log_filter = logging_utils.DistributedRankFilter()
        record = _make_log_record("should be suppressed")
        assert log_filter.filter(record) is False

    def test_filter_handles_unknown_rank(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("pyine.utils.distrib.get_world_size", lambda default=None: 4)
        monkeypatch.setattr("pyine.utils.distrib.get_global_rank", lambda default=None: None)
        log_filter = logging_utils.DistributedRankFilter()
        record = _make_log_record("no rank info")
        assert log_filter.filter(record) is True
        assert record.msg.startswith("[rank ?/4] no rank info")
        assert record.rank_info == "[rank ?/4]"
        assert record.levelno == logging.INFO

    def test_filter_preserves_primary_rank_records(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(logging_utils.REDUCE_NON_PRIMARY_LOG_LEVEL_ENV, "1")
        monkeypatch.setattr("pyine.utils.distrib.get_world_size", lambda default=None: 4)
        monkeypatch.setattr("pyine.utils.distrib.get_global_rank", lambda default=None: 0)
        log_filter = logging_utils.DistributedRankFilter()
        record = _make_log_record("primary rank info")
        record.levelno = logging.INFO
        record.levelname = "INFO"
        assert log_filter.filter(record) is True
        assert record.msg.startswith("[rank 0/4] primary rank info")
        assert record.levelno == logging.INFO
        assert record.levelname == "INFO"

    def test_filter_secondary_warning_is_emitted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(logging_utils.REDUCE_NON_PRIMARY_LOG_LEVEL_ENV, "1")
        monkeypatch.setattr("pyine.utils.distrib.get_world_size", lambda default=None: 5)
        monkeypatch.setattr("pyine.utils.distrib.get_global_rank", lambda default=None: 3)
        log_filter = logging_utils.DistributedRankFilter()
        record = _make_log_record("secondary warning")
        record.levelno = logging.WARNING
        record.levelname = "WARNING"
        assert log_filter.filter(record) is True
        assert record.msg.startswith("[rank 3/5] secondary warning")
        assert record.levelno == logging.WARNING
        assert record.levelname == "WARNING"

    def test_filter_suppression_can_be_disabled_via_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(logging_utils.REDUCE_NON_PRIMARY_LOG_LEVEL_ENV, "0")
        monkeypatch.setattr("pyine.utils.distrib.get_world_size", lambda default=None: 5)
        monkeypatch.setattr("pyine.utils.distrib.get_global_rank", lambda default=None: 3)
        log_filter = logging_utils.DistributedRankFilter()
        record = _make_log_record("secondary rank info")
        assert log_filter.filter(record) is True
        assert record.msg.startswith("[rank 3/5] secondary rank info")
        assert record.levelno == logging.INFO
        assert record.levelname == "INFO"


class _CapturingHandler(logging.Handler):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(
        self,
        record: logging.LogRecord,
    ) -> None:
        self.records.append(record.getMessage())


class TestEnsureDistributedRankFilterAttached:
    def test_filter_attached_to_handlers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("pyine.utils.distrib.get_world_size", lambda default=None: 2)
        monkeypatch.setattr("pyine.utils.distrib.get_global_rank", lambda default=None: 1)
        test_logger = logging.getLogger("pyine.tests.logging_filter")
        test_logger.setLevel(logging.INFO)
        handler_one = _CapturingHandler()
        handler_two = _CapturingHandler()
        test_logger.addHandler(handler_one)
        test_logger.addHandler(handler_two)
        try:
            logging_utils.ensure_distributed_rank_filter_attached(test_logger)
            for handler in (handler_one, handler_two):
                distributed_filters = [
                    filt for filt in handler.filters if isinstance(filt, logging_utils.DistributedRankFilter)
                ]
                assert len(distributed_filters) == 1
            test_logger.propagate = False
            test_logger.warning("prefixed warning")
            assert handler_one.records == handler_two.records
            assert handler_one.records[0].startswith("[rank 1/2] prefixed warning")
        finally:
            test_logger.handlers.clear()
