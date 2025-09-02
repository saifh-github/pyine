import logging
import logging.handlers
import os
import pathlib

import pytest

import pyine.utils.logging as log_utils


def test_setup_logging_console_only(tmp_path: str, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)  # run in temp dir to avoid writing to repo root
    log_utils.setup_logging(level=logging.DEBUG, log_to_file=False)
    out = capsys.readouterr().out
    assert "Logging has been configured." in out
    main_logger = logging.getLogger(log_utils.PROJECT_LOGGER_NAME)
    assert main_logger.level == logging.DEBUG
    assert not os.path.exists(f"{log_utils.PROJECT_LOGGER_NAME}.log")


def test_setup_logging_with_file(tmp_path: str, monkeypatch: pytest.MonkeyPatch):
    log_path = pathlib.Path(tmp_path) / "some_dir" / "dummy.log"
    log_utils.setup_logging(level=logging.INFO, log_to_file=True, log_path=log_path)
    main_logger = logging.getLogger(log_utils.PROJECT_LOGGER_NAME)
    has_file_handler = any(isinstance(h, logging.handlers.RotatingFileHandler) for h in main_logger.handlers)
    assert has_file_handler
    test_message = "hello-file-logging"
    main_logger.info(test_message)
    for h in main_logger.handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            h.flush()
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert test_message in content
