import logging
import logging.config
import pathlib

PROJECT_LOGGER_NAME = "pyine"
"""The name of the root logger for the entire PyINE framework."""


def get_default_log_file_path() -> pathlib.Path:
    """Returns the default path to the log file."""
    import pyine.utils.filesystem

    return pyine.utils.filesystem.get_logs_root_path() / f"{PROJECT_LOGGER_NAME}.log"


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_path: pathlib.Path | str | None = None,  # `None` = auto-decide, if log_to_file = True
):
    """Configures logging for the entire framework.

    This function should be called ONCE at the very beginning of your application's entry point
    (e.g., your main.py or app.py).

    Libraries should NOT call this function. They should only get and use loggers, relying on the
    application to have configured the root logger.

    Args:
        level: The minimum logging level to capture (e.g., logging.INFO, logging.DEBUG).
        log_to_file: If True, logs will also be written to a file.
        log_path: The path to write log files to; if `None` and required, we will use a default path.
    """
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s - %(levelname)-8s - %(name)s.%(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            PROJECT_LOGGER_NAME: {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
        },
        "root": {
            "level": logging.INFO,
            "handlers": ["console"],
        },
    }

    if log_to_file:
        default_log_path = get_default_log_file_path() if log_path is None else pathlib.Path(log_path)
        default_log_path.parent.mkdir(parents=True, exist_ok=True)
        logging_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": str(default_log_path),
            "maxBytes": 1024 * 1024 * 10,  # 10 MB
            "backupCount": 3,
            "encoding": "utf-8",
        }
        logging_config["loggers"][PROJECT_LOGGER_NAME]["handlers"].append("file")  # noqa
        logging_config["root"]["handlers"].append("file")  # noqa

    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(f"{PROJECT_LOGGER_NAME}.utils.logging")
    logger.info("Logging has been configured.")


if __name__ == "__main__":
    setup_logging(level=logging.DEBUG)
    main_logger = logging.getLogger(PROJECT_LOGGER_NAME)
    main_logger.debug("debug message")
    main_logger.info("info message")
    main_logger.warning("warning message")
    main_logger.error("error message")
    main_logger.critical("critical message")
