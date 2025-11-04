import logging
import logging.config
import pathlib
import typing

PROJECT_LOGGER_NAME = "pyine"
"""The name of the root logger for the entire PyINE framework."""


class DistributedRankFilter(logging.Filter):
    """Logging filter that prefixes log messages with distributed rank information.

    When running in distributed mode (world size > 1), this filter prefixes log messages with a
    `[rank X/Y]` tag indicating the current process rank and total world size. The prefix is
    injected exactly once per log record, regardless of how many handlers process the record.
    """

    @typing.override
    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:
        """Determine if the specified record is to be logged.

        Returns True if the record should be logged, or False otherwise.
        """
        record.rank_info = ""
        try:
            import pyine.utils.distrib as distrib_utils
        except Exception:
            return True
        world_size = distrib_utils.get_world_size(default=None)
        if world_size is None or world_size <= 1:
            return True
        rank = distrib_utils.get_global_rank(default=None)
        rank_display = "?" if rank is None else str(rank)
        rank_token = f"[rank {rank_display}/{world_size}]"
        record.rank_info = rank_token
        if not getattr(record, "_pyine_rank_prefixed", False):
            record.msg = f"{rank_token} {record.msg}"
            record._pyine_rank_prefixed = True
        return True


def ensure_distributed_rank_filter_attached(
    logger: logging.Logger | None = None,
) -> None:
    """Ensure that the distributed rank filter is attached to the provided logger handlers.

    Args:
        logger: Specific logger to patch. When omitted, both the root logger and the project
            logger are considered.
    """
    target_loggers: list[logging.Logger] = []
    if logger is None:
        target_loggers.append(logging.getLogger())
        project_logger = logging.getLogger(PROJECT_LOGGER_NAME)
        if project_logger not in target_loggers:
            target_loggers.append(project_logger)
    else:
        target_loggers.append(logger)
    seen_ids: set[int] = set()
    for target in target_loggers:
        if id(target) in seen_ids:
            continue
        seen_ids.add(id(target))
        for handler in list(target.handlers):
            if any(isinstance(existing_filter, DistributedRankFilter) for existing_filter in handler.filters):
                continue
            handler.addFilter(DistributedRankFilter())


def get_default_log_file_path() -> pathlib.Path:
    """Returns the default path to the log file."""
    import pyine.utils.filesystem

    return pyine.utils.filesystem.get_logs_root_path() / f"{PROJECT_LOGGER_NAME}.log"


def setup_logging(
    level: int | str = logging.INFO,
    log_to_file: bool = False,
    log_path: pathlib.Path | str | None = None,  # `None` = auto-decide, if log_to_file = True
) -> None:
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
    logging_config: dict[str, typing.Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "distributed_rank": {
                "()": "pyine.utils.logging.DistributedRankFilter",
            },
        },
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
                "filters": ["distributed_rank"],
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
            "filters": ["distributed_rank"],
        }
        project_handlers = typing.cast("list[str]", logging_config["loggers"][PROJECT_LOGGER_NAME]["handlers"])
        project_handlers.append("file")
        root_handlers = typing.cast("list[str]", logging_config["root"]["handlers"])
        root_handlers.append("file")

    logging.config.dictConfig(logging_config)
    ensure_distributed_rank_filter_attached()
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
