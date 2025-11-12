"""Logger configuration."""

import logging
import logging.config
from typing import Optional

import colorama
from colorama import Fore, Style

colorama.init()


class _ColoredFormatter(logging.Formatter):
    """Formatter that colors the levelname for terminal output."""

    LEVEL_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        levelno = record.levelno
        color = self.LEVEL_COLORS.get(levelno, "")
        if color:
            original = record.levelname
            try:
                record.levelname = f"{color}{original}{Style.RESET_ALL}"
                return super().format(record)
            finally:
                record.levelname = original
        return super().format(record)


_logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "[%(levelname)s] %(message)s"},
        "verbose": {
            "format": "[%(levelname)s] %(asctime)s: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
        "color_verbose": {
            "()": "logger._ColoredFormatter",
            "format": "[%(levelname)s] %(asctime)s: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "color_verbose",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "verbose",
            "filename": ".log",
            "mode": "a",
            "encoding": "utf-8",
        },
    },
    # Propagate this logger and library loggers, then root logs to stdout and file.
    "loggers": {
        "local_logger": {"level": "DEBUG", "propagate": True},
        "git": {"level": "DEBUG", "propagate": True},
        "git.cmd": {"level": "DEBUG", "propagate": True},
    },
    "root": {"level": "DEBUG", "handlers": ["stdout", "file"]},
}

_log = logging.getLogger("local_logger")
logging.config.dictConfig(_logging_config)


def get() -> logging.Logger:
    """Get a configured logger instance."""
    return _log
