"""Singleton logger built on loguru.

Writes to ``<LOG_DIR>/system_<date>_<time>.log`` with the format
``time | level | message``. The file sink is configured on first use (not at
import time) and the file is only created when the first message is written,
so importing modules or running tooling does not leave empty log files behind.
"""

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

LOG_FORMAT = "{time:YYYY-MM-DD_HH:mm:ss} | {level} | {message}"


def _configure_file_sink() -> None:
    """Attach the rotating file sink to loguru, reading settings from the environment."""
    load_dotenv()
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_file = log_dir / f"system_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    logger.add(
        log_file,
        rotation=os.getenv("LOG_ROTATION", "10 MB"),
        retention=os.getenv("LOG_RETENTION", "30 days"),
        level=os.getenv("LOG_LEVEL", "INFO"),
        format=LOG_FORMAT,
        delay=True,  # create the file (and LOG_DIR) on the first write, not now
    )


class Logger:
    """Singleton Logger class to provide a consistent logging interface across the project."""

    _instance: "Logger | None" = None

    def __new__(cls) -> "Logger":
        if cls._instance is None:
            _configure_file_sink()
            cls._instance = super().__new__(cls)
        return cls._instance

    def info(self, message: str) -> None:
        """Log an info message."""
        logger.info(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        logger.error(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        logger.warning(message)

    def debug(self, message: str) -> None:
        """Log a debug message."""
        logger.debug(message)
