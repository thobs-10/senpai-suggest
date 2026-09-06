"""Make the whole script to be a singlton logger class using loguru library, create a new folder in root level for logger is not exist and create a new
log file with the name_date_timestamp.log format, the logger should have a method to get the logger instance and the log
messages should be in the format of time - level - message"""

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"system_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_ROTATION = os.getenv("LOG_ROTATION", "10 MB")
LOG_RETENTION = os.getenv("LOG_RETENTION", "30 days")

logger.add(
    LOG_FILE,
    rotation=LOG_ROTATION,
    retention=LOG_RETENTION,
    level=LOG_LEVEL,
    format="{time:YYYY-MM-DD_HH:mm:ss} | {level} | {message}",
)


class Logger:
    """Singleton Logger class to provide a consistent logging interface across the project."""

    _instance: "Logger | None" = None

    def __new__(cls) -> "Logger":
        if cls._instance is None:
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
