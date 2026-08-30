"""Make the whole script to be a singlton logger class using loguru library, create a new folder in root level for logger is not exist and create a new
log file with the name_date_timestamp.log format, the logger should have a method to get the logger instance and the log
messages should be in the format of time - level - message"""

import os
from datetime import datetime
from loguru import logger

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)


class SingletonLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        LOGS_DIR = "logs"
        os.makedirs(LOGS_DIR, exist_ok=True)
        # the file will have the  name_date_timestamp.log format
        LOG_FILE = os.path.join(
            LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )
        logger.add(LOG_FILE, format="{time} - {level} - {message}", level="INFO")

    def get_logger(self):
        return logger


_singleton_logger = SingletonLogger()


def get_logger(name: str | None = None):
    """Backward-compatible logger accessor used across the codebase."""
    if name:
        return _singleton_logger.get_logger().bind(module=name)
    return _singleton_logger.get_logger()
