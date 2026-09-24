"""Unit tests for the singleton Logger."""

from pathlib import Path

import pytest
from loguru import logger as loguru_logger
from pytest_mock import MockerFixture

from src.senpai_suggest.backend.logger import logger as logger_module
from src.senpai_suggest.backend.logger.logger import Logger


def test_logger_is_singleton() -> None:
    """It should return the same instance every time."""
    assert Logger() is Logger()


def test_file_sink_is_configured_only_once(mocker: MockerFixture) -> None:
    """It should attach the file sink on first instantiation only."""
    configure = mocker.patch.object(logger_module, "_configure_file_sink")
    mocker.patch.object(Logger, "_instance", None)

    Logger()
    Logger()

    configure.assert_called_once()


def test_configure_file_sink_uses_env_and_delays_file_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """It should read LOG_* env vars and pass delay=True so no empty file is created."""
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    add = mocker.patch.object(loguru_logger, "add")

    logger_module._configure_file_sink()

    log_file, kwargs = add.call_args.args[0], add.call_args.kwargs
    assert log_file.parent == tmp_path / "logs"
    assert log_file.name.startswith("system_") and log_file.suffix == ".log"
    assert kwargs["delay"] is True
    assert kwargs["level"] == "DEBUG"


def test_configure_file_sink_creates_no_file_until_first_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It should leave the log dir absent until something is logged."""
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("LOG_DIR", str(log_dir))
    handlers_before = set(loguru_logger._core.handlers)

    logger_module._configure_file_sink()
    new_ids = set(loguru_logger._core.handlers) - handlers_before
    try:
        assert not log_dir.exists()
        loguru_logger.info("first message")
        assert len(list(log_dir.glob("system_*.log"))) == 1
    finally:
        for handler_id in new_ids:
            loguru_logger.remove(handler_id)


@pytest.mark.parametrize("level", ["info", "error", "warning", "debug"])
def test_logger_methods_forward_to_loguru(mocker: MockerFixture, level: str) -> None:
    """It should forward each level method to the matching loguru method."""
    forwarded = mocker.patch.object(loguru_logger, level)

    getattr(Logger(), level)("hello")

    forwarded.assert_called_once_with("hello")
