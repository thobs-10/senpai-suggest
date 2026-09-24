"""Unit tests for the ingestion runner; the individual ingestions are mocked."""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from pytest_mock import MockerFixture

from src.senpai_suggest.backend.data_quality import ValidationResult
from src.senpai_suggest.backend.pipelines import run_ingestion_pipelines as runner
from src.senpai_suggest.backend.utils.main_utils import read_yaml

CONFIG: dict[str, Any] = {"num_rows": 10}


def _result(name: str) -> ValidationResult:
    return ValidationResult(name, pa.table({"dataset": [name]}))


def test_run_all_ingestion_pipelines_runs_each_and_returns_by_name(
    mocker: MockerFixture,
) -> None:
    """It should call every registered ingestion with the config and key results by name."""
    ingestions = {name: mocker.Mock(return_value=_result(name)) for name in ("a", "b")}
    mocker.patch.dict(runner.INGESTIONS, ingestions, clear=True)

    results = runner.run_all_ingestion_pipelines(CONFIG)

    assert {name: r.dataset for name, r in results.items()} == {"a": "a", "b": "b"}
    for ingestion in ingestions.values():
        ingestion.assert_called_once_with(CONFIG)


def test_run_all_ingestion_pipelines_reraises_worker_errors(mocker: MockerFixture) -> None:
    """It should surface an error from any ingestion instead of swallowing it."""
    mocker.patch.dict(
        runner.INGESTIONS,
        {
            "ok": mocker.Mock(return_value=_result("ok")),
            "broken": mocker.Mock(side_effect=RuntimeError("S3 down")),
        },
        clear=True,
    )

    with pytest.raises(RuntimeError, match="S3 down"):
        runner.run_all_ingestion_pipelines(CONFIG)


def test_ingestions_registry_covers_all_datasets() -> None:
    """It should register the anime, synopsis, and ratings ingestions."""
    assert set(runner.INGESTIONS) == {"anime", "synopsis", "ratings"}


def test_main_passes_ingestion_section_of_config(mocker: MockerFixture, tmp_path: Path) -> None:
    """It should run all ingestions with the `ingestion` section and write raw reports."""
    config = {"ingestion": CONFIG, "data_quality": {"reports_dir": str(tmp_path)}}
    read_yaml_mock = mocker.patch(f"{runner.__name__}.read_yaml", return_value=config)
    run_all = mocker.patch.object(runner, "run_all_ingestion_pipelines")
    write_stage = mocker.patch.object(runner, "write_stage_reports", return_value={"a": {}})
    write_summary = mocker.patch.object(runner, "write_run_summary")

    runner.main()

    read_yaml_mock.assert_called_once_with(str(runner.CONFIG_PATH))
    run_all.assert_called_once_with(CONFIG)
    results, stage, report_dir = write_stage.call_args.args
    assert results is run_all.return_value
    assert stage == "raw"
    assert report_dir.parent == tmp_path
    write_summary.assert_called_once_with([{}], report_dir)


def test_config_path_points_to_existing_config_yaml() -> None:
    """It should resolve to the checked-in config.yaml with an ingestion section."""
    config = read_yaml(str(runner.CONFIG_PATH))

    assert set(config["ingestion"]) >= {"anime", "ratings", "synopsis"}
    assert "reports_dir" in config["data_quality"]
