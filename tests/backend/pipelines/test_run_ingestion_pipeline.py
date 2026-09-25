"""Unit tests for the ingestion Prefect flow and its tasks; S3 is mocked."""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pytest_mock import MockerFixture

from src.senpai_suggest.backend.data_quality import DataQualityError
from src.senpai_suggest.backend.pipelines import run_ingestion_pipeline as pipeline


@pytest.fixture
def raw_tables(
    raw_anime_table: pa.Table, raw_synopsis_table: pa.Table, sample_ratings_table: pa.Table
) -> dict[str, pa.Table]:
    """Raw tables as fetch_from_s3 would return them, keyed by dataset."""
    return {
        "anime": raw_anime_table,
        "synopsis": raw_synopsis_table,
        "ratings": sample_ratings_table,
    }


def _write_config(tmp_path: Path, ingestion_config: dict[str, Any]) -> Path:
    config = {
        "ingestion": {**ingestion_config, "output_dir": str(tmp_path / "ingested")},
        "data_quality": {"reports_dir": str(tmp_path / "reports")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _fake_fetch(raw_tables: dict[str, pa.Table], failing: str | None = None) -> Any:
    """Stand-in for fetch_from_s3 that picks the table from the S3 path's file name."""

    def fetch(raw_path: str, local_path: str, **_: Any) -> pa.Table:
        name = {"anime.csv": "anime", "synopsis.csv": "synopsis", "animelist.csv": "ratings"}[
            Path(raw_path).name
        ]
        if name == failing:
            raise RuntimeError("S3 down")
        return raw_tables[name]

    return fetch


def test_fetch_dataset_passes_dataset_config(
    mocker: MockerFixture, ingestion_config: dict[str, Any], sample_ratings_table: pa.Table
) -> None:
    """It should read the configured path, columns and row limit."""
    fetch = mocker.patch.object(pipeline, "fetch_from_s3", return_value=sample_ratings_table)

    pipeline.fetch_dataset.fn("ratings", ingestion_config["ratings"], 100)

    fetch.assert_called_once_with(
        "s3://bucket/animelist.csv",
        "cache/ratings.parquet",
        columns=["user_id", "anime_id", "rating"],
        num_rows=100,
    )


def test_fetch_dataset_is_the_only_retried_task() -> None:
    """It should retry S3 reads, and nothing downstream."""
    assert pipeline.fetch_dataset.retries == 3
    for downstream in (pipeline.shape_dataset, pipeline.validate_dataset, pipeline.save_ingested):
        assert downstream.retries == 0


def test_shape_dataset_applies_each_datasets_shaper(raw_tables: dict[str, pa.Table]) -> None:
    """It should rename anime/synopsis columns and pass ratings through."""
    assert pipeline.shape_dataset.fn("anime", raw_tables["anime"]).column_names[0] == "anime_id"
    assert "synopsis" in pipeline.shape_dataset.fn("synopsis", raw_tables["synopsis"]).column_names
    assert pipeline.shape_dataset.fn("ratings", raw_tables["ratings"]) is raw_tables["ratings"]


def test_validate_dataset_reports_value_failures_without_raising(
    ratings_table_with_duplicates: pa.Table,
) -> None:
    """It should return duplicate pairs as failures in report mode."""
    result = pipeline.validate_dataset.fn("ratings", ratings_table_with_duplicates)

    assert not result.passed
    assert result.table.num_rows == ratings_table_with_duplicates.num_rows


def test_validate_dataset_raises_on_missing_column(sample_ratings_table: pa.Table) -> None:
    """It should still fail fast when a required column is missing."""
    with pytest.raises(DataQualityError, match="Ratings data failed validation"):
        pipeline.validate_dataset.fn("ratings", sample_ratings_table.drop(["rating"]))


def test_save_ingested_writes_validated_table(
    tmp_path: Path, sample_ratings_table: pa.Table
) -> None:
    """It should write the validated table as {name}.parquet."""
    result = pipeline.validate_dataset.fn("ratings", sample_ratings_table)

    path = pipeline.save_ingested.fn("ratings", result, tmp_path / "run")

    assert path == tmp_path / "run" / "ratings.parquet"
    assert pq.read_table(path).equals(result.table)


def test_report_raw_quality_writes_raw_report(
    tmp_path: Path, sample_ratings_table: pa.Table
) -> None:
    """It should write the raw-stage report and return its summary."""
    result = pipeline.validate_dataset.fn("ratings", sample_ratings_table)

    summary = pipeline.report_raw_quality.fn("ratings", result, tmp_path)

    assert summary["stage"] == "raw" and summary["passed"]
    assert (tmp_path / "ratings_raw.html").exists()


def test_run_ingestion_pipeline_end_to_end(
    prefect_backend: None,
    mocker: MockerFixture,
    tmp_path: Path,
    ingestion_config: dict[str, Any],
    raw_tables: dict[str, pa.Table],
) -> None:
    """It should save every validated dataset and write reports for the run."""
    mocker.patch.object(pipeline, "fetch_from_s3", side_effect=_fake_fetch(raw_tables))

    paths = pipeline.run_ingestion_pipeline(
        config_path=str(_write_config(tmp_path, ingestion_config)), run_id="run1"
    )

    assert set(paths) == {"anime", "synopsis", "ratings"}
    assert paths["anime"] == tmp_path / "ingested" / "run1" / "anime.parquet"
    assert pq.read_table(paths["anime"]).schema.field("Score").type == pa.float64()
    summary = json.loads((tmp_path / "reports" / "run1" / "summary.json").read_text())
    assert {s["dataset"] for s in summary} == {"anime", "synopsis", "ratings"}


def test_run_ingestion_pipeline_stops_at_the_failed_dataset(
    prefect_backend: None,
    mocker: MockerFixture,
    tmp_path: Path,
    ingestion_config: dict[str, Any],
    raw_tables: dict[str, pa.Table],
) -> None:
    """It should keep datasets saved before the failure and skip the ones after it."""
    mocker.patch.object(
        pipeline, "fetch_from_s3", side_effect=_fake_fetch(raw_tables, failing="synopsis")
    )
    # No retry delays in tests.
    mocker.patch.object(pipeline, "fetch_dataset", pipeline.fetch_dataset.with_options(retries=0))

    with pytest.raises(RuntimeError, match="S3 down"):
        pipeline.run_ingestion_pipeline(
            config_path=str(_write_config(tmp_path, ingestion_config)), run_id="run2"
        )

    run_dir = tmp_path / "ingested" / "run2"
    assert (run_dir / "anime.parquet").exists()
    assert not (run_dir / "synopsis.parquet").exists()
    assert not (run_dir / "ratings.parquet").exists()


def test_datasets_run_smallest_first() -> None:
    """It should ingest ratings, the largest dataset, last."""
    assert pipeline.DATASETS == ("anime", "synopsis", "ratings")
