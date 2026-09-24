"""Unit tests for synopsis ingestion; `fetch_from_s3` is mocked."""

from typing import Any

import pyarrow as pa
import pytest
from pytest_mock import MockerFixture

from src.senpai_suggest.backend.data_quality import DataQualityError
from src.senpai_suggest.backend.pipelines.steps import synopsis_ingestion


def test_rename_synopsis_columns_maps_known_and_keeps_others() -> None:
    """It should rename MAL_ID/sypnopsis and leave other columns alone."""
    table = pa.table({"MAL_ID": [1], "Name": ["A"], "sypnopsis": ["s"]})

    renamed = synopsis_ingestion.rename_synopsis_columns(table)

    assert renamed.column_names == ["anime_id", "Name", "synopsis"]


def test_run_synopsis_ingestion_fetches_with_config_and_validates(
    mocker: MockerFixture, ingestion_config: dict[str, Any], raw_synopsis_table: pa.Table
) -> None:
    """It should fetch using the synopsis config and return the renamed, validated table."""
    fetch = mocker.patch.object(
        synopsis_ingestion, "fetch_from_s3", return_value=raw_synopsis_table
    )

    result = synopsis_ingestion.run_synopsis_ingestion(ingestion_config)

    fetch.assert_called_once_with(
        "s3://bucket/synopsis.csv",
        "cache/synopsis.parquet",
        columns=["MAL_ID", "Name", "Genres", "sypnopsis"],
        num_rows=100,
    )
    assert result.passed
    assert result.table.column_names == ["anime_id", "Name", "Genres", "synopsis"]
    assert result.table["synopsis"].to_pylist()[0] == "Space bounty hunters."


def test_ingest_synopsis_reports_missing_name_without_raising(
    mocker: MockerFixture, raw_synopsis_table: pa.Table
) -> None:
    """It should return a null Name as a failure for the raw report."""
    names = pa.array([None, "Gurren Lagann"], type=pa.string())
    bad = raw_synopsis_table.set_column(1, "Name", names)
    mocker.patch.object(synopsis_ingestion, "fetch_from_s3", return_value=bad)

    result = synopsis_ingestion.ingest_synopsis("s3://bucket/x.csv", "x.parquet")

    assert not result.passed
    assert result.failure_cases["column"].tolist() == ["Name"]


def test_ingest_synopsis_raises_on_missing_column(
    mocker: MockerFixture, raw_synopsis_table: pa.Table
) -> None:
    """It should still fail fast when the source lacks a required column."""
    mocker.patch.object(
        synopsis_ingestion, "fetch_from_s3", return_value=raw_synopsis_table.drop(["Name"])
    )

    with pytest.raises(DataQualityError, match="Synopsis data failed validation"):
        synopsis_ingestion.ingest_synopsis("s3://bucket/x.csv", "x.parquet")
