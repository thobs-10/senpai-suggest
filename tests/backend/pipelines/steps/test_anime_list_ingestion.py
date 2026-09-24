"""Unit tests for anime metadata ingestion; `fetch_from_s3` is mocked."""

from typing import Any

import pyarrow as pa
import pytest
from pytest_mock import MockerFixture

from src.senpai_suggest.backend.configs.backend_config import get_anime_columns
from src.senpai_suggest.backend.pipelines.steps import anime_list_ingestion


def test_clean_anime_metadata_resolves_names_and_unknowns(raw_anime_table: pa.Table) -> None:
    """It should rename IDs, fall back to the original name, and null out 'Unknown'."""
    cleaned = anime_list_ingestion.clean_anime_metadata(raw_anime_table)

    assert cleaned.column_names == get_anime_columns()
    assert cleaned["anime_id"].to_pylist() == [1, 5]
    assert cleaned["eng_version"].to_pylist() == ["Cowboy Bebop", "Tengen Toppa Gurren Lagann"]
    assert cleaned["Score"].to_pylist() == ["8.78", None]


def test_clean_anime_metadata_drops_unused_columns(raw_anime_table: pa.Table) -> None:
    """It should not carry raw-only columns like Studios through."""
    cleaned = anime_list_ingestion.clean_anime_metadata(raw_anime_table)

    assert "Studios" not in cleaned.column_names
    assert "English name" not in cleaned.column_names


def test_run_anime_list_ingestion_fetches_with_config_and_validates(
    mocker: MockerFixture, ingestion_config: dict[str, Any], raw_anime_table: pa.Table
) -> None:
    """It should fetch using the anime config and return a validated, coerced table."""
    fetch = mocker.patch.object(anime_list_ingestion, "fetch_from_s3", return_value=raw_anime_table)

    result = anime_list_ingestion.run_anime_list_ingestion(ingestion_config)

    fetch.assert_called_once_with(
        "s3://bucket/anime.csv",
        "cache/anime.parquet",
        num_rows=100,
    )
    assert result.passed
    assert result.table.num_rows == 2
    assert result.table.schema.field("Score").type == pa.float64()


def test_ingest_anime_list_reports_duplicate_ids_without_raising(
    mocker: MockerFixture, raw_anime_table: pa.Table
) -> None:
    """It should return duplicate anime IDs as failures for the raw report."""
    mocker.patch.object(
        anime_list_ingestion,
        "fetch_from_s3",
        return_value=pa.concat_tables([raw_anime_table, raw_anime_table]),
    )

    result = anime_list_ingestion.ingest_anime_list("s3://bucket/x.csv", "x.parquet")

    assert not result.passed
    assert result.table.num_rows == 4
    assert result.table.schema.field("Score").type == pa.float64()


def test_ingest_anime_list_propagates_fetch_errors(mocker: MockerFixture) -> None:
    """It should surface S3 fetch failures unchanged."""
    mocker.patch.object(
        anime_list_ingestion, "fetch_from_s3", side_effect=RuntimeError("fetch failed")
    )

    with pytest.raises(RuntimeError, match="fetch failed"):
        anime_list_ingestion.ingest_anime_list("s3://bucket/x.csv", "x.parquet")
