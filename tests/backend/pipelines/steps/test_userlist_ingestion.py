"""Unit tests for user ratings ingestion; `fetch_from_s3` is mocked."""

from typing import Any

import pyarrow as pa
import pytest
from pytest_mock import MockerFixture

from src.senpai_suggest.backend.data_quality import DataQualityError
from src.senpai_suggest.backend.pipelines.steps import userlist_ingestion


def test_run_userlist_ingestion_fetches_with_config_and_validates(
    mocker: MockerFixture, ingestion_config: dict[str, Any], sample_ratings_table: pa.Table
) -> None:
    """It should pass the ratings config to fetch_from_s3 and return the validated table."""
    fetch = mocker.patch.object(
        userlist_ingestion, "fetch_from_s3", return_value=sample_ratings_table
    )

    table = userlist_ingestion.run_userlist_ingestion(ingestion_config)

    fetch.assert_called_once_with(
        "s3://bucket/animelist.csv",
        "cache/ratings.parquet",
        columns=["user_id", "anime_id", "rating"],
        num_rows=100,
    )
    assert table.to_pydict() == sample_ratings_table.to_pydict()


def test_run_userlist_ingestion_defaults_optional_config(
    mocker: MockerFixture, sample_ratings_table: pa.Table
) -> None:
    """It should read all columns and rows when `columns` and `num_rows` are omitted."""
    fetch = mocker.patch.object(
        userlist_ingestion, "fetch_from_s3", return_value=sample_ratings_table
    )
    config = {"ratings": {"raw_path": "s3://bucket/r.csv", "local_path": "r.parquet"}}

    userlist_ingestion.run_userlist_ingestion(config)

    fetch.assert_called_once_with("s3://bucket/r.csv", "r.parquet", columns=None, num_rows=None)


@pytest.mark.parametrize(
    "bad_table_fixture", ["ratings_table_with_duplicates", "ratings_table_with_nulls"]
)
def test_ingest_user_ratings_rejects_bad_data(
    mocker: MockerFixture, request: pytest.FixtureRequest, bad_table_fixture: str
) -> None:
    """It should fail fast on duplicate (user, anime) pairs or nulls."""
    mocker.patch.object(
        userlist_ingestion,
        "fetch_from_s3",
        return_value=request.getfixturevalue(bad_table_fixture),
    )

    with pytest.raises(DataQualityError, match="Ratings data failed validation"):
        userlist_ingestion.ingest_user_ratings("s3://bucket/x.csv", "x.parquet")


def test_ingest_user_ratings_propagates_fetch_errors(mocker: MockerFixture) -> None:
    """It should surface S3 fetch failures unchanged."""
    mocker.patch.object(
        userlist_ingestion, "fetch_from_s3", side_effect=RuntimeError("fetch failed")
    )

    with pytest.raises(RuntimeError, match="fetch failed"):
        userlist_ingestion.ingest_user_ratings("s3://bucket/x.csv", "x.parquet")
