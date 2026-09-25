"""Pytest fixtures shared by pipeline step and flow tests."""

from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pytest
from prefect.testing.utilities import prefect_test_harness


@pytest.fixture(scope="session")
def prefect_backend() -> Iterator[None]:
    """Run flows against one temporary local Prefect server for the whole session."""
    with prefect_test_harness():
        yield


@pytest.fixture
def sample_ratings_table() -> pa.Table:
    """Create a sample valid ratings table."""
    return pa.table(
        {
            "user_id": pa.array([1, 1, 2, 2, 3], type=pa.int64()),
            "anime_id": pa.array([101, 102, 101, 103, 102], type=pa.int64()),
            "rating": pa.array([8, 7, 9, 6, 8], type=pa.int64()),
        }
    )


@pytest.fixture
def ratings_table_with_duplicates() -> pa.Table:
    """Create a ratings table where user 1 rates anime 101 twice."""
    return pa.table(
        {
            "user_id": pa.array([1, 1, 1, 2, 2], type=pa.int64()),
            "anime_id": pa.array([101, 102, 101, 101, 103], type=pa.int64()),
            "rating": pa.array([8, 7, 8, 9, 6], type=pa.int64()),
        }
    )


@pytest.fixture
def ratings_table_with_nulls() -> pa.Table:
    """Create a ratings table with null values."""
    return pa.table(
        {
            "user_id": pa.array([1, 1, None, 2, 2], type=pa.int64()),
            "anime_id": pa.array([101, 102, 101, None, 103], type=pa.int64()),
            "rating": pa.array([8, 7, 9, 6, None], type=pa.int64()),
        }
    )


@pytest.fixture
def raw_anime_table() -> pa.Table:
    """Create a raw anime table shaped like anime.csv, including 'Unknown' placeholders."""
    return pa.table(
        {
            "MAL_ID": [1, 5],
            "Name": ["Cowboy Bebop", "Tengen Toppa Gurren Lagann"],
            "English name": ["Cowboy Bebop", "Unknown"],
            "Score": ["8.78", "Unknown"],
            "Genres": ["Action, Sci-Fi", "Action, Mecha"],
            "Episodes": ["26", "27"],
            "Type": ["TV", "TV"],
            "Premiered": ["Spring 1998", "Spring 2007"],
            "Members": [1251960, 890000],
            "Studios": ["Sunrise", "Gainax"],
        }
    )


@pytest.fixture
def raw_synopsis_table() -> pa.Table:
    """Create a raw synopsis table shaped like anime_with_synopsis.csv."""
    return pa.table(
        {
            "MAL_ID": [1, 5],
            "Name": ["Cowboy Bebop", "Tengen Toppa Gurren Lagann"],
            "Genres": ["Action, Sci-Fi", "Action, Mecha"],
            "sypnopsis": ["Space bounty hunters.", "Drill into the heavens."],
        }
    )


@pytest.fixture
def ingestion_config() -> dict[str, Any]:
    """Create the `ingestion` section of config.yaml."""
    return {
        "num_rows": 100,
        "anime": {"raw_path": "s3://bucket/anime.csv", "local_path": "cache/anime.parquet"},
        "ratings": {
            "raw_path": "s3://bucket/animelist.csv",
            "local_path": "cache/ratings.parquet",
            "columns": ["user_id", "anime_id", "rating"],
        },
        "synopsis": {
            "raw_path": "s3://bucket/synopsis.csv",
            "local_path": "cache/synopsis.parquet",
            "columns": ["MAL_ID", "Name", "Genres", "sypnopsis"],
        },
    }
