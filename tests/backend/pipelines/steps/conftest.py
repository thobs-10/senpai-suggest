"""Pytest fixtures for userlist ingestion tests."""

import pyarrow as pa
import pytest

from src.senpai_suggest.backend.pipelines.steps.userlist_ingestion import (
    UserListIngestion,
)


@pytest.fixture
def ingestion_instance() -> UserListIngestion:
    """Create a UserListIngestion instance."""
    return UserListIngestion()


@pytest.fixture
def sample_ratings_table() -> pa.Table:
    """Create a sample valid ratings table."""
    return pa.table(
        {
            "user_id": pa.array([1, 1, 2, 2, 3], type=pa.int64()),
            "anime_id": pa.array([101, 102, 101, 103, 102], type=pa.int64()),
            "rating": pa.array([8.0, 7.0, 9.0, 6.0, 8.5], type=pa.float64()),
        }
    )


@pytest.fixture
def ratings_table_with_duplicates() -> pa.Table:
    """Create a table with duplicate rows."""
    return pa.table(
        {
            "user_id": pa.array([1, 1, 1, 2, 2], type=pa.int64()),
            "anime_id": pa.array([101, 102, 101, 101, 103], type=pa.int64()),
            "rating": pa.array([8.0, 7.0, 8.0, 9.0, 6.0], type=pa.float64()),
        }
    )


@pytest.fixture
def ratings_table_with_nulls() -> pa.Table:
    """Create a table with null values."""
    return pa.table(
        {
            "user_id": pa.array([1, 1, None, 2, 2], type=pa.int64()),
            "anime_id": pa.array([101, 102, 101, None, 103], type=pa.int64()),
            "rating": pa.array([8.0, 7.0, 9.0, 6.0, None], type=pa.float64()),
        }
    )


@pytest.fixture
def config_dict() -> dict[str, dict[str, str]]:
    """Create a sample configuration dictionary."""
    return {
        "ingestion": {
            "raw_ratings_path": "s3://bucket/raw_ratings.csv",
            "output_file_path": "/tmp/ratings.parquet",
            "ratings_path": "s3://bucket/ratings.parquet",
            "user_filename": "processed_users.parquet",
        },
        "aws": {
            "s3_bucket": "test-bucket",
        },
    }
