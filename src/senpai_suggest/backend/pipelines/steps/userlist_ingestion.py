"""Ingest user anime ratings from S3.

Ingestion only fetches and validates. Normalising, shuffling and ID encoding
live in `preprocessing.py`.
"""

from typing import Any

import pyarrow as pa

from src.senpai_suggest.backend.data_quality import DataQualityValidator
from src.senpai_suggest.backend.logger.logger import Logger
from src.senpai_suggest.backend.utils.main_utils import fetch_from_s3

logger: Logger = Logger()
validator = DataQualityValidator()


def ingest_user_ratings(
    raw_path: str,
    local_path: str,
    columns: list[str] | None = None,
    num_rows: int | None = None,
) -> pa.Table:
    """Fetch user ratings from S3, cache them locally as Parquet, and validate them.

    Args:
        raw_path: S3 URI of the ratings CSV.
        local_path: Local Parquet path for the fetched data.
        columns: Subset of columns to read; None reads all.
        num_rows: Maximum rows to read; None reads the whole file.

    Returns:
        Validated ratings table (user_id, anime_id, rating).

    Raises:
        RuntimeError: If the S3 fetch fails.
        DataQualityError: If the data breaks `RATINGS_SCHEMA`.
    """
    table = fetch_from_s3(raw_path, local_path, columns=columns, num_rows=num_rows)
    return validator.validate_ratings(table)


def run_userlist_ingestion(ingestion_config: dict[str, Any]) -> pa.Table:
    """Run the user ratings ingestion using the `ingestion` section of config.yaml."""
    ratings_cfg = ingestion_config["ratings"]
    logger.info("Starting user ratings ingestion.")
    table = ingest_user_ratings(
        raw_path=ratings_cfg["raw_path"],
        local_path=ratings_cfg["local_path"],
        columns=ratings_cfg.get("columns"),
        num_rows=ingestion_config.get("num_rows"),
    )
    logger.info(f"User ratings ingestion completed: {table.num_rows} rows.")
    return table
