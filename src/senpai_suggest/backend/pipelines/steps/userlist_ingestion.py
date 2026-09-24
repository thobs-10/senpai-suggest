"""Ingest user anime ratings from S3.

Ingestion only fetches and validates. Normalising, shuffling and ID encoding
live in `preprocessing.py`.
"""

from typing import Any

from src.senpai_suggest.backend.data_quality import DataQualityValidator, ValidationResult
from src.senpai_suggest.backend.logger.logger import Logger
from src.senpai_suggest.backend.utils.main_utils import fetch_from_s3

logger: Logger = Logger()
validator = DataQualityValidator()


def ingest_user_ratings(
    raw_path: str,
    local_path: str,
    columns: list[str] | None = None,
    num_rows: int | None = None,
) -> ValidationResult:
    """Fetch user ratings from S3, cache them locally as Parquet, and validate them.

    Args:
        raw_path: S3 URI of the ratings CSV.
        local_path: Local Parquet path for the fetched data.
        columns: Subset of columns to read; None reads all.
        num_rows: Maximum rows to read; None reads the whole file.

    Returns:
        Validation result: the coerced table plus any `RATINGS_SCHEMA` failures,
        for the raw data quality report.

    Raises:
        RuntimeError: If the S3 fetch fails.
        DataQualityError: If a required column is missing (structural);
            value-level failures are reported, not raised.
    """
    table = fetch_from_s3(raw_path, local_path, columns=columns, num_rows=num_rows)
    return validator.validate_ratings(table, raise_on_fail=False)


def run_userlist_ingestion(ingestion_config: dict[str, Any]) -> ValidationResult:
    """Run the user ratings ingestion using the `ingestion` section of config.yaml."""
    ratings_cfg = ingestion_config["ratings"]
    logger.info("Starting user ratings ingestion.")
    result = ingest_user_ratings(
        raw_path=ratings_cfg["raw_path"],
        local_path=ratings_cfg["local_path"],
        columns=ratings_cfg.get("columns"),
        num_rows=ingestion_config.get("num_rows"),
    )
    logger.info(
        f"User ratings ingestion completed: {result.table.num_rows} rows, "
        f"{len(result.failure_cases)} quality failures."
    )
    return result
