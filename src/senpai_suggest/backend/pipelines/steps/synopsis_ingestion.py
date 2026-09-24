"""Ingest anime synopses from S3.

Fetches the synopsis CSV, renames its columns to the pipeline's naming
(see `SYNOPSIS_SCHEMA`), and checks it in report mode.
"""

from typing import Any

import pyarrow as pa

from src.senpai_suggest.backend.data_quality import DataQualityValidator, ValidationResult
from src.senpai_suggest.backend.logger.logger import Logger
from src.senpai_suggest.backend.utils.main_utils import fetch_from_s3

logger: Logger = Logger()
validator = DataQualityValidator()

# Raw -> pipeline column names. The source export misspells "synopsis".
SYNOPSIS_RENAMES = {"MAL_ID": "anime_id", "sypnopsis": "synopsis"}


def rename_synopsis_columns(table: pa.Table) -> pa.Table:
    """Rename raw synopsis columns using `SYNOPSIS_RENAMES`; other columns are unchanged."""
    return table.rename_columns([SYNOPSIS_RENAMES.get(name, name) for name in table.column_names])


def ingest_synopsis(
    raw_path: str,
    local_path: str,
    columns: list[str] | None = None,
    num_rows: int | None = None,
) -> ValidationResult:
    """Fetch anime synopses from S3, rename columns, and validate them.

    Args:
        raw_path: S3 URI of the synopsis CSV.
        local_path: Local Parquet path for the fetched (raw) data.
        columns: Raw columns to read; None reads all.
        num_rows: Maximum rows to read; None reads the whole file.

    Returns:
        Validation result: the coerced table plus any `SYNOPSIS_SCHEMA` failures,
        for the raw data quality report.

    Raises:
        RuntimeError: If the S3 fetch fails.
        DataQualityError: If a required column is missing (structural);
            value-level failures are reported, not raised.
    """
    raw_table = fetch_from_s3(raw_path, local_path, columns=columns, num_rows=num_rows)
    return validator.validate_synopsis(rename_synopsis_columns(raw_table), raise_on_fail=False)


def run_synopsis_ingestion(ingestion_config: dict[str, Any]) -> ValidationResult:
    """Run the synopsis ingestion using the `ingestion` section of config.yaml."""
    synopsis_cfg = ingestion_config["synopsis"]
    logger.info("Starting synopsis ingestion.")
    result = ingest_synopsis(
        raw_path=synopsis_cfg["raw_path"],
        local_path=synopsis_cfg["local_path"],
        columns=synopsis_cfg.get("columns"),
        num_rows=ingestion_config.get("num_rows"),
    )
    logger.info(
        f"Synopsis ingestion completed: {result.table.num_rows} rows, "
        f"{len(result.failure_cases)} quality failures."
    )
    return result
