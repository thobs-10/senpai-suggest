"""Ingest anime metadata from S3.

Fetches the raw anime CSV, normalises it into the columns the rest of the
pipeline relies on (see `ANIME_SCHEMA`), and checks it in report mode.
"""

from typing import Any

import pyarrow as pa

from src.senpai_suggest.backend.configs.backend_config import IngestionConfig, get_anime_columns
from src.senpai_suggest.backend.data_quality import DataQualityValidator, ValidationResult
from src.senpai_suggest.backend.logger.logger import Logger
from src.senpai_suggest.backend.utils.main_utils import fetch_from_s3

logger: Logger = Logger()
validator = DataQualityValidator()


def clean_anime_metadata(table: pa.Table) -> pa.Table:
    """Rename IDs, resolve display names, null out placeholders, and keep `ANIME_COLUMNS`.

    `eng_version` is the English title when there is one, otherwise the original name.

    Args:
        table: Raw anime table as read from anime.csv.

    Returns:
        Table with exactly `ANIME_COLUMNS`, in that order.
    """
    df = table.to_pandas()
    text_columns = df.select_dtypes("object").columns
    df[text_columns] = df[text_columns].mask(df[text_columns] == IngestionConfig.unknown_marker)
    df = df.rename(columns={IngestionConfig.raw_id_column: "anime_id"})
    df["eng_version"] = df[IngestionConfig.raw_english_name_column].fillna(
        df[IngestionConfig.raw_name_column]
    )
    return pa.Table.from_pandas(df[get_anime_columns()], preserve_index=False)


def ingest_anime_list(
    raw_path: str,
    local_path: str,
    num_rows: int | None = None,
) -> ValidationResult:
    """Fetch anime metadata from S3, clean it, and validate it.

    Args:
        raw_path: S3 URI of the anime CSV.
        local_path: Local Parquet path for the fetched (raw) data.
        num_rows: Maximum rows to read; None reads the whole file.

    Returns:
        Validation result: the coerced table plus any `ANIME_SCHEMA` failures,
        for the raw data quality report.

    Raises:
        RuntimeError: If the S3 fetch fails.
        DataQualityError: If a required column is missing (structural);
            value-level failures are reported, not raised.
    """
    raw_table = fetch_from_s3(raw_path, local_path, num_rows=num_rows)
    return validator.validate_anime(clean_anime_metadata(raw_table), raise_on_fail=False)


def run_anime_list_ingestion(ingestion_config: dict[str, Any]) -> ValidationResult:
    """Run the anime metadata ingestion using the `ingestion` section of config.yaml."""
    anime_cfg = ingestion_config["anime"]
    logger.info("Starting anime list ingestion.")
    result = ingest_anime_list(
        raw_path=anime_cfg["raw_path"],
        local_path=anime_cfg["local_path"],
        num_rows=ingestion_config.get("num_rows"),
    )
    logger.info(
        f"Anime list ingestion completed: {result.table.num_rows} rows, "
        f"{len(result.failure_cases)} quality failures."
    )
    return result
