"""Utility script that has helper functions required by more than one function."""

from pathlib import Path
from typing import Optional

import awswrangler as wr
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.senpai_suggest.backend.logger import logger


def convert_to_parquet(data: pd.DataFrame, output_file: str) -> pq.ParquetDataset:
    """
    Convert a DataFrame to a Parquet file.

    Args:
        data: DataFrame to convert.
        output_file: Local path where Parquet file will be written.

    Returns:
        ParquetDataset loaded from the written file.

    Raises:
        ValueError: If data is None or empty.
        RuntimeError: If conversion or write fails.
    """
    if data is None or data.empty:
        raise ValueError("Cannot convert empty or None DataFrame to Parquet.")

    try:
        logger.info(f"Converting {len(data)} rows to Parquet: {output_file}")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(data)
        pq.write_table(table, str(output_path))
        logger.info(f"Successfully wrote Parquet file: {output_file}")

        return pq.ParquetDataset(str(output_path))
    except Exception as e:
        logger.error(f"Error converting data to Parquet: {e}")
        raise RuntimeError(f"Failed to convert data to Parquet: {e}") from e


def fetch_from_s3(
    bucket_name: str,
    key: str,
    output_file: str,
    columns: Optional[list[str]] = None,
    num_rows: Optional[int] = 1000,
) -> pq.ParquetDataset:
    """
    Fetch CSV data from S3, limit to 1000 rows, and convert to Parquet.

    Args:
        bucket_name: S3 bucket name.
        key: S3 object key (path to CSV file).
        output_file: Local path where Parquet file will be written.

    Returns:
        ParquetDataset loaded from the converted Parquet file.

    Raises:
        ValueError: If required parameters are missing.
        RuntimeError: If S3 read or conversion fails.
    """
    if not output_file:
        raise ValueError("Output file path must be provided.")
    if not bucket_name:
        raise ValueError("Bucket name must be provided.")
    if not key:
        raise ValueError("S3 key must be provided.")

    try:
        s3_path = f"s3://{bucket_name}/{key}"
        logger.info(f"Reading CSV from S3: {s3_path}, limit=1000 rows")

        df = wr.s3.read_csv(
            path=s3_path,
            use_threads=True,
            usecols=columns,
            nrows=num_rows,
        )
        logger.info(f"Successfully read {len(df)} rows from S3")

        parquet_data = convert_to_parquet(df, output_file)
        return parquet_data
    except Exception as e:
        logger.error(f"Failed to fetch and convert CSV from S3: {e}")
        raise RuntimeError(f"Failed to fetch data from S3: {e}") from e
