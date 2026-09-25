"""Utility script that has helper functions required by more than one function."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import awswrangler as wr
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.senpai_suggest.backend.logger.logger import Logger

logger: Logger = Logger()


def convert_to_parquet(data: pd.DataFrame, output_file: str) -> pa.Table:
    """
    Convert a DataFrame to a Parquet file.

    Args:
        data: DataFrame to convert.
        output_file: Local path where Parquet file will be written.

    Returns:
        PyArrow Table created from the written Parquet file.

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
        pq.write_table(table, str(output_path))  # type: ignore[no-untyped-call, unused-ignore]  # no pyarrow stubs
        logger.info(f"Successfully wrote Parquet file: {output_file}")

        return table
    except Exception as e:
        logger.error(f"Error converting data to Parquet: {e}")
        raise RuntimeError(f"Failed to convert data to Parquet: {e}") from e


def fetch_from_s3(
    raw_data_path: str,
    output_file: str,
    columns: Optional[list[str]] = None,
    num_rows: Optional[int] = 1000,
) -> pa.Table:
    """
    Fetch CSV data from S3 and convert it to a local Parquet file.

    Retries are deliberately not handled here: callers run inside Prefect
    tasks, which own retry policy (``@task(retries=..., retry_delay_seconds=...)``).
    Failures are raised as ``RuntimeError`` so the task can retry or fail.

    Args:
        raw_data_path: S3 URI of the CSV file (e.g. ``s3://bucket/key.csv``).
        output_file: Local path where Parquet file will be written.
        columns: Optional subset of columns to read; None reads all columns.
        num_rows: Maximum number of rows to read; None reads the whole file.

    Returns:
        PyArrow Table loaded from the converted Parquet file.

    Raises:
        ValueError: If required parameters are missing.
        RuntimeError: If S3 read or conversion fails.
    """
    if not output_file:
        raise ValueError("Output file path must be provided.")
    if not raw_data_path:
        raise ValueError("S3 raw data path must be provided.")

    try:
        row_limit = num_rows if num_rows is not None else "all"
        logger.info(f"Reading CSV from S3: {raw_data_path}, limit={row_limit} rows")

        df = wr.s3.read_csv(
            path=raw_data_path,
            use_threads=True,
            usecols=columns,
            nrows=num_rows,
        )
        logger.info(f"Successfully read {len(df)} rows from S3")

        return convert_to_parquet(df, output_file)
    except Exception as e:
        logger.error(f"Failed to fetch and convert CSV from S3: {e}")
        raise RuntimeError(f"Failed to fetch data from S3: {e}") from e


def save_to_s3(
    df: pa.Table,
    # file_name: str,
    bucket_name: str,
) -> None:
    """
    Save the dataframe to an S3 bucket.

    Args:
        df: PyArrow Table to be saved.
        bucket_name: S3 bucket name.
        # file_name: The S3 object key (path where the file will be stored).

    Raises:
        ValueError: If required parameters are missing.
        RuntimeError: If S3 upload fails.
    """
    if not bucket_name:
        raise ValueError("Bucket name must be provided.")
    if df is None:
        raise ValueError("DataFrame must be provided.")

    try:
        s3_path = f"s3://{bucket_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        logger.info(f"Uploading dataframe to S3: {s3_path}")
        wr.s3.to_parquet(df=df, path=s3_path)
        logger.info(f"Successfully uploaded dataframe to S3: {s3_path}")
    except Exception as e:
        logger.error(f"Failed to upload file to S3: {e}")
        raise RuntimeError(f"Failed to upload file to S3: {e}") from e


# read yaml file for configuration
def read_yaml(file_path: str) -> dict[str, Any]:
    """
    Read a YAML configuration file and return its contents as a dictionary.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary containing the YAML configuration.

    Raises:
        ValueError: If the file path is not provided.
        RuntimeError: If reading or parsing the YAML file fails.
    """
    if not file_path:
        raise ValueError("File path must be provided.")

    try:
        with open(file_path, "r") as f:
            config: dict[str, Any] = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to read YAML file: {e}")
        raise RuntimeError(f"Failed to read YAML file: {e}") from e


def new_run_id() -> str:
    """Return a UTC timestamp run ID (e.g. ``20260925_142501``) for naming run folders."""
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def latest_run_id(runs_dir: Path) -> str:
    """Return the newest run folder name under `runs_dir`.

    Run IDs from `new_run_id` are timestamps, so the newest sorts last.

    Raises:
        FileNotFoundError: If `runs_dir` has no run folders.
    """
    run_ids = sorted(path.name for path in runs_dir.glob("*") if path.is_dir())
    if not run_ids:
        raise FileNotFoundError(f"No runs found in '{runs_dir}'.")
    return run_ids[-1]
