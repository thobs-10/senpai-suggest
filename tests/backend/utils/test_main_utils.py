"""Unit tests for the shared backend utilities; S3 calls are mocked."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pytest_mock import MockerFixture

from src.senpai_suggest.backend.utils import main_utils

RATINGS_DF = pd.DataFrame({"user_id": [1, 2], "anime_id": [10, 20], "rating": [8, 9]})


# ---- convert_to_parquet ----


def test_convert_to_parquet_writes_file_and_returns_table(tmp_path: Path) -> None:
    """It should create missing parent dirs, write Parquet, and return the same data."""
    output = tmp_path / "nested" / "ratings.parquet"

    table = main_utils.convert_to_parquet(RATINGS_DF, str(output))

    assert output.exists()
    assert pq.read_table(output).equals(table)  # type: ignore[no-untyped-call, unused-ignore]  # no pyarrow stubs
    assert table.num_rows == 2


@pytest.mark.parametrize("data", [None, pd.DataFrame()])
def test_convert_to_parquet_rejects_missing_or_empty_data(
    tmp_path: Path, data: pd.DataFrame | None
) -> None:
    """It should refuse to write None or empty DataFrames."""
    with pytest.raises(ValueError, match="empty or None"):
        main_utils.convert_to_parquet(data, str(tmp_path / "x.parquet"))


def test_convert_to_parquet_wraps_write_errors(tmp_path: Path, mocker: MockerFixture) -> None:
    """It should wrap write failures in RuntimeError."""
    mocker.patch("pyarrow.parquet.write_table", side_effect=OSError("disk full"))

    with pytest.raises(RuntimeError, match="disk full"):
        main_utils.convert_to_parquet(RATINGS_DF, str(tmp_path / "x.parquet"))


# ---- fetch_from_s3 ----


def test_fetch_from_s3_reads_csv_with_options_and_caches_parquet(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """It should pass columns/num_rows to awswrangler and write the local Parquet cache."""
    read_csv = mocker.patch("awswrangler.s3.read_csv", return_value=RATINGS_DF)
    output = tmp_path / "ratings.parquet"

    table = main_utils.fetch_from_s3(
        "s3://bucket/ratings.csv", str(output), columns=["user_id"], num_rows=5
    )

    read_csv.assert_called_once_with(
        path="s3://bucket/ratings.csv", use_threads=True, usecols=["user_id"], nrows=5
    )
    assert output.exists()
    assert table.to_pydict() == pa.Table.from_pandas(RATINGS_DF).to_pydict()


@pytest.mark.parametrize(
    ("raw_path", "output", "message"),
    [("s3://bucket/x.csv", "", "Output file path"), ("", "x.parquet", "S3 raw data path")],
)
def test_fetch_from_s3_requires_paths(raw_path: str, output: str, message: str) -> None:
    """It should reject missing S3 or output paths before calling S3."""
    with pytest.raises(ValueError, match=message):
        main_utils.fetch_from_s3(raw_path, output)


def test_fetch_from_s3_wraps_s3_errors(tmp_path: Path, mocker: MockerFixture) -> None:
    """It should wrap S3 read errors in RuntimeError so Prefect can retry the task."""
    mocker.patch("awswrangler.s3.read_csv", side_effect=OSError("timeout"))

    with pytest.raises(RuntimeError, match="Failed to fetch data from S3: timeout"):
        main_utils.fetch_from_s3("s3://bucket/x.csv", str(tmp_path / "x.parquet"))


# ---- save_to_s3 ----


def test_save_to_s3_uploads_timestamped_parquet(mocker: MockerFixture) -> None:
    """It should upload to a timestamped Parquet key in the bucket."""
    to_parquet = mocker.patch("awswrangler.s3.to_parquet")
    table = pa.Table.from_pandas(RATINGS_DF)

    main_utils.save_to_s3(table, "my-bucket")

    path = to_parquet.call_args.kwargs["path"]
    assert path.startswith("s3://my-bucket/") and path.endswith(".parquet")


@pytest.mark.parametrize(
    ("table", "bucket", "message"),
    [(pa.table({"a": [1]}), "", "Bucket name"), (None, "my-bucket", "DataFrame must be provided")],
)
def test_save_to_s3_requires_table_and_bucket(
    table: pa.Table | None, bucket: str, message: str
) -> None:
    """It should reject a missing table or bucket name."""
    with pytest.raises(ValueError, match=message):
        main_utils.save_to_s3(table, bucket)


def test_save_to_s3_wraps_upload_errors(mocker: MockerFixture) -> None:
    """It should wrap upload failures in RuntimeError."""
    mocker.patch("awswrangler.s3.to_parquet", side_effect=OSError("denied"))

    with pytest.raises(RuntimeError, match="denied"):
        main_utils.save_to_s3(pa.table({"a": [1]}), "my-bucket")


# ---- read_yaml ----


def test_read_yaml_returns_parsed_dict(tmp_path: Path) -> None:
    """It should parse a YAML file into a dict."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("ingestion:\n  num_rows: 5\n")

    assert main_utils.read_yaml(str(config_file)) == {"ingestion": {"num_rows": 5}}


def test_read_yaml_requires_path() -> None:
    """It should reject an empty path."""
    with pytest.raises(ValueError, match="File path must be provided"):
        main_utils.read_yaml("")


def test_read_yaml_wraps_missing_file(tmp_path: Path) -> None:
    """It should wrap file errors in RuntimeError."""
    with pytest.raises(RuntimeError, match="Failed to read YAML file"):
        main_utils.read_yaml(str(tmp_path / "missing.yaml"))
