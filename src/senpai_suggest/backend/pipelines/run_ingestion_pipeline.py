"""Prefect flow that ingests the raw datasets from S3 and reports their quality.

Each dataset goes through its own chain of tasks:

    fetch (retried) -> shape -> validate (report mode) -> save + raw report

Datasets run one after another, smallest first, so a config or S3 problem
shows up before the slow ratings read and only one dataset is in memory at a
time. The flow stops at the first failed dataset. Only `fetch` retries: a data
quality failure would fail the same way again, and report mode never raises
for value-level problems anyway.

Outputs per run:
    {ingestion.output_dir}/{run_id}/{dataset}.parquet   validated tables for preprocessing
    {data_quality.reports_dir}/{run_id}/                 raw reports + summary.json

Usage:
    uv run python -m src.senpai_suggest.backend.pipelines.run_ingestion_pipeline
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from src.senpai_suggest.backend.configs.backend_config import CONFIG_PATH
from src.senpai_suggest.backend.data_quality import DataQualityValidator, ValidationResult
from src.senpai_suggest.backend.data_quality.report import (
    Summary,
    write_run_summary,
    write_stage_reports,
)
from src.senpai_suggest.backend.logger.logger import Logger
from src.senpai_suggest.backend.pipelines.steps.anime_list_ingestion import clean_anime_metadata
from src.senpai_suggest.backend.pipelines.steps.synopsis_ingestion import rename_synopsis_columns
from src.senpai_suggest.backend.utils.main_utils import fetch_from_s3, new_run_id, read_yaml

logger: Logger = Logger()

# Dataset name -> shaping step, in run order (smallest first). Ratings arrive in
# pipeline shape already (columns are picked at fetch time), so shaping passes
# them through.
SHAPERS: dict[str, Callable[[pa.Table], pa.Table]] = {
    "anime": clean_anime_metadata,
    "synopsis": rename_synopsis_columns,
    "ratings": lambda table: table,
}
DATASETS = tuple(SHAPERS)

validator = DataQualityValidator()


# NO_CACHE on every task: Prefect's default cache key hashes task inputs,
# which is slow and pointless for large tables that change every run.
@task(
    name="fetch-dataset",
    task_run_name="fetch-{name}",
    retries=3,
    retry_delay_seconds=[10, 30, 60],
    cache_policy=NO_CACHE,
)
def fetch_dataset(name: str, dataset_config: dict[str, Any], num_rows: int | None) -> pa.Table:
    """Read one dataset's CSV from S3 (also cached locally as raw Parquet).

    Raises:
        RuntimeError: If the S3 read fails; Prefect retries it with backoff.
    """
    return fetch_from_s3(
        dataset_config["raw_path"],
        dataset_config["local_path"],
        columns=dataset_config.get("columns"),
        num_rows=num_rows,
    )


@task(name="shape-dataset", task_run_name="shape-{name}", cache_policy=NO_CACHE)
def shape_dataset(name: str, table: pa.Table) -> pa.Table:
    """Rename and select columns so the table matches its schema's names."""
    return SHAPERS[name](table)


@task(name="validate-dataset", task_run_name="validate-{name}", cache_policy=NO_CACHE)
def validate_dataset(name: str, table: pa.Table) -> ValidationResult:
    """Check the table in report mode: value failures are collected, not raised.

    Raises:
        DataQualityError: If a required column is missing.
    """
    return validator.validate(table, name, raise_on_fail=False)


@task(name="save-ingested", task_run_name="save-{name}", cache_policy=NO_CACHE)
def save_ingested(name: str, result: ValidationResult, output_dir: Path) -> Path:
    """Write the validated table to `{output_dir}/{name}.parquet` for preprocessing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.parquet"
    pq.write_table(result.table, path)
    logger.info(f"Saved {result.table.num_rows} ingested {name} rows to {path}.")
    return path


@task(name="report-raw-quality", task_run_name="report-{name}", cache_policy=NO_CACHE)
def report_raw_quality(name: str, result: ValidationResult, report_dir: Path) -> Summary:
    """Write the raw-stage HTML/JSON report and publish the Prefect table artifact."""
    return write_stage_reports({name: result}, "raw", report_dir)[name]


@flow(name="ingestion-pipeline")
def run_ingestion_pipeline(
    config_path: str = str(CONFIG_PATH),
    run_id: str | None = None,
) -> dict[str, Path]:
    """Ingest every dataset, save the validated tables, and report their quality.

    Args:
        config_path: Path to config.yaml.
        run_id: Output folder name; a new UTC timestamp when omitted. Pass the
            same ID to the preprocessing flow to read this run's output.

    Returns:
        Path of each saved table, keyed by dataset name.

    Raises:
        Exception: The first failed task's error; later datasets are not run.
    """
    config = read_yaml(config_path)
    ingestion = config["ingestion"]
    run_id = run_id or new_run_id()
    ingested_dir = Path(ingestion["output_dir"]) / run_id
    report_dir = Path(config["data_quality"]["reports_dir"]) / run_id

    paths: dict[str, Path] = {}
    summaries: list[Summary] = []
    for name in DATASETS:
        raw = fetch_dataset(name, ingestion[name], ingestion.get("num_rows"))
        result = validate_dataset(name, shape_dataset(name, raw))
        paths[name] = save_ingested(name, result, ingested_dir)
        summaries.append(report_raw_quality(name, result, report_dir))

    write_run_summary(summaries, report_dir)
    logger.info(f"Ingestion run {run_id} completed; reports in {report_dir}.")
    return paths


if __name__ == "__main__":
    run_ingestion_pipeline()
