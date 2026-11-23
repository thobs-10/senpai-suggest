"""Unit tests for the parent pipeline flow; S3 is mocked."""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pytest_mock import MockerFixture

from src.senpai_suggest.backend.pipelines import main as pipeline
from src.senpai_suggest.backend.pipelines import run_ingestion_pipeline as ingestion


def _write_config(tmp_path: Path, ingestion_config: dict[str, Any]) -> Path:
    config = {
        "ingestion": {**ingestion_config, "output_dir": str(tmp_path / "ingested")},
        "data_quality": {"reports_dir": str(tmp_path / "reports")},
        "preprocessing": {
            "output_dir": str(tmp_path / "processed"),
            # The fixture ratings are tiny, so keep every user and anime.
            "ratings": {"min_user_interactions": 1, "min_anime_interactions": 1},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_run_pipeline_runs_every_stage_under_one_run_id(
    prefect_backend: None,
    mocker: MockerFixture,
    tmp_path: Path,
    ingestion_config: dict[str, Any],
    raw_anime_table: pa.Table,
    raw_synopsis_table: pa.Table,
    sample_ratings_table: pa.Table,
) -> None:
    """It should ingest then preprocess, writing both stages' outputs under the same run ID."""
    tables = {
        "anime.csv": raw_anime_table,
        "synopsis.csv": raw_synopsis_table,
        "animelist.csv": sample_ratings_table,
    }
    mocker.patch.object(
        ingestion,
        "fetch_from_s3",
        side_effect=lambda raw_path, *_, **__: tables[Path(raw_path).name],
    )

    outputs = pipeline.run_pipeline(
        config_path=str(_write_config(tmp_path, ingestion_config)), run_id="run1"
    )

    assert outputs["ingestion"]["ratings"] == tmp_path / "ingested" / "run1" / "ratings.parquet"
    assert outputs["preprocessing"]["train"].parent == tmp_path / "processed" / "run1"
    assert pq.read_table(outputs["preprocessing"]["train"]).num_rows > 0
    assert (tmp_path / "reports" / "run1" / "summary.json").exists()


def test_run_pipeline_stops_when_a_stage_fails(
    prefect_backend: None, mocker: MockerFixture
) -> None:
    """It should not start preprocessing when ingestion fails."""
    mocker.patch.object(pipeline, "run_ingestion_pipeline", side_effect=RuntimeError("S3 down"))
    preprocess = mocker.patch.object(pipeline, "run_preprocessing_pipeline")

    with pytest.raises(RuntimeError, match="S3 down"):
        pipeline.run_pipeline(run_id="run2")

    preprocess.assert_not_called()
