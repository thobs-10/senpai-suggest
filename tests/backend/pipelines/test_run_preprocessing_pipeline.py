"""Unit tests for the preprocessing Prefect flow and its tasks."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from prefect import Task

from src.senpai_suggest.backend.pipelines import run_preprocessing_pipeline as pipeline
from src.senpai_suggest.backend.pipelines.steps import preprocessing as steps


def _ingested_ratings() -> pa.Table:
    """Six users x ten anime; user 0 leaves everything unrated (0)."""
    rows = [(u, 100 + a, 0 if u == 0 else 1 + (u + a) % 10) for u in range(6) for a in range(10)]
    user_ids, anime_ids, ratings = zip(*rows, strict=True)
    return pa.table({"user_id": user_ids, "anime_id": anime_ids, "rating": ratings})


def _write_ingested_run(tmp_path: Path, run_id: str) -> None:
    run_dir = tmp_path / "ingested" / run_id
    run_dir.mkdir(parents=True)
    pq.write_table(_ingested_ratings(), run_dir / "ratings.parquet")


def _write_config(tmp_path: Path) -> Path:
    config = {
        "ingestion": {"output_dir": str(tmp_path / "ingested")},
        "preprocessing": {
            "output_dir": str(tmp_path / "processed"),
            "ratings": {"min_user_interactions": 5, "min_anime_interactions": 5},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_every_step_is_a_named_task_wrapping_the_pure_function() -> None:
    """It should expose each preprocessing step as its own task in the Prefect UI."""
    wrapped: dict[Task[..., Any], Callable[..., Any]] = {
        pipeline.drop_invalid_rows: steps.drop_invalid_rows,
        pipeline.dedupe_interactions: steps.dedupe_interactions,
        pipeline.flag_unrated: steps.flag_unrated,
        pipeline.filter_sparse: steps.filter_sparse,
        pipeline.add_confidence: steps.add_confidence,
        pipeline.normalize_ratings: steps.normalize_ratings,
        pipeline.split_per_user: steps.split_per_user,
        pipeline.encode_splits: steps.encode_splits,
    }
    for task_, fn in wrapped.items():
        assert task_.fn is fn
        assert task_.name == fn.__name__.replace("_", "-")


def test_load_ingested_ratings_reads_parquet(tmp_path: Path) -> None:
    """It should return the table saved by the ingestion flow."""
    path = tmp_path / "ratings.parquet"
    pq.write_table(_ingested_ratings(), path)

    assert pipeline.load_ingested_ratings.fn(path).num_rows == 60


def test_load_ingested_ratings_explains_missing_file(tmp_path: Path) -> None:
    """It should tell the user to run ingestion when the file is missing."""
    with pytest.raises(FileNotFoundError, match="Run the ingestion pipeline first"):
        pipeline.load_ingested_ratings.fn(tmp_path / "missing.parquet")


def test_save_preprocessed_ratings_writes_splits_and_id_maps(tmp_path: Path) -> None:
    """It should write both splits and index-ordered ID lists."""
    train, test = steps.split_per_user(_ingested_ratings(), test_fraction=0.2, seed=0)
    encoded = steps.encode_splits(train, test)

    paths = pipeline.save_preprocessed_ratings.fn(encoded, tmp_path / "run")

    assert pq.read_table(paths["train"]).equals(encoded.train)
    assert pq.read_table(paths["test"]).equals(encoded.test)
    id_maps = json.loads(paths["id_maps"].read_text(encoding="utf-8"))
    assert id_maps["user_ids"] == [encoded.user_decoder[i] for i in range(6)]
    assert len(id_maps["anime_ids"]) == len(encoded.anime_encoder)


def test_run_preprocessing_pipeline_end_to_end(prefect_backend: None, tmp_path: Path) -> None:
    """It should clean, split and encode a run's ratings, keeping unrated users as implicit."""
    _write_ingested_run(tmp_path, "run1")

    paths = pipeline.run_preprocessing_pipeline(
        config_path=str(_write_config(tmp_path)), run_id="run1"
    )

    assert paths["train"] == tmp_path / "processed" / "run1" / "ratings_train.parquet"
    train = pq.read_table(paths["train"])
    test = pq.read_table(paths["test"])
    assert train.num_rows == 6 * 8 and test.num_rows == 6 * 2
    assert train.column_names == [
        "user_id",
        "anime_id",
        "rating",
        "is_rated",
        "confidence",
        "user",
        "anime",
    ]
    user_ids = json.loads(paths["id_maps"].read_text(encoding="utf-8"))["user_ids"]
    for row in train.to_pylist():
        assert user_ids[row["user"]] == row["user_id"]
    unrated_user = [r for r in train.to_pylist() if r["user_id"] == 0]
    assert unrated_user and all(
        r["rating"] is None and not r["is_rated"] and r["confidence"] == 1.0 for r in unrated_user
    )


def test_run_preprocessing_pipeline_defaults_to_latest_ingestion_run(
    prefect_backend: None, tmp_path: Path
) -> None:
    """It should read the newest ingestion run and reuse its ID for the output."""
    _write_ingested_run(tmp_path, "20260101_000000")
    _write_ingested_run(tmp_path, "20260925_120000")

    paths = pipeline.run_preprocessing_pipeline(config_path=str(_write_config(tmp_path)))

    assert paths["train"].parent == tmp_path / "processed" / "20260925_120000"
