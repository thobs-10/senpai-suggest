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
from src.senpai_suggest.backend.pipelines.steps import catalog_preprocessing as catalog
from src.senpai_suggest.backend.pipelines.steps import preprocessing as steps


def _ingested_ratings() -> pa.Table:
    """Six users x ten anime; user 0 leaves everything unrated (0)."""
    rows = [(u, 100 + a, 0 if u == 0 else 1 + (u + a) % 10) for u in range(6) for a in range(10)]
    user_ids, anime_ids, ratings = zip(*rows, strict=True)
    return pa.table({"user_id": user_ids, "anime_id": anime_ids, "rating": ratings})


def _ingested_anime() -> pa.Table:
    """Anime 100-109 (the rated ones) plus 200, which nobody rated."""
    ids = [*range(100, 110), 200]
    return pa.table(
        {
            "anime_id": ids,
            "eng_version": [f"Anime {i}" for i in ids],
            "Score": [7.5] * len(ids),
            "Genres": ["Action, Comedy"] * len(ids),
            "Episodes": ["12"] * (len(ids) - 1) + [None],
            "Type": ["TV"] * len(ids),
            "Premiered": ["Spring 2006"] * (len(ids) - 1) + [None],
            "Members": [1000] * len(ids),
        }
    )


def _ingested_synopsis() -> pa.Table:
    """Synopses for anime 100 and 101; 101's is MAL's placeholder text."""
    return pa.table(
        {
            "anime_id": [100, 101],
            "Name": ["Anime 100", "Anime 101"],
            "Genres": ["Action, Comedy", "Action, Comedy"],
            "synopsis": [
                "A story.  (Source: ANN)",
                "No synopsis information has been added to this title.",
            ],
        }
    )


def _write_ingested_run(tmp_path: Path, run_id: str) -> None:
    run_dir = tmp_path / "ingested" / run_id
    run_dir.mkdir(parents=True)
    pq.write_table(_ingested_ratings(), run_dir / "ratings.parquet")
    pq.write_table(_ingested_anime(), run_dir / "anime.parquet")
    pq.write_table(_ingested_synopsis(), run_dir / "synopsis.parquet")


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
    wrapped: list[tuple[Task[..., Any], Callable[..., Any], str]] = [
        (pipeline.drop_invalid_rows, steps.drop_invalid_rows, "drop-invalid-rows"),
        (pipeline.dedupe_interactions, steps.dedupe_interactions, "dedupe-interactions"),
        (pipeline.flag_unrated, steps.flag_unrated, "flag-unrated"),
        (pipeline.filter_sparse, steps.filter_sparse, "filter-sparse"),
        (pipeline.add_confidence, steps.add_confidence, "add-confidence"),
        (pipeline.normalize_ratings, steps.normalize_ratings, "normalize-ratings"),
        (pipeline.split_per_user, steps.split_per_user, "split-per-user"),
        (pipeline.encode_splits, steps.encode_splits, "encode-splits"),
        (pipeline.drop_duplicate_anime, catalog.drop_duplicate_ids, "drop-duplicate-anime"),
        (pipeline.parse_episodes, catalog.parse_episodes, "parse-episodes"),
        (pipeline.parse_premiered, catalog.parse_premiered, "parse-premiered"),
        (pipeline.split_genres, catalog.split_genres, "split-genres"),
        (pipeline.drop_duplicate_synopses, catalog.drop_duplicate_ids, "drop-duplicate-synopses"),
        (pipeline.clean_synopsis_text, catalog.clean_synopsis_text, "clean-synopsis-text"),
        (pipeline.join_catalog, catalog.join_catalog, "join-catalog"),
        (pipeline.flag_rated_anime, catalog.flag_rated_anime, "flag-rated-anime"),
        (pipeline.add_embedding_text, catalog.add_embedding_text, "add-embedding-text"),
    ]
    for task_, fn, name in wrapped:
        assert task_.fn is fn
        assert task_.name == name


def test_load_ingested_reads_the_datasets_parquet(tmp_path: Path) -> None:
    """It should return the table the ingestion flow saved for that dataset."""
    pq.write_table(_ingested_ratings(), tmp_path / "ratings.parquet")

    assert pipeline.load_ingested.fn("ratings", tmp_path).num_rows == 60


def test_load_ingested_explains_missing_file(tmp_path: Path) -> None:
    """It should tell the user to run ingestion when the file is missing."""
    with pytest.raises(FileNotFoundError, match="No ingested anime .* Run the ingestion"):
        pipeline.load_ingested.fn("anime", tmp_path)


def test_save_anime_catalog_writes_parquet(tmp_path: Path) -> None:
    """It should write the catalog as anime_catalog.parquet."""
    table = pa.table({"anime_id": [1, 2], "has_ratings": [True, False]})

    path = pipeline.save_anime_catalog.fn(table, tmp_path / "run")

    assert path == tmp_path / "run" / "anime_catalog.parquet"
    assert pq.read_table(path).equals(table)


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


def test_run_preprocessing_pipeline_builds_the_anime_catalog(
    prefect_backend: None, tmp_path: Path
) -> None:
    """It should keep every anime, attach cleaned synopses and flag rated anime."""
    _write_ingested_run(tmp_path, "run1")

    paths = pipeline.run_preprocessing_pipeline(
        config_path=str(_write_config(tmp_path)), run_id="run1"
    )

    rows = {r["anime_id"]: r for r in pq.read_table(paths["catalog"]).to_pylist()}
    assert set(rows) == {*range(100, 110), 200}
    assert rows[100]["synopsis"] == "A story."
    assert rows[100]["embedding_text"] == "Anime 100. Genres: Action, Comedy. A story."
    assert rows[101]["synopsis"] is None  # placeholder text
    assert rows[100]["has_ratings"] and not rows[200]["has_ratings"]
    assert rows[100]["Episodes"] == 12 and rows[100]["year"] == 2006
    assert rows[200]["Episodes"] is None and rows[200]["season"] is None


def test_run_preprocessing_pipeline_defaults_to_latest_ingestion_run(
    prefect_backend: None, tmp_path: Path
) -> None:
    """It should read the newest ingestion run and reuse its ID for the output."""
    _write_ingested_run(tmp_path, "20260101_000000")
    _write_ingested_run(tmp_path, "20260925_120000")

    paths = pipeline.run_preprocessing_pipeline(config_path=str(_write_config(tmp_path)))

    assert paths["train"].parent == tmp_path / "processed" / "20260925_120000"
