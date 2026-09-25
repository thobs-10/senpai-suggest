"""Prefect flow that turns one ingestion run into model-ready tables.

Reads `{ingestion.output_dir}/{run_id}/{ratings,anime,synopsis}.parquet` from
the ingestion flow and runs every preprocessing step as its own task, so the
Prefect UI shows where time goes and which step failed:

    ratings:  load -> drop-invalid-rows -> dedupe-interactions -> flag-unrated
              -> filter-sparse -> add-confidence -> normalize-ratings
              -> split-per-user -> encode-splits -> save
    anime:    load -> drop-duplicate-anime -> parse-episodes -> parse-premiered
              -> split-genres
    synopsis: load -> drop-duplicate-synopses -> clean-synopsis-text
    catalog:  join-catalog -> flag-rated-anime -> add-embedding-text -> save

Writes to `{preprocessing.output_dir}/{run_id}/`, using the ingestion run's ID:

    ratings_train.parquet
    ratings_test.parquet
    id_maps.json            # model index -> raw ID, for users and anime
    anime_catalog.parquet   # every anime, with synopsis and embedding text

The processed-stage data quality gate joins once the processed schemas exist
(step 2e).

Usage:
    uv run python -m src.senpai_suggest.backend.pipelines.run_preprocessing_pipeline
"""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from prefect import Task, flow, task
from prefect.cache_policies import NO_CACHE

from src.senpai_suggest.backend.configs.backend_config import (
    CONFIG_PATH,
    EncodedRatings,
    RatingsPreprocessingConfig,
)
from src.senpai_suggest.backend.logger.logger import Logger
from src.senpai_suggest.backend.pipelines.steps import catalog_preprocessing as catalog
from src.senpai_suggest.backend.pipelines.steps import preprocessing as steps
from src.senpai_suggest.backend.utils.main_utils import latest_run_id, read_yaml

logger: Logger = Logger()

# A step list: each task with the extra arguments it takes after the table.
Steps = list[tuple[Task[..., Any], tuple[Any, ...]]]


# One task per step, applied here so the step functions stay Prefect-free.
# NO_CACHE: Prefect's default cache key hashes task inputs, which is slow and
# pointless for large tables.
drop_invalid_rows = task(name="drop-invalid-rows", cache_policy=NO_CACHE)(steps.drop_invalid_rows)
dedupe_interactions = task(name="dedupe-interactions", cache_policy=NO_CACHE)(
    steps.dedupe_interactions
)
flag_unrated = task(name="flag-unrated", cache_policy=NO_CACHE)(steps.flag_unrated)
filter_sparse = task(name="filter-sparse", cache_policy=NO_CACHE)(steps.filter_sparse)
add_confidence = task(name="add-confidence", cache_policy=NO_CACHE)(steps.add_confidence)
normalize_ratings = task(name="normalize-ratings", cache_policy=NO_CACHE)(steps.normalize_ratings)
split_per_user = task(name="split-per-user", cache_policy=NO_CACHE)(steps.split_per_user)
encode_splits = task(name="encode-splits", cache_policy=NO_CACHE)(steps.encode_splits)

drop_duplicate_anime = task(name="drop-duplicate-anime", cache_policy=NO_CACHE)(
    catalog.drop_duplicate_ids
)
parse_episodes = task(name="parse-episodes", cache_policy=NO_CACHE)(catalog.parse_episodes)
parse_premiered = task(name="parse-premiered", cache_policy=NO_CACHE)(catalog.parse_premiered)
split_genres = task(name="split-genres", cache_policy=NO_CACHE)(catalog.split_genres)
drop_duplicate_synopses = task(name="drop-duplicate-synopses", cache_policy=NO_CACHE)(
    catalog.drop_duplicate_ids
)
clean_synopsis_text = task(name="clean-synopsis-text", cache_policy=NO_CACHE)(
    catalog.clean_synopsis_text
)
join_catalog = task(name="join-catalog", cache_policy=NO_CACHE)(catalog.join_catalog)
flag_rated_anime = task(name="flag-rated-anime", cache_policy=NO_CACHE)(catalog.flag_rated_anime)
add_embedding_text = task(name="add-embedding-text", cache_policy=NO_CACHE)(
    catalog.add_embedding_text
)


@task(name="load-ingested", task_run_name="load-{name}", cache_policy=NO_CACHE)
def load_ingested(name: str, run_dir: Path) -> pa.Table:
    """Read `{run_dir}/{name}.parquet` saved by the ingestion flow.

    Raises:
        FileNotFoundError: If that ingestion run has no such file.
    """
    path = run_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No ingested {name} at '{path}'. Run the ingestion pipeline first."
        )
    table: pa.Table = pq.read_table(path)
    logger.info(f"Loaded {table.num_rows} ingested {name} rows from {path}.")
    return table


@task(name="save-preprocessed-ratings", cache_policy=NO_CACHE)
def save_preprocessed_ratings(encoded: EncodedRatings, output_dir: Path) -> dict[str, Path]:
    """Write the train/test splits and the ID maps into `output_dir`.

    The ID maps are stored as lists where position = model index, which is
    all serving needs to turn model output back into raw IDs.

    Returns:
        Paths of the written files, keyed by artifact name.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "ratings_train.parquet",
        "test": output_dir / "ratings_test.parquet",
        "id_maps": output_dir / "id_maps.json",
    }
    pq.write_table(encoded.train, paths["train"])
    pq.write_table(encoded.test, paths["test"])
    paths["id_maps"].write_text(json.dumps(_id_maps(encoded)), encoding="utf-8")

    logger.info(
        f"Saved {encoded.train.num_rows} train / {encoded.test.num_rows} test rows, "
        f"{len(encoded.user_encoder)} users, {len(encoded.anime_encoder)} anime to {output_dir}."
    )
    return paths


@task(name="save-anime-catalog", cache_policy=NO_CACHE)
def save_anime_catalog(catalog_table: pa.Table, output_dir: Path) -> Path:
    """Write the anime catalog to `{output_dir}/anime_catalog.parquet`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "anime_catalog.parquet"
    pq.write_table(catalog_table, path)
    rated = pc.sum(catalog_table["has_ratings"]).as_py() or 0
    logger.info(f"Saved {catalog_table.num_rows} anime ({rated} with ratings) to {path}.")
    return path


@flow(name="preprocessing-pipeline")
def run_preprocessing_pipeline(
    config_path: str = str(CONFIG_PATH),
    run_id: str | None = None,
) -> dict[str, Path]:
    """Preprocess one ingestion run and save the outputs under the same run ID.

    Args:
        config_path: Path to config.yaml.
        run_id: Ingestion run to read; the latest one when omitted.

    Returns:
        Paths of the written files, keyed by artifact name.

    Raises:
        FileNotFoundError: If there is no ingestion run, or a dataset is missing from it.
    """
    config = read_yaml(config_path)
    ingested_root = Path(config["ingestion"]["output_dir"])
    run_id = run_id or latest_run_id(ingested_root)
    ratings_config = RatingsPreprocessingConfig.from_dict(config["preprocessing"]["ratings"])
    run_dir = ingested_root / run_id
    output_dir = Path(config["preprocessing"]["output_dir"]) / run_id
    logger.info(f"Preprocessing ingestion run {run_id}.")

    encoded = _preprocess_ratings(run_dir, ratings_config)
    paths: dict[str, Path] = save_preprocessed_ratings(encoded, output_dir)

    rated_anime_ids = pc.unique(encoded.train["anime_id"])
    catalog_table = _build_catalog(run_dir, rated_anime_ids)
    paths["catalog"] = save_anime_catalog(catalog_table, output_dir)
    return paths


def _preprocess_ratings(run_dir: Path, config: RatingsPreprocessingConfig) -> EncodedRatings:
    """Clean, split and encode the run's ratings."""
    ratings = load_ingested("ratings", run_dir)
    cleaned = _run_steps(
        ratings,
        [
            (drop_invalid_rows, (config.unrated_value, config.min_rating, config.max_rating)),
            (dedupe_interactions, ()),
            (flag_unrated, (config.unrated_value,)),
            (filter_sparse, (config.min_user_interactions, config.min_anime_interactions)),
            (add_confidence, (config.confidence_alpha, config.unrated_confidence)),
            (normalize_ratings, (config.min_rating, config.max_rating)),
        ],
    )
    train, test = split_per_user(cleaned, config.test_fraction, config.seed)
    encoded: EncodedRatings = encode_splits(train, test)
    return encoded


def _build_catalog(run_dir: Path, rated_anime_ids: pa.Array) -> pa.Table:
    """Clean anime metadata and synopses, then join them into one catalog."""
    anime_steps: Steps = [
        (drop_duplicate_anime, ()),
        (parse_episodes, ()),
        (parse_premiered, ()),
        (split_genres, ()),
    ]
    anime = _run_steps(load_ingested("anime", run_dir), anime_steps)
    synopsis_steps: Steps = [(drop_duplicate_synopses, ()), (clean_synopsis_text, ())]
    synopsis = _run_steps(load_ingested("synopsis", run_dir), synopsis_steps)

    table = join_catalog(anime, synopsis)
    table = flag_rated_anime(table, rated_anime_ids)
    return add_embedding_text(table)


def _run_steps(table: pa.Table, step_list: Steps) -> pa.Table:
    """Run steps in order, logging how many rows each one keeps."""
    for step, args in step_list:
        rows_before = table.num_rows
        table = step(table, *args)
        logger.info(f"{step.name}: {rows_before} -> {table.num_rows} rows.")
    return table


def _id_maps(encoded: EncodedRatings) -> dict[str, list[Any]]:
    # Decoders map 0..n-1 to raw IDs, so reading them in index order gives the lookup lists.
    return {
        "user_ids": [encoded.user_decoder[i] for i in range(len(encoded.user_decoder))],
        "anime_ids": [encoded.anime_decoder[i] for i in range(len(encoded.anime_decoder))],
    }


if __name__ == "__main__":
    run_preprocessing_pipeline()
