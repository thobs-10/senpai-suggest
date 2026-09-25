"""Prefect flow that turns ingested ratings into model-ready train/test tables.

Reads `{ingestion.output_dir}/{run_id}/ratings.parquet` from the ingestion flow
and runs every preprocessing step as its own task, so the Prefect UI shows
where time goes and which step failed:

    load -> drop-invalid-rows -> dedupe-interactions -> flag-unrated
         -> filter-sparse -> add-confidence -> normalize-ratings
         -> split-per-user -> encode-splits -> save

Writes to `{preprocessing.output_dir}/{run_id}/`, using the ingestion run's ID:

    ratings_train.parquet
    ratings_test.parquet
    id_maps.json      # model index -> raw ID, for users and anime

The processed-stage data quality gate joins once the processed schemas exist
(step 2e); anime and synopsis preprocessing join as they land.

Usage:
    uv run python -m src.senpai_suggest.backend.pipelines.run_preprocessing_pipeline
"""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from prefect import Task, flow, task
from prefect.cache_policies import NO_CACHE

from src.senpai_suggest.backend.configs.backend_config import (
    CONFIG_PATH,
    EncodedRatings,
    RatingsPreprocessingConfig,
)
from src.senpai_suggest.backend.logger.logger import Logger
from src.senpai_suggest.backend.pipelines.steps import preprocessing as steps
from src.senpai_suggest.backend.utils.main_utils import latest_run_id, read_yaml

logger: Logger = Logger()


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


@task(name="load-ingested-ratings", cache_policy=NO_CACHE)
def load_ingested_ratings(path: Path) -> pa.Table:
    """Read the ratings Parquet saved by the ingestion flow.

    Raises:
        FileNotFoundError: If that ingestion run has no ratings file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No ingested ratings at '{path}'. Run the ingestion pipeline first."
        )
    table: pa.Table = pq.read_table(path)
    logger.info(f"Loaded {table.num_rows} ingested ratings from {path}.")
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


@flow(name="preprocessing-pipeline")
def run_preprocessing_pipeline(
    config_path: str = str(CONFIG_PATH),
    run_id: str | None = None,
) -> dict[str, Path]:
    """Preprocess one ingestion run's ratings and save the outputs under the same run ID.

    Args:
        config_path: Path to config.yaml.
        run_id: Ingestion run to read; the latest one when omitted.

    Returns:
        Paths of the written files, keyed by artifact name.

    Raises:
        FileNotFoundError: If there is no ingestion run (or no ratings in it).
    """
    config = read_yaml(config_path)
    ingested_dir = Path(config["ingestion"]["output_dir"])
    run_id = run_id or latest_run_id(ingested_dir)
    ratings_config = RatingsPreprocessingConfig.from_dict(config["preprocessing"]["ratings"])
    logger.info(f"Preprocessing ingestion run {run_id}.")

    ratings = load_ingested_ratings(ingested_dir / run_id / "ratings.parquet")
    cleaned = _clean_ratings(ratings, ratings_config)
    train, test = split_per_user(cleaned, ratings_config.test_fraction, ratings_config.seed)
    encoded = encode_splits(train, test)

    output_dir = Path(config["preprocessing"]["output_dir"]) / run_id
    paths: dict[str, Path] = save_preprocessed_ratings(encoded, output_dir)
    return paths


def _clean_ratings(table: pa.Table, config: RatingsPreprocessingConfig) -> pa.Table:
    """Run the value-fixing steps in order, logging how many rows each one keeps."""
    fixes: list[tuple[Task[..., Any], tuple[Any, ...]]] = [
        (drop_invalid_rows, (config.unrated_value, config.min_rating, config.max_rating)),
        (dedupe_interactions, ()),
        (flag_unrated, (config.unrated_value,)),
        (filter_sparse, (config.min_user_interactions, config.min_anime_interactions)),
        (add_confidence, (config.confidence_alpha, config.unrated_confidence)),
        (normalize_ratings, (config.min_rating, config.max_rating)),
    ]
    for step, args in fixes:
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
