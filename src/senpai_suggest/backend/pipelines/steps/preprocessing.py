"""Preprocess validated ratings into model-ready form.

Takes the output of `userlist_ingestion` and normalises ratings, shuffles
rows, and encodes user/anime IDs into contiguous integers. The encoding maps
are returned alongside the table because serving needs them to translate
between raw IDs and model indices.
"""

from typing import Any

import numpy as np
import pyarrow as pa

from src.senpai_suggest.backend.configs.backend_config import EncodedRatings
from src.senpai_suggest.backend.logger.logger import Logger

logger: Logger = Logger()


def normalize_ratings(
    table: pa.Table,
    min_rating: float,
    max_rating: float,
) -> pa.Table:
    """Scale the `rating` column to the 0-1 range.

    Raises:
        ValueError: If `rating` is missing or `min_rating == max_rating`.
    """
    if "rating" not in table.column_names:
        raise ValueError("Table must contain 'rating' column.")

    scale = max_rating - min_rating
    if scale == 0:
        raise ValueError("max_rating and min_rating cannot be the same.")

    ratings = table["rating"].combine_chunks().cast(pa.float64())
    normalized = ((ratings - min_rating) / scale).cast(pa.float64())
    return table.set_column(table.column_names.index("rating"), "rating", normalized)


def shuffle_rows(table: pa.Table, seed: int) -> pa.Table:
    """Shuffle rows reproducibly; the same seed always gives the same order."""
    indices = np.random.default_rng(seed).permutation(table.num_rows)
    return table.take(pa.array(indices, type=pa.int64()))


def encode_ids(
    table: pa.Table,
    source_column: str,
    target_column: str,
) -> tuple[pa.Table, dict[Any, int], dict[int, Any]]:
    """Map the values of `source_column` to contiguous ints in a new `target_column`.

    IDs are numbered in order of first appearance.

    Returns:
        The table with `target_column` appended, the raw->encoded map, and the
        encoded->raw map.

    Raises:
        ValueError: If `source_column` is missing.
    """
    if source_column not in table.column_names:
        raise ValueError(f"Table must contain '{source_column}' column.")

    raw_ids = table[source_column].to_pylist()
    encoder = {raw_id: i for i, raw_id in enumerate(dict.fromkeys(raw_ids))}
    decoder = {i: raw_id for raw_id, i in encoder.items()}
    encoded = pa.array([encoder[raw_id] for raw_id in raw_ids], type=pa.int32())
    return table.append_column(target_column, encoded), encoder, decoder


def run_ratings_preprocessing(
    ratings: pa.Table,
    min_rating: float = 0,
    max_rating: float = 10,
    seed: int = 42,
) -> EncodedRatings:
    """Normalise, shuffle, and encode validated ratings."""
    logger.info("Starting ratings preprocessing.")
    table = normalize_ratings(ratings, min_rating=min_rating, max_rating=max_rating)
    table = shuffle_rows(table, seed=seed)
    table, user_encoder, user_decoder = encode_ids(table, "user_id", "user")
    table, anime_encoder, anime_decoder = encode_ids(table, "anime_id", "anime")
    logger.info(f"Encoded {len(user_encoder)} users and {len(anime_encoder)} anime.")
    return EncodedRatings(table, user_encoder, user_decoder, anime_encoder, anime_decoder)
