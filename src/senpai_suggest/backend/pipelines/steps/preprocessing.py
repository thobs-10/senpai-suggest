"""Steps that turn ingested ratings into model-ready train/test tables.

Preprocessing is the only stage that changes values; data quality only
reports. The functions here are pure and Prefect-free; `run_preprocessing_pipeline`
runs each one as its own task, in this order:

1. Drops rows with null IDs or out-of-range ratings.
2. Keeps one row per (user, anime) pair.
3. Treats rating 0 as "listed but not scored": `rating` becomes null and
   `is_rated` false. It stays an implicit interaction, not a dislike.
4. Drops users and anime with too few interactions (rated or not).
5. Adds the ALS `confidence` weight, then scales ratings to 0-1.
6. Holds out a share of each user's interactions as the test split.
7. Encodes user/anime IDs as contiguous ints, fitted on train only.

The encoding maps are returned with the tables because serving needs them to
translate between raw IDs and model indices.
"""

from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from src.senpai_suggest.backend.configs.backend_config import (
    EncodedRatings,
    get_preprocessing_key_columns,
)
from src.senpai_suggest.backend.logger.logger import Logger
from src.senpai_suggest.backend.utils.table_utils import replace_column, require_columns

logger: Logger = Logger()

KEY_COLUMNS = tuple(get_preprocessing_key_columns().keys())


def drop_invalid_rows(
    table: pa.Table,
    unrated_value: int,
    min_rating: float,
    max_rating: float,
) -> pa.Table:
    """Drop rows with a null user/anime ID or a rating outside the valid values.

    A null rating is read as "not scored" and becomes `unrated_value`, so the
    interaction itself is kept.

    args:
        table: The input PyArrow table containing user-anime interactions.
        unrated_value: The value representing an unrated interaction.
        min_rating: The minimum valid rating.
        max_rating: The maximum valid rating.

    Returns:
        pa.Table: The filtered PyArrow table containing only valid rows.

    Raises:
        ValueError: If a key column or `rating` is missing.
    """
    require_columns(table, *KEY_COLUMNS, "rating")
    rating = pc.fill_null(table["rating"], unrated_value)
    table = replace_column(table, "rating", rating)

    in_range = pc.and_(pc.greater_equal(rating, min_rating), pc.less_equal(rating, max_rating))
    valid_rating = pc.or_(pc.equal(rating, unrated_value), in_range)
    valid_keys = pc.and_(pc.is_valid(table["user_id"]), pc.is_valid(table["anime_id"]))
    return table.filter(pc.and_(valid_keys, valid_rating))


def dedupe_interactions(table: pa.Table) -> pa.Table:
    """Keep one row per (user_id, anime_id), with the highest rating.

    Taking the max means a real score wins over an unrated 0. The output is
    sorted by the key columns so later seeded steps are reproducible no matter
    the input order.

    args:
        table: The input PyArrow table containing user-anime interactions. Only the first
            occurrence of each (user_id, anime_id) pair is kept.

    Returns:
        pa.Table: The deduplicated PyArrow table containing only the highest rating per (user_id, anime_id) pair.

    Raises:
        ValueError: If a key column or `rating` is missing.
    """
    require_columns(table, *KEY_COLUMNS, "rating")
    deduped = table.group_by(list(KEY_COLUMNS)).aggregate([("rating", "max")])
    deduped = deduped.rename_columns(
        ["rating" if c == "rating_max" else c for c in deduped.column_names]
    )
    return deduped.select([*KEY_COLUMNS, "rating"]).sort_by([(c, "ascending") for c in KEY_COLUMNS])


def flag_unrated(table: pa.Table, unrated_value: int) -> pa.Table:
    """Add `is_rated`, and null out `rating` where it equals `unrated_value`.

    args:
        table: The input PyArrow table containing user-anime interactions.
        unrated_value: The value representing an unrated interaction.
    Returns:
        pa.Table: The PyArrow table with the `is_rated` column added and `rating` nulled where unrated.
    Raises:
        ValueError: If `rating` is missing.
    """
    require_columns(table, "rating")
    is_rated = pc.not_equal(table["rating"], unrated_value)
    rating = pc.if_else(is_rated, table["rating"], pa.scalar(None, table["rating"].type))
    return replace_column(table, "rating", rating).append_column("is_rated", is_rated)


def filter_sparse(
    table: pa.Table,
    min_user_interactions: int,
    min_anime_interactions: int,
    max_rounds: int = 10,
) -> pa.Table:
    """Drop users and anime with too few interactions.

    args:
        table: The input PyArrow table containing user-anime interactions.
        min_user_interactions: Minimum number of interactions a user must have to be kept.
        min_anime_interactions: Minimum number of interactions an anime must have to be kept.
        max_rounds: Maximum number of iterations to apply the filtering.

    Returns:
        pa.Table: The filtered PyArrow table containing only users and anime with sufficient interactions.

    Removing sparse anime can push a user under the cutoff and vice versa, so
    both filters repeat until nothing changes (or `max_rounds` is reached).

    Raises:
        ValueError: If a key column is missing.
    """
    require_columns(table, *KEY_COLUMNS)
    for _ in range(max_rounds):
        rows_before = table.num_rows
        table = _keep_frequent(table, "anime_id", min_anime_interactions)
        table = _keep_frequent(table, "user_id", min_user_interactions)
        if table.num_rows == rows_before:
            break
    return table


def add_confidence(
    table: pa.Table,
    alpha: float,
    unrated_confidence: float,
) -> pa.Table:
    """Add the ALS `confidence` column: 1 + alpha * rating, or `unrated_confidence` if unrated.

    Must run before `normalize_ratings`, because it uses the raw score.

    args:
        table: The input PyArrow table containing user-anime interactions.
        alpha: The ALS confidence slope for rated rows.
        unrated_confidence: The ALS confidence for listed-but-unrated rows.

    Returns:
        pa.Table: The PyArrow table with the `confidence` column added.

    Raises:
        ValueError: If `rating` is missing.
    """
    require_columns(table, "rating")
    rating = table["rating"].cast(pa.float64())
    confidence = pc.fill_null(pc.add(1.0, pc.multiply(rating, alpha)), unrated_confidence)
    return table.append_column("confidence", confidence)


def normalize_ratings(
    table: pa.Table,
    min_rating: float,
    max_rating: float,
) -> pa.Table:
    """Scale the `rating` column to the 0-1 range; null ratings stay null.

    args:
        table: The input PyArrow table containing user-anime interactions.
        min_rating: The minimum possible rating.
        max_rating: The maximum possible rating.

    Returns:
        pa.Table: The PyArrow table with the `rating` column normalized to [0, 1].

    Raises:
        ValueError: If `rating` is missing or `min_rating == max_rating`.
    """
    require_columns(table, "rating")
    scale = max_rating - min_rating
    if scale == 0:
        raise ValueError("max_rating and min_rating cannot be the same.")

    ratings = table["rating"].cast(pa.float64())
    return replace_column(table, "rating", pc.divide(pc.subtract(ratings, min_rating), scale))


def shuffle_rows(table: pa.Table, seed: int) -> pa.Table:
    """Shuffle rows reproducibly; the same seed always gives the same order."""
    indices = np.random.default_rng(seed).permutation(table.num_rows)
    return table.take(pa.array(indices, type=pa.int64()))


# ? This is not a train test split, it's for the model to know each user beforehand.
def split_per_user(
    table: pa.Table,
    test_fraction: float,
    seed: int,
) -> tuple[pa.Table, pa.Table]:
    """Hold out `floor(n * test_fraction)` random interactions of each user as test.

    Every user keeps at least one training row, so all test users are known
    to the model.

    Returns:
        (train, test) tables with the input's columns.

    Raises:
        ValueError: If `user_id` is missing or `test_fraction` is not in [0, 1).
    """
    require_columns(table, "user_id")
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must be in [0, 1).")

    shuffled = shuffle_rows(table, seed)
    is_test = _holdout_mask(shuffled["user_id"].to_numpy(), test_fraction)
    mask = pa.array(is_test)
    return shuffled.filter(pc.invert(mask)), shuffled.filter(mask)


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
    require_columns(table, source_column)
    raw_ids = pc.unique(table[source_column]).to_pylist()
    encoder = {raw_id: i for i, raw_id in enumerate(raw_ids)}
    decoder = dict(enumerate(raw_ids))
    return apply_encoding(table, source_column, target_column, encoder), encoder, decoder


def apply_encoding(
    table: pa.Table,
    source_column: str,
    target_column: str,
    encoder: dict[Any, int],
) -> pa.Table:
    """Append `target_column` using an existing encoder; rows with unseen IDs are dropped.

    Raises:
        ValueError: If `source_column` is missing.
    """
    require_columns(table, source_column)
    # Dict order matches the codes (encoder values are 0..n-1 in insertion order).
    value_set = pa.array(list(encoder), type=table[source_column].type)
    codes = pc.index_in(table[source_column], value_set=value_set).cast(pa.int32())
    encoded = table.append_column(target_column, codes)
    return encoded.filter(pc.is_valid(encoded[target_column]))


def encode_splits(train: pa.Table, test: pa.Table) -> EncodedRatings:
    """Fit ID encoders on `train` and apply them to both splits."""
    train, user_encoder, user_decoder = encode_ids(train, "user_id", "user")
    train, anime_encoder, anime_decoder = encode_ids(train, "anime_id", "anime")
    encoded_test = apply_encoding(test, "user_id", "user", user_encoder)
    encoded_test = apply_encoding(encoded_test, "anime_id", "anime", anime_encoder)

    dropped = test.num_rows - encoded_test.num_rows
    if dropped:
        logger.info(f"Dropped {dropped} test rows whose anime never appears in train.")
    return EncodedRatings(
        train, encoded_test, user_encoder, user_decoder, anime_encoder, anime_decoder
    )


def _keep_frequent(table: pa.Table, column: str, min_count: int) -> pa.Table:
    """Keep only rows where the specified column has at least `min_count` occurrences.

    Args:
        table: The input PyArrow table.
        column: The column to check for frequency.
        min_count: The minimum number of occurrences required to keep a row.

    Returns:
        pa.Table: The filtered PyArrow table containing only rows with frequent values in the specified column.
    """
    counts = table.group_by(column).aggregate([([], "count_all")])
    frequent = counts.filter(pc.greater_equal(counts["count_all"], min_count))[column]
    return table.filter(pc.is_in(table[column], value_set=frequent.combine_chunks()))


def _holdout_mask(
    user_ids: np.ndarray,
    test_fraction: float,
) -> np.ndarray:
    """Mark the first floor(n * test_fraction) rows of each user as test.

    Rows are already shuffled, so "first" within a user is random. A stable
    sort groups each user's rows while keeping that shuffled order.

    Args:
        user_ids: Array of user IDs corresponding to each row.
        test_fraction: Fraction of each user's rows to mark as test.

    Returns:
        np.ndarray: Boolean mask indicating test rows.
    """
    order = np.argsort(user_ids, kind="stable")
    sorted_ids = user_ids[order]

    _, starts, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
    position = np.arange(len(sorted_ids)) - np.repeat(starts, counts)
    is_test_sorted = position < np.floor(np.repeat(counts, counts) * test_fraction)

    is_test = np.empty(len(user_ids), dtype=bool)
    is_test[order] = is_test_sorted
    return is_test
