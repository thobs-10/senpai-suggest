"""Unit tests for ratings preprocessing."""

from collections.abc import Sequence

import pyarrow as pa
import pytest

from src.senpai_suggest.backend.configs.backend_config import (
    CONFIG_PATH,
    RatingsPreprocessingConfig,
)
from src.senpai_suggest.backend.pipelines.steps.preprocessing import (
    add_confidence,
    apply_encoding,
    dedupe_interactions,
    drop_invalid_rows,
    encode_ids,
    encode_splits,
    filter_sparse,
    flag_unrated,
    normalize_ratings,
    shuffle_rows,
    split_per_user,
)
from src.senpai_suggest.backend.utils.main_utils import read_yaml


def _ratings(rows: Sequence[tuple[int | None, int | None, int | None]]) -> pa.Table:
    user_ids, anime_ids, ratings = zip(*rows, strict=True) if rows else ((), (), ())
    return pa.table(
        {
            "user_id": pa.array(user_ids, type=pa.int64()),
            "anime_id": pa.array(anime_ids, type=pa.int64()),
            "rating": pa.array(ratings, type=pa.int64()),
        }
    )


def test_normalize_ratings_scales_rating_column(sample_ratings_table: pa.Table) -> None:
    """It should normalize ratings into the 0 to 1 range as float64."""
    normalized = normalize_ratings(sample_ratings_table, min_rating=0, max_rating=10)

    assert normalized.column("rating").to_pylist() == [0.8, 0.7, 0.9, 0.6, 0.8]
    assert normalized.schema.field("rating").type == pa.float64()


def test_normalize_ratings_rejects_missing_rating_column(sample_ratings_table: pa.Table) -> None:
    """It should reject tables without a rating column."""
    with pytest.raises(ValueError, match="Table must contain 'rating' column."):
        normalize_ratings(sample_ratings_table.drop(["rating"]), min_rating=0, max_rating=10)


def test_normalize_ratings_rejects_zero_scale(sample_ratings_table: pa.Table) -> None:
    """It should reject identical min and max rating values."""
    with pytest.raises(ValueError, match="max_rating and min_rating cannot be the same."):
        normalize_ratings(sample_ratings_table, min_rating=1, max_rating=1)


def test_shuffle_rows_preserves_rows(sample_ratings_table: pa.Table) -> None:
    """It should reorder rows without adding, dropping, or changing any."""
    shuffled = shuffle_rows(sample_ratings_table, seed=0)

    def key(row: dict[str, int]) -> tuple[int, int]:
        return (row["user_id"], row["anime_id"])

    assert sorted(shuffled.to_pylist(), key=key) == sorted(
        sample_ratings_table.to_pylist(), key=key
    )


def test_shuffle_rows_is_reproducible_with_same_seed(sample_ratings_table: pa.Table) -> None:
    """It should produce the same order for the same seed."""
    assert shuffle_rows(sample_ratings_table, seed=7).equals(
        shuffle_rows(sample_ratings_table, seed=7)
    )


def test_encode_ids_adds_contiguous_column_and_maps(sample_ratings_table: pa.Table) -> None:
    """It should number IDs by first appearance and return both maps."""
    encoded, encoder, decoder = encode_ids(sample_ratings_table, "anime_id", "anime")

    assert encoded.column_names == ["user_id", "anime_id", "rating", "anime"]
    assert encoded.column("anime").to_pylist() == [0, 1, 0, 2, 1]
    assert encoder == {101: 0, 102: 1, 103: 2}
    assert decoder == {0: 101, 1: 102, 2: 103}


def test_encode_ids_rejects_missing_column(sample_ratings_table: pa.Table) -> None:
    """It should raise when the source column is missing."""
    with pytest.raises(ValueError, match="Table must contain 'user_id' column."):
        encode_ids(sample_ratings_table.drop(["user_id"]), "user_id", "user")


def test_drop_invalid_rows_removes_null_ids_and_out_of_range_ratings() -> None:
    """It should drop null IDs and ratings outside 1-10, keeping unrated 0s."""
    table = _ratings(
        [(1, 10, 8), (None, 10, 8), (1, None, 8), (1, 11, 11), (1, 12, 0), (1, 13, -1)]
    )

    cleaned = drop_invalid_rows(table, unrated_value=0, min_rating=1, max_rating=10)

    assert cleaned.to_pylist() == [
        {"user_id": 1, "anime_id": 10, "rating": 8},
        {"user_id": 1, "anime_id": 12, "rating": 0},
    ]


def test_drop_invalid_rows_treats_null_rating_as_unrated() -> None:
    """It should keep a null rating as an unrated interaction."""
    cleaned = drop_invalid_rows(_ratings([(1, 10, None)]), 0, 1, 10)

    assert cleaned["rating"].to_pylist() == [0]


def test_dedupe_interactions_keeps_highest_rating_sorted_by_keys() -> None:
    """It should keep one row per pair, prefer a real score over 0, and sort by keys."""
    table = _ratings([(2, 10, 5), (1, 11, 0), (1, 11, 7), (1, 10, 3), (1, 10, 3)])

    assert dedupe_interactions(table).to_pylist() == [
        {"user_id": 1, "anime_id": 10, "rating": 3},
        {"user_id": 1, "anime_id": 11, "rating": 7},
        {"user_id": 2, "anime_id": 10, "rating": 5},
    ]


def test_flag_unrated_nulls_zero_ratings_and_adds_flag() -> None:
    """It should turn rating 0 into null with is_rated false."""
    flagged = flag_unrated(_ratings([(1, 10, 0), (1, 11, 9)]), unrated_value=0)

    assert flagged["rating"].to_pylist() == [None, 9]
    assert flagged["is_rated"].to_pylist() == [False, True]


def test_filter_sparse_repeats_until_both_cutoffs_hold() -> None:
    """It should drop a user left sparse after a sparse anime is removed."""
    # Anime 99 has one interaction; removing it leaves user 3 with one row.
    table = _ratings([(1, 10, 5), (1, 11, 5), (2, 10, 5), (2, 11, 5), (3, 10, 5), (3, 99, 5)])

    filtered = filter_sparse(table, min_user_interactions=2, min_anime_interactions=2)

    assert set(filtered["user_id"].to_pylist()) == {1, 2}
    assert set(filtered["anime_id"].to_pylist()) == {10, 11}


def test_filter_sparse_counts_unrated_interactions() -> None:
    """It should count unrated rows towards the user cutoff."""
    table = _ratings([(1, 10, 0), (1, 11, 0), (2, 10, 5), (2, 11, 5)])

    filtered = filter_sparse(table, min_user_interactions=2, min_anime_interactions=1)

    assert filtered.num_rows == 4


def test_add_confidence_weights_rated_rows_and_defaults_unrated() -> None:
    """It should compute 1 + alpha * rating, or the unrated confidence for null ratings."""
    table = flag_unrated(_ratings([(1, 10, 10), (1, 11, 0)]), unrated_value=0)

    confidence = add_confidence(table, alpha=0.4, unrated_confidence=1.0)["confidence"]

    assert confidence.to_pylist() == pytest.approx([5.0, 1.0])


def test_normalize_ratings_keeps_nulls() -> None:
    """It should leave unrated (null) ratings null."""
    table = flag_unrated(_ratings([(1, 10, 10), (1, 11, 0)]), unrated_value=0)

    assert normalize_ratings(table, 1, 10)["rating"].to_pylist() == [1.0, None]


def test_split_per_user_holds_out_fraction_of_each_user() -> None:
    """It should hold out floor(n * fraction) rows per user and keep the rest in train."""
    table = _ratings([(u, a, 5) for u in (1, 2) for a in range(10)] + [(3, 0, 5)])

    train, test = split_per_user(table, test_fraction=0.2, seed=0)

    test_counts = {u: test["user_id"].to_pylist().count(u) for u in (1, 2, 3)}
    assert test_counts == {1: 2, 2: 2, 3: 0}
    assert train.num_rows + test.num_rows == table.num_rows
    assert set(train["user_id"].to_pylist()) == {1, 2, 3}


def test_split_per_user_is_reproducible_and_disjoint() -> None:
    """It should give the same split for the same seed with no shared rows."""
    table = _ratings([(u, a, 5) for u in range(5) for a in range(10)])

    train_a, test_a = split_per_user(table, 0.3, seed=1)
    train_b, test_b = split_per_user(table, 0.3, seed=1)

    assert test_a.equals(test_b) and train_a.equals(train_b)
    pairs = {(r["user_id"], r["anime_id"]) for r in train_a.to_pylist()}
    assert not pairs & {(r["user_id"], r["anime_id"]) for r in test_a.to_pylist()}


@pytest.mark.parametrize("fraction", [-0.1, 1.0])
def test_split_per_user_rejects_invalid_fraction(fraction: float) -> None:
    """It should reject fractions outside [0, 1)."""
    with pytest.raises(ValueError, match="test_fraction"):
        split_per_user(_ratings([(1, 10, 5)]), fraction, seed=0)


def test_apply_encoding_drops_unseen_ids() -> None:
    """It should encode known IDs and drop rows with IDs missing from the encoder."""
    encoded = apply_encoding(_ratings([(1, 10, 5), (1, 99, 5)]), "anime_id", "anime", {10: 0})

    assert encoded["anime"].to_pylist() == [0]
    assert encoded["anime_id"].to_pylist() == [10]


def test_ratings_config_from_dict_uses_defaults_for_missing_keys() -> None:
    """It should build from a partial config section."""
    config = RatingsPreprocessingConfig.from_dict({"test_fraction": 0.1})

    assert config.test_fraction == 0.1
    assert config.min_user_interactions == 20


def test_ratings_config_matches_config_yaml() -> None:
    """It should accept every key in the checked-in preprocessing.ratings section."""
    config = read_yaml(str(CONFIG_PATH))

    ratings_config = RatingsPreprocessingConfig.from_dict(config["preprocessing"]["ratings"])

    assert ratings_config == RatingsPreprocessingConfig()


def test_encode_splits_fits_on_train_and_drops_unseen_test_anime() -> None:
    """It should encode test with train's maps and drop anime that only appear in test."""
    train = _ratings([(1, 10, 5), (2, 11, 5)])
    test = _ratings([(1, 11, 5), (2, 99, 5)])

    result = encode_splits(train, test)

    assert set(result.anime_encoder) == {10, 11}
    assert result.test.to_pylist() == [
        {"user_id": 1, "anime_id": 11, "rating": 5, "user": 0, "anime": 1}
    ]
