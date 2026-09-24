"""Unit tests for ratings preprocessing."""

import pyarrow as pa
import pytest

from src.senpai_suggest.backend.pipelines.steps.preprocessing import (
    encode_ids,
    normalize_ratings,
    run_ratings_preprocessing,
    shuffle_rows,
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


def test_run_ratings_preprocessing_returns_encoded_table_and_maps(
    sample_ratings_table: pa.Table,
) -> None:
    """It should normalize, shuffle, and encode, returning maps that decode the table."""
    result = run_ratings_preprocessing(sample_ratings_table, seed=0)

    assert result.table.column_names == ["user_id", "anime_id", "rating", "user", "anime"]
    assert result.table.num_rows == sample_ratings_table.num_rows
    assert set(result.user_encoder) == {1, 2, 3}
    assert set(result.anime_encoder) == {101, 102, 103}
    for row in result.table.to_pylist():
        assert result.user_decoder[row["user"]] == row["user_id"]
        assert result.anime_decoder[row["anime"]] == row["anime_id"]
