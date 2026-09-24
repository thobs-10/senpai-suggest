"""Unit tests for the pandera schemas' contracts."""

import pandas as pd
import pandera.pandas as pdr
import pytest

from src.senpai_suggest.backend.data_quality.schemas import (
    ANIME_SCHEMA,
    RATINGS_SCHEMA,
    SCHEMAS,
    SYNOPSIS_SCHEMA,
)


def test_schemas_registry_maps_dataset_names() -> None:
    """It should expose each schema under its dataset name."""
    assert SCHEMAS == {
        "ratings": RATINGS_SCHEMA,
        "anime": ANIME_SCHEMA,
        "synopsis": SYNOPSIS_SCHEMA,
    }


@pytest.mark.parametrize(
    "ratings",
    [
        pd.DataFrame({"user_id": [1, 1], "anime_id": [10, 10], "rating": [5, 6]}),  # duplicate
        pd.DataFrame({"user_id": [-1], "anime_id": [10], "rating": [5]}),  # negative id
        pd.DataFrame({"user_id": [1], "anime_id": [10], "rating": [11]}),  # out of range
    ],
)
def test_ratings_schema_rejects_invalid_rows(ratings: pd.DataFrame) -> None:
    """It should reject duplicate pairs, negative IDs, and ratings outside 0-10."""
    with pytest.raises(pdr.errors.SchemaErrors):
        RATINGS_SCHEMA.validate(ratings, lazy=True)


def test_anime_schema_coerces_score_and_allows_missing_optionals() -> None:
    """It should coerce Score strings to float and allow nulls in optional columns."""
    anime = pd.DataFrame(
        {
            "anime_id": [1],
            "eng_version": ["Cowboy Bebop"],
            "Score": ["8.78"],
            "Genres": [None],
            "Episodes": [None],
            "Type": [None],
            "Premiered": [None],
            "Members": [100],
        }
    )

    validated = ANIME_SCHEMA.validate(anime)

    assert validated["Score"].dtype == "float64"


def test_anime_schema_requires_display_name() -> None:
    """It should reject anime without an eng_version."""
    anime = pd.DataFrame({"anime_id": [1], "eng_version": [None], "Members": [1]})

    with pytest.raises(pdr.errors.SchemaErrors):
        ANIME_SCHEMA.validate(anime, lazy=True)


def test_synopsis_schema_rejects_duplicate_anime() -> None:
    """It should reject two synopses for the same anime."""
    synopses = pd.DataFrame(
        {"anime_id": [1, 1], "Name": ["A", "A"], "Genres": ["x", "x"], "synopsis": ["s", "t"]}
    )

    with pytest.raises(pdr.errors.SchemaErrors):
        SYNOPSIS_SCHEMA.validate(synopses, lazy=True)
