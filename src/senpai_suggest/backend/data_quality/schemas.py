"""Pandera schemas: the contract each dataset must satisfy before moving on.

Keyed by dataset name in `SCHEMAS`, which `DataQualityValidator` uses by default.
"""

from pandera.pandas import Check, Column, DataFrameSchema

# Raw user ratings, as read from animelist.csv (user_id, anime_id, rating).
# `unique` + `nullable=False` replace separate duplicate/null checks.
RATINGS_SCHEMA = DataFrameSchema(
    {
        "user_id": Column(int, Check.ge(0), nullable=False),
        "anime_id": Column(int, Check.ge(0), nullable=False),
        "rating": Column(int, Check.in_range(0, 10), nullable=False),
    },
    unique=["user_id", "anime_id"],
    coerce=True,
    strict=False,
)

# Anime metadata after ingestion cleaning (renamed + filtered columns).
ANIME_SCHEMA = DataFrameSchema(
    {
        "anime_id": Column(int, Check.ge(0), nullable=False),
        "eng_version": Column(str, nullable=False),
        "Score": Column(float, Check.in_range(0.0, 10.0), nullable=True),
        "Genres": Column(str, nullable=True),
        "Episodes": Column(str, nullable=True),
        "Type": Column(str, nullable=True),
        "Premiered": Column(str, nullable=True),
        "Members": Column(int, Check.ge(0), nullable=True),
    },
    unique=["anime_id"],
    coerce=True,
    strict=False,
)

# Anime synopses after ingestion cleaning (renamed columns).
SYNOPSIS_SCHEMA = DataFrameSchema(
    {
        "anime_id": Column(int, Check.ge(0), nullable=False),
        "Name": Column(str, nullable=False),
        "Genres": Column(str, nullable=True),
        "synopsis": Column(str, nullable=True),
    },
    unique=["anime_id"],
    coerce=True,
    strict=False,
)

SCHEMAS: dict[str, DataFrameSchema] = {
    "ratings": RATINGS_SCHEMA,
    "anime": ANIME_SCHEMA,
    "synopsis": SYNOPSIS_SCHEMA,
}
