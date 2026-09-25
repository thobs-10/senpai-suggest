"""this is the configuration for data ingestion."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

# The checked-in pipeline configuration; flows take a `config_path` to override it.
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass(frozen=True)
class IngestionConfig:
    """Raw column names in the MyAnimeList anime.csv export."""

    raw_id_column: str = "MAL_ID"
    raw_name_column: str = "Name"
    raw_english_name_column: str = "English name"
    unknown_marker: str = "Unknown"


def get_anime_columns() -> list[str]:
    """Return the list of output columns for anime metadata."""
    ANIME_COLUMNS = [
        "anime_id",
        "eng_version",
        "Score",
        "Genres",
        "Episodes",
        "Type",
        "Premiered",
        "Members",
    ]
    return ANIME_COLUMNS


@dataclass(frozen=True)
class RatingsPreprocessingConfig:
    """Settings for ratings preprocessing; mirrors `preprocessing.ratings` in config.yaml.

    Attributes:
        unrated_value: Raw rating meaning "on the list, not scored" (0 in MAL).
        min_rating: Lowest real score, used for 0-1 normalisation.
        max_rating: Highest real score.
        min_user_interactions: Users with fewer interactions (rated or not) are dropped.
        min_anime_interactions: Anime with fewer interactions are dropped.
        test_fraction: Share of each user's interactions held out for testing.
        confidence_alpha: ALS confidence slope: 1 + alpha * rating for rated rows.
        unrated_confidence: ALS confidence for listed-but-unrated rows.
        seed: Seed for the per-user split.
    """

    unrated_value: int = 0
    min_rating: float = 1
    max_rating: float = 10
    min_user_interactions: int = 20
    min_anime_interactions: int = 10
    test_fraction: float = 0.2
    confidence_alpha: float = 0.4
    unrated_confidence: float = 1.0
    seed: int = 42

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RatingsPreprocessingConfig":
        """Build from the `preprocessing.ratings` config section; missing keys use defaults."""
        return cls(**values)


@dataclass(frozen=True)
class EncodedRatings:
    """Preprocessed train/test ratings plus the ID maps needed to decode model output.

    Both tables carry `user` and `anime` columns encoded with maps fitted on
    `train` only; test rows whose IDs never appear in train are dropped.
    """

    train: pa.Table
    test: pa.Table
    user_encoder: dict[Any, int]
    user_decoder: dict[int, Any]
    anime_encoder: dict[Any, int]
    anime_decoder: dict[int, Any]


def get_preprocessing_key_columns() -> dict[str, str]:
    """Return the key columns used during preprocessing."""
    return {
        "user_id": "user",
        "anime_id": "anime",
    }
