"""this is the configuration for data ingestion."""

from dataclasses import dataclass
from typing import Any

import pyarrow as pa


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
class EncodedRatings:
    """Preprocessed ratings plus the ID maps needed to decode model output."""

    table: pa.Table
    user_encoder: dict[Any, int]
    user_decoder: dict[int, Any]
    anime_encoder: dict[Any, int]
    anime_decoder: dict[int, Any]
