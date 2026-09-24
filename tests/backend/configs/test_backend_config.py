"""Unit tests for the backend column configuration."""

import dataclasses

import pytest

from src.senpai_suggest.backend.configs.backend_config import IngestionConfig, get_anime_columns
from src.senpai_suggest.backend.data_quality.schemas import ANIME_SCHEMA


def test_ingestion_config_defaults_match_raw_anime_export() -> None:
    """It should default to the MyAnimeList anime.csv column names."""
    config = IngestionConfig()

    assert config.raw_id_column == "MAL_ID"
    assert config.raw_name_column == "Name"
    assert config.raw_english_name_column == "English name"
    assert config.unknown_marker == "Unknown"


def test_ingestion_config_is_immutable() -> None:
    """It should be frozen so shared config can't be changed at runtime."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        IngestionConfig().raw_id_column = "id"


def test_get_anime_columns_matches_anime_schema() -> None:
    """It should list exactly the columns ANIME_SCHEMA validates, in output order."""
    assert get_anime_columns()[:2] == ["anime_id", "eng_version"]
    assert set(get_anime_columns()) == set(ANIME_SCHEMA.columns)


def test_get_anime_columns_returns_fresh_list() -> None:
    """It should return a new list each call so callers can't mutate shared state."""
    columns = get_anime_columns()
    columns.append("extra")

    assert "extra" not in get_anime_columns()
