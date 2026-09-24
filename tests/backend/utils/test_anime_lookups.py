"""Unit tests for anime lookup helpers."""

import pyarrow as pa
import pytest

from src.senpai_suggest.backend.utils.anime_lookups import (
    get_anime,
    get_anime_name,
    get_synopsis,
)

ANIME = pa.table({"anime_id": [1, 5], "eng_version": ["Cowboy Bebop", "Gurren Lagann"]})
SYNOPSES = pa.table(
    {"anime_id": [1], "Name": ["Cowboy Bebop"], "synopsis": ["Space bounty hunters."]}
)


def test_get_anime_by_id_and_title() -> None:
    """It should find the same row by ID or by English title."""
    assert get_anime(ANIME, 5).equals(get_anime(ANIME, "Gurren Lagann"))
    assert get_anime(ANIME, 5).num_rows == 1


def test_get_anime_name_returns_none_when_missing() -> None:
    """It should return the display name, or None for unknown IDs."""
    assert get_anime_name(ANIME, 1) == "Cowboy Bebop"
    assert get_anime_name(ANIME, 999) is None


def test_get_synopsis_by_id_or_name() -> None:
    """It should return the synopsis by ID or original name, or None if missing."""
    assert get_synopsis(SYNOPSES, 1) == "Space bounty hunters."
    assert get_synopsis(SYNOPSES, "Cowboy Bebop") == "Space bounty hunters."
    assert get_synopsis(SYNOPSES, 5) is None


def test_lookups_reject_unsupported_key_types() -> None:
    """It should reject keys that are neither int nor str."""
    with pytest.raises(TypeError):
        get_anime(ANIME, 1.0)  # type: ignore[arg-type]
