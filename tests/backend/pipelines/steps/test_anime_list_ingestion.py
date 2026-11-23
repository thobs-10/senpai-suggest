"""Unit tests for anime metadata shaping."""

import pyarrow as pa

from src.senpai_suggest.backend.configs.backend_config import get_anime_columns
from src.senpai_suggest.backend.pipelines.steps import anime_list_ingestion


def test_clean_anime_metadata_resolves_names_and_unknowns(raw_anime_table: pa.Table) -> None:
    """It should rename IDs, fall back to the original name, and null out 'Unknown'."""
    cleaned = anime_list_ingestion.clean_anime_metadata(raw_anime_table)

    assert cleaned.column_names == get_anime_columns()
    assert cleaned["anime_id"].to_pylist() == [1, 5]
    assert cleaned["eng_version"].to_pylist() == ["Cowboy Bebop", "Tengen Toppa Gurren Lagann"]
    assert cleaned["Score"].to_pylist() == ["8.78", None]


def test_clean_anime_metadata_drops_unused_columns(raw_anime_table: pa.Table) -> None:
    """It should not carry raw-only columns like Studios through."""
    cleaned = anime_list_ingestion.clean_anime_metadata(raw_anime_table)

    assert "Studios" not in cleaned.column_names
    assert "English name" not in cleaned.column_names
