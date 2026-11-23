"""Unit tests for synopsis shaping."""

import pyarrow as pa

from src.senpai_suggest.backend.pipelines.steps import synopsis_ingestion


def test_rename_synopsis_columns_maps_known_and_keeps_others() -> None:
    """It should rename MAL_ID/sypnopsis and leave other columns alone."""
    table = pa.table({"MAL_ID": [1], "Name": ["A"], "sypnopsis": ["s"]})

    renamed = synopsis_ingestion.rename_synopsis_columns(table)

    assert renamed.column_names == ["anime_id", "Name", "synopsis"]
