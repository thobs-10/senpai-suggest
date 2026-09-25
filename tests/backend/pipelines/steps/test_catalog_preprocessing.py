"""Unit tests for anime, synopsis and catalog preprocessing."""

import pyarrow as pa
import pytest

from src.senpai_suggest.backend.pipelines.steps import catalog_preprocessing as catalog


def _anime(**columns: list[object]) -> pa.Table:
    defaults: dict[str, list[object]] = {
        "anime_id": [1, 5],
        "eng_version": ["Cowboy Bebop", "Gurren Lagann"],
        "Genres": ["Action, Sci-Fi", "Action, Mecha"],
        "Episodes": ["26", None],
        "Premiered": ["Spring 1998", None],
    }
    return pa.table({**defaults, **columns})


def _synopsis(texts: list[str | None], ids: list[int] | None = None) -> pa.table:
    ids = ids or list(range(1, len(texts) + 1))
    return pa.table({"anime_id": ids, "synopsis": pa.array(texts, type=pa.string())})


def test_drop_duplicate_ids_keeps_first_row_and_drops_null_ids() -> None:
    """It should keep the first row per ID, in input order, and drop null IDs."""
    table = pa.table({"anime_id": [5, 1, 5, None], "name": ["a", "b", "c", "d"]})

    assert catalog.drop_duplicate_ids(table).to_pydict() == {"anime_id": [5, 1], "name": ["a", "b"]}


def test_parse_episodes_turns_non_numbers_into_null() -> None:
    """It should parse whole numbers and null out everything else."""
    parsed = catalog.parse_episodes(pa.table({"Episodes": ["26", "Unknown", "12a", None, "1"]}))

    assert parsed["Episodes"].to_pylist() == [26, None, None, None, 1]
    assert parsed.schema.field("Episodes").type == pa.int64()


def test_parse_premiered_splits_season_and_year() -> None:
    """It should replace Premiered with season and year, null when it does not match."""
    parsed = catalog.parse_premiered(pa.table({"Premiered": ["Spring 1998", "Unknown", None]}))

    assert "Premiered" not in parsed.column_names
    assert parsed["season"].to_pylist() == ["Spring", None, None]
    assert parsed["year"].to_pylist() == [1998, None, None]


def test_split_genres_makes_lists_and_keeps_nulls() -> None:
    """It should split the comma-separated genres into lists."""
    parsed = catalog.split_genres(_anime(Genres=["Action, Sci-Fi", None]))

    assert parsed["Genres"].to_pylist() == [["Action", "Sci-Fi"], None]


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("Space  bounty\n hunters. (Source: ANN)", "Space bounty hunters."),
        ("Drill into the heavens. [Written by MAL Rewrite]", "Drill into the heavens."),
        ("  Plain text.  ", "Plain text."),
        ("No synopsis information has been added to this title. Help improve ...", None),
        ("No synopsis has been added for this series yet. Click here ...", None),
        ("   ", None),
        (None, None),
    ],
)
def test_clean_synopsis_text(raw: str | None, cleaned: str | None) -> None:
    """It should strip attributions and whitespace, and null out placeholders."""
    result = catalog.clean_synopsis_text(_synopsis([raw]))

    assert result["synopsis"].to_pylist() == [cleaned]


def test_join_catalog_keeps_every_anime_sorted_by_id() -> None:
    """It should left-join synopses, leaving null for anime without one."""
    # Split genres: list columns must survive the join.
    anime = catalog.split_genres(_anime(anime_id=[5, 1]))
    synopsis = _synopsis(["Space bounty hunters.", "Orphan synopsis."], ids=[1, 99])

    joined = catalog.join_catalog(anime, synopsis)

    assert joined["anime_id"].to_pylist() == [1, 5]
    assert joined["synopsis"].to_pylist() == ["Space bounty hunters.", None]
    assert joined["Genres"].to_pylist() == [["Action", "Mecha"], ["Action", "Sci-Fi"]]


def test_flag_rated_anime_marks_anime_in_training_ratings() -> None:
    """It should flag anime that appear in the ratings."""
    flagged = catalog.flag_rated_anime(_anime(), pa.array([5, 42]))

    assert flagged["has_ratings"].to_pylist() == [False, True]


def test_add_embedding_text_joins_available_parts() -> None:
    """It should join name, genres and synopsis, skipping the missing ones."""
    table = catalog.split_genres(_anime(anime_id=[1, 2], Genres=["Action, Sci-Fi", None]))
    table = table.append_column("synopsis", pa.array(["Bounty hunters.", None]))

    texts = catalog.add_embedding_text(table)["embedding_text"].to_pylist()

    assert texts == [
        "Cowboy Bebop. Genres: Action, Sci-Fi. Bounty hunters.",
        "Gurren Lagann",
    ]


@pytest.mark.parametrize(
    ("step", "column"),
    [
        (catalog.parse_episodes, "Episodes"),
        (catalog.parse_premiered, "Premiered"),
        (catalog.split_genres, "Genres"),
        (catalog.clean_synopsis_text, "synopsis"),
    ],
)
def test_steps_reject_missing_column(step: object, column: str) -> None:
    """It should name the missing column."""
    with pytest.raises(ValueError, match=f"'{column}'"):
        step(pa.table({"anime_id": [1]}))  # type: ignore[operator]
