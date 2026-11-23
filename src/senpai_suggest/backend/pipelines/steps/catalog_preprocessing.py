"""Steps that turn ingested anime metadata and synopses into one anime catalog.

Pure and Prefect-free, like `preprocessing.py`; `run_preprocessing_pipeline`
runs each one as its own task:

- anime:    drop duplicate IDs -> parse episodes -> parse premiered -> split genres
- synopsis: drop duplicate IDs -> clean synopsis text
- catalog:  anime LEFT JOIN synopsis -> flag rated anime -> build embedding text

The catalog keeps every anime, including ones nobody rated: the content
model can still recommend them, which is what cold start relies on.
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from src.senpai_suggest.backend.utils.table_utils import replace_column, require_columns

# "Spring 1998" -> season "Spring", year 1998. Anything else becomes null.
PREMIERED_PATTERN = r"^(?P<season>Spring|Summer|Fall|Winter) (?P<year>\d{4})$"
GENRE_SEPARATOR = ", "
# MyAnimeList's stand-in texts for anime without a real synopsis.
SYNOPSIS_PLACEHOLDER_PATTERN = r"^No synopsis (information )?has been added"
# Trailing attribution such as "(Source: ANN)" or "[Written by MAL Rewrite]".
SYNOPSIS_ATTRIBUTION_PATTERN = r"\s*[\(\[](?:Source|Written by)[^\)\]]*[\)\]]\s*$"


def drop_duplicate_ids(table: pa.Table, id_column: str = "anime_id") -> pa.Table:
    """Drop rows with a null ID and keep the first row of each repeated ID.

    Raises:
        ValueError: If `id_column` is missing.
    """
    require_columns(table, id_column)
    table = table.filter(pc.is_valid(table[id_column]))
    _, first_rows = np.unique(table[id_column].to_numpy(), return_index=True)
    return table.take(pa.array(np.sort(first_rows)))


def parse_episodes(table: pa.Table) -> pa.Table:
    """Turn the `Episodes` text into a nullable int; anything not a whole number becomes null.

    Raises:
        ValueError: If `Episodes` is missing.
    """
    require_columns(table, "Episodes")
    text = table["Episodes"].cast(pa.string())
    is_number = pc.match_substring_regex(text, r"^\d+$")
    episodes = pc.if_else(is_number, text, _null(pa.string())).cast(pa.int64())
    return replace_column(table, "Episodes", episodes)


def parse_premiered(table: pa.Table) -> pa.Table:
    """Replace `Premiered` ("Spring 1998") with `season` and `year` columns.

    Values that do not match (e.g. null or "Unknown") give null season and year.

    Raises:
        ValueError: If `Premiered` is missing.
    """
    require_columns(table, "Premiered")
    parts = pc.extract_regex(table["Premiered"].cast(pa.string()), PREMIERED_PATTERN)
    # Non-matching rows get empty strings in the fields, so mask them to null.
    matched = pc.is_valid(parts)
    season = pc.if_else(matched, pc.struct_field(parts, "season"), _null(pa.string()))
    year = pc.if_else(matched, pc.struct_field(parts, "year"), _null(pa.string()))
    table = table.drop(["Premiered"])
    return table.append_column("season", season).append_column("year", year.cast(pa.int64()))


def split_genres(table: pa.Table) -> pa.Table:
    """Turn the `Genres` text ("Action, Comedy") into a list of genre names.

    Raises:
        ValueError: If `Genres` is missing.
    """
    require_columns(table, "Genres")
    genres = pc.split_pattern(table["Genres"].cast(pa.string()), GENRE_SEPARATOR)
    return replace_column(table, "Genres", genres)


def clean_synopsis_text(table: pa.Table) -> pa.Table:
    """Strip attributions and extra whitespace; placeholder or empty synopses become null.

    Raises:
        ValueError: If `synopsis` is missing.
    """
    require_columns(table, "synopsis")
    text = pc.replace_substring_regex(
        table["synopsis"].cast(pa.string()), SYNOPSIS_ATTRIBUTION_PATTERN, ""
    )
    text = pc.utf8_trim_whitespace(pc.replace_substring_regex(text, r"\s+", " "))
    is_missing = pc.or_(
        pc.match_substring_regex(text, SYNOPSIS_PLACEHOLDER_PATTERN), pc.equal(text, "")
    )
    synopsis = pc.if_else(pc.fill_null(is_missing, True), _null(pa.string()), text)
    return replace_column(table, "synopsis", synopsis)


def join_catalog(anime: pa.Table, synopsis: pa.Table) -> pa.Table:
    """Left-join synopses onto anime by `anime_id`; anime without one get a null synopsis.

    Only `synopsis` is taken from the synopsis table: name and genres come
    from the anime metadata, which covers more titles. `synopsis` must have
    unique IDs (run `drop_duplicate_ids` first).

    Raises:
        ValueError: If `anime_id` is missing from either table, or `synopsis` from the second.
    """
    require_columns(anime, "anime_id")
    require_columns(synopsis, "anime_id", "synopsis")
    # A lookup instead of Table.join: Arrow's hash join cannot carry list
    # columns such as the split Genres. index_in gives null for no match,
    # and take() turns a null index into a null synopsis.
    rows = pc.index_in(anime["anime_id"], value_set=synopsis["anime_id"].combine_chunks())
    joined = anime.append_column("synopsis", synopsis["synopsis"].take(rows))
    return joined.sort_by("anime_id")


def flag_rated_anime(catalog: pa.Table, rated_anime_ids: pa.Array) -> pa.Table:
    """Add `has_ratings`: true for anime that appear in the ratings training split.

    Raises:
        ValueError: If `anime_id` is missing.
    """
    require_columns(catalog, "anime_id")
    has_ratings = pc.is_in(catalog["anime_id"], value_set=rated_anime_ids)
    return catalog.append_column("has_ratings", has_ratings)


def add_embedding_text(catalog: pa.Table) -> pa.Table:
    """Add `embedding_text` = "{name}. Genres: {genres}. {synopsis}", skipping missing parts.

    This is the text the sentence transformer embeds in feature engineering.

    Raises:
        ValueError: If `eng_version`, `Genres` or `synopsis` is missing.
    """
    require_columns(catalog, "eng_version", "Genres", "synopsis")
    genres = pc.binary_join(catalog["Genres"], GENRE_SEPARATOR)
    genres_part = pc.binary_join_element_wise("Genres: ", genres, "")  # null if no genres
    text = pc.binary_join_element_wise(
        catalog["eng_version"], genres_part, catalog["synopsis"], ". ", null_handling="skip"
    )
    return catalog.append_column("embedding_text", text)


def _null(type_: pa.DataType) -> pa.Scalar:
    return pa.scalar(None, type_)
