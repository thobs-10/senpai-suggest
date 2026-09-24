"""Look up anime details and synopses by ID or title.

Used when presenting recommendations (API responses, notebooks): the model
returns anime IDs, and these helpers turn them into names and synopses.
They expect tables produced by the ingestion steps (`anime_id`, `eng_version`
for metadata; `anime_id`, `Name`, `synopsis` for synopses).
"""

import pyarrow as pa
import pyarrow.compute as pc


def _filter_by_id_or_title(table: pa.Table, anime: int | str, title_column: str) -> pa.Table:
    """Return rows whose `anime_id` equals an int, or whose `title_column` equals a str."""
    if isinstance(anime, bool) or not isinstance(anime, (int, str)):
        raise TypeError(f"anime must be an int (ID) or str (title), got {type(anime).__name__}.")
    column = "anime_id" if isinstance(anime, int) else title_column
    return table.filter(pc.equal(table[column], anime))  # type: ignore[attr-defined, unused-ignore]  # no pyarrow stubs


def get_anime(anime_table: pa.Table, anime: int | str) -> pa.Table:
    """Get the metadata row(s) for an anime by ID or English title (`eng_version`)."""
    return _filter_by_id_or_title(anime_table, anime, "eng_version")


def get_anime_name(anime_table: pa.Table, anime_id: int) -> str | None:
    """Get the display name (English title, else original) for an anime ID, or None."""
    names = get_anime(anime_table, anime_id)["eng_version"].to_pylist()
    return names[0] if names else None


def get_synopsis(synopsis_table: pa.Table, anime: int | str) -> str | None:
    """Get the synopsis for an anime by ID or original title (`Name`), or None."""
    synopses = _filter_by_id_or_title(synopsis_table, anime, "Name")["synopsis"].to_pylist()
    return synopses[0] if synopses else None
