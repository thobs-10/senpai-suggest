"""Shape raw anime metadata into the columns the pipeline relies on.

Ingestion only reshapes (renames, picks columns, reads placeholders as null);
it never fixes values. Fetching, validation and retries are Prefect tasks in
`run_ingestion_pipeline`.
"""

import pyarrow as pa

from src.senpai_suggest.backend.configs.backend_config import IngestionConfig, get_anime_columns


def clean_anime_metadata(table: pa.Table) -> pa.Table:
    """Rename IDs, resolve display names, null out placeholders, and keep `ANIME_COLUMNS`.

    `eng_version` is the English title when there is one, otherwise the original name.

    Args:
        table: Raw anime table as read from anime.csv.

    Returns:
        Table with exactly `ANIME_COLUMNS`, in that order.
    """
    df = table.to_pandas()
    text_columns = df.select_dtypes("object").columns
    df[text_columns] = df[text_columns].mask(df[text_columns] == IngestionConfig.unknown_marker)
    df = df.rename(columns={IngestionConfig.raw_id_column: "anime_id"})
    df["eng_version"] = df[IngestionConfig.raw_english_name_column].fillna(
        df[IngestionConfig.raw_name_column]
    )
    return pa.Table.from_pandas(df[get_anime_columns()], preserve_index=False)
