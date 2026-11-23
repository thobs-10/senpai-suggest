"""Shape raw anime synopses into the pipeline's column names (see `SYNOPSIS_SCHEMA`).

Fetching, validation and retries are Prefect tasks in `run_ingestion_pipeline`.
"""

import pyarrow as pa

# Raw -> pipeline column names. The source export misspells "synopsis".
SYNOPSIS_RENAMES = {"MAL_ID": "anime_id", "sypnopsis": "synopsis"}


def rename_synopsis_columns(table: pa.Table) -> pa.Table:
    """Rename raw synopsis columns using `SYNOPSIS_RENAMES`; other columns are unchanged."""
    return table.rename_columns([SYNOPSIS_RENAMES.get(name, name) for name in table.column_names])
