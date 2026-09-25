"""Small pyarrow helpers shared by the preprocessing step modules."""

import pyarrow as pa


def require_columns(table: pa.Table, *columns: str) -> None:
    """Raise if any of `columns` is missing from `table`.

    Raises:
        ValueError: Naming the first missing column.
    """
    for column in columns:
        if column not in table.column_names:
            raise ValueError(f"Table must contain '{column}' column.")


def replace_column(table: pa.Table, name: str, values: pa.ChunkedArray | pa.Array) -> pa.Table:
    """Return `table` with column `name` replaced by `values`, keeping its position."""
    return table.set_column(table.column_names.index(name), name, values)
