"""Data quality checks for pipeline tables, backed by pandera.

`DataQualityValidator` holds the schemas and exposes one method per dataset.
It never stores the tables it checks, so one instance can be shared safely
across pipeline steps (and threads).

Each schema is the contract for *processed* data and runs in two modes:
- report (`raise_on_fail=False`): used on raw data after ingestion. Failures
  are collected for the "before" report and the pipeline continues.
- gate (`raise_on_fail=True`): used on processed data. Any failure stops the
  pipeline.

Structural failures (a missing column) raise in both modes, because nothing
downstream can work without the column.

Usage:
    from src.senpai_suggest.backend.data_quality import DataQualityValidator

    validator = DataQualityValidator()
    result = validator.validate_ratings(raw_table, raise_on_fail=False)
    result.table, result.failure_cases, result.passed
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd
import pandera.pandas as pdr
import pyarrow as pa
from pandera.pandas import DataFrameSchema

from src.senpai_suggest.backend.data_quality.schemas import SCHEMAS
from src.senpai_suggest.backend.logger.logger import Logger

logger: Logger = Logger()

# pandera check names that mean the table's shape is broken, not its values.
STRUCTURAL_CHECKS = frozenset({"column_in_dataframe"})


class DataQualityError(RuntimeError):
    """Raised when a table fails schema validation."""


# eq=False: comparing DataFrames with == is element-wise, not a bool.
@dataclass(frozen=True, eq=False)
class ValidationResult:
    """Outcome of validating one table.

    Attributes:
        dataset: Name of the schema the table was checked against.
        table: The table after pandera's dtype coercion. In report mode with
            failures, columns that could not be coerced keep their raw type.
        failure_cases: pandera's failure cases (`column`, `check`,
            `failure_case`, `index`, ...); empty when the table passed.
    """

    dataset: str
    table: pa.Table
    failure_cases: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def passed(self) -> bool:
        """True when no check failed."""
        return bool(self.failure_cases.empty)


class DataQualityValidator:
    """Validate pipeline tables against their pandera schemas.

    Args:
        schemas: Dataset name -> schema. Defaults to `SCHEMAS`; pass your own
            to add datasets or to test with simpler schemas.
    """

    def __init__(self, schemas: Mapping[str, DataFrameSchema] = SCHEMAS) -> None:
        self.schemas = dict(schemas)

    def validate(
        self,
        table: pa.Table,
        dataset: str,
        raise_on_fail: bool = True,
    ) -> ValidationResult:
        """Validate `table` against the schema for `dataset`.

        Pipeline steps pass `pyarrow.Table`s, but pandera validates DataFrames,
        so the table goes through pandas here and nowhere else.

        Args:
            table: Table to check.
            dataset: Key of the schema in `self.schemas`.
            raise_on_fail: True for gate mode, False for report mode.

        Raises:
            KeyError: If no schema is registered for `dataset`.
            DataQualityError: In gate mode on any failure, and in both modes
                on a structural failure. The message lists the failure cases.
        """
        if dataset not in self.schemas:
            raise KeyError(f"No schema registered for dataset '{dataset}'.")

        try:
            validated = self.schemas[dataset].validate(table.to_pandas(), lazy=True)
        except pdr.errors.SchemaErrors as exc:
            return self._handle_failures(exc, dataset, raise_on_fail)

        logger.info(f"{dataset} data passed validation: {len(validated)} rows.")
        return ValidationResult(dataset, _to_table(validated))

    def validate_ratings(self, table: pa.Table, raise_on_fail: bool = True) -> ValidationResult:
        """Validate ratings (user_id, anime_id, rating)."""
        return self.validate(table, "ratings", raise_on_fail)

    def validate_anime(self, table: pa.Table, raise_on_fail: bool = True) -> ValidationResult:
        """Validate anime metadata."""
        return self.validate(table, "anime", raise_on_fail)

    def validate_synopsis(self, table: pa.Table, raise_on_fail: bool = True) -> ValidationResult:
        """Validate anime synopses."""
        return self.validate(table, "synopsis", raise_on_fail)

    @staticmethod
    def _handle_failures(
        exc: pdr.errors.SchemaErrors,
        dataset: str,
        raise_on_fail: bool,
    ) -> ValidationResult:
        """Raise in gate mode or on structural failures; otherwise return them as a result."""
        failures = exc.failure_cases
        structural = failures["check"].isin(STRUCTURAL_CHECKS).any()
        if raise_on_fail or structural:
            logger.error(f"{dataset} data failed validation ({len(failures)} failures).")
            raise DataQualityError(
                f"{dataset.capitalize()} data failed validation:\n{failures}"
            ) from exc

        logger.warning(f"{dataset} data has {len(failures)} quality failures (report mode).")
        # exc.data is the frame after coercion, so downstream steps still get typed columns.
        return ValidationResult(dataset, _to_table(exc.data), failures)


def _to_table(df: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(df, preserve_index=False)
