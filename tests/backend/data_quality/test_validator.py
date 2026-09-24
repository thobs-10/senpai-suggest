"""Unit tests for DataQualityValidator."""

import pyarrow as pa
import pytest
from pandera.pandas import Check, Column, DataFrameSchema

from src.senpai_suggest.backend.data_quality import (
    DataQualityError,
    DataQualityValidator,
    ValidationResult,
)

RATINGS = pa.table({"user_id": [1, 2], "anime_id": [10, 10], "rating": [8, 9]})
BAD_RATING = pa.table({"user_id": [1, 2], "anime_id": [10, 10], "rating": [8, 11]})


def test_validate_ratings_returns_passing_result_with_coerced_table() -> None:
    """It should return a passing result whose table matches the input content."""
    result = DataQualityValidator().validate_ratings(RATINGS)

    assert isinstance(result, ValidationResult)
    assert result.dataset == "ratings"
    assert result.passed
    assert result.failure_cases.empty
    assert result.table.to_pydict() == RATINGS.to_pydict()


def test_gate_mode_raises_on_value_failure() -> None:
    """It should raise DataQualityError listing the failures by default (gate mode)."""
    with pytest.raises(DataQualityError, match="Ratings data failed validation"):
        DataQualityValidator().validate_ratings(BAD_RATING)


def test_report_mode_returns_failures_without_raising() -> None:
    """It should collect value failures and keep every row in report mode."""
    result = DataQualityValidator().validate_ratings(BAD_RATING, raise_on_fail=False)

    assert not result.passed
    assert result.table.num_rows == 2
    assert result.failure_cases["column"].tolist() == ["rating"]
    assert result.failure_cases["failure_case"].tolist() == [11]


def test_report_mode_still_coerces_dtypes() -> None:
    """It should return coerced columns even when other checks fail."""
    schema = DataFrameSchema({"score": Column(float, Check.ge(0))}, coerce=True)
    validator = DataQualityValidator(schemas={"scores": schema})

    result = validator.validate(pa.table({"score": [1, -1]}), "scores", raise_on_fail=False)

    assert not result.passed
    assert result.table.schema.field("score").type == pa.float64()


def test_report_mode_raises_on_missing_column() -> None:
    """It should treat a missing column as structural and raise even in report mode."""
    with pytest.raises(DataQualityError, match="column_in_dataframe"):
        DataQualityValidator().validate_ratings(RATINGS.drop(["rating"]), raise_on_fail=False)


def test_validate_rejects_unknown_dataset() -> None:
    """It should raise KeyError when no schema is registered for the dataset."""
    with pytest.raises(KeyError, match="No schema registered for dataset 'users'"):
        DataQualityValidator().validate(RATINGS, "users")


def test_validator_accepts_injected_schemas() -> None:
    """It should validate against custom schemas passed to the constructor."""
    schema = DataFrameSchema({"score": Column(int, Check.ge(0))})
    validator = DataQualityValidator(schemas={"scores": schema})

    assert validator.validate(pa.table({"score": [1, 2]}), "scores").table.num_rows == 2
    with pytest.raises(DataQualityError):
        validator.validate(pa.table({"score": [-1]}), "scores")
