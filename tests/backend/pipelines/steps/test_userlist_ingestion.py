"""Unit tests for the UserListIngestion class."""

import pyarrow as pa
import pytest
from pytest_mock import MockerFixture

from src.senpai_suggest.backend.pipelines.steps.userlist_ingestion import (
    UserListIngestion,
)


def test_ingest_user_list_success(
    mocker: MockerFixture,
    config_dict: dict[str, dict[str, str]],
    sample_ratings_table: pa.Table,
) -> None:
    """It should fetch CSV data, convert it to a PyArrow table, and return it."""
    ingestion = UserListIngestion()

    mock_fetch = mocker.patch(
        "src.senpai_suggest.backend.pipelines.steps.userlist_ingestion.fetch_from_s3",
        return_value=sample_ratings_table,
    )

    table = ingestion.ingest_user_list(config_dict)

    mock_fetch.assert_called_once_with(
        raw_data_path=config_dict["ingestion"]["raw_ratings_path"],
        output_file=config_dict["ingestion"]["output_file_path"],
    )
    assert table is sample_ratings_table
    assert table.num_rows == 5
    assert table.column_names == ["user_id", "anime_id", "rating"]


def test_ingest_user_list_raises_when_fetch_fails(
    mocker: MockerFixture,
    config_dict: dict[str, dict[str, str]],
) -> None:
    """It should surface fetch errors as RuntimeError."""
    ingestion = UserListIngestion()

    mock_fetch = mocker.patch(
        "src.senpai_suggest.backend.pipelines.steps.userlist_ingestion.fetch_from_s3",
        side_effect=RuntimeError("fetch failed"),
    )

    with pytest.raises(RuntimeError, match="fetch failed"):
        ingestion.ingest_user_list(config_dict)

    mock_fetch.assert_called_once_with(
        raw_data_path=config_dict["ingestion"]["raw_ratings_path"],
        output_file=config_dict["ingestion"]["output_file_path"],
    )


def test_ingest_user_list_returns_table_with_correct_schema(
    mocker: MockerFixture,
    config_dict: dict[str, dict[str, str]],
    sample_ratings_table: pa.Table,
) -> None:
    """It should return a table with the correct schema."""
    ingestion = UserListIngestion()

    mocker.patch(
        "src.senpai_suggest.backend.pipelines.steps.userlist_ingestion.fetch_from_s3",
        return_value=sample_ratings_table,
    )

    table = ingestion.ingest_user_list(config_dict)

    assert table.schema.names == ["user_id", "anime_id", "rating"]
    assert table.schema.types == [pa.int64(), pa.int64(), pa.float64()]
    assert table.num_rows == 5
    assert table.to_pydict() == {
        "user_id": [1, 1, 2, 2, 3],
        "anime_id": [101, 102, 101, 103, 102],
        "rating": [8.0, 7.0, 9.0, 6.0, 8.5],
    }


def test_normalize_ratings_scales_rating_column(
    ingestion_instance: UserListIngestion,
    sample_ratings_table: pa.Table,
) -> None:
    """It should normalize ratings into the 0 to 1 range."""
    normalized = ingestion_instance.normalize_ratings(
        sample_ratings_table,
        min_rating=0,
        max_rating=10,
    )

    assert normalized.column("rating").to_pylist() == [0.8, 0.7, 0.9, 0.6, 0.85]
    assert normalized.schema.field("rating").type == pa.float64()


def test_normalize_ratings_rejects_missing_rating_column(
    ingestion_instance: UserListIngestion,
) -> None:
    """It should reject tables without a rating column."""
    table = pa.table(
        {
            "user_id": pa.array([1, 2], type=pa.int64()),
            "anime_id": pa.array([101, 102], type=pa.int64()),
        }
    )

    with pytest.raises(ValueError, match="Table must contain 'rating' column."):
        ingestion_instance.normalize_ratings(table, min_rating=0, max_rating=10)


def test_normalize_ratings_rejects_zero_scale(
    ingestion_instance: UserListIngestion,
    sample_ratings_table: pa.Table,
) -> None:
    """It should reject identical min and max rating values."""
    with pytest.raises(ValueError, match="max_rating and min_rating cannot be the same."):
        ingestion_instance.normalize_ratings(sample_ratings_table, min_rating=1, max_rating=1)


def test_check_duplicates_detects_duplicate_rows(
    ingestion_instance: UserListIngestion,
    ratings_table_with_duplicates: pa.Table,
) -> None:
    """It should detect duplicate rows in a table."""
    assert ingestion_instance.check_duplicates(ratings_table_with_duplicates) is True


def test_check_duplicates_returns_false_for_unique_rows(
    ingestion_instance: UserListIngestion,
    sample_ratings_table: pa.Table,
) -> None:
    """It should return False when rows are unique."""
    assert ingestion_instance.check_duplicates(sample_ratings_table) is False


def test_check_duplicates_returns_false_for_table_with_nulls(
    ingestion_instance: UserListIngestion,
    ratings_table_with_nulls: pa.Table,
) -> None:
    """It should return False when there are no duplicate rows, even if there are null values."""
    assert ingestion_instance.check_duplicates(ratings_table_with_nulls) is False


def test_check_duplicates_returns_false_for_empty_table(
    ingestion_instance: UserListIngestion,
) -> None:
    """It should return False when the table is empty."""
    empty_table = pa.table(
        {
            "user_id": pa.array([], type=pa.int64()),
            "anime_id": pa.array([], type=pa.int64()),
            "rating": pa.array([], type=pa.float64()),
        }
    )
    assert ingestion_instance.check_duplicates(empty_table) is False


def test_check_duplicates_returns_true_for_table_with_duplicates(
    ingestion_instance: UserListIngestion,
    ratings_table_with_duplicates: pa.Table,
) -> None:
    """It should return True when there are duplicate rows."""
    assert ingestion_instance.check_duplicates(ratings_table_with_duplicates) is True


def test_check_nulls_detects_null_values(
    ingestion_instance: UserListIngestion,
    ratings_table_with_nulls: pa.Table,
) -> None:
    """It should detect null values in the table."""
    assert ingestion_instance.check_nulls(ratings_table_with_nulls) is True


def test_check_nulls_returns_false_for_clean_table(
    ingestion_instance: UserListIngestion,
    sample_ratings_table: pa.Table,
) -> None:
    """It should return False when there are no null values."""
    assert ingestion_instance.check_nulls(sample_ratings_table) is False


def test_encode_users_adds_encoded_user_column(
    ingestion_instance: UserListIngestion,
    sample_ratings_table: pa.Table,
) -> None:
    """It should encode unique user IDs and append a user column."""
    encoded_table, user_map, user_reverse_map = ingestion_instance.encode_users(
        sample_ratings_table
    )

    assert encoded_table.column_names == ["user_id", "anime_id", "rating", "user"]
    assert encoded_table.column("user").to_pylist() == [0, 0, 1, 1, 2]
    assert user_map == {1: 0, 2: 1, 3: 2}
    assert user_reverse_map == {0: 1, 1: 2, 2: 3}


def test_encode_users_missing_user_id_column(
    ingestion_instance: UserListIngestion,
    sample_ratings_table: pa.Table,
) -> None:
    """It should raise an error when the 'user_id' column is missing."""
    table_without_user_id = sample_ratings_table.drop(["user_id"])

    with pytest.raises(ValueError, match="Table must contain 'user_id' column."):
        ingestion_instance.encode_users(table_without_user_id)


def test_encode_anime_missing_anime_id_column(
    ingestion_instance: UserListIngestion,
    sample_ratings_table: pa.Table,
) -> None:
    """It should raise an error when the 'anime_id' column is missing."""
    table_without_anime_id = sample_ratings_table.drop(["anime_id"])

    with pytest.raises(ValueError, match="Table must contain 'anime_id' column."):
        ingestion_instance.encode_anime(table_without_anime_id)


def test_encode_anime_adds_encoded_anime_column(
    ingestion_instance: UserListIngestion,
    sample_ratings_table: pa.Table,
) -> None:
    """It should encode unique anime IDs and append an anime column."""
    encoded_table, anime_map, anime_reverse_map = ingestion_instance.encode_anime(
        sample_ratings_table
    )

    assert encoded_table.column_names == ["user_id", "anime_id", "rating", "anime"]
    assert encoded_table.column("anime").to_pylist() == [0, 1, 0, 2, 1]
    assert anime_map == {101: 0, 102: 1, 103: 2}
    assert anime_reverse_map == {0: 101, 1: 102, 2: 103}


def test_sort_user_list_preserves_rows(
    ingestion_instance: UserListIngestion,
    sample_ratings_table: pa.Table,
) -> None:
    """It should shuffle the rows without changing the row set."""
    shuffled = ingestion_instance.sort_user_list(sample_ratings_table)

    assert shuffled.num_rows == sample_ratings_table.num_rows
    # assert set(shuffled.to_pydict().items()) == set(sample_ratings_table.to_pydict().items())
    assert sorted(
        shuffled.to_pylist(), key=lambda row: (row["user_id"], row["anime_id"], row["rating"])
    ) == sorted(
        sample_ratings_table.to_pylist(),
        key=lambda row: (row["user_id"], row["anime_id"], row["rating"]),
    )


def test_save_user_list_missing_table(
    ingestion_instance: UserListIngestion,
) -> None:
    """It should raise an error when the underlying saver receives no table."""
    with pytest.raises(ValueError, match="DataFrame must be provided."):
        ingestion_instance.save_user_list(None, "test-bucket")


def test_save_user_list_with_valid_table(
    mocker: MockerFixture,
    ingestion_instance: UserListIngestion,
    sample_ratings_table: pa.Table,
) -> None:
    """It should delegate to the shared save helper with the expected arguments."""
    save_mock = mocker.patch(
        "src.senpai_suggest.backend.pipelines.steps.userlist_ingestion.save_to_s3"
    )

    ingestion_instance.save_user_list(sample_ratings_table, "test-bucket")

    save_mock.assert_called_once_with(sample_ratings_table, "test-bucket")


def test_save_user_list_with_invalid_path(
    ingestion_instance: UserListIngestion,
    sample_ratings_table: pa.Table,
) -> None:
    """It should raise the bucket validation error from the shared saver."""
    with pytest.raises(ValueError, match="Bucket name must be provided."):
        ingestion_instance.save_user_list(sample_ratings_table, "")


def test_save_user_list_with_none_path(
    ingestion_instance: UserListIngestion,
    sample_ratings_table: pa.Table,
) -> None:
    """It should raise the bucket validation error from the shared saver."""
    with pytest.raises(ValueError, match="Bucket name must be provided."):
        ingestion_instance.save_user_list(sample_ratings_table, None)  # type: ignore[arg-type]


def test_run_userlist_ingestion_calls_steps_and_saves(
    mocker: MockerFixture,
    ingestion_instance: UserListIngestion,
    config_dict: dict[str, dict[str, str]],
    sample_ratings_table: pa.Table,
) -> None:
    """It should orchestrate the whole pipeline and save the final table."""
    mocker.patch.object(ingestion_instance, "ingest_user_list", return_value=sample_ratings_table)
    mocker.patch.object(ingestion_instance, "check_duplicates", return_value=False)
    mocker.patch.object(ingestion_instance, "check_nulls", return_value=False)
    mocker.patch.object(
        ingestion_instance,
        "normalize_ratings",
        side_effect=lambda table, min_rating, max_rating: table,
    )
    mocker.patch.object(
        ingestion_instance,
        "sort_user_list",
        side_effect=lambda table: table,
    )
    mocker.patch.object(
        ingestion_instance,
        "encode_users",
        side_effect=lambda table: (
            table.append_column("user", pa.array([0, 0, 1, 1, 2], type=pa.int32())),
            {1: 0, 2: 1, 3: 2},
            {0: 1, 1: 2, 2: 3},
        ),
    )
    mocker.patch.object(
        ingestion_instance,
        "encode_anime",
        side_effect=lambda table: (
            table.append_column("anime", pa.array([0, 1, 0, 2, 1], type=pa.int32())),
            {101: 0, 102: 1, 103: 2},
            {0: 101, 1: 102, 2: 103},
        ),
    )
    save_mock = mocker.patch.object(ingestion_instance, "save_user_list")

    final_table = ingestion_instance.run_userlist_ingestion(config_dict)

    save_mock.assert_called_once()
    assert final_table.column_names == ["user_id", "anime_id", "rating", "user", "anime"]
