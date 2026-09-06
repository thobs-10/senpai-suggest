"""Data ingestion tasks for loading data from AWS S3."""

from typing import Any, Dict, Tuple

import pyarrow as pa
from dotenv import load_dotenv

from src.senpai_suggest.backend.logger.logger import Logger
from src.senpai_suggest.backend.utils.main_utils import fetch_from_s3, save_to_s3

logger: Logger = Logger()

load_dotenv()


class UserListIngestion:
    """Class for handling user list ingestion tasks."""

    def __init__(self) -> None:
        self.user_ratings_data: pa.Table | None = None

    def ingest_user_list(self, config: Dict[str, Dict[str, str]]) -> pa.Table:
        """
        Load user list from S3 and convert the CSV to a PyArrow table.

        Args:
            config: Dictionary containing ingestion configuration.

        Returns:
            PyArrow Table containing user list data.

        Raises:
            RuntimeError: If S3 access or data parsing fails.
        """
        self.user_ratings_data = fetch_from_s3(
            config["ingestion"]["raw_ratings_path"],
            config["ingestion"]["output_file_path"],
        )
        return self.user_ratings_data

    def normalize_ratings(
        self,
        rating_table: pa.Table,
        min_rating: float,
        max_rating: float,
    ) -> pa.Table:
        """
        Normalize the ratings in the table to a range of 0 to 1.

        Args:
            rating_table: PyArrow Table containing user ratings.
            min_rating: The minimum rating value.
            max_rating: The maximum rating value.

        Returns:
            PyArrow Table with normalized ratings.
        """
        if "rating" not in rating_table.column_names:
            raise ValueError("Table must contain 'rating' column.")

        scale = max_rating - min_rating
        if scale == 0:
            raise ValueError("max_rating and min_rating cannot be the same.")

        rating_col = rating_table["rating"].combine_chunks().cast(pa.float64())
        normalized = ((rating_col - min_rating) / scale).cast(pa.float64())
        idx = rating_table.column_names.index("rating")
        return rating_table.set_column(idx, "rating", normalized)

    def check_duplicates(self, rating_table: pa.Table) -> bool:
        """Check for duplicate rows in the table."""
        if len(rating_table) == 0:
            return False

        unique_rows = rating_table.group_by(rating_table.column_names).aggregate([])
        return len(unique_rows) != len(rating_table)

    def check_nulls(self, rating_table: pa.Table) -> bool:
        """Check for null values in the table."""
        return any(
            rating_table[column_name].null_count > 0 for column_name in rating_table.column_names
        )

    def encode_users(
        self,
        rating_table: pa.Table,
    ) -> Tuple[pa.Table, Dict[Any, int], Dict[int, Any]]:
        """
        Encode user IDs in the table to a continuous range of integers.

        Args:
            rating_table: The input table containing user ratings.

        Returns:
            Tuple containing the table with encoded user IDs, a mapping from original
            user IDs to encoded IDs, and the reverse mapping.
        """
        if "user_id" not in rating_table.column_names:
            raise ValueError("Table must contain 'user_id' column.")

        user_ids = rating_table["user_id"].combine_chunks().to_pylist()
        user2user_encoded = {user_id: i for i, user_id in enumerate(dict.fromkeys(user_ids))}
        user2user_decoded = {i: user_id for user_id, i in user2user_encoded.items()}
        encoded_user = pa.array(
            [user2user_encoded[user_id] for user_id in user_ids], type=pa.int32()
        )
        return (
            rating_table.append_column("user", encoded_user),
            user2user_encoded,
            user2user_decoded,
        )

    def encode_anime(
        self,
        rating_table: pa.Table,
    ) -> Tuple[pa.Table, Dict[Any, int], Dict[int, Any]]:
        """
        Encode anime IDs in the table to a continuous range of integers.

        Args:
            rating_table: The input table containing anime ratings.

        Returns:
            Tuple containing the table with encoded anime IDs, a mapping from original
            anime IDs to encoded IDs, and the reverse mapping.
        """
        if "anime_id" not in rating_table.column_names:
            raise ValueError("Table must contain 'anime_id' column.")

        anime_ids = rating_table["anime_id"].combine_chunks().to_pylist()
        anime2anime_encoded = {anime_id: i for i, anime_id in enumerate(dict.fromkeys(anime_ids))}
        anime2anime_decoded = {i: anime_id for anime_id, i in anime2anime_encoded.items()}
        encoded_anime = pa.array(
            [anime2anime_encoded[anime_id] for anime_id in anime_ids], type=pa.int32()
        )
        return (
            rating_table.append_column("anime", encoded_anime),
            anime2anime_encoded,
            anime2anime_decoded,
        )

    def sort_user_list(self, rating_table: pa.Table) -> pa.Table:
        """Shuffle the table rows."""
        import random

        indices = list(range(len(rating_table)))
        random.shuffle(indices)
        return rating_table.take(pa.array(indices, type=pa.int64()))

    def save_user_list(self, ratings_table: pa.Table, bucket_name: str) -> None:
        """Save the table to S3 as Parquet via the shared utility."""
        save_to_s3(ratings_table, bucket_name)

    def run_userlist_ingestion(self, config: Dict[str, Dict[str, str]]) -> pa.Table:
        """
        Run the complete user list ingestion pipeline.

        Args:
            config: Dictionary containing ingestion and AWS configuration.

        Returns:
            The processed PyArrow table after normalization, shuffling, and encoding.

        Raises:
            RuntimeError: If any of the fetch or save steps fail.
        """
        logger.info("Starting user list ingestion process.")

        ratings_table = self.ingest_user_list(config)
        self.check_duplicates(ratings_table)
        self.check_nulls(ratings_table)

        ratings_table = self.normalize_ratings(ratings_table, min_rating=0, max_rating=10)
        ratings_table = self.sort_user_list(ratings_table)

        ratings_table, user2user_encoded, user2user_decoded = self.encode_users(ratings_table)
        ratings_table, anime2anime_encoded, anime2anime_decoded = self.encode_anime(ratings_table)

        logger.info(f"Encoded {len(user2user_encoded)} users and {len(anime2anime_encoded)} anime")

        self.save_user_list(ratings_table, config["ingestion"]["user_filename"])

        logger.info("User list ingestion process completed.")
        return ratings_table


# @task(retries=3, retry_delay_seconds=60)
# def ingest_anime_data(s3_path: str) -> dict:
#     """
#     Load anime metadata from S3.

#     Args:
#         s3_path: S3 path to the anime data CSV file.

#     Returns:
#         Dictionary containing anime metadata.

#     Raises:
#         Exception: If S3 access or data parsing fails.
#     """
#     # Implement anime data ingestion from S3
#     return fetch_data_from_s3(s3_path)


# @task(retries=3, retry_delay_seconds=60)
# def ingest_user_ratings(s3_path: str) -> dict:
#     """
#     Load user ratings from S3.

#     Args:
#         s3_path: S3 path to the user ratings CSV file.

#     Returns:
#         Dictionary containing user ratings data.

#     Raises:
#         Exception: If S3 access or data parsing fails.
#     """
#     # TODO: Implement user ratings ingestion from S3
#     pass


# @task(retries=3, retry_delay_seconds=60)
# def ingest_user_profiles(s3_path: str) -> dict:
#     """
#     Load user profiles from S3.

#     Args:
#         s3_path: S3 path to the user profiles CSV file.

#     Returns:
#         Dictionary containing user profile data.

#     Raises:
#         Exception: If S3 access or data parsing fails.
#     """
#     # TODO: Implement user profiles ingestion from S3
#     pass
