"""
Data ingestion tasks for loading data from AWS S3.

All tasks are logged and orchestrated using Prefect flows.
"""

from typing import Tuple
from prefect import task
import pandas as pd
import numpy as np

from src.senpai_suggest.backend.utils.main_utils import fetch_from_s3


class UserListIngestion:
    """Class for handling user list ingestion tasks."""

    def __init__(self):
        self.user_ratings_data = None

    def ingest_user_list(self, s3_path: str) -> dict:
        """
        Load user list from S3.

        Args:
            s3_path: S3 path to the user list CSV file.

        Returns:
            Dictionary containing user list data.

        Raises:
            Exception: If S3 access or data parsing fails.
        """
        self.user_ratings_data = fetch_from_s3(
            bucket_name="your-bucket-name",
            key="your-key",
            output_file="your-output-file",
        )
        return self.user_ratings_data

    ## normalize user ratings
    def normalize_ratings(
        self,
        rating_df: pd.DataFrame,
        min_rating: float,
        max_rating: float,
    ) -> pd.DataFrame:
        """
        Normalize the ratings in the dataframe to a range of 0 to 1.

        Parameters:
        rating_df (pd.DataFrame): The input dataframe containing user ratings.
        min_rating (float): The minimum rating value.
        max_rating (float): The maximum rating value.

        Returns:
        pd.DataFrame: The dataframe with normalized ratings.
        """
        if "rating" not in rating_df.columns:
            raise ValueError("DataFrame must contain 'rating' column.")

        scale = max_rating - min_rating
        if scale == 0:
            raise ValueError("max_rating and min_rating cannot be the same.")

        rating_df["rating"] = ((rating_df["rating"] - min_rating) / scale).astype(np.float64)
        return rating_df

    def check_duplicates(self, rating_df: pd.DataFrame) -> bool:
        """
        Check for duplicate rows in the dataframe.

        Args:
            rating_df: DataFrame to check for duplicates.

        Returns:
            True if duplicates are found, False otherwise.

        """
        return rating_df.duplicated().any()

    def check_nulls(
        self,
        rating_df: pd.DataFrame,
    ) -> bool:
        """
        Check for null values in the dataframe.

        Parameters:
        rating_df (pd.DataFrame): The input dataframe to check for null values.

        Returns:
        bool: True if null values are found, False otherwise.
        """
        return rating_df.isnull().values.any()

    def encode_users(
        self,
        rating_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, dict, dict]:
        """
        Encode user IDs in the dataframe to a continuous range of integers.

        Parameters:
        rating_df (pd.DataFrame): The input dataframe containing user ratings.

        Returns:
        Tuple[pd.DataFrame, dict, dict]: A tuple containing the dataframe with encoded user IDs,
                                        a dictionary mapping original user IDs to encoded IDs,
                                        and a dictionary mapping encoded IDs back to original user IDs.
        """
        if "user_id" not in rating_df.columns:
            raise ValueError("DataFrame must contain 'user_id' column.")

        user_ids = rating_df["user_id"].unique().tolist()
        user2user_encoded = {user_id: i for i, user_id in enumerate(user_ids)}
        user2user_decoded = {i: user_id for i, user_id in enumerate(user_ids)}
        # create a new column in the dataframe with the encoded user IDs(from userid = 123456 to user = 0)
        rating_df["user"] = rating_df["user_id"].map(user2user_encoded)

        return rating_df, user2user_encoded, user2user_decoded

    def encode_anime() -> None:
        # TODO: Implement anime ID encoding similar to user ID encoding.
        raise NotImplementedError("encode_anime method is not implemented yet.")

    def sort_user_list() -> None:
        # TODO: Implement sorting of the user list based on specific criteria.
        raise NotImplementedError("sort_user_list method is not implemented yet.")

    def save_user_list() -> None:
        # TODO: Implement saving of the user list to a persistent storage.
        raise NotImplementedError("save_user_list method is not implemented yet.")


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
