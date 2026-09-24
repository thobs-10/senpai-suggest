"""Main Prefect flows orchestrating the data pipeline."""

from prefect import flow

from .steps.data_ingestion import (
    ingest_anime_data,
    ingest_user_profiles,
    ingest_user_ratings,
)


@flow(name="anime-data-pipeline")
def main_pipeline(
    anime_s3_path: str = "s3://anime-data/anime.csv",
    ratings_s3_path: str = "s3://anime-data/ratings.csv",
    profiles_s3_path: str = "s3://anime-data/profiles.csv",
) -> dict:
    """
    Main data pipeline orchestrating data ingestion and preprocessing.

    Args:
        anime_s3_path: S3 path to anime metadata CSV.
        ratings_s3_path: S3 path to user ratings CSV.
        profiles_s3_path: S3 path to user profiles CSV.

    Returns:
        Dictionary with ingested and processed data.
    """
    # TODO: Phase 3 - Implement parallel data ingestion
    anime_data = ingest_anime_data(anime_s3_path)
    ratings_data = ingest_user_ratings(ratings_s3_path)
    profiles_data = ingest_user_profiles(profiles_s3_path)

    return {
        "anime": anime_data,
        "ratings": ratings_data,
        "profiles": profiles_data,
    }


if __name__ == "__main__":
    main_pipeline()
