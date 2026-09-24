"""
Configuration loader for senpai-suggest application.

Loads settings from environment variables and config files.
"""

import os
from typing import Optional

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Backend settings
    backend_host: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))
    environment: str = os.getenv("ENVIRONMENT", "development")

    # AWS settings
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_bucket: str = os.getenv("S3_BUCKET", "senpai-suggest-data")

    # Model settings
    model_name: str = os.getenv("MODEL_NAME", "als_recommender")
    model_version: str = os.getenv("MODEL_VERSION", "v1")
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "/tmp/models")

    # Logging settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
