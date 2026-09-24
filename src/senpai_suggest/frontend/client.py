"""API client for communicating with the recommendation backend."""

import os
from typing import Any, Dict

import requests


class RecommenderClient:
    """Client for interacting with the FastAPI recommendation service."""

    def __init__(self, base_url: str | None = None, timeout: int = 10):
        """
        Initialize the recommendation client.

        Args:
            base_url: Base URL of the API. Defaults to BACKEND_URL env var.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://localhost:8000")
        self.timeout = timeout

    def get_recommendations(self, user_id: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get anime recommendations for a user.

        Args:
            user_id: The user's unique identifier.
            limit: Maximum number of recommendations to return.

        Returns:
            Dictionary containing recommendations.

        Raises:
            requests.RequestException: If the API call fails.
        """
        url = f"{self.base_url}/recommendations/{user_id}"
        params = {"limit": limit}

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.ConnectionError as e:
            raise requests.RequestException(
                f"Failed to connect to backend at {self.base_url}"
            ) from e
        except requests.Timeout as e:
            raise requests.RequestException("Backend request timed out") from e
        except requests.HTTPError as e:
            raise requests.RequestException(
                f"Backend error: {response.status_code} {response.text}"
            ) from e

    def health_check(self) -> bool:
        """
        Check if the backend is healthy.

        Returns:
            True if backend is healthy, False otherwise.
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            return response.status_code == 200
        except Exception:
            return False
