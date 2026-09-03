"""Main FastAPI application for anime recommendations."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage FastAPI application lifespan.

    On startup: Load pre-trained models into memory.
    On shutdown: Clean up resources.
    """
    # TODO: Phase 2 - Load pre-trained model from S3 or cache
    logger.info("FastAPI startup: Loading recommendation model...")

    yield

    logger.info("FastAPI shutdown: Cleaning up resources...")


app = FastAPI(
    title="Senpai Suggest API",
    description="Anime recommendation engine API",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for ALB monitoring."""
    return {"status": "healthy", "service": "senpai-suggest-api"}


# TODO: Phase 2 - Add recommendation endpoints
# @app.get("/recommendations/{user_id}")
# async def get_recommendations(user_id: str, limit: int = 10):
#     """Get anime recommendations for a user."""
#     pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
