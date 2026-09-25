"""Parent Prefect flow: the single entry point for the whole data pipeline.

Runs each stage as a subflow, in order, with one shared run ID so every
output of a run lands in folders with the same name:

    ingestion-pipeline      -> data/ingested/{run_id}/, data/reports/{run_id}/
    preprocessing-pipeline  -> data/processed/{run_id}/

A failed stage stops the run; later stages are not started. Each stage can
still be run on its own (see its module) to redo just that part.
Feature engineering and training join here as later subflows.

Usage:
    uv run python -m src.senpai_suggest.backend.pipelines.main
"""

from pathlib import Path

from prefect import flow

from src.senpai_suggest.backend.configs.backend_config import CONFIG_PATH
from src.senpai_suggest.backend.logger.logger import Logger
from src.senpai_suggest.backend.pipelines.run_ingestion_pipeline import run_ingestion_pipeline
from src.senpai_suggest.backend.pipelines.run_preprocessing_pipeline import (
    run_preprocessing_pipeline,
)
from src.senpai_suggest.backend.utils.main_utils import new_run_id

logger: Logger = Logger()


@flow(name="senpai-suggest-pipeline")
def run_pipeline(
    config_path: str = str(CONFIG_PATH),
    run_id: str | None = None,
) -> dict[str, dict[str, Path]]:
    """Run ingestion then preprocessing under one run ID.

    Args:
        config_path: Path to config.yaml, passed to every stage.
        run_id: Shared output folder name; a new UTC timestamp when omitted.

    Returns:
        Each stage's written files, keyed by stage name.
    """
    run_id = run_id or new_run_id()
    logger.info(f"Starting pipeline run {run_id}.")

    ingested: dict[str, Path] = run_ingestion_pipeline(config_path=config_path, run_id=run_id)
    processed: dict[str, Path] = run_preprocessing_pipeline(config_path=config_path, run_id=run_id)

    logger.info(f"Pipeline run {run_id} completed.")
    return {"ingestion": ingested, "preprocessing": processed}


if __name__ == "__main__":
    run_pipeline()
