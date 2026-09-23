"""Run all ingestion pipelines for the Senpai Suggest backend.

The three ingestions are independent, so they run concurrently (they are
I/O-bound on S3, so threads are enough). This runner is a stand-in until the
steps are wired into a Prefect flow.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa

from src.senpai_suggest.backend.logger.logger import Logger
from src.senpai_suggest.backend.pipelines.steps.anime_list_ingestion import (
    run_anime_list_ingestion,
)
from src.senpai_suggest.backend.pipelines.steps.synopsis_ingestion import run_synopsis_ingestion
from src.senpai_suggest.backend.pipelines.steps.userlist_ingestion import run_userlist_ingestion
from src.senpai_suggest.backend.utils.main_utils import read_yaml

logger: Logger = Logger()

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"

INGESTIONS: dict[str, Callable[[dict[str, Any]], pa.Table]] = {
    "anime": run_anime_list_ingestion,
    "synopsis": run_synopsis_ingestion,
    "ratings": run_userlist_ingestion,
}


def run_all_ingestion_pipelines(ingestion_config: dict[str, Any]) -> dict[str, pa.Table]:
    """Run every ingestion concurrently and return the tables by dataset name.

    Raises:
        Exception: The first ingestion error, once all ingestions have finished.
    """
    logger.info("Starting all ingestion pipelines.")
    with ThreadPoolExecutor(max_workers=len(INGESTIONS)) as pool:
        futures = {name: pool.submit(run, ingestion_config) for name, run in INGESTIONS.items()}
    # .result() re-raises any exception from a worker thread instead of losing it.
    results = {name: future.result() for name, future in futures.items()}
    logger.info("All ingestion pipelines have completed.")
    return results


def main() -> None:
    """Load config.yaml and run all ingestions."""
    config = read_yaml(str(CONFIG_PATH))
    run_all_ingestion_pipelines(config["ingestion"])


if __name__ == "__main__":
    main()
