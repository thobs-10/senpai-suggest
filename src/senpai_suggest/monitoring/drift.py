"""Data and prediction drift reports, backed by Evidently.

Run `generate_drift_report` as a pipeline step (or a small scheduled Prefect
flow) after each retraining ingestion, comparing the freshly ingested batch
against a stored reference sample. The HTML report is an artifact you can
upload to S3 next to the model version it corresponds to; `drift_score`
is a single number cheap enough to also push to Prometheus later via a
Pushgateway, if you want it on a dashboard instead of read as a file.

This intentionally stays a batch report generator rather than a live
service: Evidently's realtime monitoring stack is more than a learning
project on a t2.micro needs.
"""

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path,
) -> float:
    """Compare a current data batch against a reference batch for drift.

    Args:
        reference: A known-good sample (e.g. the training set, or last
            week's ingested batch) with the same columns as `current`.
        current: The batch to check, such as today's ingested ratings.
        output_path: Where to write the HTML report.

    Returns:
        The share of columns flagged as drifted, in [0, 1].

    Raises:
        ValueError: If `reference` and `current` have different columns.
    """
    if set(reference.columns) != set(current.columns):
        raise ValueError("reference and current must share the same columns.")

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(output_path))

    summary = result.dict()
    return float(summary["metrics"][0]["value"]["share"])
