"""Turn validation results into lean data quality reports.

For each dataset and stage (`raw` after ingestion, `processed` after
preprocessing) this writes:
- `{dataset}_{stage}.html`: human-readable report, with a before -> after
  comparison when the raw summary is passed in.
- `{dataset}_{stage}.json`: the same summary, machine-readable.
Plus one `summary.json` per run, and a Prefect table artifact when called
inside a flow run.

Only pandas and the standard library are used, so no extra dependencies.
Uploading the report directory to S3 is left to the pipeline flow.

Usage:
    raw = summarize(raw_result, "raw")
    write_report(raw, raw_result.failure_cases, run_dir)
    processed = summarize(processed_result, "processed")
    write_report(processed, processed_result.failure_cases, run_dir, before=raw)
    publish_prefect_summary(processed)
"""

import html
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
from prefect.artifacts import create_table_artifact
from prefect.context import FlowRunContext, TaskRunContext

from src.senpai_suggest.backend.data_quality.validator import ValidationResult
from src.senpai_suggest.backend.logger.logger import Logger

logger: Logger = Logger()

Stage = Literal["raw", "processed"]
Summary = dict[str, Any]

FAILURE_SAMPLE_SIZE = 20
# Label for checks that span the whole table, e.g. multi-column uniqueness.
TABLE_LEVEL = "<table>"

_CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem; color: #1f2328; }
table { border-collapse: collapse; margin: 0.5rem 0 1.5rem; font-size: 0.9rem; }
th, td { border: 1px solid #d0d7de; padding: 0.3rem 0.6rem; text-align: left; }
th { background: #f6f8fa; }
.pass { color: #1a7f37; } .fail { color: #cf222e; }
"""


@dataclass(frozen=True)
class ReportPaths:
    """Where `write_report` put the files."""

    html: Path
    json: Path


def column_profile(table: pa.Table) -> list[dict[str, Any]]:
    """Type, null count and null rate for every column."""
    rows = max(table.num_rows, 1)
    return [
        {
            "column": name,
            "type": str(table.schema.field(name).type),
            "nulls": table[name].null_count,
            "null_rate": round(table[name].null_count / rows, 4),
        }
        for name in table.column_names
    ]


def check_counts(failure_cases: pd.DataFrame) -> list[dict[str, Any]]:
    """Number of failing values per (column, check), from pandera's failure cases."""
    if failure_cases.empty:
        return []
    columns = failure_cases["column"].fillna(TABLE_LEVEL).astype(str)
    counts = failure_cases.assign(column=columns).groupby(["column", "check"]).size()
    return [
        {"column": column, "check": str(check), "failures": int(n)}
        for (column, check), n in counts.items()
    ]


def summarize(result: ValidationResult, stage: Stage) -> Summary:
    """Build the JSON-serialisable summary of one validation result."""
    return {
        "dataset": result.dataset,
        "stage": stage,
        "passed": result.passed,
        "rows": result.table.num_rows,
        "columns": result.table.num_columns,
        "failure_count": len(result.failure_cases),
        "checks": check_counts(result.failure_cases),
        "profile": column_profile(result.table),
    }


def compare_checks(before: Summary, after: Summary) -> list[dict[str, Any]]:
    """Failures per (column, check) before and after preprocessing.

    A check missing from a summary passed at that stage, so it counts as 0.
    """
    counts_before = {(c["column"], c["check"]): c["failures"] for c in before["checks"]}
    counts_after = {(c["column"], c["check"]): c["failures"] for c in after["checks"]}
    return [
        {
            "column": column,
            "check": check,
            "before": counts_before.get((column, check), 0),
            "after": counts_after.get((column, check), 0),
        }
        for column, check in sorted(counts_before.keys() | counts_after.keys())
    ]


def render_html(
    summary: Summary,
    failure_sample: pd.DataFrame,
    before: Summary | None = None,
) -> str:
    """Render one dataset/stage report as a standalone HTML page."""
    title = f"{summary['dataset']} · {summary['stage']}"
    status = "pass" if summary["passed"] else "fail"
    sections = [
        f"<h1>Data quality: {html.escape(title)}</h1>",
        f"<p class='{status}'><strong>{status.upper()}</strong> · {summary['rows']:,} rows · "
        f"{summary['columns']} columns · {summary['failure_count']:,} failing values</p>",
        "<h2>Failed checks</h2>",
        _rows_to_html(summary["checks"], "All checks passed."),
    ]
    if before is not None:
        sections += ["<h2>Before → after</h2>", _rows_to_html(compare_checks(before, summary), "")]
    sections += [
        "<h2>Columns</h2>",
        _rows_to_html(summary["profile"], "No columns."),
        f"<h2>Failure sample (first {FAILURE_SAMPLE_SIZE})</h2>",
        _frame_to_html(failure_sample.head(FAILURE_SAMPLE_SIZE), "No failures."),
    ]
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{_CSS}</style></head><body>{''.join(sections)}</body></html>"
    )


def write_report(
    summary: Summary,
    failure_cases: pd.DataFrame,
    output_dir: Path,
    before: Summary | None = None,
) -> ReportPaths:
    """Write `{dataset}_{stage}.html` and `.json` into `output_dir` (created if missing).

    Raises:
        OSError: If the files cannot be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{summary['dataset']}_{summary['stage']}"
    paths = ReportPaths(html=output_dir / f"{stem}.html", json=output_dir / f"{stem}.json")

    payload = dict(summary)
    if before is not None:
        payload["comparison"] = compare_checks(before, summary)
    paths.html.write_text(render_html(summary, failure_cases, before), encoding="utf-8")
    paths.json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    logger.info(f"Wrote data quality report for {stem} to {output_dir}.")
    return paths


def write_run_summary(summaries: list[Summary], output_dir: Path) -> Path:
    """Write one `summary.json` with the headline numbers of every report in a run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    headline_keys = ("dataset", "stage", "passed", "rows", "failure_count")
    headlines = [{key: s[key] for key in headline_keys} for s in summaries]
    path = output_dir / "summary.json"
    path.write_text(json.dumps(headlines, indent=2), encoding="utf-8")
    return path


def write_stage_reports(
    results: Mapping[str, ValidationResult],
    stage: Stage,
    output_dir: Path,
    before: Mapping[str, Summary] | None = None,
) -> dict[str, Summary]:
    """Summarise, write and publish reports for every dataset of one stage.

    Args:
        results: Validation results by dataset name.
        stage: Which stage these results come from.
        output_dir: The run's report directory.
        before: Raw-stage summaries by dataset, for the before -> after view.

    Returns:
        Summaries by dataset name; pass them as `before` for the next stage.
    """
    summaries: dict[str, Summary] = {}
    for name, result in results.items():
        summary = summarize(result, stage)
        previous = before.get(name) if before else None
        write_report(summary, result.failure_cases, output_dir, before=previous)
        publish_prefect_summary(summary)
        summaries[name] = summary
    return summaries


def publish_prefect_summary(summary: Summary) -> bool:
    """Publish the failed checks as a Prefect table artifact.

    Returns:
        False without publishing when not inside a Prefect flow or task run,
        so steps can call this from plain scripts and tests.
    """
    if FlowRunContext.get() is None and TaskRunContext.get() is None:
        logger.debug("No Prefect run context; skipping data quality artifact.")
        return False

    rows = summary["checks"] or [{"column": "-", "check": "all checks passed", "failures": 0}]
    create_table_artifact(
        table=rows,
        key=f"dq-{summary['dataset']}-{summary['stage']}".lower().replace("_", "-"),
        description=(
            f"Data quality for **{summary['dataset']}** ({summary['stage']}): "
            f"{'passed' if summary['passed'] else 'failed'}, {summary['rows']:,} rows."
        ),
    )
    return True


def _rows_to_html(rows: list[dict[str, Any]], empty_message: str) -> str:
    return _frame_to_html(pd.DataFrame(rows), empty_message)


def _frame_to_html(frame: pd.DataFrame, empty_message: str) -> str:
    if frame.empty:
        return f"<p>{html.escape(empty_message)}</p>"
    return str(frame.to_html(index=False, border=0, escape=True, na_rep=""))
