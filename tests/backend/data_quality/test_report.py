"""Unit tests for data quality report rendering and writing."""

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
from prefect.context import FlowRunContext
from pytest_mock import MockerFixture

from src.senpai_suggest.backend.data_quality import DataQualityValidator, ValidationResult, report

DUPLICATES = pa.table(
    {"user_id": [1, 1, 2], "anime_id": [10, 10, 11], "rating": [8, 8, None]},
)
CLEAN = pa.table({"user_id": [1, 2], "anime_id": [10, 11], "rating": [8, 7]})


def _raw_result() -> ValidationResult:
    return DataQualityValidator().validate_ratings(DUPLICATES, raise_on_fail=False)


def _clean_result() -> ValidationResult:
    return DataQualityValidator().validate_ratings(CLEAN)


def test_column_profile_reports_types_and_null_rates() -> None:
    """It should list every column with its null count and rate."""
    profile = report.column_profile(pa.table({"a": [1, None], "b": ["x", "y"]}))

    assert profile == [
        {"column": "a", "type": "int64", "nulls": 1, "null_rate": 0.5},
        {"column": "b", "type": "string", "nulls": 0, "null_rate": 0.0},
    ]


def test_column_profile_handles_empty_table() -> None:
    """It should not divide by zero on an empty table."""
    profile = report.column_profile(pa.table({"a": pa.array([], type=pa.int64())}))

    assert profile[0]["null_rate"] == 0.0


def test_check_counts_groups_failures_and_labels_table_level_checks() -> None:
    """It should count failures per (column, check), labelling column-less checks."""
    failures = pd.DataFrame(
        {"column": ["rating", "rating", None], "check": ["not_nullable", "not_nullable", "u"]}
    )

    assert report.check_counts(failures) == [
        {"column": report.TABLE_LEVEL, "check": "u", "failures": 1},
        {"column": "rating", "check": "not_nullable", "failures": 2},
    ]


def test_check_counts_is_empty_when_nothing_failed() -> None:
    """It should return no rows for an empty failure frame."""
    assert report.check_counts(pd.DataFrame()) == []


def test_summarize_captures_headline_numbers() -> None:
    """It should summarise the dataset, stage, pass flag, size and failures."""
    summary = report.summarize(_raw_result(), "raw")

    assert summary["dataset"] == "ratings"
    assert summary["stage"] == "raw"
    assert summary["passed"] is False
    assert summary["rows"] == 3
    assert summary["failure_count"] > 0
    assert {c["column"] for c in summary["profile"]} == {"user_id", "anime_id", "rating"}


def test_compare_checks_counts_fixed_checks_as_zero_after() -> None:
    """It should show every failing check with before/after counts."""
    before = report.summarize(_raw_result(), "raw")
    after = report.summarize(_clean_result(), "processed")

    comparison = report.compare_checks(before, after)

    assert comparison
    assert all(row["before"] > 0 and row["after"] == 0 for row in comparison)


def test_render_html_escapes_values_and_shows_status() -> None:
    """It should mark the status and escape failure values."""
    summary = report.summarize(_clean_result(), "processed")
    sample = pd.DataFrame({"failure_case": ["<script>"]})

    page = report.render_html(summary, sample)

    assert "PASS" in page
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_write_report_writes_html_and_json_with_comparison(tmp_path: Path) -> None:
    """It should write both files and include the comparison when `before` is given."""
    raw = report.summarize(_raw_result(), "raw")
    processed_result = _clean_result()
    processed = report.summarize(processed_result, "processed")

    paths = report.write_report(
        processed, processed_result.failure_cases, tmp_path / "run", before=raw
    )

    assert paths.html == tmp_path / "run" / "ratings_processed.html"
    assert "Before → after" in paths.html.read_text(encoding="utf-8")
    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["comparison"]


def test_write_run_summary_keeps_headline_fields(tmp_path: Path) -> None:
    """It should write one summary.json with only the headline fields."""
    summary = report.summarize(_raw_result(), "raw")

    path = report.write_run_summary([summary], tmp_path)

    assert json.loads(path.read_text()) == [
        {
            "dataset": "ratings",
            "stage": "raw",
            "passed": False,
            "rows": 3,
            "failure_count": summary["failure_count"],
        }
    ]


def test_write_stage_reports_writes_and_publishes_each_dataset(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    """It should write a report and publish an artifact per dataset."""
    publish = mocker.patch.object(report, "publish_prefect_summary")

    summaries = report.write_stage_reports({"ratings": _raw_result()}, "raw", tmp_path)

    assert (tmp_path / "ratings_raw.html").exists()
    assert summaries["ratings"]["stage"] == "raw"
    publish.assert_called_once_with(summaries["ratings"])


def test_publish_prefect_summary_skips_outside_a_run(mocker: MockerFixture) -> None:
    """It should not call Prefect when there is no flow or task run."""
    create = mocker.patch.object(report, "create_table_artifact")

    assert report.publish_prefect_summary(report.summarize(_clean_result(), "raw")) is False
    create.assert_not_called()


def test_publish_prefect_summary_creates_artifact_inside_a_run(mocker: MockerFixture) -> None:
    """It should publish the failed checks with a Prefect-safe key inside a run."""
    mocker.patch.object(FlowRunContext, "get", return_value=object())
    create = mocker.patch.object(report, "create_table_artifact")

    assert report.publish_prefect_summary(report.summarize(_raw_result(), "raw")) is True
    assert create.call_args.kwargs["key"] == "dq-ratings-raw"
    assert create.call_args.kwargs["table"]
