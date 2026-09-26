"""Aggregate final classification_report.json files into a single CSV.

Usage:
    python scripts/aggregate_classification_reports.py
    python scripts/aggregate_classification_reports.py --input eval_results/final --output reports.csv
"""

import argparse
import csv
import json
from pathlib import Path


METRICS = ("precision", "recall", "f1-score", "support")
SUMMARY_KEYS = ("accuracy", "macro avg", "weighted avg")


def flatten_report(report):
    """Return one flat row containing all standard sklearn report values."""
    row = {}
    for key, values in report.items():
        if isinstance(values, dict):
            for metric in METRICS:
                if metric in values:
                    row[f"{key}_{metric}"] = values[metric]
        elif key == "accuracy":
            row["accuracy"] = values
    return row


def collect(input_dir):
    rows = []
    for report_path in sorted(input_dir.rglob("classification_report.json")):
        try:
            with report_path.open(encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: skipped {report_path}: {exc}")
            continue
        relative = report_path.relative_to(input_dir)
        parts = relative.parts
        row = {
            "model": parts[0] if parts else "",
            "checkpoint": parts[1] if len(parts) > 2 else "",
            "report_path": relative.as_posix(),
        }
        row.update(flatten_report(report))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/media/user/BF45460F1FCF00571/Tcy/MRI_Code/ADMAN/eval_results/final"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    input_dir = args.input.resolve()
    output = args.output or input_dir / "classification_reports_summary.csv"
    rows = collect(input_dir)
    if not rows:
        raise SystemExit(f"No classification_report.json found under {input_dir}")

    preferred = ["model", "checkpoint", "report_path", "accuracy"]
    metric_columns = sorted({key for row in rows for key in row if key not in preferred})
    columns = preferred + metric_columns
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} reports to {output}")


if __name__ == "__main__":
    main()
