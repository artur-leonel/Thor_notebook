#!/usr/bin/env python3
"""Smoke-check the THOR→MRV handoff row and artifact outputs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thor.io.handoff import load_table
from thor.io.inputs import txt
from thor.io.openscad import export_mrv_to_openscad
from thor.mrv.handoff import (
    REQUIRED_METRIC_FIELDS,
    load_metrics,
    validate_mrv_oml_row,
)


def main() -> int:
    config_path = txt("mrv", "config_path", default="configs/mrv3.yaml")
    out_dir = txt("mrv", "output_dir", default="outputs/mrv3_notebook")

    # Re-run the handoff flow used by notebooks/scripts so required artifacts are current.
    _, _, _ = export_mrv_to_openscad(config_path, out_dir)

    table = load_table("mrv_oml")
    if table is None or table.height == 0:
        print("mrv_oml smoke check FAILED")
        print("missing table:")
        print("  - mrv_oml")
        return 1

    row = table.to_dicts()[0]
    fields_missing, files_missing = _validate_row(row)
    if fields_missing or files_missing:
        print("mrv_oml smoke check FAILED")
        if fields_missing:
            print("missing fields:")
            for field in fields_missing:
                print(f"  - {field}")
        if files_missing:
            print("missing files:")
            for value in files_missing:
                print(f"  - {value}")
        return 1

    required_metric_missing = _validate_metric_fields(row["metrics_json_path"])
    if required_metric_missing:
        print("mrv_oml smoke check FAILED")
        print("missing metrics keys:")
        for key in required_metric_missing:
            print(f"  - {key}")
        return 1

    print("mrv_oml smoke check OK")
    print(f"case: {Path(row['out_dir']).name}")
    print(f"scad: {row['scad_path']}")
    print(f"three_view: {row['three_view_path']}")
    print(f"metrics: {row['metrics_json_path']}")
    return 0


def _validate_row(row: dict[str, object]) -> tuple[list[str], list[str]]:
    required_fields_ok, missing_fields, missing_files = validate_mrv_oml_row(row, require_files=True)
    if not required_fields_ok:
        return missing_fields, missing_files
    return [], []


def _validate_metric_fields(metrics_path: object) -> list[str]:
    try:
        metrics = load_metrics(metrics_path) if isinstance(metrics_path, (str, Path)) else {}
    except (TypeError, FileNotFoundError, ValueError):
        return REQUIRED_METRIC_FIELDS.copy()
    return [field for field in REQUIRED_METRIC_FIELDS if field not in metrics]


if __name__ == "__main__":
    raise SystemExit(main())
