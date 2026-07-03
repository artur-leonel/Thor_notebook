#!/usr/bin/env python3
"""Synchronize MRV-derived THOR CSV rows with the selected YAML geometry."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from astreia_mrv.config import load_design
from astreia_mrv.geometry.lifting_body import generate_lifting_body
from thor.io.inputs import INPUTS_CSV, txt


def _expected_rows(config_path: str) -> dict[tuple[str, str, str], str]:
    design = load_design(config_path)
    geometry = generate_lifting_body(design)
    bounds = geometry.mesh.bounds
    span = bounds[1] - bounds[0]

    return {
        ("aero", "", "reference_area_m2"): repr(float(geometry.reference_area_m2)),
        ("aero", "", "nose_radius_m"): repr(float(geometry.metadata["parameters"]["R_N"])),
        ("geometry", "", "length_m"): repr(float(geometry.reference_length_m)),
        ("geometry", "", "width_m"): repr(float(span[1])),
        ("geometry", "", "s_ref_factor"): "1.0",
        ("tps", "", "shield_area_m2"): repr(float(geometry.wetted_area_m2)),
        ("mass_props", "", "height_m"): repr(float(span[2])),
        ("mrv_payload", "", "name"): design.payload.name,
        ("mrv_payload", "", "mass_kg"): repr(float(design.payload.mass_kg)),
        ("mrv_payload", "", "length_m"): repr(float(design.payload.length_m)),
        ("mrv_payload", "", "width_m"): repr(float(design.payload.width_m)),
        ("mrv_payload", "", "height_m"): repr(float(design.payload.height_m)),
    }


def sync_inputs(csv_path: Path, config_path: str) -> int:
    expected = _expected_rows(config_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    changed = 0
    for row in rows:
        key = (row["section"], row["item"], row["parameter"])
        if key not in expected:
            continue
        new_value = expected[key]
        if row["value"] != new_value:
            row["value"] = new_value
            changed += 1

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames or rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=INPUTS_CSV)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config_path = args.config or txt("mrv", "config_path", default="configs/mrv3.yaml")
    changed = sync_inputs(args.csv, config_path)
    print(f"synced {changed} MRV THOR input rows in {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
