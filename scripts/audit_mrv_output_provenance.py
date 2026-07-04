#!/usr/bin/env python3
"""Verify current MRV artifacts are generated from THOR-selected inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from astreia_mrv.cli import generate_geometry
from astreia_mrv.config import load_design
from thor.io.handoff import load_table
from thor.io.inputs import INPUTS_CSV, load_inputs, txt


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return data


def _mrv_input_rows() -> list[dict[str, str]]:
    inputs = load_inputs()
    df = inputs.filter(inputs["section"].is_in(["mrv", "mrv_payload"]))
    return [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in df.to_dicts()
    ]


def _close(name: str, actual: float, expected: float, errors: list[str], rel_tol: float = 1e-7) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=rel_tol, abs_tol=1e-9):
        errors.append(f"{name}: artifact={actual!r} expected={expected!r}")


def _compare_numeric_mapping(name: str, actual: dict[str, Any], expected: dict[str, Any], errors: list[str]) -> None:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        errors.append(f"{name}: missing keys {missing}")
    if extra:
        errors.append(f"{name}: unexpected keys {extra}")
    for key in sorted(set(expected) & set(actual)):
        _close(f"{name}.{key}", float(actual[key]), float(expected[key]), errors)


def audit(out_dir: Path) -> list[str]:
    errors: list[str] = []
    config_path = Path(txt("mrv", "config_path", default="configs/mrv3.yaml"))
    expected_out_dir = Path(txt("mrv", "output_dir", default="outputs/mrv3_notebook"))
    if out_dir != expected_out_dir:
        errors.append(f"output dir is not THOR-selected mrv.output_dir: {out_dir} != {expected_out_dir}")

    design = load_design(config_path)
    geometry = generate_geometry(design)
    geometry_json = _read_json(out_dir / "geometry.json")
    metrics_json = _read_json(out_dir / "metrics.json")
    brep_spec = _read_json(out_dir / "onshape_brep_spec.json")

    for label, payload in (("geometry.json", geometry_json), ("metrics.json", metrics_json)):
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            errors.append(f"{label}: missing provenance object")
            continue

        expected = {
            "schema": "astreia_mrv_thor_provenance_v1",
            "generator": "thor.io.openscad.export_mrv_to_openscad",
            "notebook": "notebooks/80_oml_parametric.py",
            "preview_notebook": "notebooks/82_openscad_export.py",
            "thor_inputs_csv": str(INPUTS_CSV),
            "thor_inputs_sha256": _sha256(INPUTS_CSV),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "output_dir": str(expected_out_dir),
        }
        for key, value in expected.items():
            if provenance.get(key) != value:
                errors.append(f"{label}: provenance.{key}={provenance.get(key)!r}, expected {value!r}")

        if provenance.get("mrv_input_rows") != _mrv_input_rows():
            errors.append(f"{label}: provenance.mrv_input_rows does not match current thor_inputs.csv")

        vehicle = provenance.get("vehicle", {})
        payload_spec = provenance.get("payload", {})
        if vehicle.get("name") != design.scale.name:
            errors.append(f"{label}: provenance.vehicle.name does not match YAML")
        _close(f"{label}.provenance.vehicle.body_length_m", vehicle.get("body_length_m", -1), design.scale.body_length_m, errors)
        if payload_spec.get("name") != design.payload.name:
            errors.append(f"{label}: provenance.payload.name does not match YAML")
        _close(f"{label}.provenance.payload.mass_kg", payload_spec.get("mass_kg", -1), design.payload.mass_kg, errors)
        _compare_numeric_mapping(f"{label}.provenance.rx", provenance.get("rx", {}), dict(sorted(design.rx.items())), errors)

    _compare_numeric_mapping("geometry.json.normalized_rx", geometry_json.get("normalized_rx", {}), design.rx, errors)
    _compare_numeric_mapping(
        "geometry.json.parameters",
        geometry_json.get("parameters", {}),
        geometry.metadata.get("parameters", {}),
        errors,
    )
    _compare_numeric_mapping(
        "onshape_brep_spec.source_parameters",
        brep_spec.get("source_parameters", {}),
        geometry.metadata.get("parameters", {}),
        errors,
    )
    _compare_numeric_mapping(
        "onshape_brep_spec.normalized_rx",
        brep_spec.get("normalized_rx", {}),
        design.rx,
        errors,
    )
    _close("geometry.reference_area_m2", geometry_json.get("reference_area_m2", -1), geometry.reference_area_m2, errors)
    _close("geometry.reference_length_m", geometry_json.get("reference_length_m", -1), geometry.reference_length_m, errors)
    _close("geometry.volume_m3", geometry_json.get("volume_m3", -1), geometry.volume_m3, errors)
    _close("geometry.wetted_area_m2", geometry_json.get("wetted_area_m2", -1), geometry.wetted_area_m2, errors)
    for artifact_name in ("onshape_parameters.csv", "onshape_loft_sections.csv", "onshape_variables.fs"):
        if not (out_dir / artifact_name).exists():
            errors.append(f"missing Onshape editable handoff artifact: {out_dir / artifact_name}")
    step_manifest_path = out_dir / "onshape_step" / "onshape_step_manifest.json"
    if step_manifest_path.exists():
        step_manifest = _read_json(step_manifest_path)
        if step_manifest.get("export_kind") != "faceted_brep_solid_from_generated_mrv_mesh":
            errors.append(
                "onshape_step_manifest.export_kind is not faceted_brep_solid_from_generated_mrv_mesh"
            )
        if step_manifest.get("valid_body") is not True:
            errors.append("onshape_step_manifest.valid_body is not true")
        if step_manifest.get("valid_assembly") is not True:
            errors.append("onshape_step_manifest.valid_assembly is not true")
        assembly_path = Path(str(step_manifest.get("assembly_step", "")))
        if not assembly_path.exists():
            errors.append(f"Onshape assembly STEP is missing: {assembly_path}")

    metrics_geometry = metrics_json.get("geometry", {})
    _close("metrics.geometry.reference_area_m2", metrics_geometry.get("reference_area_m2", -1), geometry.reference_area_m2, errors)
    _close("metrics.geometry.reference_length_m", metrics_geometry.get("reference_length_m", -1), geometry.reference_length_m, errors)
    _close("metrics.geometry.volume_m3", metrics_geometry.get("volume_m3", -1), geometry.volume_m3, errors)
    _close("metrics.geometry.wetted_area_m2", metrics_geometry.get("wetted_area_m2", -1), geometry.wetted_area_m2, errors)

    table = load_table("mrv_oml")
    if table is None or table.height == 0:
        errors.append("missing mrv_oml handoff table")
    else:
        row = table.to_dicts()[0]
        expected_paths = {
            "out_dir": out_dir,
            "geometry_json_path": out_dir / "geometry.json",
            "metrics_json_path": out_dir / "metrics.json",
            "scad_path": out_dir / "geometry.scad",
            "obj_path": out_dir / "geometry.obj",
            "stl_path": out_dir / "geometry.stl",
        }
        for key, path in expected_paths.items():
            if row.get(key) != str(path):
                errors.append(f"mrv_oml.{key}={row.get(key)!r}, expected {str(path)!r}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=txt("mrv", "output_dir", default="outputs/mrv3_notebook"))
    args = parser.parse_args()

    errors = audit(Path(args.out_dir))
    if errors:
        print("MRV output provenance audit FAILED")
        for error in errors:
            print(f" - {error}")
        return 1

    print("MRV output provenance audit OK")
    print(f"thor_inputs: {INPUTS_CSV}")
    print(f"config: {txt('mrv', 'config_path', default='configs/mrv3.yaml')}")
    print(f"out_dir: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
