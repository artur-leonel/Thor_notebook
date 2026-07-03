"""OpenSCAD handoff for Astreia-MRV geometry notebooks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from shutil import which
import subprocess

import polars as pl

from astreia_mrv.analysis.aerodynamics import alpha_sweep, write_polar_csv
from astreia_mrv.analysis.constraints import evaluate_constraints
from astreia_mrv.analysis.heating import heating_summary, write_heating_csv
from astreia_mrv.cli import generate_geometry
from astreia_mrv.config import VehicleDesign, load_design
from astreia_mrv.geometry.cad_export import export_geometry
from astreia_mrv.geometry.mesh import VehicleGeometry
from thor.io.inputs import INPUTS_CSV, load_inputs
from astreia_mrv.viz.plot_geometry import (
    save_control_surface_inspection,
    save_engineering_sheet,
    save_shaded_render,
    save_three_view,
)


@dataclass(frozen=True)
class OpenSCADExport:
    out_dir: Path
    scad_path: Path
    obj_path: Path
    stl_path: Path
    geometry_json_path: Path
    metrics_json_path: Path
    three_view_path: Path
    shaded_render_path: Path
    engineering_views_path: Path
    control_inspection_path: Path
    openscad_available: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mrv_input_rows() -> list[dict[str, str]]:
    df = load_inputs().filter(pl.col("section").is_in(["mrv", "mrv_payload"]))
    return [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in df.to_dicts()
    ]


def _provenance(config_path: str | Path, out_dir: str | Path, design: VehicleDesign) -> dict:
    config = Path(config_path)
    return {
        "schema": "astreia_mrv_thor_provenance_v1",
        "generator": "thor.io.openscad.export_mrv_to_openscad",
        "notebook": "notebooks/80_oml_parametric.py",
        "preview_notebook": "notebooks/82_openscad_export.py",
        "thor_inputs_csv": str(INPUTS_CSV),
        "thor_inputs_sha256": _sha256(INPUTS_CSV),
        "mrv_input_rows": _mrv_input_rows(),
        "config_path": str(config),
        "config_sha256": _sha256(config),
        "output_dir": str(out_dir),
        "vehicle": {
            "name": design.scale.name,
            "geometry_family": design.geometry_family,
            "body_length_m": design.scale.body_length_m,
            "payload_target_kg": design.scale.payload_target_kg,
            "recovery_mode": design.recovery_mode,
        },
        "payload": {
            "name": design.payload.name,
            "mass_kg": design.payload.mass_kg,
            "length_m": design.payload.length_m,
            "width_m": design.payload.width_m,
            "height_m": design.payload.height_m,
        },
        "rx": dict(sorted(design.rx.items())),
    }


def export_mrv_to_openscad(config_path: str | Path, out_dir: str | Path) -> tuple[VehicleDesign, VehicleGeometry, OpenSCADExport]:
    """Generate MRV geometry and write OpenSCAD/mesh analysis handoff files."""
    design = load_design(config_path)
    geometry = generate_geometry(design)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(config_path, out_dir, design)
    export_geometry(geometry, out, extra_metadata={"provenance": provenance})
    save_three_view(geometry, out / "three_view.png")
    save_shaded_render(geometry, out / "shaded_render.png")
    save_engineering_sheet(geometry, out / "engineering_views.png")
    save_control_surface_inspection(geometry, out / "control_surface_inspection.png")

    polar = alpha_sweep(geometry, mach=20.0)
    heat = heating_summary(design, geometry)
    write_polar_csv(polar, out / "aero_polar.csv")
    write_heating_csv(heat, out / "heating_summary.csv")

    # Keep a lightweight notebook metrics file beside the package-level geometry.json.
    metrics = {
        "provenance": provenance,
        "geometry": {
            "reference_area_m2": geometry.reference_area_m2,
            "reference_length_m": geometry.reference_length_m,
            "volume_m3": geometry.volume_m3,
            "wetted_area_m2": geometry.wetted_area_m2,
            "mesh_quality": geometry.metadata.get("mesh_quality", {}),
        },
        "heating": heat,
        "constraints": [c.__dict__ for c in evaluate_constraints(design, geometry)],
        "openscad": {
            "path": str(out / "geometry.scad"),
            "available_locally": openscad_available(),
        },
        "previews": {
            "three_view": str(out / "three_view.png"),
            "shaded_render": str(out / "shaded_render.png"),
            "engineering_views": str(out / "engineering_views.png"),
            "control_surface_inspection": str(out / "control_surface_inspection.png"),
        },
        "assumptions": [
            "OpenSCAD receives the sampled parametric MRV as a native polyhedron.",
            "Conceptual aero/heating only; TODO: CFD/FEA/6DOF integration.",
            "Landing accuracy is terminal recovery into an authorized recovery zone.",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    export = OpenSCADExport(
        out_dir=out,
        scad_path=out / "geometry.scad",
        obj_path=out / "geometry.obj",
        stl_path=out / "geometry.stl",
        geometry_json_path=out / "geometry.json",
        metrics_json_path=out / "metrics.json",
        three_view_path=out / "three_view.png",
        shaded_render_path=out / "shaded_render.png",
        engineering_views_path=out / "engineering_views.png",
        control_inspection_path=out / "control_surface_inspection.png",
        openscad_available=openscad_available(),
    )
    return design, geometry, export


def openscad_available() -> bool:
    return which("openscad") is not None


def render_scad_to_stl(scad_path: str | Path, stl_path: str | Path) -> bool:
    """Render with OpenSCAD if installed; return False when unavailable."""
    if not openscad_available():
        return False
    subprocess.run(["openscad", "-o", str(stl_path), str(scad_path)], check=True)
    return True


def openscad_command(scad_path: str | Path) -> str:
    return f"openscad {Path(scad_path)}"
