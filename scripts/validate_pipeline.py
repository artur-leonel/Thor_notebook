#!/usr/bin/env python3
"""Validate THOR inputs plus Astreia-MRV OpenSCAD notebook handoff."""

from __future__ import annotations

import math
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl

from astreia_mrv.analysis.constraints import evaluate_constraints, payload_fit
from astreia_mrv.analysis.heating import heating_summary
from thor.io.handoff import load_state, save_state, save_table
from thor.io.inputs import entry_velocity_m_s, float_list, item_table, items_with_params, load_inputs, num, phase_rows, txt
from thor.io.openscad import export_mrv_to_openscad
from thor.models.mission_spec import MissionPhase, MissionPhaseSpec, MissionSpec
from thor.models.vehicle_state import EntryState, MassBudget, MassItem, OrbitState
from thor.physics.entry import integrate_entry_3dof
from thor.mrv.handoff import (
    load_mass_sanity_config,
    MASS_SANITY_LEVEL_FAIL,
    MASS_SANITY_LEVEL_WARNING,
    evaluate_mass_sanity,
    load_metrics,
)


def _massage_oml_table(export, design, geometry) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "vehicle": [design.scale.name],
            "family": [design.geometry_family],
            "volume_m3": [geometry.volume_m3],
            "wetted_area_m2": [geometry.wetted_area_m2],
            "reference_area_m2": [geometry.reference_area_m2],
            "out_dir": [str(export.out_dir)],
            "scad_path": [str(export.scad_path)],
            "obj_path": [str(export.obj_path)],
            "stl_path": [str(export.stl_path)],
            "geometry_json_path": [str(export.geometry_json_path)],
            "metrics_json_path": [str(export.metrics_json_path)],
            "three_view_path": [str(export.three_view_path)],
            "shaded_render_path": [str(export.shaded_render_path)],
            "engineering_views_path": [str(export.engineering_views_path)],
        }
    )


def _write_mass_sanity_metric(export, config_path: str, dry_mass_kg: float, payload_mass_kg: float, mrv_volume_m3: float) -> None:
    config = load_mass_sanity_config(config_path)
    mass_sanity = evaluate_mass_sanity(
        dry_mass_kg,
        payload_mass_kg,
        mrv_volume_m3,
        config,
    )
    metrics = load_metrics(export.metrics_json_path)
    metrics["mass_sanity"] = mass_sanity
    export.metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _assert_mrv_inputs_aligned(design, geometry) -> None:
    if txt("mrv_payload", "name") != design.payload.name:
        raise AssertionError(
            f"THOR CSV mismatch for mrv_payload.name: csv={txt('mrv_payload', 'name')} "
            f"generated={design.payload.name}"
        )

    bounds = geometry.mesh.bounds
    span = bounds[1] - bounds[0]
    checks = {
        "geometry.length_m": (num("geometry", "length_m"), geometry.reference_length_m),
        "geometry.width_m": (num("geometry", "width_m"), float(span[1])),
        "aero.reference_area_m2": (num("aero", "reference_area_m2"), geometry.reference_area_m2),
        "aero.nose_radius_m": (num("aero", "nose_radius_m"), geometry.metadata["parameters"]["R_N"]),
        "tps.shield_area_m2": (num("tps", "shield_area_m2"), geometry.wetted_area_m2),
        "mass_props.height_m": (num("mass_props", "height_m"), float(span[2])),
        "mrv_payload.mass_kg": (num("mrv_payload", "mass_kg"), design.payload.mass_kg),
        "mrv_payload.length_m": (num("mrv_payload", "length_m"), design.payload.length_m),
        "mrv_payload.width_m": (num("mrv_payload", "width_m"), design.payload.width_m),
        "mrv_payload.height_m": (num("mrv_payload", "height_m"), design.payload.height_m),
    }
    for label, (csv_value, generated_value) in checks.items():
        if not math.isclose(csv_value, generated_value, rel_tol=1e-6, abs_tol=1e-9):
            raise AssertionError(f"THOR CSV mismatch for {label}: csv={csv_value} generated={generated_value}")


def _format_pipeline_output(
    g_peak: float,
    dry_mass_kg: float,
    geometry_volume_m3: float,
    mass_sanity: dict,
) -> str:
    status = "pipeline OK"
    if mass_sanity["status"] == MASS_SANITY_LEVEL_WARNING:
        status = "pipeline OK with warnings"
    if mass_sanity["status"] == MASS_SANITY_LEVEL_FAIL and mass_sanity["fail_on_mass_sanity"]:
        status = "pipeline failed mass sanity"

    output = (
        f"{status} - g_peak={g_peak:.2f} g, dry_mass={dry_mass_kg:.0f} kg, "
        f"mrv_volume={geometry_volume_m3:.3f} m^3, bulk_density={mass_sanity['bulk_density_kg_m3']:.0f} kg/m^3"
    )
    if mass_sanity["status"] != "ok":
        output += f"\nWARNING: {mass_sanity['message']}"
    return output


def main() -> None:
    df = load_inputs()
    print(f"inputs: {len(df)} rows")

    phases = [
        MissionPhaseSpec(
            phase=MissionPhase(r["item"]),
            duration_s=float(r["duration_s"]),
            delta_v_m_s=float(r["delta_v_m_s"]),
            power_wh=float(r["power_wh"]),
        )
        for r in phase_rows()
    ]
    mission = MissionSpec(
        name=txt("mission", "name"),
        vehicle_class=txt("mission", "vehicle_class"),
        phases=phases,
    )

    growth = num("mass", "growth_allowance", "_config")
    mass_rows = items_with_params("mass", "dry_kg")
    mass = MassBudget(
        items=[MassItem(name=r["item"], dry_kg=float(r["dry_kg"]), growth_kg=float(r["dry_kg"]) * growth) for r in mass_rows],
        growth_allowance=growth,
        propellant_kg=num("mass", "propellant_kg", "_config"),
    )

    state = load_state()
    state.mission = mission
    state.mass = mass
    state.orbit = OrbitState(altitude_km=num("orbit", "altitude_km"), inclination_deg=num("orbit", "inclination_deg"))
    state.entry = EntryState(
        altitude_m=num("entry", "altitude_m"),
        velocity_m_s=entry_velocity_m_s(),
        flight_path_angle_deg=num("entry", "flight_path_angle_deg"),
        ballistic_coefficient_kg_m2=num("entry", "ballistic_coefficient_kg_m2"),
    )
    state.aero.cd = num("aero", "cd")
    state.aero.reference_area_m2 = num("geometry", "length_m") * num("geometry", "width_m") * num("geometry", "s_ref_factor")

    traj = integrate_entry_3dof(
        state.entry.altitude_m,
        state.entry.velocity_m_s,
        math.radians(state.entry.flight_path_angle_deg),
        state.entry.ballistic_coefficient_kg_m2,
        dt=2.0,
    )
    g_peak = float(traj["g_load"].max())

    config_path = txt("mrv", "config_path", default="configs/mrv3.yaml")
    out_dir = txt("mrv", "output_dir", default="outputs/mrv3_notebook")
    design, geometry, export = export_mrv_to_openscad(config_path, out_dir)
    _assert_mrv_inputs_aligned(design, geometry)
    constraints = evaluate_constraints(design, geometry)
    heat = heating_summary(design, geometry)
    fit = payload_fit(design, geometry)
    mass_sanity = evaluate_mass_sanity(
        mass.dry_mass_kg,
        design.payload.mass_kg,
        geometry.volume_m3,
        load_mass_sanity_config(config_path),
    )

    state.aero.reference_area_m2 = geometry.reference_area_m2
    state.aero.nose_radius_m = geometry.metadata["parameters"].get("R_N", state.aero.nose_radius_m)
    state.notes["mrv_openscad"] = str(export.scad_path)
    state.notes["mrv_recovery"] = "Terminal recovery into authorized recovery zone; no arbitrary impact targeting."
    save_state(state)

    save_table(
        "mrv_oml",
        _massage_oml_table(export, design, geometry),
    )
    _write_mass_sanity_metric(export, config_path, mass.dry_mass_kg, design.payload.mass_kg, geometry.volume_m3)
    save_table("mrv_constraints", pl.DataFrame([c.__dict__ for c in constraints]))
    save_table(
        "mrv_heating",
        pl.DataFrame({k: [v] for k, v in heat.items() if isinstance(v, (int, float))}),
    )

    assert len(phases) == 10
    assert len(float_list("aero_db", "mach_list")) >= 3
    assert len(item_table("docking")) == 4
    assert len(item_table("nav")) == 4
    assert export.scad_path.exists()
    assert "polyhedron(" in export.scad_path.read_text(encoding="utf-8")
    assert fit.passed
    assert all(c.passed for c in constraints)
    print(_format_pipeline_output(g_peak, mass.dry_mass_kg, geometry.volume_m3, mass_sanity))
    if mass_sanity["status"] == MASS_SANITY_LEVEL_FAIL and mass_sanity["fail_on_mass_sanity"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
