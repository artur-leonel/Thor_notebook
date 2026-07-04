from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from astreia_mrv.geometry.mesh import SurfaceMesh, VehicleGeometry, export_surface


def _fmt_float(value: float) -> str:
    return f"{float(value):.9g}"


def _body_profile_points(
    x: float,
    half_width: float,
    top: float,
    belly: float,
    shoulder_y_scale: float = 0.58,
    shoulder_z_scale: float = 0.50,
    chine_y_scale: float = 1.00,
    chine_z_scale: float = -0.08,
    belly_scale: float = 0.92,
) -> list[list[float]]:
    """Six-point chine body section used by the Onshape BREP handoff."""
    yz = [
        (0.0, top),
        (shoulder_y_scale * half_width, shoulder_z_scale * top),
        (chine_y_scale * half_width, chine_z_scale * belly),
        (0.0, -belly_scale * belly),
        (-chine_y_scale * half_width, chine_z_scale * belly),
        (-shoulder_y_scale * half_width, shoulder_z_scale * top),
    ]
    return [[float(x), float(y), float(z)] for y, z in yz]


def _scaled_body_profile(
    x: float,
    half_width: float,
    top: float,
    belly: float,
    y_scale: float,
    z_scale: float,
) -> list[list[float]]:
    return _body_profile_points(x, half_width * y_scale, top * z_scale, belly * z_scale)


def _section_point_rows(sections: list[dict[str, Any]]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for section in sections:
        for index, point in enumerate(section["points"]):
            x, y, z = point
            rows.append(
                {
                    "section": str(section["name"]),
                    "point_index": index,
                    "x_m": float(x),
                    "y_m": float(y),
                    "z_m": float(z),
                }
            )
    return rows


def _cad_measurements(geometry: VehicleGeometry) -> dict[str, float]:
    bounds = geometry.mesh.bounds
    span = bounds[1] - bounds[0]
    p = geometry.metadata.get("parameters", {})
    return {
        "length_m": float(span[0]),
        "total_span_m": float(span[1]),
        "height_m": float(span[2]),
        "x_min_m": float(bounds[0, 0]),
        "x_max_m": float(bounds[1, 0]),
        "y_min_m": float(bounds[0, 1]),
        "y_max_m": float(bounds[1, 1]),
        "z_min_m": float(bounds[0, 2]),
        "z_max_m": float(bounds[1, 2]),
        "core_body_width_m": float(2.0 * p.get("body_half_width", 0.0)),
        "reference_area_m2": float(geometry.reference_area_m2),
        "wetted_area_m2": float(geometry.wetted_area_m2),
        "volume_m3": float(geometry.volume_m3),
        "payload_length_m": float(geometry.payload_box[0]),
        "payload_width_m": float(geometry.payload_box[1]),
        "payload_height_m": float(geometry.payload_box[2]),
    }


def _onshape_brep_spec(geometry: VehicleGeometry) -> dict[str, Any]:
    """Create a BREP-oriented loft/fin spec for Onshape STEP export.

    This deliberately uses analytic sections and thick fin solids rather than
    the sampled mesh so Onshape receives editable solid bodies instead of STL
    facets. Units are meters.
    """
    p = geometry.metadata.get("parameters", {})
    length = float(p["L_body"])
    half_width = float(p["body_half_width"])
    top = float(p["body_top_height"])
    belly = float(p["body_belly_depth"])
    rn = float(p["R_N"])
    theta = np.deg2rad(float(p["theta_N_deg"]))
    spherical_x1 = rn * (1.0 - np.cos(theta))
    nose_profile_code = int(float(p.get("nose_profile_code", 0.0)))
    x1 = max(spherical_x1, float(p.get("nose_station_frac", 0.0)) * length)
    r1 = rn * np.sin(theta)
    x2 = min(0.70 * length, spherical_x1 + float(p["dx1"]))
    x3 = min(0.94 * length, x2 + float(p["dx2"]))
    if nose_profile_code == 1:
        nose_points = _body_profile_points(
            float(x1),
            min(0.82 * r1, half_width),
            0.94 * r1,
            0.94 * r1,
            shoulder_y_scale=0.82,
            shoulder_z_scale=0.47,
            chine_y_scale=0.82,
            chine_z_scale=-0.47,
            belly_scale=1.0,
        )
    elif nose_profile_code == 2:
        nose_points = _body_profile_points(
            float(x1),
            min(0.92 * r1, half_width),
            0.92 * r1,
            0.70 * r1,
            shoulder_y_scale=0.92,
            shoulder_z_scale=0.08,
            chine_y_scale=0.62,
            chine_z_scale=-0.42,
            belly_scale=1.0,
        )
    else:
        nose_points = _body_profile_points(
            float(x1),
            min(r1, half_width),
            float(p["nose_top_scale"]) * r1,
            float(p["nose_belly_scale"]) * r1,
            shoulder_y_scale=float(p["nose_shoulder_scale"]),
            shoulder_z_scale=0.46,
            chine_y_scale=float(p["nose_chine_scale"]),
            chine_z_scale=-0.05,
            belly_scale=1.0,
        )

    sections = [
        {
            "name": "nose_blunt_start",
            "x_m": 0.004 * length,
            "points": _scaled_body_profile(0.004 * length, half_width, top, belly, 0.06, 0.06),
        },
        {
            "name": "nose_spherical_match",
            "x_m": float(x1),
            "points": nose_points,
        },
        {
            "name": "mid_body",
            "x_m": float(x2),
            "points": _body_profile_points(
                float(x2),
                half_width * float(p["forebody_width_scale"]),
                top * 0.96,
                belly * float(p["station2_belly_scale"]),
                shoulder_y_scale=float(p["station2_shoulder_y_scale"]),
                shoulder_z_scale=float(p["station2_shoulder_z_scale"]),
                chine_y_scale=float(p["station2_chine_y_scale"]),
                chine_z_scale=float(p["station2_chine_z_scale"]),
            ),
        },
        {
            "name": "aft_body",
            "x_m": float(x3),
            "points": _body_profile_points(
                float(x3),
                half_width * float(p["station3_width_scale"]),
                top * 0.64,
                belly * 0.70,
                shoulder_y_scale=float(p["station3_shoulder_y_scale"]),
                shoulder_z_scale=float(p["station3_shoulder_z_scale"]),
                chine_y_scale=1.0,
                chine_z_scale=-0.08,
            ),
        },
    ]
    for x_frac, y_scale, z_scale in (
        (0.74, 0.88, 0.90),
        (0.82, 0.80, 0.82),
        (0.90, 0.68, 0.70),
        (0.955, 0.54, 0.56),
        (0.985, 0.40, 0.42),
        (1.000, 0.30, 0.34),
    ):
        x = length * x_frac
        if x > sections[-1]["x_m"] + 1e-6:
            sections.append(
                {
                    "name": f"tail_{x_frac:.3f}",
                    "x_m": float(x),
                    "points": _scaled_body_profile(float(x), half_width, top, belly, y_scale, z_scale),
                }
            )
    sections = sorted(sections, key=lambda row: row["x_m"])

    root_y = 1.03 * half_width
    fin_root_z = 0.02 * top
    fin_x0 = 0.70 * length
    fin_x1 = 0.995 * length
    fin_span_y = min(0.070 * length, 1.20 * half_width) * float(p.get("fin_span_scale", 1.0))
    fin_rise_z = 0.62 * fin_span_y
    fin_thickness = max(0.022, 0.17 * fin_span_y)
    # Swept, clipped trapezoid in each fin center plane. The STEP exporter
    # thickens this around z so the part is a real solid, not a sheet.
    parts: list[dict[str, Any]] = []
    for sign, side in ((1.0, "right"), (-1.0, "left")):
        y_root = sign * root_y
        span_vector = [0.04 * length, sign * fin_span_y, fin_rise_z]
        root_le = [fin_x0, y_root, fin_root_z]
        root_te = [fin_x1, y_root * 0.58, fin_root_z - 0.012 * length]
        points = [
            root_le,
            root_te,
            [root_te[0] + span_vector[0], root_te[1] + span_vector[1], root_te[2] + span_vector[2]],
            [root_le[0] + span_vector[0], root_le[1] + span_vector[1], root_le[2] + span_vector[2]],
        ]
        parts.append(
            {
                "name": f"{side}_canted_aft_fin",
                "kind": "thick_polygon",
                "points": points,
                "thickness_m": fin_thickness,
                "thickness_axis": [0.0, 0.0, 1.0],
                "role": "canted aft fin for conceptual pitch/roll/yaw allocation; TODO: CFD/6DOF control allocation",
            }
        )

    tail_fin_x0 = 0.76 * length
    tail_fin_x1 = 0.995 * length
    tail_fin_height = min(0.075 * length, 1.25 * half_width)
    for sign_z, name in ((1.0, "dorsal_tail_fin"), (-1.0, "ventral_tail_fin")):
        root_z = 0.88 * top if sign_z > 0.0 else -0.88 * belly
        points = [
            [tail_fin_x0, 0.0, root_z],
            [tail_fin_x1, 0.0, root_z + sign_z * 0.016 * length],
            [tail_fin_x1 - 0.10 * length, 0.0, root_z + sign_z * tail_fin_height],
            [tail_fin_x0 + 0.06 * length, 0.0, root_z + sign_z * 0.52 * tail_fin_height],
        ]
        parts.append(
            {
                "name": name,
                "kind": "thick_polygon",
                "points": points,
                "thickness_m": max(0.018, 0.10 * half_width),
                "thickness_axis": [0.0, 1.0, 0.0],
                "role": "aft vertical stabilizer for conceptual yaw damping/allocation; TODO: CFD/6DOF validation",
            }
        )

    flap_x0 = 0.76 * length
    flap_x1 = 0.995 * length
    flap_half_width = 0.42 * half_width
    flap_z = -0.88 * belly
    parts.append(
        {
            "name": "center_body_flap",
            "kind": "thick_polygon",
            "points": [
                [flap_x0, -0.34 * flap_half_width, flap_z],
                [flap_x0, 0.34 * flap_half_width, flap_z],
                [flap_x1, flap_half_width, 0.72 * flap_z],
                [flap_x1, -flap_half_width, 0.72 * flap_z],
            ],
            "thickness_m": max(0.018, 0.08 * half_width),
            "thickness_axis": [0.0, 0.0, 1.0],
            "role": "aft windward pitch trim flap; TODO: hinge/actuator design",
        }
    )

    return {
        "schema": "astreia_mrv_onshape_brep_v1",
        "units": "m",
        "note": "Analytic BREP-oriented sections for STEP export; not an arbitrary precision targeting model.",
        "measurements": _cad_measurements(geometry),
        "source_parameters": dict(
            sorted(
                (str(key), float(value) if isinstance(value, (int, float, np.integer, np.floating)) else str(value))
                for key, value in p.items()
            )
        ),
        "normalized_rx": dict(sorted((str(key), float(value)) for key, value in geometry.metadata.get("normalized_rx", {}).items())),
        "body": {
            "name": "astreia_mrv_body",
            "sections": sections,
        },
        "parts": parts,
        "metadata": {
            "reference_length_m": geometry.reference_length_m,
            "payload_box_m": geometry.payload_box,
            "recovery": geometry.metadata.get("recovery", {}),
        },
    }


def export_onshape_brep_spec(geometry: VehicleGeometry, path: str | Path) -> None:
    path = Path(path)
    spec = _onshape_brep_spec(geometry)
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")


def export_onshape_parameter_csv(geometry: VehicleGeometry, path: str | Path) -> None:
    """Write named dimensions for Onshape variable studios or manual edits."""
    path = Path(path)
    p = geometry.metadata.get("parameters", {})
    rx = geometry.metadata.get("normalized_rx", {})
    rows: list[dict[str, str | float]] = []
    descriptions = {
        "L_body": "Overall generated body length.",
        "R_N": "Nose radius used by heating and blunt-ogive geometry.",
        "theta_N_deg": "Nose spherical-match angle.",
        "body_half_width": "Core half-width before integrated fin extension.",
        "body_top_height": "Core upper body height from centerline.",
        "body_belly_depth": "Core lower body depth from centerline.",
        "fin_span_scale": "Normalized scale applied to canted aft fin span.",
        "x_cg_frac": "Conceptual center-of-gravity fraction of body length.",
    }
    for name, value in sorted(p.items()):
        if isinstance(value, (int, float, np.integer, np.floating)):
            unit = "deg" if name.endswith("_deg") else "m" if name.startswith(("L_", "R_", "body_", "payload_", "dx")) else "dimensionless"
            rows.append(
                {
                    "name": f"mrv_{name}",
                    "value": float(value),
                    "unit": unit,
                    "source": "physical_parameter",
                    "description": descriptions.get(name, "Generated from configs/mrv3.yaml normalized rx mapping."),
                }
            )
    for name, value in sorted(rx.items()):
        rows.append(
            {
                "name": f"rx_{name}",
                "value": float(value),
                "unit": "dimensionless",
                "source": "normalized_rx",
                "description": "Original normalized design variable from config.",
            }
        )
    for name, value in sorted(_cad_measurements(geometry).items()):
        rows.append(
            {
                "name": f"measure_{name}",
                "value": float(value),
                "unit": "m2" if name.endswith("_m2") else "m3" if name.endswith("_m3") else "m",
                "source": "generated_measurement",
                "description": "Measured from generated geometry mesh bounds or metrics.",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "value", "unit", "source", "description"])
        writer.writeheader()
        writer.writerows(rows)


def export_onshape_loft_sections_csv(geometry: VehicleGeometry, path: str | Path) -> None:
    spec = _onshape_brep_spec(geometry)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "point_index", "x_m", "y_m", "z_m"])
        writer.writeheader()
        writer.writerows(_section_point_rows(spec["body"]["sections"]))


def export_onshape_feature_variables(geometry: VehicleGeometry, path: str | Path) -> None:
    """Write a small FeatureScript helper that creates named variables in Onshape."""
    p = geometry.metadata.get("parameters", {})
    measurements = _cad_measurements(geometry)
    scalar_rows = [
        ("mrv_L_body", float(p["L_body"]), "meter"),
        ("mrv_R_N", float(p["R_N"]), "meter"),
        ("mrv_theta_N", float(p["theta_N_deg"]), "degree"),
        ("mrv_body_half_width", float(p["body_half_width"]), "meter"),
        ("mrv_body_top_height", float(p["body_top_height"]), "meter"),
        ("mrv_body_belly_depth", float(p["body_belly_depth"]), "meter"),
        ("mrv_core_body_width", measurements["core_body_width_m"], "meter"),
        ("mrv_total_span", measurements["total_span_m"], "meter"),
        ("mrv_height", measurements["height_m"], "meter"),
        ("mrv_volume", measurements["volume_m3"], "meter ^ 3"),
        ("mrv_wetted_area", measurements["wetted_area_m2"], "meter ^ 2"),
        ("mrv_payload_l", measurements["payload_length_m"], "meter"),
        ("mrv_payload_w", measurements["payload_width_m"], "meter"),
        ("mrv_payload_h", measurements["payload_height_m"], "meter"),
    ]
    lines = [
        "FeatureScript 2521;",
        'import(path : "onshape/std/common.fs", version : "2521.0");',
        "",
        'annotation { "Feature Type Name" : "Astreia MRV Parameters" }',
        "export const astreiaMRVParameters = defineFeature(function(context is Context, id is Id, definition is map)",
        "    precondition",
        "    {",
        "    }",
        "    {",
        "        // Generated from the same config/THOR pipeline as the MRV STEP handoff.",
        "        // Use these variables while recreating/editing sketches and loft sections in Onshape.",
    ]
    for name, value, unit in scalar_rows:
        lines.append(f'        setVariable(context, "{name}", {_fmt_float(value)} * {unit});')
    lines.extend(
        [
            "    });",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def export_onshape_readme(path: str | Path) -> None:
    Path(path).write_text(
        "\n".join(
            [
                "# Astreia-MRV Onshape Handoff",
                "",
                "Use `onshape_step/geometry_onshape.step` for a true solid/BREP import in Onshape.",
                "The STL/OBJ files are kept only for visual mesh comparison.",
                "",
                "The STEP exporter builds a faceted BREP solid directly from the generated",
                "watertight MRV OML. It should match the plotted geometry; it is not the",
                "older simplified analytic loft. STEP does not preserve the Python/THOR",
                "feature history. To make reconstruction/editing easier, this folder also includes:",
                "- `onshape_parameters.csv`: named dimensions, normalized rx values, and generated measurements",
                "- `onshape_loft_sections.csv`: x/y/z points for each analytic loft section",
                "- `onshape_variables.fs`: FeatureScript helper that creates common MRV variables",
                "- `onshape_brep_spec.json`: full analytic body/fin reconstruction spec",
                "",
                "Expected STEP body:",
                "- `astreia_mrv_body`: one continuous faceted BREP solid generated from the MRV mesh",
                "",
                "Units are meters. The conceptual control surfaces are placeholders",
                "for CFD, FEA, hinge, actuator, and 6DOF control-allocation studies.",
                "Landing remains terminal recovery into an authorized recovery zone.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def export_openscad_polyhedron(geometry: VehicleGeometry, path: str | Path) -> None:
    """Export the generated mesh as an OpenSCAD polyhedron.

    OpenSCAD is the v0.1 CAD handoff format. The Python geometry remains the
    parametric source of truth; OpenSCAD receives the sampled solid/plate mesh
    as a native `polyhedron(...)` that can be inspected, rendered, or exported
    from OpenSCAD.
    """
    path = Path(path)
    mesh = geometry.mesh
    family = geometry.metadata.get("family", "vehicle")
    lines = [
        "// Astreia-MRV OpenSCAD export",
        "// Units: meters",
        "// Generated from parametric Python geometry as an OpenSCAD polyhedron.",
        "// Landing accuracy is represented as terminal recovery into an authorized recovery zone.",
        f"vehicle_family = \"{family}\";",
        "",
        "module astreia_vehicle() {",
        "  polyhedron(",
        "    points=[",
    ]
    for vertex in mesh.vertices:
        lines.append(
            "      ["
            + ", ".join(_fmt_float(v) for v in vertex)
            + "],"
        )
    lines.extend(["    ],", "    faces=["])
    for face in mesh.faces:
        lines.append("      [" + ", ".join(str(int(i)) for i in face) + "],")
    lines.extend(
        [
            "    ],",
            "    convexity=10",
            "  );",
            "}",
            "",
            "astreia_vehicle();",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def export_editable_openscad_wrapper(geometry: VehicleGeometry, path: str | Path) -> None:
    """Write a hand-editable OpenSCAD wrapper around the generated mesh.

    The sampled MRV OML remains in `geometry.scad`; this wrapper is the file to
    open when sketching panels, adding simple solids, or creating projected DXF
    views from OpenSCAD.
    """
    path = Path(path)
    bounds = geometry.mesh.bounds
    span = bounds[1] - bounds[0]
    payload_l, payload_w, payload_h = geometry.payload_box
    params = geometry.metadata.get("parameters", {})
    cg_x = float(params.get("x_cg_frac", 0.52)) * geometry.reference_length_m
    lines = [
        "// Astreia-MRV editable OpenSCAD handoff",
        "// Units: meters. Open this file for sketching/modification.",
        "// The underlying sampled OML is in geometry.scad and is imported as astreia_vehicle().",
        "// Keep recovery/landing concepts as terminal recovery into an authorized recovery zone.",
        "",
        "use <geometry.scad>;",
        "",
        f"L = {_fmt_float(span[0])};",
        f"B = {_fmt_float(span[1])};",
        f"H = {_fmt_float(span[2])};",
        f"x_min = {_fmt_float(bounds[0, 0])};",
        f"x_max = {_fmt_float(bounds[1, 0])};",
        f"y_min = {_fmt_float(bounds[0, 1])};",
        f"y_max = {_fmt_float(bounds[1, 1])};",
        f"z_min = {_fmt_float(bounds[0, 2])};",
        f"z_max = {_fmt_float(bounds[1, 2])};",
        f"cg_x = {_fmt_float(cg_x)};",
        f"payload_l = {_fmt_float(payload_l)};",
        f"payload_w = {_fmt_float(payload_w)};",
        f"payload_h = {_fmt_float(payload_h)};",
        "",
        "// Change this to: \"solid\", \"top_projection\", \"side_projection\", \"aft_projection\", or \"stations\".",
        "mode = \"solid\";",
        "show_payload = true;",
        "show_reference_planes = false;",
        "",
        "module base_oml() {",
        "  astreia_vehicle();",
        "}",
        "",
        "module payload_box(alpha=0.25) {",
        "  color([0.25, 0.62, 1.0, alpha])",
        "    translate([0.52 * L, 0, 0])",
        "      cube([payload_l, payload_w, payload_h], center=true);",
        "}",
        "",
        "module cg_marker() {",
        "  color([1.0, 0.30, 0.18, 0.85])",
        "    translate([cg_x, 0, 0]) sphere(r=0.018, $fn=32);",
        "}",
        "",
        "module reference_planes() {",
        "  // Thin axes only; filled planes obscure the vehicle mesh in top views.",
        "  color([0.2, 0.55, 1.0, 0.45]) translate([L/2, 0, 0]) cube([L, 0.003, 0.003], center=true);",
        "  color([0.9, 0.9, 0.9, 0.40]) translate([L/2, 0, 0]) cube([0.003, B * 1.15, 0.003], center=true);",
        "  color([1.0, 0.30, 0.18, 0.45]) translate([L/2, 0, 0]) cube([0.003, 0.003, H * 1.25], center=true);",
        "}",
        "",
        "// Put additive sketches/features here.",
        "module user_additions() {",
        "  // Example: dorsal antenna/fairing placeholder",
        "  // color([0.1, 0.1, 0.1]) translate([1.15, 0, z_max + 0.005]) cube([0.20, 0.03, 0.01], center=true);",
        "}",
        "",
        "// Put subtractive cuts here.",
        "module user_cuts() {",
        "  // Example panel/cavity cut:",
        "  // translate([1.55, 0, -0.02]) cube([0.25, 0.18, 0.04], center=true);",
        "}",
        "",
        "module working_solid() {",
        "  difference() {",
        "    union() {",
        "      color([0.30, 0.35, 0.39]) base_oml();",
        "      user_additions();",
        "      if (show_payload) payload_box();",
        "      cg_marker();",
        "      if (show_reference_planes) reference_planes();",
        "    }",
        "    user_cuts();",
        "  }",
        "}",
        "",
        "// Projection modes are useful for Export as DXF from OpenSCAD.",
        "module top_projection() { projection(cut=false) base_oml(); }",
        "module side_projection() { projection(cut=false) rotate([90, 0, 0]) base_oml(); }",
        "module aft_projection() { projection(cut=false) rotate([0, 90, 0]) base_oml(); }",
        "",
        "module station_planes() {",
        "  working_solid();",
        "  for (x = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75])",
        "    color([1, 1, 0, 0.18]) translate([x, 0, 0]) cube([0.004, B * 1.25, H * 1.35], center=true);",
        "}",
        "",
        "if (mode == \"top_projection\") top_projection();",
        "else if (mode == \"side_projection\") side_projection();",
        "else if (mode == \"aft_projection\") aft_projection();",
        "else if (mode == \"stations\") station_planes();",
        "else working_solid();",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _face_part_labels(geometry: VehicleGeometry) -> list[str]:
    mesh = geometry.mesh
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    length = max(geometry.reference_length_m, 1e-9)
    labels = []
    for zone, centroid in zip(mesh.face_zone, centroids):
        x_frac = float(centroid[0] / length)
        y = float(centroid[1])
        if zone == "body_flap":
            labels.append("body_flap")
        elif zone == "elevon":
            labels.append("right_elevon" if y >= 0.0 else "left_elevon")
        elif zone == "strake":
            labels.append("right_strake" if y >= 0.0 else "left_strake")
        elif zone == "dorsal_fin":
            labels.append("dorsal_tail_fin")
        elif zone == "ventral_fin":
            labels.append("ventral_tail_fin")
        elif zone == "nose_cap" or x_frac < 0.18:
            labels.append("nose_forebody")
        elif zone == "aft_closeout" or x_frac > 0.94:
            labels.append("tail_closeout")
        elif x_frac < 0.55:
            labels.append("mid_body")
        else:
            labels.append("aft_body")
    return labels


def _surface_subset(surface: SurfaceMesh, mask: np.ndarray, label: str) -> SurfaceMesh | None:
    faces = surface.faces[mask]
    if len(faces) == 0:
        return None
    used = np.unique(faces)
    remap = {int(old): new for new, old in enumerate(used)}
    remapped_faces = np.vectorize(remap.__getitem__)(faces)
    return SurfaceMesh(surface.vertices[used], remapped_faces, [label] * len(remapped_faces))


def _part_colors() -> dict[str, tuple[float, float, float]]:
    return {
        "nose_forebody": (0.36, 0.40, 0.43),
        "mid_body": (0.31, 0.36, 0.40),
        "aft_body": (0.27, 0.32, 0.36),
        "tail_closeout": (0.20, 0.24, 0.28),
        "body_flap": (0.42, 0.46, 0.49),
        "left_strake": (0.25, 0.30, 0.34),
        "right_strake": (0.25, 0.30, 0.34),
        "left_elevon": (0.18, 0.22, 0.26),
        "right_elevon": (0.18, 0.22, 0.26),
        "dorsal_tail_fin": (0.16, 0.19, 0.23),
        "ventral_tail_fin": (0.16, 0.19, 0.23),
    }


def export_grouped_obj(geometry: VehicleGeometry, path: str | Path) -> None:
    path = Path(path)
    labels = _face_part_labels(geometry)
    colors = _part_colors()
    groups = [group for group in colors if group in set(labels)]
    mtl_path = path.with_suffix(".mtl")

    mtl_lines = [
        "# Astreia-MRV grouped OBJ materials",
        "# Units: meters",
    ]
    for name in groups:
        r, g, b = colors[name]
        mtl_lines.extend(
            [
                f"newmtl {name}",
                f"Kd {_fmt_float(r)} {_fmt_float(g)} {_fmt_float(b)}",
                "Ka 0.05 0.05 0.05",
                "Ks 0.18 0.18 0.18",
                "Ns 24",
                "",
            ]
        )
    mtl_path.write_text("\n".join(mtl_lines), encoding="utf-8")

    lines = [
        "# Astreia-MRV grouped OBJ",
        "# Units: meters",
        "# Groups are selectable CAD/mesh objects; full watertight solid is geometry.stl/scad.",
        f"mtllib {mtl_path.name}",
        "",
    ]
    for vertex in geometry.mesh.vertices:
        lines.append("v " + " ".join(_fmt_float(v) for v in vertex))

    label_array = np.asarray(labels, dtype=object)
    for group in groups:
        lines.extend(["", f"o {group}", f"g {group}", f"usemtl {group}"])
        for face in geometry.mesh.faces[label_array == group]:
            # OBJ indices are 1-based.
            lines.append("f " + " ".join(str(int(idx) + 1) for idx in face))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_part_meshes(geometry: VehicleGeometry, out_dir: str | Path) -> list[str]:
    parts_dir = Path(out_dir) / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(_face_part_labels(geometry), dtype=object)
    part_names = [name for name in _part_colors() if name in set(labels.tolist())]
    manifest = []
    for name in part_names:
        subset = _surface_subset(geometry.mesh, labels == name, name)
        if subset is None:
            continue
        export_surface(subset, parts_dir / f"{name}.stl")
        export_surface(subset, parts_dir / f"{name}.obj")
        manifest.append(
            {
                "part": name,
                "faces": int(len(subset.faces)),
                "stl": str(parts_dir / f"{name}.stl"),
                "obj": str(parts_dir / f"{name}.obj"),
                "note": "Exterior face subset for CAD selection/sketching; full watertight OML remains geometry.stl/scad.",
            }
        )
    (parts_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return part_names


def export_parts_openscad_assembly(part_names: list[str], path: str | Path) -> None:
    path = Path(path)
    colors = _part_colors()
    lines = [
        "// Astreia-MRV part assembly for CAD-style selection",
        "// Units: meters. Individual imported STL files live in ./parts/.",
        "// These are separate selectable mesh objects; geometry.scad/stl remains the single watertight OML.",
        "",
        "show_body = true;",
        "show_controls = true;",
        "",
    ]
    for name in part_names:
        r, g, b = colors[name]
        lines.extend(
            [
                f"module {name}() {{",
                f"  color([{_fmt_float(r)}, {_fmt_float(g)}, {_fmt_float(b)}, 1.0]) import(\"parts/{name}.stl\");",
                "}",
                "",
            ]
        )
    body_names = [name for name in ("nose_forebody", "mid_body", "aft_body", "tail_closeout") if name in part_names]
    control_names = [
        name
        for name in (
            "body_flap",
            "left_strake",
            "right_strake",
            "left_elevon",
            "right_elevon",
            "dorsal_tail_fin",
            "ventral_tail_fin",
        )
        if name in part_names
    ]
    lines.append("if (show_body) {")
    lines.extend(f"  {name}();" for name in body_names)
    lines.extend(["}", "", "if (show_controls) {"])
    lines.extend(f"  {name}();" for name in control_names)
    lines.extend(["}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def export_geometry(geometry: VehicleGeometry, out_dir: str | Path, extra_metadata: dict[str, Any] | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    export_surface(geometry.mesh, out / "geometry.obj")
    export_surface(geometry.mesh, out / "geometry.stl")
    export_openscad_polyhedron(geometry, out / "geometry.scad")
    export_editable_openscad_wrapper(geometry, out / "geometry_editable.scad")
    export_grouped_obj(geometry, out / "geometry_grouped.obj")
    part_names = export_part_meshes(geometry, out)
    export_parts_openscad_assembly(part_names, out / "geometry_parts.scad")
    export_onshape_brep_spec(geometry, out / "onshape_brep_spec.json")
    export_onshape_parameter_csv(geometry, out / "onshape_parameters.csv")
    export_onshape_loft_sections_csv(geometry, out / "onshape_loft_sections.csv")
    export_onshape_feature_variables(geometry, out / "onshape_variables.fs")
    export_onshape_readme(out / "ONSHAPE_README.md")
    metadata = {
        **geometry.metadata,
        "cad_backend": "OpenSCAD mesh handoff plus optional Onshape BREP STEP spec",
        "onshape_brep_spec": str(out / "onshape_brep_spec.json"),
        "onshape_parameters_csv": str(out / "onshape_parameters.csv"),
        "onshape_loft_sections_csv": str(out / "onshape_loft_sections.csv"),
        "onshape_variables_featurescript": str(out / "onshape_variables.fs"),
        "reference_area_m2": geometry.reference_area_m2,
        "reference_length_m": geometry.reference_length_m,
        "volume_m3": geometry.volume_m3,
        "wetted_area_m2": geometry.wetted_area_m2,
        "payload_box": geometry.payload_box,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    with (out / "geometry.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with (out / "parameters.json").open("w", encoding="utf-8") as f:
        json.dump(geometry.metadata.get("parameters", {}), f, indent=2)


def openscad_available() -> bool:
    from shutil import which

    return which("openscad") is not None
