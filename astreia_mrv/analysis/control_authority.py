from __future__ import annotations

import math
from typing import Any

import numpy as np

from astreia_mrv.geometry.mesh import SurfaceMesh, VehicleGeometry


def surface_stats(surface: SurfaceMesh) -> dict[str, float]:
    """Area and centroid for one generated control-surface subset."""
    if len(surface.faces) == 0:
        return {
            "area_m2": 0.0,
            "centroid_x_m": 0.0,
            "centroid_y_m": 0.0,
            "centroid_z_m": 0.0,
        }
    areas = surface.face_areas
    tri = surface.vertices[surface.faces]
    centroids = tri.mean(axis=1)
    area = float(areas.sum())
    centroid = np.average(centroids, axis=0, weights=np.maximum(areas, 1e-12))
    return {
        "area_m2": area,
        "centroid_x_m": float(centroid[0]),
        "centroid_y_m": float(centroid[1]),
        "centroid_z_m": float(centroid[2]),
    }


def control_surface_table(geometry: VehicleGeometry) -> list[dict[str, float | str]]:
    params = geometry.metadata.get("parameters", {})
    core_half_width = float(params.get("body_half_width", 0.0))
    rows: list[dict[str, float | str]] = []
    for name, surface in sorted(geometry.control_surfaces.items()):
        bounds = surface.bounds
        stats = surface_stats(surface)
        rows.append(
            {
                "surface": name,
                "zone": surface.face_zone[0] if surface.face_zone else "",
                "faces": float(len(surface.faces)),
                **stats,
                "x_min_m": float(bounds[0, 0]),
                "x_max_m": float(bounds[1, 0]),
                "y_min_m": float(bounds[0, 1]),
                "y_max_m": float(bounds[1, 1]),
                "z_min_m": float(bounds[0, 2]),
                "z_max_m": float(bounds[1, 2]),
                "outboard_extension_m": float(
                    max(0.0, max(abs(bounds[0, 1]), abs(bounds[1, 1])) - core_half_width)
                ),
            }
        )
    return rows


def estimate_control_authority(
    geometry: VehicleGeometry,
    cm_required: float = 0.05,
    cl_required: float = 0.03,
    cn_required: float = 0.02,
    deflection_limit_deg: float = 20.0,
    yaw_rcs_assist_coeff: float = 0.016,
) -> list[dict[str, Any]]:
    """Conceptual static control authority from generated control-surface areas.

    This is a low-order sizing check, not a replacement for CFD, wind-tunnel
    derivatives, actuator sizing, or 6DOF trim simulation. It is intentionally
    conservative for yaw because horizontal/canted aft fins mostly provide
    pitch and roll; yaw should remain an RCS or canted-vertical-surface design
    item until later analysis proves otherwise.
    """
    params = geometry.metadata.get("parameters", {})
    cg_x = float(params.get("x_cg_frac", 0.52)) * geometry.reference_length_m
    s_ref = max(geometry.reference_area_m2, 1e-9)
    l_ref = max(geometry.reference_length_m, 1e-9)
    span_ref = max(float(geometry.mesh.bounds[1, 1] - geometry.mesh.bounds[0, 1]), 1e-9)
    delta_rad = math.radians(deflection_limit_deg)

    stats = {name: surface_stats(surface) for name, surface in geometry.control_surfaces.items()}

    body = stats.get("body_flap", {})
    right = stats.get("right_elevon", {})
    left = stats.get("left_elevon", {})
    right_strake = stats.get("right_strake", {})
    left_strake = stats.get("left_strake", {})
    dorsal_fin = stats.get("dorsal_fin", {})
    ventral_fin = stats.get("ventral_fin", {})

    elevon_area = float(right.get("area_m2", 0.0)) + float(left.get("area_m2", 0.0))
    elevon_x_arm = _weighted_abs_arm_x([right, left], cg_x)
    strake_area = float(right_strake.get("area_m2", 0.0)) + float(left_strake.get("area_m2", 0.0))
    strake_x_arm = _weighted_abs_arm_x([right_strake, left_strake], cg_x)
    flap_x_arm = abs(float(body.get("centroid_x_m", cg_x)) - cg_x)
    flap_area = float(body.get("area_m2", 0.0))
    # TODO: replace these canted-fin effectiveness factors with CFD/6DOF
    # derivatives. They are conservative conceptual credits for aft fin panels
    # that are shaped to carry deflectable/effective trailing regions.
    pitch_derivative = (
        1.35
        * (flap_area * flap_x_arm + 0.90 * elevon_area * elevon_x_arm + 0.18 * strake_area * strake_x_arm)
        / (s_ref * l_ref)
    )
    pitch_available = pitch_derivative * delta_rad

    roll_y_arm_area = _area_y_arm([right, left]) + 0.55 * _area_y_arm([right_strake, left_strake])
    roll_derivative = 0.90 * roll_y_arm_area / (s_ref * span_ref)
    roll_available = roll_derivative * delta_rad

    yaw_y_arm_area = _area_y_arm([right_strake, left_strake, right, left])
    vertical_tail_area = float(dorsal_fin.get("area_m2", 0.0)) + float(ventral_fin.get("area_m2", 0.0))
    vertical_tail_x_arm = _weighted_abs_arm_x([dorsal_fin, ventral_fin], cg_x)
    yaw_derivative = 0.12 * yaw_y_arm_area / (s_ref * span_ref) + 0.40 * (
        vertical_tail_area * vertical_tail_x_arm
    ) / (s_ref * l_ref)
    yaw_aero_available = yaw_derivative * delta_rad
    yaw_available = yaw_aero_available + max(0.0, yaw_rcs_assist_coeff)

    return [
        _authority_row(
            axis="pitch",
            mode="symmetric body flap + elevons",
            derivative_per_rad=pitch_derivative,
            available_coeff=pitch_available,
            required_coeff=cm_required,
            aero_only_coeff=pitch_available,
            assist_coeff=0.0,
            assist_mode="none",
            note="Conceptual normal-force moment only; TODO: CFD/6DOF trim.",
        ),
        _authority_row(
            axis="roll",
            mode="differential elevons",
            derivative_per_rad=roll_derivative,
            available_coeff=roll_available,
            required_coeff=cl_required,
            aero_only_coeff=roll_available,
            assist_coeff=0.0,
            assist_mode="none",
            note="Conceptual differential normal-force roll check.",
        ),
        _authority_row(
            axis="yaw",
            mode="canted-fin residual + RCS assist",
            derivative_per_rad=yaw_derivative,
            available_coeff=yaw_available,
            required_coeff=cn_required,
            aero_only_coeff=yaw_aero_available,
            assist_coeff=max(0.0, yaw_rcs_assist_coeff),
            assist_mode="RCS placeholder",
            note="Yaw is not credited to fins alone; TODO: replace RCS placeholder with 6DOF control allocation.",
        ),
    ]


def _weighted_abs_arm_x(stats: list[dict[str, float]], cg_x: float) -> float:
    area = sum(float(row.get("area_m2", 0.0)) for row in stats)
    if area <= 0.0:
        return 0.0
    return sum(
        float(row.get("area_m2", 0.0)) * abs(float(row.get("centroid_x_m", cg_x)) - cg_x)
        for row in stats
    ) / area


def _area_y_arm(stats: list[dict[str, float]]) -> float:
    return sum(
        float(row.get("area_m2", 0.0)) * abs(float(row.get("centroid_y_m", 0.0)))
        for row in stats
    )


def _authority_row(
    axis: str,
    mode: str,
    derivative_per_rad: float,
    available_coeff: float,
    required_coeff: float,
    aero_only_coeff: float,
    assist_coeff: float,
    assist_mode: str,
    note: str,
) -> dict[str, Any]:
    margin = float(available_coeff - required_coeff)
    return {
        "axis": axis,
        "mode": mode,
        "derivative_per_rad": float(derivative_per_rad),
        "aero_only_coeff": float(aero_only_coeff),
        "assist_coeff": float(assist_coeff),
        "assist_mode": assist_mode,
        "available_coeff": float(available_coeff),
        "required_coeff": float(required_coeff),
        "margin": margin,
        "passes": bool(margin >= 0.0),
        "status": "pass" if margin >= 0.0 else "needs revision",
        "note": note,
    }
