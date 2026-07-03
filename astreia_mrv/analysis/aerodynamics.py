from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from astreia_mrv.geometry.mesh import VehicleGeometry


def cp_stag_modified_newtonian(mach: float, gamma: float = 1.4) -> float:
    """Approximate stagnation pressure coefficient cap for modified Newtonian aero."""
    if mach <= 1.0:
        return 1.0
    # A bounded conceptual approximation; high-Mach limit approaches about 2.
    return min(2.0, 2.0 - 0.15 / max(mach * mach, 1.0))


def freestream_hat(alpha_deg: float, beta_deg: float = 0.0) -> np.ndarray:
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    # Body axes: +x nose-to-tail, +z upward. Positive alpha exposes the
    # windward lower surface, so the freestream z component is negative.
    v = np.array([math.cos(alpha) * math.cos(beta), math.sin(beta), -math.sin(alpha) * math.cos(beta)])
    return v / np.linalg.norm(v)


def aerodynamic_coefficients(
    geometry: VehicleGeometry,
    mach: float,
    alpha_deg: float,
    beta_deg: float = 0.0,
    q_dyn_pa: float = 1000.0,
    cg: np.ndarray | None = None,
) -> dict[str, float]:
    mesh = geometry.mesh
    vertices = mesh.vertices
    faces = mesh.faces
    normals = mesh.face_normals
    if normals is None:
        mesh.update_normals()
        normals = mesh.face_normals
    tri = vertices[faces]
    centroids = tri.mean(axis=1)
    areas = mesh.face_areas
    vhat = freestream_hat(alpha_deg, beta_deg)
    cp_max = cp_stag_modified_newtonian(mach)
    cp_leeward = -0.02 / max(mach, 1.0)
    force = np.zeros(3)
    moment = np.zeros(3)
    cg_vec = cg
    if cg_vec is None:
        p = geometry.metadata.get("parameters", {})
        cg_vec = np.array([p.get("x_cg_frac", 0.52) * geometry.reference_length_m, 0.0, 0.0])
    for n, area, ctr in zip(normals, areas, centroids):
        sin_theta = -float(np.dot(vhat, n))
        cp = cp_max * sin_theta**2 if sin_theta > 0.0 else cp_leeward
        f = -cp * q_dyn_pa * area * n
        force += f
        moment += np.cross(ctr - cg_vec, f)
    s_ref = max(geometry.reference_area_m2, 1e-9)
    l_ref = max(geometry.reference_length_m, 1e-9)
    drag_axis = vhat
    side_axis = np.array([0.0, 1.0, 0.0])
    lift_axis = np.cross(side_axis, drag_axis)
    lift_axis /= max(np.linalg.norm(lift_axis), 1e-9)
    normal_force_z = float(force[2])
    x_cp_m = float(cg_vec[0] - moment[1] / normal_force_z) if abs(normal_force_z) > 1e-9 else float("nan")
    return {
        "alpha_deg": float(alpha_deg),
        "mach": float(mach),
        "CD": float(np.dot(force, drag_axis) / (q_dyn_pa * s_ref)),
        "CL": float(np.dot(force, lift_axis) / (q_dyn_pa * s_ref)),
        "CY": float(np.dot(force, side_axis) / (q_dyn_pa * s_ref)),
        "Cl": float(moment[0] / (q_dyn_pa * s_ref * l_ref)),
        "Cm": float(moment[1] / (q_dyn_pa * s_ref * l_ref)),
        "Cn": float(moment[2] / (q_dyn_pa * s_ref * l_ref)),
        "x_cp_m": x_cp_m,
    }


def alpha_sweep(
    geometry: VehicleGeometry,
    mach: float = 20.0,
    alpha_deg: np.ndarray | None = None,
    q_dyn_pa: float = 1000.0,
) -> list[dict[str, float]]:
    if alpha_deg is None:
        alpha_deg = np.linspace(0.0, 40.0, 17)
    return [aerodynamic_coefficients(geometry, mach, float(a), q_dyn_pa=q_dyn_pa) for a in alpha_deg]


def write_polar_csv(rows: list[dict[str, float]], path: str | Path) -> None:
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
