from __future__ import annotations

import math

import numpy as np

from astreia_mrv.config import VehicleDesign
from astreia_mrv.geometry.mesh import SurfaceMesh, VehicleGeometry, make_trimesh, validate_mesh
from astreia_mrv.geometry.primitives import revolve_profile
from astreia_mrv.parameters import PhysicalParameters, map_capsule_parameters


def capsule_profile(params: PhysicalParameters, n: int = 96) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    p = params.values
    rn = p["R_N"]
    rs = p["R_S"]
    rm = min(p["R_m"], 0.92 * rn)
    theta_c = math.radians(p["theta_c_deg"])
    lc = p["L_c"]

    theta_max = math.asin(np.clip((rm - rs) / max(rn - rs, 1e-6), 0.1, 0.95))
    x1 = rn * (1.0 - math.cos(theta_max))
    r1 = rn * math.sin(theta_max)

    shoulder_theta = np.linspace(theta_max, theta_c, max(8, n // 8))
    shoulder_x = x1 + rs * (np.sin(theta_max) - np.sin(shoulder_theta))
    shoulder_r = rm - rs * (1.0 - np.cos(shoulder_theta))
    shoulder_x -= shoulder_x[0] - x1
    shoulder_r -= shoulder_r[0] - r1

    cone_x = np.linspace(shoulder_x[-1], shoulder_x[-1] + lc, max(12, n // 6))
    cone_r = np.linspace(shoulder_r[-1], max(0.18 * rm, shoulder_r[-1] - lc * math.tan(theta_c)), len(cone_x))

    rear_radius = max(0.15 * rm, cone_r[-1])
    rear_len = max(0.18 * p["L_body"], rear_radius * 0.7)
    phi = np.linspace(0.0, math.pi / 2.0, max(12, n // 8))
    rear_x = cone_x[-1] + rear_len * np.sin(phi)
    rear_r = cone_r[-1] * np.cos(phi)

    nose_theta = np.linspace(0.0, theta_max, max(16, n // 5))
    nose_x = rn * (1.0 - np.cos(nose_theta))
    nose_r = rn * np.sin(nose_theta)

    x = np.concatenate([nose_x, shoulder_x[1:], cone_x[1:], rear_x[1:]])
    r = np.concatenate([nose_r, shoulder_r[1:], cone_r[1:], rear_r[1:]])
    x *= p["L_body"] / x[-1]
    metadata = {
        "R_sp1": rn,
        "R_t1": rs,
        "R_t2": rm - rs,
        "theta_sp1_max_rad": float(theta_max),
        "theta_c_rad": float(theta_c),
        "L_total_unscaled_m": float(x[-1]),
    }
    return x, r, metadata


def generate_capsule(design: VehicleDesign, n_profile: int = 120, n_theta: int = 96) -> VehicleGeometry:
    params = map_capsule_parameters(design)
    x, r, profile_meta = capsule_profile(params, n=n_profile)
    surface = revolve_profile(x, r, n_theta=n_theta, zone="capsule_aeroshell")
    tri = make_trimesh(surface)
    bounds = np.asarray(tri.bounds)
    reference_area = math.pi * (0.5 * (bounds[1, 1] - bounds[0, 1])) ** 2
    payload_box = (design.payload.length_m, design.payload.width_m, design.payload.height_m)
    metadata = {
        "family": "capsule",
        "parameters": params.values,
        "normalized_rx": params.normalized,
        "profile": profile_meta,
        "mesh_quality": validate_mesh(surface),
    }
    return VehicleGeometry(
        mesh=surface,
        reference_area_m2=float(reference_area),
        reference_length_m=design.scale.body_length_m,
        volume_m3=float(tri.volume),
        wetted_area_m2=float(tri.area),
        payload_box=payload_box,
        control_surfaces={},
        metadata=metadata,
    )
