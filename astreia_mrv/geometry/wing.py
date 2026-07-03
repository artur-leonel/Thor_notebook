from __future__ import annotations

import numpy as np

from astreia_mrv.geometry.mesh import SurfaceMesh, polygon_prism


def make_strake_pair(
    params: dict[str, float],
    side_z: float = -0.03,
    root_anchors: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, SurfaceMesh]:
    """Generate low-aspect hypersonic chines blended from the body side."""
    l = params["L_body"]
    thickness = max(0.020, 0.20 * params["R_LE"])
    x0 = 0.42 * l
    x1 = 0.84 * l
    x_tip_le = 0.60 * l
    x_tip_te = l - 0.012
    y_root = 0.82 * params["body_half_width"]
    y_tip = params["body_half_width"] + 0.18 * params["semi_span"]
    z_root = -0.42 * params["body_belly_depth"]
    z_tip = -0.64 * params["body_belly_depth"]

    out: dict[str, SurfaceMesh] = {}
    for sign in (1.0, -1.0):
        key = "right_strake" if sign > 0 else "left_strake"
        if root_anchors and key in root_anchors:
            p0, p1 = root_anchors[key]
        else:
            p0 = np.array([x0, sign * y_root, z_root])
            p1 = np.array([x1, sign * y_root, z_root - 0.018])
        p2 = np.array([x_tip_le, sign * y_tip, z_tip + 0.012])
        p3 = np.array([x_tip_te, sign * y_tip, z_tip - 0.020])
        out[key] = polygon_prism(np.vstack([p0, p1, p3, p2]), thickness, "strake")
    return out


def make_winglet_pair(params: dict[str, float], side_z: float = 0.0) -> dict[str, SurfaceMesh]:
    """Reference-inspired aft layout does not use separate winglets."""
    return {}


def make_wing_fairing_pair(params: dict[str, float], side_z: float = -0.03) -> dict[str, SurfaceMesh]:
    """Generate a short pad blending each aft fairing into the rear elevon."""
    l = params["L_body"]
    thickness = max(0.020, 0.18 * params["R_LE"])
    x0 = 0.82 * l
    x1 = 0.985 * l
    y0 = 0.42 * params["body_half_width"]
    y1 = 0.96 * params["body_half_width"]
    z0 = -0.74 * params["body_belly_depth"]
    z1 = -0.88 * params["body_belly_depth"]

    out: dict[str, SurfaceMesh] = {}
    for sign in (1.0, -1.0):
        key = "right_wing_fairing" if sign > 0 else "left_wing_fairing"
        out[key] = polygon_prism(
            np.vstack(
                [
                    [x0, sign * y0, z0],
                    [x1, sign * y0, z1],
                    [x1, sign * y1, z1 - 0.035],
                    [x0, sign * y1, z0 - 0.02],
                ]
            ),
            thickness,
            "strake",
        )
    return out
