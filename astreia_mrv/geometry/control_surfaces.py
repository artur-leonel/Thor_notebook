from __future__ import annotations

import numpy as np

from astreia_mrv.geometry.mesh import SurfaceMesh, polygon_prism


def make_body_flap(params: dict[str, float]) -> SurfaceMesh:
    l = params["L_body"]
    width = min(0.58 * params["body_half_width"], 0.22 * l)
    length = min(max(params["L_bf"], 0.14 * l), 0.24 * l)
    thickness = max(0.025, 0.20 * params["R_LE"])
    x_hinge = l - length
    x_trailing = l - 0.006
    z_hinge = -0.92 * params["body_belly_depth"]
    z_trailing = z_hinge - 0.035
    p0 = np.array([x_hinge, -0.5 * width, z_hinge])
    p1 = np.array([x_hinge, 0.5 * width, z_hinge])
    p2 = np.array([x_trailing, 0.5 * width, z_trailing])
    p3 = np.array([x_trailing, -0.5 * width, z_trailing])
    return polygon_prism(np.vstack([p0, p1, p2, p3]), thickness, "body_flap")


def make_elevon_pair(params: dict[str, float]) -> dict[str, SurfaceMesh]:
    l = params["L_body"]
    thickness = max(0.020, 0.16 * params["R_LE"])
    length = min(0.18 * l, 0.30 * params["L_w"])
    x_hinge = max(0.78 * l, l - length)
    x_trailing = l - 0.006
    y_root = 0.72 * params["body_half_width"]
    y_tip = params["body_half_width"] + 0.18 * params["semi_span"]
    z_hinge = -0.60 * params["body_belly_depth"]
    z_trailing = -0.70 * params["body_belly_depth"]
    out: dict[str, SurfaceMesh] = {}
    for sign, name in [(1.0, "right_elevon"), (-1.0, "left_elevon")]:
        p0 = np.array([x_hinge, sign * y_root, z_hinge])
        p1 = np.array([x_hinge, sign * y_tip, z_hinge - 0.012])
        p2 = np.array([x_trailing, sign * y_tip, z_trailing])
        p3 = np.array([x_trailing, sign * y_root, z_trailing + 0.016])
        out[name] = polygon_prism(np.vstack([p0, p1, p2, p3]), thickness, "elevon")
    return out
