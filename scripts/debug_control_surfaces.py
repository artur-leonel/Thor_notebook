#!/usr/bin/env python3
"""Print focused geometry diagnostics for all control surfaces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from astreia_mrv.config import load_design
from astreia_mrv.geometry.lifting_body import _build_fuselage, generate_lifting_body
from astreia_mrv.parameters import map_lifting_body_parameters


def point_triangle_distance(point: np.ndarray, tri: np.ndarray) -> float:
    a, b, c = tri
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        projection = a + v * ab
        return float(np.linalg.norm(point - projection))

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        projection = a + w * ac
        return float(np.linalg.norm(point - projection))

    va = d3 * d6 - d5 * d4
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    projection = a + ab * v + ac * w
    return float(np.linalg.norm(point - projection))


def min_body_distances(surface_points: np.ndarray, body_tris: np.ndarray) -> np.ndarray:
    # KD-tree distances to body vertices/face centroids are a fast diagnostic
    # proxy; exact point-triangle checks are too slow for routine smoke runs.
    body_cloud = np.vstack([body_tris.reshape(-1, 3), body_tris.mean(axis=1)])
    distances, _ = cKDTree(body_cloud).query(surface_points)
    return np.asarray(distances, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mrv3.yaml"))
    args = parser.parse_args()

    config_path = args.config
    design = load_design(config_path)
    geometry = generate_lifting_body(design)
    tri_mesh = trimesh.Trimesh(vertices=geometry.mesh.vertices, faces=geometry.mesh.faces, process=False)
    fin_geometry = geometry.metadata["control_surfaces"].get("geometry", {})
    core_half_width = 0.5 * float(fin_geometry.get("core_body_width_m", 0.0))

    body = _build_fuselage(map_lifting_body_parameters(design))
    body_mask = np.array([z in {"fuselage", "nose_cap", "aft_closeout"} for z in body.face_zone])
    body_tris = body.vertices[body.faces[body_mask]]

    print(f"control surfaces @ {config_path}")
    print("=" * 56)
    rows = []
    for name, surf in sorted(geometry.control_surfaces.items()):
        d = min_body_distances(surf.vertices, body_tris) if len(surf.vertices) else np.array([float("nan")])
        rows.append(
            (
                name,
                len(surf.faces),
                float(surf.face_areas.sum()),
                float(max(0.0, np.max(np.abs(surf.vertices[:, 1])) - core_half_width)) if len(surf.vertices) else 0.0,
                float(d.min()),
                float(d.mean()),
                float(d.max()),
                float(surf.bounds[0, 0]) if len(surf.faces) else float("nan"),
                float(surf.bounds[1, 0]) if len(surf.faces) else float("nan"),
            )
        )

    w0 = max(len(r[0]) for r in rows)
    print(
        f"{'surface'.ljust(w0)} | faces | area (m^2) | outbd (m) | x range (m) | "
        "min gap (m) | mean gap (m) | max gap (m)"
    )
    print("-" * 111)
    for name, faces, area, outboard, d_min, d_mean, d_max, x_min, x_max in rows:
        print(
            f"{name.ljust(w0)} | {faces:5d} | {area:10.5f} | {outboard:9.5f} | {x_min:5.2f}-{x_max:4.2f} | "
            f"{d_min:10.6f} | {d_mean:11.6f} | {d_max:11.6f}"
        )

    bounds = tri_mesh.bounds
    span = bounds[1] - bounds[0]
    print(f"\nmesh components: {len(tri_mesh.split(only_watertight=False))}")
    print(f"bounds: x={bounds[0,0]:.3f}..{bounds[1,0]:.3f} m, y={bounds[0,1]:.3f}..{bounds[1,1]:.3f} m, z={bounds[0,2]:.3f}..{bounds[1,2]:.3f} m")
    print(f"span:   L={span[0]:.3f} m, W={span[1]:.3f} m, H={span[2]:.3f} m, L/W={span[0]/span[1]:.2f}, W/H={span[1]/span[2]:.2f}")
    if fin_geometry:
        print(
            "fins:   "
            f"core W={float(fin_geometry['core_body_width_m']):.3f} m, "
            f"total W={float(fin_geometry['total_span_m']):.3f} m, "
            f"per-side extension={float(fin_geometry['per_side_fin_extension_m']):.3f} m"
        )
    print("\nmetadata control-surface groups:")
    for group, payload in geometry.metadata["control_surfaces"].items():
        print(f" - {group}: {payload}")
    print(f"\nwetted area: {geometry.wetted_area_m2:.6f} m^2")
    print(f"volume:      {geometry.volume_m3:.6f} m^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
