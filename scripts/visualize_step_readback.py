#!/usr/bin/env python3
"""Read an exported STEP with OpenCascade and render that imported geometry.

This is a verification tool for the Onshape handoff: it plots the geometry that
is actually inside the STEP file, not the source MRV mesh used to create it.

Typical local use with the repo's split Python environments:

    PYTHONPATH=.cad-venv/lib/python3.10/site-packages .cad-venv/bin/python \
        scripts/visualize_step_readback.py --extract-only
    python scripts/visualize_step_readback.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _canonical_triangles(triangles: np.ndarray) -> np.ndarray:
    rows = []
    for face in np.round(triangles, 12):
        ordered_face = face[np.lexsort((face[:, 2], face[:, 1], face[:, 0]))]
        rows.append(ordered_face.reshape(-1))
    canonical = np.asarray(rows)
    return canonical[np.lexsort(tuple(canonical[:, i] for i in range(canonical.shape[1] - 1, -1, -1)))]


def _read_source_geometry(config_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    from astreia_mrv.config import load_design
    from astreia_mrv.geometry.lifting_body import generate_lifting_body

    geometry = generate_lifting_body(load_design(config_path))
    vertices = geometry.mesh.vertices
    faces = geometry.mesh.faces
    return vertices.min(axis=0), vertices.max(axis=0), vertices[faces], (len(vertices), len(faces))


def extract_step_mesh(step_path: Path, mesh_npz: Path, report_path: Path, config_path: Path, linear_deflection: float) -> dict[str, Any]:
    from OCP.BRep import BRep_Tool
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"OpenCascade could not read STEP file: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    is_valid = bool(BRepCheck_Analyzer(shape).IsValid())
    shape_type = str(shape.ShapeType()).split(".")[-1]

    BRepMesh_IncrementalMesh(shape, linear_deflection)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    face_count = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, loc)
        if triangulation is not None:
            face_count += 1
            offset = len(vertices)
            transform = loc.Transformation()
            for node_index in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(node_index).Transformed(transform)
                vertices.append([float(point.X()), float(point.Y()), float(point.Z())])
            for tri_index in range(1, triangulation.NbTriangles() + 1):
                a, b, c = triangulation.Triangle(tri_index).Get()
                faces.append([offset + int(a) - 1, offset + int(b) - 1, offset + int(c) - 1])
        exp.Next()

    vertex_array = np.asarray(vertices, dtype=float)
    face_array = np.asarray(faces, dtype=np.int64)
    if len(vertex_array) == 0 or len(face_array) == 0:
        raise RuntimeError(f"No renderable triangulation was extracted from STEP file: {step_path}")

    mesh_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mesh_npz, vertices=vertex_array, faces=face_array)

    source_min, source_max, source_triangles, source_shape = _read_source_geometry(config_path)
    step_min = vertex_array.min(axis=0)
    step_max = vertex_array.max(axis=0)
    step_triangles = vertex_array[face_array]
    source_canonical = _canonical_triangles(source_triangles)
    step_canonical = _canonical_triangles(step_triangles)
    same_triangle_shape = source_canonical.shape == step_canonical.shape
    triangle_error = (
        float(np.max(np.abs(source_canonical - step_canonical)))
        if same_triangle_shape
        else None
    )
    report: dict[str, Any] = {
        "step_path": str(step_path),
        "shape_type": shape_type,
        "valid_shape": is_valid,
        "linear_deflection_m": linear_deflection,
        "brep_faces_with_triangulation": face_count,
        "readback_vertices": int(len(vertex_array)),
        "readback_triangles": int(len(face_array)),
        "source_vertices": int(source_shape[0]),
        "source_triangles": int(source_shape[1]),
        "step_bounds_min_m": step_min.tolist(),
        "step_bounds_max_m": step_max.tolist(),
        "source_bounds_min_m": source_min.tolist(),
        "source_bounds_max_m": source_max.tolist(),
        "bounds_abs_error_m": np.maximum(np.abs(step_min - source_min), np.abs(step_max - source_max)).tolist(),
        "max_bounds_abs_error_m": float(
            max(np.max(np.abs(step_min - source_min)), np.max(np.abs(step_max - source_max)))
        ),
        "canonical_triangle_point_max_abs_error_m": triangle_error,
        "canonical_triangles_match_at_1e_11_m": bool(
            same_triangle_shape and np.allclose(source_canonical, step_canonical, atol=1e-11, rtol=0.0)
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _make_readback_geometry(vertices: np.ndarray, faces: np.ndarray):
    from astreia_mrv.geometry.mesh import SurfaceMesh, VehicleGeometry

    surface = SurfaceMesh(vertices, faces, ["fuselage"] * len(faces))
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return VehicleGeometry(
        mesh=surface,
        reference_area_m2=float((maxs[1] - mins[1]) * (maxs[2] - mins[2])),
        reference_length_m=float(maxs[0] - mins[0]),
        volume_m3=0.0,
        wetted_area_m2=float(surface.face_areas.sum()),
        payload_box=(0.0, 0.0, 0.0),
        control_surfaces={},
        metadata={"source": "step_readback"},
    )


def plot_step_mesh(mesh_npz: Path, out_dir: Path) -> dict[str, str]:
    from astreia_mrv.viz.plot_geometry import save_engineering_sheet, save_shaded_render, save_three_view

    data = np.load(mesh_npz)
    geometry = _make_readback_geometry(data["vertices"], data["faces"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "step_readback_render": str(out_dir / "step_readback_render.png"),
        "step_readback_three_view": str(out_dir / "step_readback_three_view.png"),
        "step_readback_engineering": str(out_dir / "step_readback_engineering.png"),
    }
    save_shaded_render(geometry, paths["step_readback_render"])
    save_three_view(geometry, paths["step_readback_three_view"])
    save_engineering_sheet(geometry, paths["step_readback_engineering"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="outputs/mrv3_notebook/onshape_step/geometry_onshape.step")
    parser.add_argument("--config", default="configs/mrv3.yaml")
    parser.add_argument("--out-dir", default="outputs/mrv3_notebook/step_readback")
    parser.add_argument("--linear-deflection", type=float, default=0.003)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    mesh_npz = out_dir / "step_readback_mesh.npz"
    report_path = out_dir / "step_readback_report.json"

    result: dict[str, Any] = {}
    if not args.plot_only:
        result["report"] = extract_step_mesh(
            Path(args.step),
            mesh_npz,
            report_path,
            Path(args.config),
            args.linear_deflection,
        )
    if not args.extract_only:
        result["plots"] = plot_step_mesh(mesh_npz, out_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
