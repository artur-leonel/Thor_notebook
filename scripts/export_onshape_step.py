#!/usr/bin/env python3
"""Export the Astreia-MRV Onshape BREP handoff as STEP.

The default export is a faceted OpenCascade BREP solid built directly from the
same watertight MRV geometry mesh used by the plots. This intentionally avoids
the older simplified loft approximation, which was editable but could drift
away from the generated vehicle. Run it with the local CAD virtual environment
or with OCP on PYTHONPATH:

    PYTHONPATH=.cad-venv/lib/python3.10/site-packages python scripts/export_onshape_step.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.gp import gp_Pnt

from astreia_mrv.config import load_design
from astreia_mrv.geometry.lifting_body import generate_lifting_body
from astreia_mrv.geometry.mesh import VehicleGeometry


Point = tuple[float, float, float]


def _point(row: Iterable[float]) -> Point:
    x, y, z = row
    return (float(x), float(y), float(z))


def _wire(points: list[Point]):
    polygon = BRepBuilderAPI_MakePolygon()
    for x, y, z in points:
        polygon.Add(gp_Pnt(x, y, z))
    polygon.Close()
    if not polygon.IsDone():
        raise RuntimeError("Failed to build closed wire")
    return polygon.Wire()


def _shape_is_valid(shape: TopoDS_Shape) -> bool:
    return bool(BRepCheck_Analyzer(shape).IsValid())


def _write_step(shape: TopoDS_Shape, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: {path}")


def _triangle_face(points: list[Point]) -> TopoDS_Shape:
    face_builder = BRepBuilderAPI_MakeFace(_wire(points))
    if not face_builder.IsDone():
        raise RuntimeError("Failed to build triangular BREP face")
    return face_builder.Face()


def _faceted_solid_from_geometry(geometry: VehicleGeometry) -> TopoDS_Shape:
    sewer = BRepBuilderAPI_Sewing(1.0e-6)
    vertices = geometry.mesh.vertices
    for face in geometry.mesh.faces:
        points = [_point(vertices[int(index)]) for index in face]
        sewer.Add(_triangle_face(points))
    sewer.Perform()
    shell = TopoDS.Shell_s(sewer.SewedShape())
    if not _shape_is_valid(shell):
        raise RuntimeError("Sewed faceted MRV shell is invalid")
    solid = BRepBuilderAPI_MakeSolid(shell).Solid()
    if not _shape_is_valid(solid):
        raise RuntimeError("Faceted MRV BREP solid is invalid")
    return solid


def export_step(spec_path: Path, output_dir: Path, config_path: Path = Path("configs/mrv3.yaml")) -> dict[str, str | bool]:
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {"units": "m"}
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_step in output_dir.glob("*.step"):
        stale_step.unlink()

    geometry = generate_lifting_body(load_design(config_path))
    body = _faceted_solid_from_geometry(geometry)
    body_path = output_dir / "astreia_mrv_body.step"
    _write_step(body, body_path)

    assembly_path = output_dir / "geometry_onshape.step"
    _write_step(body, assembly_path)
    manifest = {
        "assembly_step": str(assembly_path),
        "part_steps": {"body": str(body_path)},
        "valid_body": _shape_is_valid(body),
        "valid_assembly": _shape_is_valid(body),
        "units": spec.get("units", "m"),
        "export_kind": "faceted_brep_solid_from_generated_mrv_mesh",
        "config_path": str(config_path),
        "faces": int(len(geometry.mesh.faces)),
        "vertices": int(len(geometry.mesh.vertices)),
    }
    (output_dir / "onshape_step_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="outputs/mrv3_notebook/onshape_brep_spec.json")
    parser.add_argument("--config", default="configs/mrv3.yaml")
    parser.add_argument("--output-dir", default="outputs/mrv3_notebook/onshape_step")
    args = parser.parse_args()
    manifest = export_step(Path(args.spec), Path(args.output_dir), Path(args.config))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
