#!/usr/bin/env python3
"""Export the Astreia-MRV Onshape BREP handoff as STEP.

This script intentionally depends only on OCP/OpenCascade plus the generated
`onshape_brep_spec.json`. Run it with the local CAD virtual environment or with
OCP on PYTHONPATH:

    PYTHONPATH=.cad-venv/lib/python3.10/site-packages python scripts/export_onshape_step.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape
from OCP.gp import gp_Pnt, gp_Vec


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


def _loft_solid(section_points: list[list[Point]]) -> TopoDS_Shape:
    loft = BRepOffsetAPI_ThruSections(True, False, 1.0e-5)
    loft.CheckCompatibility(True)
    for points in section_points:
        loft.AddWire(_wire(points))
    loft.Build()
    if not loft.IsDone():
        raise RuntimeError("Loft operation failed")
    shape = loft.Shape()
    if not _shape_is_valid(shape):
        raise RuntimeError("Loft produced an invalid body")
    return shape


def _normalized(vector: Iterable[float]) -> Point:
    x, y, z = (float(v) for v in vector)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 0.0:
        raise ValueError("Cannot normalize zero vector")
    return (x / norm, y / norm, z / norm)


def _offset(points: list[Point], axis: Point, distance: float) -> list[Point]:
    ax, ay, az = axis
    return [(x + ax * distance, y + ay * distance, z + az * distance) for x, y, z in points]


def _thick_polygon_solid(points: list[Point], thickness: float, axis: Point) -> TopoDS_Shape:
    half = 0.5 * float(thickness)
    normal = _normalized(axis)
    start = _offset(points, normal, -half)
    face_builder = BRepBuilderAPI_MakeFace(_wire(start))
    if not face_builder.IsDone():
        raise RuntimeError("Failed to build planar face for thick polygon")
    ax, ay, az = normal
    prism = BRepPrimAPI_MakePrism(face_builder.Face(), gp_Vec(ax * thickness, ay * thickness, az * thickness))
    prism.Build()
    if not prism.IsDone():
        raise RuntimeError("Failed to thicken polygon into prism")
    shape = prism.Shape()
    if not _shape_is_valid(shape):
        raise RuntimeError("Prism produced an invalid solid")
    return shape


def _compound(shapes: list[TopoDS_Shape]) -> TopoDS_Compound:
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for shape in shapes:
        builder.Add(compound, shape)
    return compound


def export_step(spec_path: Path, output_dir: Path) -> dict[str, str | bool]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    body_sections = [[_point(point) for point in section["points"]] for section in spec["body"]["sections"]]
    body = _loft_solid(body_sections)
    body_path = output_dir / "astreia_mrv_body.step"
    _write_step(body, body_path)

    shapes = [body]
    part_paths = {"body": str(body_path)}
    for part in spec["parts"]:
        if part["kind"] != "thick_polygon":
            raise ValueError(f"Unsupported part kind: {part['kind']}")
        solid = _thick_polygon_solid(
            [_point(point) for point in part["points"]],
            float(part["thickness_m"]),
            _point(part["thickness_axis"]),
        )
        if not _shape_is_valid(solid):
            raise RuntimeError(f"Invalid part shape: {part['name']}")
        part_path = output_dir / f"{part['name']}.step"
        _write_step(solid, part_path)
        part_paths[part["name"]] = str(part_path)
        shapes.append(solid)

    assembly = _compound(shapes)
    assembly_path = output_dir / "geometry_onshape.step"
    _write_step(assembly, assembly_path)
    manifest = {
        "assembly_step": str(assembly_path),
        "part_steps": part_paths,
        "valid_body": _shape_is_valid(body),
        "valid_assembly": _shape_is_valid(assembly),
        "units": spec.get("units", "m"),
    }
    (output_dir / "onshape_step_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="outputs/mrv3_notebook/onshape_brep_spec.json")
    parser.add_argument("--output-dir", default="outputs/mrv3_notebook/onshape_step")
    args = parser.parse_args()
    manifest = export_step(Path(args.spec), Path(args.output_dir))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
