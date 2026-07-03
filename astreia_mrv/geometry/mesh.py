from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


@dataclass
class SurfaceMesh:
    vertices: np.ndarray
    faces: np.ndarray
    face_zone: list[str] = field(default_factory=list)
    face_normals: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if not self.face_zone:
            self.face_zone = ["default"] * len(self.faces)
        if len(self.face_zone) != len(self.faces):
            raise ValueError("face_zone length must match faces length")
        self.update_normals()

    def update_normals(self) -> None:
        tri = self.vertices[self.faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norms = np.linalg.norm(normals, axis=1)
        good = norms > 0.0
        normals[good] /= norms[good, None]
        self.face_normals = normals

    def to_trimesh(self, process: bool = False) -> trimesh.Trimesh:
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=process)

    @property
    def bounds(self) -> np.ndarray:
        return np.vstack([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    @property
    def face_areas(self) -> np.ndarray:
        tri = self.vertices[self.faces]
        return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


@dataclass
class VehicleGeometry:
    mesh: SurfaceMesh
    reference_area_m2: float
    reference_length_m: float
    volume_m3: float
    wetted_area_m2: float
    payload_box: tuple[float, float, float]
    control_surfaces: dict[str, SurfaceMesh]
    metadata: dict[str, Any]


def make_trimesh(surface: SurfaceMesh) -> trimesh.Trimesh:
    mesh = surface.to_trimesh(process=True)
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    elif hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def surface_from_trimesh(mesh: trimesh.Trimesh, zones: list[str] | None = None) -> SurfaceMesh:
    return SurfaceMesh(
        vertices=np.asarray(mesh.vertices, dtype=float),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        face_zone=zones or ["default"] * len(mesh.faces),
    )


def combine_meshes(meshes: list[SurfaceMesh]) -> SurfaceMesh:
    vertices = []
    faces = []
    zones: list[str] = []
    offset = 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        zones.extend(mesh.face_zone)
        offset += len(mesh.vertices)
    return SurfaceMesh(np.vstack(vertices), np.vstack(faces), zones)


def weld_vertices(surface: SurfaceMesh, tol: float = 1e-8) -> SurfaceMesh:
    """Merge coincident vertices so rooted appendage panels share topology."""
    if len(surface.vertices) == 0:
        return surface
    keys = np.round(surface.vertices / tol).astype(np.int64)
    _, unique_indices, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(unique_indices)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    vertices = surface.vertices[unique_indices[order]]
    faces = remap[inverse[surface.faces]]
    nondegenerate = np.array([len(set(face.tolist())) == 3 for face in faces], dtype=bool)
    return SurfaceMesh(vertices, faces[nondegenerate], [z for z, keep in zip(surface.face_zone, nondegenerate) if keep])


def drop_tiny_face_islands(surface: SurfaceMesh, max_fraction: float = 0.01) -> SurfaceMesh:
    """Remove face islands that are tiny compared with the main connected skin."""
    if len(surface.faces) < 2:
        return surface

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(surface.faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(a), int(b))))
            edge_to_faces.setdefault(edge, []).append(face_index)

    neighbors = [set() for _ in range(len(surface.faces))]
    for linked_faces in edge_to_faces.values():
        if len(linked_faces) > 1:
            for face_index in linked_faces:
                neighbors[face_index].update(other for other in linked_faces if other != face_index)

    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(len(surface.faces)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in neighbors[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    largest = max(len(component) for component in components)
    min_keep = max(2, int(max_fraction * largest))
    keep_faces = {face_index for component in components if len(component) >= min_keep for face_index in component}
    if len(keep_faces) == len(surface.faces):
        return surface

    keep = np.array([idx in keep_faces for idx in range(len(surface.faces))], dtype=bool)
    return SurfaceMesh(surface.vertices.copy(), surface.faces[keep].copy(), [zone for zone, keep_face in zip(surface.face_zone, keep) if keep_face])


def orient_faces_away_from_centroid(surface: SurfaceMesh) -> SurfaceMesh:
    """Flip obviously inward faces for star-shaped sampled meshes and plate components."""
    vertices = surface.vertices.copy()
    faces = surface.faces.copy()
    center = vertices.mean(axis=0)
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    centroids = tri.mean(axis=1)
    inward = np.einsum("ij,ij->i", normals, centroids - center) < 0.0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return SurfaceMesh(vertices, faces, list(surface.face_zone))


def triangle_prism(points: np.ndarray, thickness: float, zone: str) -> SurfaceMesh:
    """Create a closed thin triangular prism from three points."""
    pts = np.asarray(points, dtype=float)
    normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    norm = np.linalg.norm(normal)
    if norm == 0.0:
        raise ValueError("Cannot create prism from collinear points")
    normal /= norm
    top = pts + 0.5 * thickness * normal
    bottom = pts - 0.5 * thickness * normal
    vertices = np.vstack([top, bottom])
    faces = np.array(
        [
            [0, 1, 2],
            [5, 4, 3],
            [0, 3, 4],
            [0, 4, 1],
            [1, 4, 5],
            [1, 5, 2],
            [2, 5, 3],
            [2, 3, 0],
        ],
        dtype=np.int64,
    )
    return SurfaceMesh(vertices, faces, [zone] * len(faces))


def polygon_prism(points: np.ndarray, thickness: float, zone: str) -> SurfaceMesh:
    """Create a closed thin prism from one planar polygon.

    This is used for fins, strakes, elevons, and flaps so each appendage has one
    consistent thickness direction instead of being split into separate triangle
    prisms with slightly different normals.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        raise ValueError("Cannot create prism from fewer than three points")

    normal = None
    for i in range(1, len(pts) - 1):
        candidate = np.cross(pts[i] - pts[0], pts[i + 1] - pts[0])
        norm = np.linalg.norm(candidate)
        if norm > 0.0:
            normal = candidate / norm
            break
    if normal is None:
        raise ValueError("Cannot create prism from collinear points")

    top = pts + 0.5 * thickness * normal
    bottom = pts - 0.5 * thickness * normal
    vertices = np.vstack([top, bottom])

    faces: list[list[int]] = []
    n = len(pts)
    for i in range(1, n - 1):
        faces.append([0, i, i + 1])
        faces.append([n, n + i + 1, n + i])
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, n + i, n + j])
        faces.append([i, n + j, j])

    return SurfaceMesh(vertices, np.asarray(faces, dtype=np.int64), [zone] * len(faces))


def validate_mesh(surface: SurfaceMesh) -> dict[str, Any]:
    tri = make_trimesh(surface)
    areas = surface.face_areas
    symmetry = symmetry_error_y(surface)
    return {
        "finite_vertices": bool(np.isfinite(surface.vertices).all()),
        "watertight": bool(tri.is_watertight),
        "positive_volume": bool(tri.volume > 0.0),
        "volume_m3": float(tri.volume),
        "area_m2": float(tri.area),
        "min_panel_area_m2": float(areas.min()) if len(areas) else 0.0,
        "symmetry_error_m": float(symmetry),
    }


def symmetry_error_y(surface: SurfaceMesh) -> float:
    verts = surface.vertices
    mirrored = verts.copy()
    mirrored[:, 1] *= -1.0
    sample = mirrored[:: max(1, len(mirrored) // 500)]
    errors = []
    for p in sample:
        d = np.linalg.norm(verts - p, axis=1)
        errors.append(float(d.min()))
    return float(np.max(errors)) if errors else 0.0


def export_surface(surface: SurfaceMesh, path: str | Path) -> None:
    path = Path(path)
    mesh = make_trimesh(surface)
    mesh.export(path)
