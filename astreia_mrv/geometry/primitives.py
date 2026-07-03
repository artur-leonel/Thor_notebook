from __future__ import annotations

import numpy as np

from astreia_mrv.geometry.mesh import SurfaceMesh


def revolve_profile(x: np.ndarray, r: np.ndarray, n_theta: int = 96, zone: str = "aeroshell") -> SurfaceMesh:
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    vertices = []
    for xi, ri in zip(x, r):
        vertices.extend(np.column_stack([np.full(n_theta, xi), ri * np.cos(theta), ri * np.sin(theta)]))
    vertices_arr = np.asarray(vertices)
    faces = []
    zones = []
    for i in range(len(x) - 1):
        for j in range(n_theta):
            a = i * n_theta + j
            b = i * n_theta + (j + 1) % n_theta
            c = (i + 1) * n_theta + (j + 1) % n_theta
            d = (i + 1) * n_theta + j
            faces.append([a, d, c])
            faces.append([a, c, b])
            zones.extend([zone, zone])
    return SurfaceMesh(vertices_arr, np.asarray(faces, dtype=np.int64), zones)
