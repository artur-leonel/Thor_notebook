from __future__ import annotations

import numpy as np


def hermite_point(p0: np.ndarray, m0: np.ndarray, m1: np.ndarray, p1: np.ndarray, u: float) -> np.ndarray:
    """Evaluate a cubic Hermite segment."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    m0 = np.asarray(m0, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    h0 = (1.0 + 2.0 * u) * (1.0 - u) ** 2
    h1 = u * (1.0 - u) ** 2
    h2 = u**2 * (u - 1.0)
    h3 = u**2 * (3.0 - 2.0 * u)
    return p0 * h0 + m0 * h1 + m1 * h2 + p1 * h3


def finite_difference_tangents(points: np.ndarray, closed: bool = False) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    tangents = np.zeros_like(points)
    n = len(points)
    for i in range(n):
        if closed:
            tangents[i] = 0.5 * (points[(i + 1) % n] - points[(i - 1) % n])
        elif i == 0:
            tangents[i] = points[1] - points[0]
        elif i == n - 1:
            tangents[i] = points[-1] - points[-2]
        else:
            tangents[i] = 0.5 * (points[i + 1] - points[i - 1])
    return tangents


def hermite_to_bezier(p0: np.ndarray, m0: np.ndarray, m1: np.ndarray, p1: np.ndarray) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    m0 = np.asarray(m0, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    return np.vstack([p0, p0 + m0 / 3.0, p1 - m1 / 3.0, p1])


def _segment_intersection_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def bezier_has_obvious_self_intersection(bezier: np.ndarray) -> bool:
    """Detect the most common cubic overshoot pathology using control polygon crossing."""
    b = np.asarray(bezier, dtype=float)
    if b.shape[1] > 2:
        b = b[:, :2]
    return _segment_intersection_2d(b[0], b[1], b[2], b[3])


def fix_segment_convexity(
    p0: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    p1: np.ndarray,
    max_iter: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink tangents until the Hermite segment's Bezier control polygon is sane.

    This is intentionally conservative for v0.1 sampled geometry. TODO: replace
    with a fuller contour convexity/self-intersection proof before CAD export.
    """
    m0_fixed = np.asarray(m0, dtype=float).copy()
    m1_fixed = np.asarray(m1, dtype=float).copy()
    chord = np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float))
    for _ in range(max_iter):
        bez = hermite_to_bezier(p0, m0_fixed, m1_fixed, p1)
        long_handles = np.linalg.norm(m0_fixed) > 1.5 * chord or np.linalg.norm(m1_fixed) > 1.5 * chord
        if not bezier_has_obvious_self_intersection(bez) and not long_handles:
            return m0_fixed, m1_fixed
        m0_fixed *= 0.5
        m1_fixed *= 0.5
    return m0_fixed, m1_fixed


def sample_closed_hermite(points: np.ndarray, samples_per_segment: int = 8) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    tangents = finite_difference_tangents(points, closed=True)
    samples = []
    n = len(points)
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        m0, m1 = fix_segment_convexity(p0, tangents[i], tangents[(i + 1) % n], p1)
        for k in range(samples_per_segment):
            samples.append(hermite_point(p0, m0, m1, p1, k / samples_per_segment))
    return np.asarray(samples)


def sample_open_hermite(points: np.ndarray, samples_per_segment: int = 8) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    tangents = finite_difference_tangents(points, closed=False)
    samples = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        m0, m1 = fix_segment_convexity(p0, tangents[i], tangents[i + 1], p1)
        for k in range(samples_per_segment):
            samples.append(hermite_point(p0, m0, m1, p1, k / samples_per_segment))
    samples.append(points[-1])
    return np.asarray(samples)
