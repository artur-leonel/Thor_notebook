from __future__ import annotations

import numpy as np


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    """Return mask for nondominated points assuming all columns are minimized."""
    pts = np.asarray(points, dtype=float)
    keep = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        if not keep[i]:
            continue
        dominated_by_any = np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1))
        keep[i] = not dominated_by_any
    return keep
