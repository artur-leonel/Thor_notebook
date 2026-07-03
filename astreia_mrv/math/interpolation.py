from __future__ import annotations

import numpy as np


def lerp(a: np.ndarray | float, b: np.ndarray | float, t: float):
    return (1.0 - t) * np.asarray(a) + t * np.asarray(b)


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)
