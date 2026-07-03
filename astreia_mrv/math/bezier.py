from __future__ import annotations

import numpy as np


def cubic_bezier_point(control: np.ndarray, u: float) -> np.ndarray:
    b = np.asarray(control, dtype=float)
    return (
        (1.0 - u) ** 3 * b[0]
        + 3.0 * (1.0 - u) ** 2 * u * b[1]
        + 3.0 * (1.0 - u) * u**2 * b[2]
        + u**3 * b[3]
    )
