from __future__ import annotations

import math


def exponential_atmosphere(altitude_m: float) -> tuple[float, float, float]:
    """Return density, speed of sound, and temperature for a simple atmosphere.

    Good enough for conceptual v0.1 heating/aero sweeps. TODO: replace with a
    standard atmosphere table and high-altitude composition model.
    """
    rho0 = 1.225
    scale_height = 7200.0
    rho = rho0 * math.exp(-max(0.0, altitude_m) / scale_height)
    temp = max(190.0, 288.15 - 0.0065 * min(altitude_m, 11000.0))
    gamma = 1.4
    r_air = 287.05
    a = math.sqrt(gamma * r_air * temp)
    return rho, a, temp


def dynamic_pressure(rho_kg_m3: float, velocity_m_s: float) -> float:
    return 0.5 * rho_kg_m3 * velocity_m_s**2
