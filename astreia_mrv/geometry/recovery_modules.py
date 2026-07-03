from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryTargetZone:
    center_lat_lon: tuple[float, float]
    radius_m: float
    authorized: bool = True


def landing_error_estimate_m(mode: str, wind_factor: float = 1.0, nav_quality: float = 1.0) -> float:
    """Terminal recovery placeholder, not impact targeting."""
    base = {
        "parafoil": 25.0,
        "deployable_drone": 8.0,
        "parachute_airbag": 250.0,
        "runway_skid": 60.0,
        "splashdown": 1000.0,
    }.get(mode, 150.0)
    return base * max(0.5, wind_factor) / max(0.5, nav_quality)
