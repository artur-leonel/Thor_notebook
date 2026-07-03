from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryZone:
    center_lat_lon: tuple[float, float]
    radius_m: float
    authorized: bool = True


def recovery_zone_feasible(zone: RecoveryZone) -> bool:
    return zone.authorized and zone.radius_m > 0.0
