from __future__ import annotations

import csv
import math
from pathlib import Path

from astreia_mrv.analysis.atmosphere import exponential_atmosphere
from astreia_mrv.config import VehicleDesign
from astreia_mrv.geometry.mesh import VehicleGeometry


def stagnation_heat_flux_w_m2(rho_kg_m3: float, velocity_m_s: float, nose_radius_m: float) -> float:
    """Sutton-Graves cold-wall stagnation estimate in SI units."""
    return (1.83e-4 / math.sqrt(max(nose_radius_m, 1e-6))) * math.sqrt(max(rho_kg_m3, 0.0)) * velocity_m_s**3


def leading_edge_heat_flux_w_m2(
    rho_kg_m3: float,
    velocity_m_s: float,
    leading_edge_radius_m: float,
    sweep_deg: float,
) -> float:
    q_radius = stagnation_heat_flux_w_m2(rho_kg_m3, velocity_m_s, leading_edge_radius_m)
    q_flat_plate = 0.18 * q_radius
    sweep = math.radians(sweep_deg)
    # TODO: replace flat-plate placeholder with boundary-layer correlation.
    return 0.5 * (q_radius * math.cos(sweep) ** 2 + q_flat_plate * math.sin(sweep) ** 2)


def heating_summary(design: VehicleDesign, geometry: VehicleGeometry, altitude_m: float | None = None) -> dict[str, float]:
    p = geometry.metadata.get("parameters", {})
    altitude = design.mission.entry_interface_altitude_m if altitude_m is None else altitude_m
    rho, _, _ = exponential_atmosphere(altitude)
    velocity = design.mission.entry_velocity_m_s
    rn = float(p.get("R_N", 0.35))
    rle = float(p.get("R_LE", max(0.025, 0.05 * rn)))
    sweep = float(p.get("sweep_deg", 60.0))
    q_stag = stagnation_heat_flux_w_m2(rho, velocity, rn)
    q_le = leading_edge_heat_flux_w_m2(rho, velocity, rle, sweep)
    entry_duration_s = 450.0
    heat_load = 0.35 * q_stag * entry_duration_s
    return {
        "altitude_m": float(altitude),
        "rho_kg_m3": float(rho),
        "velocity_m_s": float(velocity),
        "q_dot_stag_w_m2": float(q_stag),
        "q_dot_leading_edge_w_m2": float(q_le),
        "integrated_heat_load_j_m2": float(heat_load),
        "assumption": "conceptual cold-wall Sutton-Graves plus leading-edge placeholder",
    }


def write_heating_csv(summary: dict[str, float], path: str | Path) -> None:
    numeric = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(numeric.keys()))
        writer.writeheader()
        writer.writerow(numeric)
