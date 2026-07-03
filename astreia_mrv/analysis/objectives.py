from __future__ import annotations

from astreia_mrv.analysis.heating import heating_summary
from astreia_mrv.config import VehicleDesign
from astreia_mrv.geometry.mesh import VehicleGeometry


def objective_summary(design: VehicleDesign, geometry: VehicleGeometry) -> dict[str, float]:
    heat = heating_summary(design, geometry)
    payload_volume = design.payload.length_m * design.payload.width_m * design.payload.height_m
    return {
        "minimize_peak_heat_flux_w_m2": max(heat["q_dot_stag_w_m2"], heat["q_dot_leading_edge_w_m2"]),
        "maximize_payload_volume_efficiency": payload_volume / max(geometry.volume_m3, 1e-9),
        "maximize_reuse_score": float(design.scale.reuse_target) / (1.0 + heat["q_dot_stag_w_m2"] / 1.0e6),
        "maximize_crossrange_proxy": geometry.reference_area_m2 / max(geometry.wetted_area_m2, 1e-9),
    }
