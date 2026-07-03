from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from astreia_mrv.analysis.control_authority import estimate_control_authority
from astreia_mrv.analysis.atmosphere import dynamic_pressure, exponential_atmosphere
from astreia_mrv.analysis.heating import heating_summary
from astreia_mrv.config import VehicleDesign
from astreia_mrv.geometry.mesh import VehicleGeometry


@dataclass(frozen=True)
class ConstraintResult:
    name: str
    passed: bool
    margin: float
    value: float
    limit: float
    note: str = ""


def launch_envelope_fit(design: VehicleDesign, geometry: VehicleGeometry) -> ConstraintResult:
    bounds = geometry.mesh.bounds
    length = bounds[1, 0] - bounds[0, 0]
    width = bounds[1, 1] - bounds[0, 1]
    height = bounds[1, 2] - bounds[0, 2]
    env = design.launch_envelope
    max_width = env.max_stowed_width_m or env.max_stowed_diameter_m
    max_height = env.max_stowed_height_m or env.max_stowed_diameter_m
    margins = [
        env.max_stowed_length_m - length,
        max_width - width,
        max_height - height,
    ]
    margin = min(margins)
    return ConstraintResult("launch_envelope", bool(margin >= 0.0), float(margin), float(max(length, width, height)), 0.0)


def payload_fit(design: VehicleDesign, geometry: VehicleGeometry) -> ConstraintResult:
    p = geometry.metadata.get("parameters", {})
    clear = float(p.get("payload_clearance_m", 0.05))
    available = np.array(
        [
            0.965 * geometry.reference_length_m,
            1.35 * float(p.get("body_half_width", 0.5)),
            float(p.get("body_top_height", 0.3)) + float(p.get("body_belly_depth", 0.2)),
        ]
    )
    required = np.array(
        [
            design.payload.length_m + 2.0 * clear,
            design.payload.width_m + 2.0 * clear,
            design.payload.height_m + 2.0 * clear,
        ]
    )
    margin = float(np.min(available - required))
    return ConstraintResult("payload_bay_fit", bool(margin >= 0.0), margin, float(np.max(required)), float(np.min(available)))


def heat_flux_limits(design: VehicleDesign, geometry: VehicleGeometry) -> list[ConstraintResult]:
    heat = heating_summary(design, geometry)
    max_stag = design.constraints.get("max_stagnation_heat_flux_w_m2", 1.2e6)
    max_le = design.constraints.get("max_leading_edge_heat_flux_w_m2", 1.5e6)
    return [
        ConstraintResult(
            "max_stagnation_heat_flux",
            heat["q_dot_stag_w_m2"] <= max_stag,
            float(max_stag - heat["q_dot_stag_w_m2"]),
            float(heat["q_dot_stag_w_m2"]),
            float(max_stag),
        ),
        ConstraintResult(
            "max_leading_edge_heat_flux",
            heat["q_dot_leading_edge_w_m2"] <= max_le,
            float(max_le - heat["q_dot_leading_edge_w_m2"]),
            float(heat["q_dot_leading_edge_w_m2"]),
            float(max_le),
        ),
    ]


def placeholder_constraints(design: VehicleDesign, geometry: VehicleGeometry) -> list[ConstraintResult]:
    rho, _, _ = exponential_atmosphere(45000.0)
    q_dyn = dynamic_pressure(rho, 2200.0)
    max_q = design.constraints.get("max_dynamic_pressure_pa", 50000.0)
    max_g = design.constraints.get("max_g_load", design.payload.max_g)
    estimated_g = 0.75 * max_g
    return [
        ConstraintResult("max_g_load_placeholder", estimated_g <= max_g, float(max_g - estimated_g), float(estimated_g), float(max_g), "TODO: 3DOF/6DOF trajectory integration"),
        ConstraintResult("max_dynamic_pressure_placeholder", q_dyn <= max_q, float(max_q - q_dyn), float(q_dyn), float(max_q), "TODO: trajectory-derived max-q"),
        *control_authority_constraints(geometry),
    ]


def control_authority_constraints(geometry: VehicleGeometry) -> list[ConstraintResult]:
    results = []
    for row in estimate_control_authority(geometry):
        results.append(
            ConstraintResult(
                f"{row['axis']}_control_authority_conceptual",
                bool(row["passes"]),
                float(row["margin"]),
                float(row["available_coeff"]),
                float(row["required_coeff"]),
                str(row["note"]),
            )
        )
    return results


def evaluate_constraints(design: VehicleDesign, geometry: VehicleGeometry) -> list[ConstraintResult]:
    return [
        launch_envelope_fit(design, geometry),
        payload_fit(design, geometry),
        *heat_flux_limits(design, geometry),
        *placeholder_constraints(design, geometry),
    ]
