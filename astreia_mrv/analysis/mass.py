from __future__ import annotations

from astreia_mrv.config import VehicleDesign
from astreia_mrv.geometry.mesh import VehicleGeometry


def conceptual_mass_breakdown(design: VehicleDesign, geometry: VehicleGeometry) -> dict[str, float]:
    aeroshell_areal_kg_m2 = 18.0
    dry_mass = aeroshell_areal_kg_m2 * geometry.wetted_area_m2 + 0.35 * design.payload.mass_kg
    return {
        "payload_kg": float(design.payload.mass_kg),
        "dry_mass_estimate_kg": float(dry_mass),
        "entry_mass_estimate_kg": float(dry_mass + design.payload.mass_kg),
        "payload_mass_fraction": float(design.payload.mass_kg / max(dry_mass + design.payload.mass_kg, 1e-9)),
    }
