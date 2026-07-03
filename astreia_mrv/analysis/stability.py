from __future__ import annotations

from astreia_mrv.geometry.mesh import VehicleGeometry


def active_control_margin_placeholder(geometry: VehicleGeometry) -> dict[str, float | str]:
    return {
        "pitch_trim_margin_deg": 5.0,
        "yaw_roll_control_margin": 0.2,
        "note": "TODO: replace placeholder with static derivatives and 6DOF trim analysis",
    }
