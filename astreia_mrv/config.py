from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


GeometryFamily = Literal["capsule", "lifting_body"]
NoseProfile = Literal["rounded", "conic", "triangular"]
RecoveryMode = Literal[
    "parafoil",
    "parachute_airbag",
    "deployable_drone",
    "runway_skid",
    "splashdown",
]


@dataclass(frozen=True)
class LaunchEnvelope:
    name: str
    max_stowed_length_m: float
    max_stowed_diameter_m: float
    max_stowed_width_m: float | None = None
    max_stowed_height_m: float | None = None
    payload_mass_limit_kg: float | None = None


@dataclass(frozen=True)
class PayloadSpec:
    name: str
    mass_kg: float
    length_m: float
    width_m: float
    height_m: float
    max_g: float
    cg_x_frac: float = 0.5


@dataclass(frozen=True)
class VehicleScale:
    name: str
    body_length_m: float
    payload_target_kg: float
    reuse_target: int


@dataclass(frozen=True)
class TPSZone:
    name: str
    material_id: str
    thickness_m: float
    replaceable: bool
    max_reuse_temp_c: float | None = None
    allowable_heat_flux_w_m2: float | None = None


@dataclass(frozen=True)
class MissionSpec:
    entry_interface_altitude_m: float = 120000.0
    entry_velocity_m_s: float = 7830.0
    flight_path_angle_deg: float = -1.5
    target_crossrange_class: str = "medium"
    landing_accuracy_requirement_m: float = 150.0
    recovery_zone_authorized: bool = True


@dataclass(frozen=True)
class RecoverySpec:
    mode: RecoveryMode = "parafoil"
    deploy_mach_max: float = 0.8
    deploy_dynamic_pressure_max_pa: float = 6000.0


@dataclass(frozen=True)
class VehicleDesign:
    scale: VehicleScale
    launch_envelope: LaunchEnvelope
    payload: PayloadSpec
    rx: dict[str, float]
    geometry_family: GeometryFamily
    recovery_mode: RecoveryMode
    nose_profile: NoseProfile = "rounded"
    mission: MissionSpec = field(default_factory=MissionSpec)
    recovery: RecoverySpec = field(default_factory=RecoverySpec)
    constraints: dict[str, float] = field(default_factory=dict)


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required config section/key: {key}")
    return mapping[key]


def load_design(path: str | Path) -> VehicleDesign:
    """Load a vehicle design YAML file into dataclasses."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    vehicle = _require(data, "vehicle")
    payload = _require(data, "payload")
    launch = _require(data, "launch_envelope")
    mission = data.get("mission", {})
    recovery = data.get("recovery", {})

    scale = VehicleScale(
        name=str(vehicle["name"]),
        body_length_m=float(vehicle["body_length_m"]),
        payload_target_kg=float(vehicle["payload_target_kg"]),
        reuse_target=int(vehicle.get("reuse_target", 1)),
    )
    launch_envelope = LaunchEnvelope(
        name=str(launch["name"]),
        max_stowed_length_m=float(launch["max_stowed_length_m"]),
        max_stowed_diameter_m=float(launch["max_stowed_diameter_m"]),
        max_stowed_width_m=(
            None if launch.get("max_stowed_width_m") is None else float(launch["max_stowed_width_m"])
        ),
        max_stowed_height_m=(
            None if launch.get("max_stowed_height_m") is None else float(launch["max_stowed_height_m"])
        ),
        payload_mass_limit_kg=(
            None if launch.get("payload_mass_limit_kg") is None else float(launch["payload_mass_limit_kg"])
        ),
    )
    payload_spec = PayloadSpec(
        name=str(payload["name"]),
        mass_kg=float(payload["mass_kg"]),
        length_m=float(payload["length_m"]),
        width_m=float(payload["width_m"]),
        height_m=float(payload["height_m"]),
        max_g=float(payload["max_g"]),
        cg_x_frac=float(payload.get("cg_x_frac", 0.5)),
    )
    mission_spec = MissionSpec(
        entry_interface_altitude_m=float(mission.get("entry_interface_altitude_m", 120000.0)),
        entry_velocity_m_s=float(mission.get("entry_velocity_m_s", 7830.0)),
        flight_path_angle_deg=float(mission.get("flight_path_angle_deg", -1.5)),
        target_crossrange_class=str(mission.get("target_crossrange_class", "medium")),
        landing_accuracy_requirement_m=float(mission.get("landing_accuracy_requirement_m", 150.0)),
        recovery_zone_authorized=bool(mission.get("recovery_zone_authorized", True)),
    )
    recovery_mode = str(recovery.get("mode", data.get("recovery_mode", "parafoil")))
    nose_profile = str(vehicle.get("nose_profile", "rounded"))
    if nose_profile not in {"rounded", "conic", "triangular"}:
        raise ValueError(f"Unsupported nose_profile {nose_profile!r}; expected rounded, conic, or triangular")
    recovery_spec = RecoverySpec(
        mode=recovery_mode,  # type: ignore[arg-type]
        deploy_mach_max=float(recovery.get("deploy_mach_max", 0.8)),
        deploy_dynamic_pressure_max_pa=float(recovery.get("deploy_dynamic_pressure_max_pa", 6000.0)),
    )

    rx = {str(k): float(v) for k, v in (data.get("rx") or {}).items()}
    for key, value in rx.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Normalized design variable {key}={value} is outside [0, 1]")

    return VehicleDesign(
        scale=scale,
        launch_envelope=launch_envelope,
        payload=payload_spec,
        rx=rx,
        geometry_family=str(vehicle.get("geometry_family", "lifting_body")),  # type: ignore[arg-type]
        recovery_mode=recovery_mode,  # type: ignore[arg-type]
        nose_profile=nose_profile,  # type: ignore[arg-type]
        mission=mission_spec,
        recovery=recovery_spec,
        constraints={str(k): float(v) for k, v in (data.get("constraints") or {}).items()},
    )
