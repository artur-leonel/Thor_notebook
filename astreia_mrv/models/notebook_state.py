from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class DesignState(BaseModel):
    config_path: str = "configs/mrv3.yaml"
    vehicle_name: str = "MRV-3"
    geometry_family: str = "lifting_body"
    recovery_mode: str = "parafoil"
    body_length_m: float = 2.0
    payload_mass_kg: float = 300.0
    payload_box_m: tuple[float, float, float] = (1.8, 0.10, 0.17)
    rx: dict[str, float] = Field(default_factory=dict)


class GeometryState(BaseModel):
    reference_area_m2: float = 0.0
    reference_length_m: float = 0.0
    volume_m3: float = 0.0
    wetted_area_m2: float = 0.0
    mesh_quality: dict[str, float | bool | str] = Field(default_factory=dict)
    output_dir: str = "outputs/notebook_mrv3"
    openscad_path: str = ""
    obj_path: str = ""
    stl_path: str = ""


class AeroThermalState(BaseModel):
    mach: float = 20.0
    peak_stagnation_heat_flux_w_m2: float = 0.0
    peak_leading_edge_heat_flux_w_m2: float = 0.0
    integrated_heat_load_j_m2: float = 0.0
    alpha_sweep_rows: int = 0


class ConstraintState(BaseModel):
    all_passed: bool = False
    results: list[dict[str, float | bool | str]] = Field(default_factory=list)


class AstreiaNotebookState(BaseModel):
    design: DesignState = Field(default_factory=DesignState)
    geometry: GeometryState = Field(default_factory=GeometryState)
    aero_thermal: AeroThermalState = Field(default_factory=AeroThermalState)
    constraints: ConstraintState = Field(default_factory=ConstraintState)
    notes: dict[str, str] = Field(default_factory=dict)

    def resolve_output_dir(self) -> Path:
        return Path(self.geometry.output_dir)
