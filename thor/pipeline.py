"""Ordered THOR notebook manifest.

The manifest is intentionally small and boring: it keeps the notebook chain,
expected handoff tables, and MRV handoff notebooks in one importable place so
scripts and tests can audit the whole pipeline instead of drifting into ad hoc
subsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class NotebookSpec:
    path: str
    layer: str
    tables: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()
    notes: str = ""

    @property
    def notebook_path(self) -> Path:
        return Path(self.path)


THOR_NOTEBOOKS: tuple[NotebookSpec, ...] = (
    NotebookSpec("notebooks/00_inputs.py", "inputs", notes="Spreadsheet browser; no handoff table."),
    NotebookSpec("notebooks/00_mission_conops.py", "mission", ("mission_phases",), ("mission", "phase")),
    NotebookSpec("notebooks/01_mass_budget.py", "mission", ("mass_budget",), ("mass",)),
    NotebookSpec("notebooks/02_delta_v_budget.py", "mission", ("delta_v_budget",), ("delta_v",)),
    NotebookSpec("notebooks/03_power_energy_budget.py", "mission", ("power_budget",), ("phase",)),
    NotebookSpec("notebooks/04_link_budget.py", "mission", ("link_budget",), ("link",)),
    NotebookSpec("notebooks/10_orbit_design.py", "orbit", required_inputs=("orbit",), notes="Updates vehicle_state.orbit."),
    NotebookSpec("notebooks/12_rendezvous_phasing.py", "orbit", ("phasing_dv",), ("rendezvous", "orbit")),
    NotebookSpec("notebooks/13_proximity_ops_docking.py", "orbit", ("docking_approach",), ("docking",)),
    NotebookSpec("notebooks/14_deorbit_targeting.py", "orbit", ("entry_interface",), ("entry", "orbit")),
    NotebookSpec("notebooks/20_atmosphere_model.py", "entry", ("atmosphere",), ("atmosphere",)),
    NotebookSpec("notebooks/21_entry_trajectory_3dof.py", "entry", ("entry_trajectory",), ("entry", "aero")),
    NotebookSpec("notebooks/22_aeroheating.py", "entry", ("aeroheating",), ("aero",)),
    NotebookSpec("notebooks/23_tps_sizing.py", "entry", ("tps_sizing",), ("tps",)),
    NotebookSpec("notebooks/24_hypersonic_aero.py", "entry", ("hypersonic_aero",), ("mrv", "aero_db")),
    NotebookSpec("notebooks/30_aero_database.py", "aero", ("aero_database",), ("mrv", "aero_db")),
    NotebookSpec("notebooks/31_stability_trim.py", "aero", ("stability_trim",), ("mrv", "stability", "control")),
    NotebookSpec(
        "notebooks/32_control_surfaces.py",
        "aero",
        ("control_surfaces", "control_authority"),
        ("mrv", "control"),
    ),
    NotebookSpec("notebooks/33_terminal_descent.py", "aero", ("terminal_descent",), ("terminal",)),
    NotebookSpec("notebooks/40_main_propulsion.py", "propulsion", ("main_propulsion",), ("propulsion",)),
    NotebookSpec("notebooks/41_rcs_sizing.py", "propulsion", ("rcs_sizing",), ("rcs",)),
    NotebookSpec("notebooks/42_propellant_tanks.py", "propulsion", ("propellant_tanks",), ("propulsion", "mass")),
    NotebookSpec("notebooks/50_mass_properties.py", "gnc", ("mass_properties",), ("geometry", "mass_props", "payload")),
    NotebookSpec("notebooks/51_attitude_dynamics_6dof.py", "gnc", ("attitude_dynamics_6dof",), ("gnc",)),
    NotebookSpec("notebooks/52_attitude_control.py", "gnc", ("attitude_control",), ("gnc", "rcs", "control")),
    NotebookSpec("notebooks/53_navigation_filters.py", "gnc", ("navigation_filters",), ("nav",)),
    NotebookSpec("notebooks/60_loads_environments.py", "structures", ("loads_environments",), ("loads", "docking")),
    NotebookSpec("notebooks/61_primary_structure.py", "structures", ("primary_structure",), ("structure", "mass")),
    NotebookSpec("notebooks/62_thermostructural.py", "structures", ("thermostructural",), ("tps",)),
    NotebookSpec("notebooks/70_thermal_control.py", "power", ("thermal_control",), ("thermal",)),
    NotebookSpec("notebooks/71_power_system.py", "power", ("power_system",), ("power", "phase")),
    NotebookSpec("notebooks/80_oml_parametric.py", "mrv", ("mrv_oml",), ("mrv", "mrv_payload")),
    NotebookSpec("notebooks/81_packaging_payload.py", "mrv", ("mrv_payload_packaging",), ("mrv", "mrv_payload")),
    NotebookSpec("notebooks/82_openscad_export.py", "mrv", required_inputs=("mrv",), notes="Artifact preview/check."),
    NotebookSpec("notebooks/90_mdao_sizing_loop.py", "mdao", ("mdao_convergence",), ("mdao", "propulsion", "entry")),
    NotebookSpec("notebooks/91_trade_studies.py", "mdao", ("trade_studies",), ("trade", "entry")),
)


def notebook_paths() -> tuple[Path, ...]:
    return tuple(spec.notebook_path for spec in THOR_NOTEBOOKS)


def expected_tables() -> tuple[str, ...]:
    tables: list[str] = []
    for spec in THOR_NOTEBOOKS:
        tables.extend(spec.tables)
    return tuple(tables)


def manifest_by_path() -> dict[str, NotebookSpec]:
    return {spec.path: spec for spec in THOR_NOTEBOOKS}
