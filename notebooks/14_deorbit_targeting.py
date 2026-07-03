"""14 — Deorbit targeting → estado na Entry Interface (122 km)."""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import math
    import polars as pl

    from thor.constants import MU_EARTH, R_EARTH
    from thor.io.handoff import load_state, save_state, save_table
    from thor.io.inputs import num
    from thor.models.vehicle_state import EntryState
    return EntryState, MU_EARTH, R_EARTH, load_state, math, mo, num, pl, save_state, save_table


@app.cell
def _(mo):
    mo.md("# Camada 1 — Deorbit Targeting\n\nInputs: `entry` section in `thor_inputs.csv`")
    return


@app.cell
def _(EntryState, MU_EARTH, R_EARTH, load_state, math, num):
    _state = load_state()
    alt_orbit = _state.orbit.altitude_km * 1000 if _state.orbit else num("orbit", "altitude_km") * 1000
    r = R_EARTH + alt_orbit
    v_circ = math.sqrt(MU_EARTH / r)
    dv_deorbit = num("entry", "deorbit_dv_m_s")
    v_override = num("entry", "velocity_m_s")
    v_ei = (
        v_override
        if v_override > 0
        else math.sqrt(max(v_circ**2 + dv_deorbit**2 - 2 * v_circ * dv_deorbit, 0))
    )
    entry = EntryState(
        altitude_m=num("entry", "altitude_m"),
        velocity_m_s=v_ei,
        flight_path_angle_deg=num("entry", "flight_path_angle_deg"),
        heading_deg=num("entry", "heading_deg"),
        ballistic_coefficient_kg_m2=num("entry", "ballistic_coefficient_kg_m2"),
    )
    return (entry,)


@app.cell
def _(entry, mo, pl):
    df = pl.DataFrame(
        {
            "parameter": ["h_EI [km]", "V_EI [m/s]", "γ [deg]", "ψ [deg]", "β [kg/m²]"],
            "value": [
                entry.altitude_m / 1000,
                round(entry.velocity_m_s, 1),
                entry.flight_path_angle_deg,
                entry.heading_deg,
                entry.ballistic_coefficient_kg_m2,
            ],
        }
    )
    mo.ui.table(df)
    return


@app.cell
def _(entry, load_state, mo, pl, save_state, save_table):
    _state = load_state()
    _state.entry = entry
    save_state(_state)
    save_table(
        "entry_interface",
        pl.DataFrame({"key": ["h", "V", "gamma"], "value": [entry.altitude_m, entry.velocity_m_s, entry.flight_path_angle_deg]}),
    )
    mo.md("✓ EntryState → vehicle_state (entrada Camada 2)")
    return


if __name__ == "__main__":
    app.run()
