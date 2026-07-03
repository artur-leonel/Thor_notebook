"""40 — Main propulsion: Tsiolkovsky, Isp, deorbit motor."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.constants import G0
    from thor.io.handoff import load_state, save_state, save_table
    from thor.io.inputs import estimated_wet_mass_kg, num
    from thor.physics.propulsion import propellant_mass, tsiolkovsky_delta_v
    return G0, estimated_wet_mass_kg, load_state, mo, num, pl, propellant_mass, save_state, save_table, tsiolkovsky_delta_v


@app.cell
def _(mo):
    mo.md("# Camada 4 — Main Propulsion\n\nInputs: `propulsion` section in `thor_inputs.csv`")
    return


@app.cell
def _(estimated_wet_mass_kg, load_state, num, propellant_mass):
    _state = load_state()
    isp = num("propulsion", "isp_s")
    dv_req = num("propulsion", "deorbit_dv_m_s")
    m0 = _state.mass.wet_mass_kg or estimated_wet_mass_kg()
    mp = propellant_mass(m0, isp, dv_req)
    mf = m0 - mp
    return dv_req, isp, m0, mf, mp


@app.cell
def _(dv_req, isp, mf, mo, m0, mp, num, pl, tsiolkovsky_delta_v):
    dv_check = tsiolkovsky_delta_v(isp, m0, mf)
    thrust = num("propulsion", "thrust_N")
    df = pl.DataFrame(
        {
            "parameter": ["Isp [s]", "ΔV req [m/s]", "ΔV check", "m_prop [kg]", "Thrust [N]"],
            "value": [isp, dv_req, round(dv_check, 1), round(mp, 1), thrust],
        }
    )
    mo.ui.table(df)
    return df, dv_check, thrust


@app.cell
def _(isp, load_state, mf, mo, mp, pl, save_state, save_table):
    _state = load_state()
    _state.mass.propellant_kg = mp
    save_state(_state)
    save_table("main_propulsion", pl.DataFrame({"isp_s": [isp], "propellant_kg": [mp], "mf_kg": [mf]}))
    mo.md("✓ Propelente → mass budget (fecha loop com 01/02)")
    return


if __name__ == "__main__":
    app.run()
