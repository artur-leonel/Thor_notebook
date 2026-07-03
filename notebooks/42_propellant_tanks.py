"""42 — Propellant tanks: volume, pressão, blowdown."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import load_state, save_state, save_table
    from thor.io.inputs import num
    return load_state, mo, num, pl, save_state, save_table


@app.cell
def _(mo):
    mo.md("# Camada 4 — Propellant Tanks\n\nInputs: `propulsion` section in `thor_inputs.csv`")
    return


@app.cell
def _(load_state, num):
    _state = load_state()
    mp = _state.mass.propellant_kg or num("mass", "propellant_kg", "_config")
    rho_prop = num("propulsion", "rho_propellant_kg_m3")
    ullage = num("propulsion", "ullage_frac")
    volume_m3 = mp / rho_prop * (1 + ullage)
    p_tank = num("propulsion", "tank_pressure_bar") * 1e5
    tank_mass = volume_m3 * num("propulsion", "tank_mass_kg_m3")
    return mp, p_tank, rho_prop, tank_mass, volume_m3


@app.cell
def _(mo, mp, p_tank, pl, tank_mass, volume_m3):
    df = pl.DataFrame(
        {
            "parameter": ["V_tank [m³]", "P [bar]", "m_prop [kg]", "m_tank [kg]"],
            "value": [round(volume_m3, 2), p_tank / 1e5, mp, round(tank_mass, 0)],
        }
    )
    mo.ui.table(df)
    return df


@app.cell
def _(load_state, mo, pl, save_state, save_table, tank_mass):
    _state = load_state()
    for item in _state.mass.items:
        if "Propulsion" in item.name:
            item.dry_kg += tank_mass
    save_state(_state)
    save_table("propellant_tanks", pl.DataFrame({"tank_mass_kg": [tank_mass]}))
    mo.md("✓ Tanques → mass budget (loop fechado com 01)")
    return


if __name__ == "__main__":
    app.run()
