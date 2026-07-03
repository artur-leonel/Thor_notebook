"""13 — Proximity ops & docking: V-bar/R-bar, closing velocity."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import load_state, save_table
    from thor.io.inputs import estimated_wet_mass_kg, item_table, num
    return estimated_wet_mass_kg, item_table, load_state, mo, num, pl, save_table


@app.cell
def _(mo):
    mo.md("# Camada 1 — Proximity Ops & Docking\n\nInputs: `docking` in `thor_inputs.csv`")
    return


@app.cell
def _(estimated_wet_mass_kg, item_table, load_state, num):
    approach = item_table("docking").rename({"item": "gate"})
    state = load_state()
    m_dock = state.mass.wet_mass_kg or state.mass.dry_mass_kg or estimated_wet_mass_kg()
    a_contact = num("docking", "contact_accel_m_s2")
    f_dock = m_dock * a_contact
    return a_contact, approach, f_dock, m_dock, state


@app.cell
def _(approach, f_dock, mo):
    mo.vstack([
        mo.md(f"**Carga de contato estimada:** {f_dock:.0f} N → alimenta 60_loads"),
        mo.ui.table(approach),
    ])
    return


@app.cell
def _(approach, mo, save_table):
    save_table("docking_approach", approach)
    mo.md("✓ Approach profile → parquet")
    return


if __name__ == "__main__":
    app.run()
