"""60 — Loads environments: launch, reentry, docking, landing."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import load_state, load_table, save_table
    from thor.io.inputs import estimated_wet_mass_kg, num
    return estimated_wet_mass_kg, load_state, load_table, mo, num, pl, save_table


@app.cell
def _(mo):
    mo.md("# Camada 6 — Loads Environments\n\nInputs: `loads` + derived reentry/docking")
    return


@app.cell
def _(estimated_wet_mass_kg, load_state, load_table, num, pl):
    state = load_state()
    traj = load_table("entry_trajectory")
    g_peak = float(traj["g_load"].max()) if traj is not None else num("mission", "max_g_load")
    m_dock = state.mass.wet_mass_kg or state.mass.dry_mass_kg or estimated_wet_mass_kg()
    f_dock = m_dock * num("docking", "contact_accel_m_s2")
    df = pl.DataFrame(
        {
            "case": ["Launch quasi-static", "Launch acoustic", "Reentry", "Docking", "Landing"],
            "load": [num("loads", "launch_g"), num("loads", "acoustic_OASPL"), g_peak, f_dock, num("loads", "landing_g")],
            "unit": ["g", "dB OASPL", "g", "N", "g"],
        }
    )
    return df, f_dock, g_peak, m_dock, state, traj


@app.cell
def _(df, mo, save_table):
    mo.ui.table(df)
    save_table("loads_environments", df)
    mo.md("✓ Casos de carga → 61_primary_structure")
    return


if __name__ == "__main__":
    app.run()
