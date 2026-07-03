"""61 — Primary structure sizing."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import load_state, save_table
    from thor.io.inputs import estimated_dry_mass_kg, num
    return estimated_dry_mass_kg, load_state, mo, num, pl, save_table


@app.cell
def _(mo):
    mo.md("# Camada 6 — Primary Structure\n\nInputs: `structure` in `thor_inputs.csv`")
    return


@app.cell
def _(estimated_dry_mass_kg, load_state, num):
    state = load_state()
    m_struct = num("structure", "m_structure_kg")
    sigma_allow = num("structure", "sigma_allow_Pa")
    load_g = num("structure", "load_g")
    mass = state.mass.dry_mass_kg or estimated_dry_mass_kg()
    area_load = mass * 9.81 * load_g / sigma_allow
    return area_load, load_g, m_struct, mass, sigma_allow, state


@app.cell
def _(area_load, m_struct, mo, num, pl):
    df = pl.DataFrame(
        {
            "item": ["m_structure [kg]", "A_min [m²]", "sigma_allow [MPa]"],
            "value": [m_struct, round(area_load, 3), num("structure", "sigma_allow_Pa") / 1e6],
        }
    )
    mo.ui.table(df)
    return df


@app.cell
def _(df, mo, save_table):
    save_table("primary_structure", df)
    mo.md("✓ Estrutura primária → parquet")
    return


if __name__ == "__main__":
    app.run()
