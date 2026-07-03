"""20 — Atmosphere: US Standard 1976 + NRLMSISE-00 placeholder."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl

    from thor.io.handoff import save_table
    from thor.io.inputs import num
    from thor.physics.atmosphere import nrlmsise00_placeholder, us76_density
    return mo, np, num, pl, save_table, us76_density, nrlmsise00_placeholder


@app.cell
def _(mo):
    mo.md("# Camada 2 — Atmosphere Model\n\nInputs: `atmosphere` in `thor_inputs.csv`")
    return


@app.cell
def _(np, num, nrlmsise00_placeholder, pl, us76_density):
    h = np.linspace(num("atmosphere", "h_min_m"), num("atmosphere", "h_max_m"), int(num("atmosphere", "n_points")))
    df = pl.DataFrame(
        {
            "h_km": h / 1000,
            "rho_us76": [us76_density(x) for x in h],
            "rho_nrlmsise": [nrlmsise00_placeholder(x) for x in h],
        }
    )
    return df, h


@app.cell
def _(df, mo):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(df["h_km"].to_numpy(), df["rho_us76"].to_numpy(), label="US76")
    ax.semilogy(df["h_km"].to_numpy(), df["rho_nrlmsise"].to_numpy(), "--", label="NRLMSISE (placeholder)")
    ax.set(xlabel="Altitude [km]", ylabel="ρ [kg/m³]")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig
    return ax, fig, plt


@app.cell
def _(df, mo, save_table):
    save_table("atmosphere", df)
    mo.md("✓ Perfil atmosférico → parquet")
    return


if __name__ == "__main__":
    app.run()
