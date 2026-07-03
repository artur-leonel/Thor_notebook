"""24 — Hypersonic aero: Newtoniano modificado → Cl/Cd/Cm."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl

    from astreia_mrv.analysis.aerodynamics import alpha_sweep
    from astreia_mrv.config import load_design
    from astreia_mrv.geometry.lifting_body import generate_lifting_body
    from thor.io.handoff import load_state, save_state, save_table
    from thor.io.inputs import float_list, num, txt
    return alpha_sweep, float_list, generate_lifting_body, load_design, load_state, mo, np, num, pl, save_state, save_table, txt


@app.cell
def _(mo):
    mo.md("# Camada 2 — Hypersonic Aero\n\nInputs: `hypersonic` in `thor_inputs.csv`")
    return


@app.cell
def _(alpha_sweep, float_list, generate_lifting_body, load_design, np, txt):
    config_path = txt("mrv", "config_path", default="configs/mrv3.yaml")
    geometry = generate_lifting_body(load_design(config_path))
    mach = max(float_list("aero_db", "mach_list"))
    alphas = np.linspace(-10, 40, 26)
    rows = []
    for row in alpha_sweep(geometry, mach=mach, alpha_deg=alphas):
        cd = row["CD"]
        cl = row["CL"]
        rows.append({**row, "Cd": cd, "Cl": cl, "L_D": cl / cd if cd else 0.0, "source": "Astreia MRV mesh"})
    df = pl.DataFrame(rows)
    return alphas, config_path, df, geometry, mach, rows


@app.cell
def _(df, mo):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["alpha_deg"].to_numpy(), df["CD"].to_numpy(), label="CD")
    ax.plot(df["alpha_deg"].to_numpy(), df["CL"].to_numpy(), label="CL")
    ax.set(xlabel="α [deg]", ylabel="Coeff", title="MRV mesh aero (modified Newtonian)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig
    return ax, fig, plt


@app.cell
def _(config_path, df, load_state, mo, pl, save_state, save_table):
    state = load_state()
    trim = df.filter(pl.col("L_D") == df["L_D"].max()).row(0, named=True)
    state.aero.cd = trim["CD"]
    state.aero.cl = trim["CL"]
    state.notes["hypersonic_aero"] = f"MRV mesh-derived alpha sweep from {config_path}"
    save_state(state)
    save_table("hypersonic_aero", df)
    mo.md(f"✓ MRV mesh aero: max L/D={trim['L_D']:.2f} @ α={trim['alpha_deg']:.0f}°")
    return state, trim


if __name__ == "__main__":
    app.run()
