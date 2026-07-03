"""30 — Aero database: Cl/Cd/Cm vs Mach × α (todos regimes)."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl

    from astreia_mrv.analysis.aerodynamics import aerodynamic_coefficients
    from astreia_mrv.config import load_design
    from astreia_mrv.geometry.lifting_body import generate_lifting_body
    from thor.io.handoff import save_table
    from thor.io.inputs import float_list, num, txt
    return aerodynamic_coefficients, float_list, generate_lifting_body, load_design, mo, np, num, pl, save_table, txt


@app.cell
def _(mo):
    mo.md("# Camada 3 — Aero Database\n\nInputs: `aero_db` in `thor_inputs.csv`")
    return


@app.cell
def _(aerodynamic_coefficients, float_list, generate_lifting_body, load_design, np, num, pl, txt):
    config_path = txt("mrv", "config_path", default="configs/mrv3.yaml")
    geometry = generate_lifting_body(load_design(config_path))
    machs = float_list("aero_db", "mach_list")
    alphas = np.linspace(num("aero_db", "alpha_min"), num("aero_db", "alpha_max"), int(num("aero_db", "alpha_steps")))
    rows = []
    for _mach in machs:
        for a in alphas:
            coeff = aerodynamic_coefficients(geometry, mach=float(_mach), alpha_deg=float(a))
            rows.append(
                {
                    "Mach": _mach,
                    "alpha_deg": a,
                    "Cd": coeff["CD"],
                    "Cl": coeff["CL"],
                    "Cm": coeff["Cm"],
                    "CY": coeff["CY"],
                    "Cl_roll": coeff["Cl"],
                    "Cn_yaw": coeff["Cn"],
                    "x_cp_m": coeff["x_cp_m"],
                    "source": "Astreia MRV mesh",
                }
            )
    df = pl.DataFrame(rows)
    return alphas, config_path, df, geometry, machs, rows


@app.cell
def _(df, mo, pl):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for _mach in df["Mach"].unique().sort().to_list():
        sub = df.filter(pl.col("Mach") == _mach)
        ax.plot(sub["alpha_deg"].to_numpy(), sub["Cl"].to_numpy(), label=f"M={_mach}")
    ax.set(xlabel="α [deg]", ylabel="Cl", title="Cl(α) por Mach")
    ax.legend(fontsize=8)
    fig
    return ax, fig, plt


@app.cell
def _(config_path, df, mo, save_table):
    save_table("aero_database", df)
    mo.md(f"✓ MRV mesh AeroDB from `{config_path}` → parquet")
    return


if __name__ == "__main__":
    app.run()
