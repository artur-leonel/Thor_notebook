"""91 — Trade studies: cápsula vs lifting body, Pareto."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl

    from thor.io.handoff import save_table
    from thor.io.inputs import entry_velocity_m_s, num
    from thor.physics.entry import integrate_entry_3dof
    return entry_velocity_m_s, integrate_entry_3dof, mo, np, num, pl, save_table


@app.cell
def _(mo):
    mo.md("# Camada 9 — Trade Studies\n\nInputs: `trade` + `entry` in `thor_inputs.csv`")
    return


@app.cell
def _(entry_velocity_m_s, integrate_entry_3dof, num, np):
    import math

    betas = np.linspace(num("trade", "beta_min"), num("trade", "beta_max"), int(num("trade", "beta_steps")))
    h_ei = num("entry", "altitude_m")
    gamma = math.radians(num("entry", "flight_path_angle_deg"))
    v_ei = entry_velocity_m_s()
    s_ref = num("trade", "s_ref_m2")
    configs = ["capsule", "lifting_body"]
    rows = []
    for _cfg in configs:
        cd = num("trade", "cd", _cfg)
        cl = num("trade", "cl", _cfg)
        for beta in betas:
            m = beta * cd * s_ref
            traj = integrate_entry_3dof(h_ei, v_ei, gamma, beta)
            rows.append(
                {
                    "config": _cfg,
                    "beta": beta,
                    "mass_kg": m,
                    "g_peak": float(traj["g_load"].max()),
                    "q_peak": float(traj["q_W_cm2"].max()),
                    "L_D": cl / cd,
                }
            )
    df = pl.DataFrame(rows)
    return betas, configs, df, gamma, h_ei, math, rows, s_ref, v_ei


@app.cell
def _(df, mo, pl):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for _cfg in df["config"].unique().to_list():
        sub = df.filter(pl.col("config") == _cfg)
        ax.scatter(sub["mass_kg"].to_numpy(), sub["q_peak"].to_numpy(), label=_cfg, s=60)
    ax.set(xlabel="Massa [kg]", ylabel="q_peak [W/cm²]", title="Pareto: massa × calor")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig
    return ax, fig, plt


@app.cell
def _(df, mo, pl, save_table):
    save_table("trade_studies", df)
    best = df.filter(pl.col("config") == "lifting_body").sort("q_peak").row(0, named=True)
    mo.md(f"**Lifting body mínimo q:** m={best['mass_kg']:.0f} kg, q={best['q_peak']:.0f} W/cm²")
    return best


if __name__ == "__main__":
    app.run()
