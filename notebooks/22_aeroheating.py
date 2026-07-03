"""22 — Aeroheating: Sutton-Graves, calor integrado → TPS."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import load_table, save_table
    from thor.io.inputs import num
    return load_table, mo, num, pl, save_table


@app.cell
def _(mo):
    mo.md("# Camada 2 — Aeroheating\n\nInputs: `aero` (Rn) + trajectory from 21")
    return


@app.cell
def _(load_table):
    traj = load_table("entry_trajectory")
    return (traj,)


@app.cell
def _(mo, num, pl, save_table, traj):
    import numpy as np

    rn = num("aero", "nose_radius_m")
    if traj is None:
        status = mo.md("⚠ Execute 21_entry_trajectory_3dof primeiro")
        dt = np.array([])
        q = np.array([])
        q_total = 0.0
        df = pl.DataFrame({"t_s": [], "q_W_cm2": [], "Q_integrated_J_cm2": []})
        fig = ax = ax2 = plt = None
    else:
        dt = np.diff(traj["t_s"].to_numpy(), prepend=0)
        q = traj["q_W_cm2"].to_numpy()
        Q_J_cm2 = np.cumsum(q * dt) / 1e4
        df = pl.DataFrame({"t_s": traj["t_s"], "q_W_cm2": q, "Q_integrated_J_cm2": Q_J_cm2})
        q_total = float(Q_J_cm2[-1])

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(df["t_s"].to_numpy(), df["q_W_cm2"].to_numpy(), label="q̇")
        ax.set(xlabel="t [s]", ylabel="W/cm²")
        ax2 = ax.twinx()
        ax2.plot(df["t_s"].to_numpy(), df["Q_integrated_J_cm2"].to_numpy(), "r--", label="∫Q")
        ax2.set_ylabel("J/cm²")
        ax.set_title(f"Calor total ≈ {q_total:.0f} J/cm² (Rn={rn} m)")
        _plot = fig

        save_table("aeroheating", df)
        status = mo.vstack([_plot, mo.md(f"✓ Q_total = **{q_total:.0f} J/cm²** → dimensiona TPS (23)")])
    status
    return ax, ax2, df, dt, fig, np, plt, q, q_total, rn, status


if __name__ == "__main__":
    app.run()
