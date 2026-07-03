"""21 — Entry trajectory 3DOF + Allen-Eggers sanity check."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import load_state, save_table
    from thor.io.inputs import num
    from thor.physics.entry import allen_eggers_peak_heat, integrate_entry_3dof
    return allen_eggers_peak_heat, integrate_entry_3dof, load_state, mo, num, pl, save_table


@app.cell
def _(mo):
    mo.md("# Camada 2 — Entry Trajectory 3DOF\n\nInputs: `entry` + `aero` in `thor_inputs.csv`")
    return


@app.cell
def _(integrate_entry_3dof, load_state):
    import math

    state = load_state()
    e = state.entry
    traj = integrate_entry_3dof(
        e.altitude_m,
        e.velocity_m_s,
        math.radians(e.flight_path_angle_deg),
        e.ballistic_coefficient_kg_m2,
    )
    return e, math, state, traj


@app.cell
def _(allen_eggers_peak_heat, e, mo, num, traj):
    import numpy as np

    g_peak = float(np.max(traj["g_load"]))
    q_peak = float(np.max(traj["q_W_cm2"]))
    rho_ei = num("entry", "rho_ei_kg_m3")
    rn = num("aero", "nose_radius_m")
    q_ae, _ = allen_eggers_peak_heat(e.velocity_m_s, rho_ei, rn)
    mo.vstack([
        mo.md(f"**Pico g-load:** {g_peak:.2f} g"),
        mo.md(f"**Pico q̇:** {q_peak:.0f} W/cm² (Allen-Eggers ~{q_ae/1e4:.0f} W/cm²)"),
    ])
    return g_peak, np, q_ae, q_peak, rn, rho_ei


@app.cell
def _(mo, pl, save_table, traj):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(traj["t_s"], traj["v_m_s"] / 1000)
    axes[0].set(xlabel="t [s]", ylabel="V [km/s]", title="Velocidade")
    axes[1].plot(traj["t_s"], traj["g_load"])
    axes[1].set(xlabel="t [s]", ylabel="g", title="Desaceleração")
    axes[2].plot(traj["t_s"], traj["q_W_cm2"])
    axes[2].set(xlabel="t [s]", ylabel="q̇ [W/cm²]", title="Calor")
    plt.tight_layout()

    df = pl.DataFrame(
        {
            "t_s": traj["t_s"].tolist(),
            "h_km": (traj["h_m"] / 1000).tolist(),
            "v_m_s": traj["v_m_s"].tolist(),
            "g_load": traj["g_load"].tolist(),
            "q_W_cm2": traj["q_W_cm2"].tolist(),
        }
    )
    save_table("entry_trajectory", df)
    mo.vstack([fig, mo.md("✓ Trajetória 3DOF → parquet")])
    return axes, df, fig, plt


if __name__ == "__main__":
    app.run()
