"""90 — MDAO sizing loop: converge massa, β, TPS, propelente."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import load_state, save_state, save_table
    from thor.io.inputs import num
    from thor.physics.entry import integrate_entry_3dof
    from thor.physics.propulsion import propellant_mass
    return integrate_entry_3dof, load_state, mo, num, pl, propellant_mass, save_state, save_table


@app.cell
def _(mo):
    mo.md("# Camada 9 — MDAO Sizing Loop\n\nInputs: `mdao` + `propulsion` + `entry`")
    return


@app.cell
def _(integrate_entry_3dof, load_state, num, propellant_mass):
    import math

    state = load_state()
    n_iter = int(num("mdao", "iterations"))
    isp = num("propulsion", "isp_s")
    dv = num("propulsion", "deorbit_dv_m_s")
    tps_base = num("mdao", "tps_base_kg")
    tps_q = num("mdao", "tps_q_factor")
    history = []
    for it in range(n_iter):
        m_dry = state.mass.dry_mass_kg
        cd, area = state.aero.cd, state.aero.reference_area_m2
        beta = m_dry / (cd * area) if cd * area else state.entry.ballistic_coefficient_kg_m2
        mp = propellant_mass(m_dry + state.mass.propellant_kg, isp, dv)
        state.mass.propellant_kg = mp
        state.entry.ballistic_coefficient_kg_m2 = beta
        traj = integrate_entry_3dof(
            state.entry.altitude_m,
            state.entry.velocity_m_s,
            math.radians(state.entry.flight_path_angle_deg),
            beta,
        )
        g_peak = float(traj["g_load"].max())
        q_peak = float(traj["q_W_cm2"].max())
        history.append({"iter": it, "m_dry": m_dry, "beta": beta, "g_peak": g_peak, "q_peak": q_peak})
        for item in state.mass.items:
            if item.name == "TPS":
                item.dry_kg = tps_base + q_peak * tps_q
    return beta, cd, g_peak, history, isp, math, m_dry, mp, n_iter, q_peak, state, tps_base, tps_q, traj


@app.cell
def _(history, mo, pl):
    df = pl.DataFrame(history)
    mo.ui.table(df)
    return df


@app.cell
def _(df, mo, save_state, save_table, state):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df["iter"].to_numpy(), df["m_dry"].to_numpy(), "o-", label="m_dry")
    ax.set(xlabel="iteração", ylabel="kg", title="Convergência MDAO")
    ax.legend()

    save_state(state)
    save_table("mdao_convergence", df)
    mo.vstack([fig, mo.md("✓ Veículo convergido → vehicle_state.json")])
    return ax, fig, plt


if __name__ == "__main__":
    app.run()
