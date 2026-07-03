"""12 — Rendezvous phasing: Hohmann + Clohessy-Wiltshire → ΔV braking."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl

    from thor.io.handoff import load_state, save_state, save_table
    from thor.io.inputs import num
    from thor.physics.astro import hohmann_delta_v, hill_clohessy_wiltshire, mean_motion
    return (
        hill_clohessy_wiltshire,
        hohmann_delta_v,
        load_state,
        mean_motion,
        mo,
        np,
        num,
        pl,
        save_state,
        save_table,
    )


@app.cell
def _(mo):
    mo.md("# Camada 1 — Rendezvous & Phasing\n\nInputs: `rendezvous` + `orbit` in `thor_inputs.csv`")
    return


@app.cell
def _(hohmann_delta_v, load_state, num):
    _state = load_state()
    chaser_alt = num("rendezvous", "chaser_alt_km") * 1000
    target_alt = _state.orbit.altitude_km * 1000 if _state.orbit else num("orbit", "altitude_km") * 1000
    dv1, dv2 = hohmann_delta_v(chaser_alt, target_alt)
    dv_braking = abs(dv2) + num("rendezvous", "braking_margin_m_s")
    return chaser_alt, dv1, dv2, dv_braking, target_alt


@app.cell
def _(dv1, dv2, dv_braking, hill_clohessy_wiltshire, mean_motion, mo, np, num, pl, target_alt):
    n = mean_motion(target_alt)
    x0 = np.array([num("rendezvous", "rel_x_m"), 0, 0, 0, num("rendezvous", "rel_vy_m_s"), 0])
    dt = num("rendezvous", "cw_dt_s")
    xf = hill_clohessy_wiltshire(n, dt, x0)
    df = pl.DataFrame(
        {"maneuver": ["Hohmann burn 1", "Hohmann burn 2", "Braking (CW)"], "delta_v_m_s": [dv1, dv2, dv_braking]}
    )
    mo.vstack([
        mo.md(f"**ΔV phasing total:** {dv1 + dv2 + dv_braking:.1f} m/s"),
        mo.md(f"Estado relativo final: x={xf[0]:.0f} m, y={xf[1]:.0f} m"),
        mo.ui.table(df),
    ])
    return df, dt, n, x0, xf


@app.cell
def _(df, dv_braking, load_state, mo, save_state, save_table):
    _state = load_state()
    for item in _state.delta_v.items:
        if "phasing" in item.phase or "rendezvous" in item.phase:
            item.delta_v_m_s = dv_braking if "rendezvous" in item.phase else item.delta_v_m_s
    save_state(_state)
    save_table("phasing_dv", df)
    mo.md("✓ ΔV braking → delta_v budget")
    return


if __name__ == "__main__":
    app.run()
