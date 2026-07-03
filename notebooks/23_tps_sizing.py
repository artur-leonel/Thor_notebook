"""23 — TPS sizing: condução 1D transiente, espessura → massa."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import load_state, load_table, save_state, save_table
    from thor.io.inputs import num
    return load_state, load_table, mo, num, pl, save_state, save_table


@app.cell
def _(mo):
    mo.md("# Camada 2 — TPS Sizing\n\nInputs: `tps` section in `thor_inputs.csv`")
    return


@app.cell
def _(load_table):
    heat = load_table("aeroheating")
    q_peak = float(heat["q_W_cm2"].max()) if heat is not None else 800.0
    return heat, q_peak


@app.cell
def _(mo, num, q_peak):
    rho_tps = num("tps", "rho_kg_m3")
    k = num("tps", "conductivity_W_mK")
    cp = num("tps", "cp_J_kgK")
    t_peak = num("tps", "exposure_time_s")
    area_tps = num("tps", "shield_area_m2")
    q_w_m2 = q_peak * 1e4
    alpha = k / (rho_tps * cp)
    thickness_m = 2 * (alpha * t_peak) ** 0.5 + q_w_m2 * 0.001 / k
    mass_tps = rho_tps * thickness_m * area_tps
    mo.vstack([
        mo.md(f"**Espessura TPS:** {thickness_m*1000:.1f} mm"),
        mo.md(f"**Massa TPS:** {mass_tps:.0f} kg → feedback 01_mass_budget"),
    ])
    return alpha, area_tps, cp, k, mass_tps, rho_tps, t_peak, thickness_m


@app.cell
def _(load_state, mass_tps, mo, pl, save_state, save_table, thickness_m):
    state = load_state()
    for item in state.mass.items:
        if item.name == "TPS":
            item.dry_kg = mass_tps
            item.growth_kg = mass_tps * state.mass.growth_allowance
    cd, area = state.aero.cd, state.aero.reference_area_m2
    state.entry.ballistic_coefficient_kg_m2 = state.mass.dry_mass_kg / (cd * area)
    save_state(state)
    save_table("tps_sizing", pl.DataFrame({"thickness_mm": [thickness_m * 1000], "mass_kg": [mass_tps]}))
    mo.md(f"✓ β atualizado: {state.entry.ballistic_coefficient_kg_m2:.0f} kg/m²")
    return cd, state


if __name__ == "__main__":
    app.run()
