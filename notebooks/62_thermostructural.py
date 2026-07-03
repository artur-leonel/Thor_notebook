"""62 — Thermostructural: heat-soak pós-pico."""

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
    mo.md("# Camada 6 — Thermostructural\n\nInputs: `tps` + aeroheating from 22")
    return


@app.cell
def _(load_table, num):
    heat = load_table("aeroheating")
    tps = load_table("tps_sizing")
    q_peak = float(heat["q_W_cm2"].max()) if heat is not None else num("mission", "peak_heat_flux_W_cm2")
    t_tps = float(tps["thickness_mm"][0] / 1000) if tps is not None else 0.05
    return heat, q_peak, tps, t_tps


@app.cell
def _(mo, num, pl, q_peak, t_tps):
    k = num("tps", "conductivity_W_mK")
    limit = num("tps", "bondline_temp_C")
    t_bondline = q_peak * 1e4 * t_tps / k
    df = pl.DataFrame(
        {
            "parameter": ["q_peak [W/cm²]", "ΔT_bondline [K]", "limit [K]"],
            "value": [q_peak, round(t_bondline, 0), limit],
        }
    )
    ok = t_bondline < limit
    mo.vstack([mo.md(f"**Bondline OK:** {ok}"), mo.ui.table(df)])
    return df, k, limit, ok, t_bondline


@app.cell
def _(df, mo, save_table):
    save_table("thermostructural", df)
    mo.md("✓ Acoplamento térmico-estrutural")
    return


if __name__ == "__main__":
    app.run()
