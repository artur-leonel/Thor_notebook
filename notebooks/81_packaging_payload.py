"""81 — Astreia-MRV payload packaging & fit."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from astreia_mrv.analysis.constraints import payload_fit
    from thor.io.openscad import export_mrv_to_openscad
    from thor.io.handoff import load_state, save_state, save_table
    from thor.io.inputs import txt

    return export_mrv_to_openscad, load_state, mo, payload_fit, pl, save_state, save_table, txt


@app.cell
def _(mo):
    mo.md("# Layer 8.1 — MRV Payload Packaging\n\nPayload dimensions come from the MRV YAML config selected by `mrv.config_path` in `thor_inputs.csv`; the synced `mrv_payload` CSV rows are checked by the pipeline tests.")
    return


@app.cell
def _(export_mrv_to_openscad, payload_fit, txt):
    config_path = txt("mrv", "config_path", default="configs/mrv3.yaml")
    out_dir = txt("mrv", "output_dir", default="outputs/mrv3_notebook")
    design, geometry, _ = export_mrv_to_openscad(config_path, out_dir)
    fit = payload_fit(design, geometry)
    return config_path, design, fit, geometry


@app.cell
def _(design, fit, mo, pl, save_table):
    df = pl.DataFrame(
        {
            "payload": [design.payload.name],
            "mass_kg": [design.payload.mass_kg],
            "length_m": [design.payload.length_m],
            "width_m": [design.payload.width_m],
            "height_m": [design.payload.height_m],
            "fit_passed": [fit.passed],
            "fit_margin_m": [fit.margin],
        }
    )
    save_table("mrv_payload_packaging", df)
    mo.vstack([mo.md(f"**Payload fit:** `{fit.passed}` margin `{fit.margin:.3f} m`"), mo.ui.table(df)])
    return (df,)


@app.cell
def _(design, load_state, mo, save_state):
    state = load_state()
    state.notes["mrv_payload"] = (
        f"{design.payload.name}, {design.payload.mass_kg:.0f} kg, terminal recovery into authorized zone."
    )
    save_state(state)
    mo.md("Saved MRV payload note to `vehicle_state.json`.")
    return (state,)


if __name__ == "__main__":
    app.run()
