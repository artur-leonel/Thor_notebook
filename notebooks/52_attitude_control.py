"""52 — Attitude control: RCS / wheels."""

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
    mo.md("# Camada 5 — Attitude Control\n\nInputs: `gnc` + `rcs` in `thor_inputs.csv`")
    return


@app.cell
def _(load_table, num):
    rcs = load_table("rcs_sizing")
    control_authority = load_table("control_authority")
    torque_req = num("gnc", "torque_required_Nm")
    torque_avail = float(rcs["torque_Nm"][0]) if rcs is not None else num("rcs", "thrust_N") * num("rcs", "arm_m")
    aero_authority_pass = bool(control_authority["passes"].all()) if control_authority is not None else False
    return aero_authority_pass, control_authority, rcs, torque_avail, torque_req


@app.cell
def _(aero_authority_pass, mo, pl, torque_avail, torque_req):
    rcs_torque_pass = torque_avail >= torque_req
    df = pl.DataFrame(
        {
            "metric": ["tau_req_Nm", "tau_RCS_Nm", "torque_margin_Nm", "aero_authority_pass", "rcs_torque_pass"],
            "value": [float(torque_req), float(torque_avail), float(torque_avail - torque_req), float(aero_authority_pass), float(rcs_torque_pass)],
            "status": ["", "", "", "pass" if aero_authority_pass else "needs revision", "pass" if rcs_torque_pass else "needs revision"],
        }
    )
    mo.ui.table(df)
    return df


@app.cell
def _(df, mo, save_table):
    save_table("attitude_control", df)
    mo.md("✓ Autoridade de controle → parquet")
    return


if __name__ == "__main__":
    app.run()
