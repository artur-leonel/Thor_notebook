"""41 — RCS sizing: thrusters, phase-plane."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import save_table
    from thor.io.inputs import num
    return mo, num, pl, save_table


@app.cell
def _(mo):
    mo.md("# Camada 4 — RCS Sizing\n\nInputs: `rcs` in `thor_inputs.csv`")
    return


@app.cell
def _(mo, num, pl):
    thrust_rcs = num("rcs", "thrust_N")
    arm = num("rcs", "arm_m")
    torque = thrust_rcs * arm
    n_thrusters = int(num("rcs", "n_thrusters"))
    df = pl.DataFrame(
        {
            "axis": ["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
            "thrusters": [2, 2, 2, 2, 2, 2],
            "thrust_N": [thrust_rcs] * 6,
            "torque_Nm": [torque] * 6,
        }
    )
    mo.vstack([
        mo.md(f"**Torque max:** {torque:.0f} N·m ({n_thrusters} thrusters)"),
        mo.ui.table(df),
    ])
    return arm, df, n_thrusters, thrust_rcs, torque


@app.cell
def _(df, mo, save_table):
    save_table("rcs_sizing", df)
    mo.md("✓ RCS → parquet")
    return


if __name__ == "__main__":
    app.run()
