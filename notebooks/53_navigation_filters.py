"""53 — Navigation filters: IMU, EKF/UKF."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import save_table
    from thor.io.inputs import item_table
    return item_table, mo, pl, save_table


@app.cell
def _(mo):
    mo.md("# Camada 5 — Navigation\n\nInputs: `nav` in `thor_inputs.csv`")
    return


@app.cell
def _(item_table, mo, pl):
    raw = item_table("nav")
    df = pl.DataFrame(
        {
            "sensor": raw["item"],
            "error": raw["error"].cast(pl.Utf8),
            "phase": raw["phase"].cast(pl.Utf8),
        }
    )
    mo.vstack([mo.md("EKF/UKF: fusão IMU + ST + GNSS (placeholder)"), mo.ui.table(df)])
    return df


@app.cell
def _(df, mo, save_table):
    save_table("navigation_filters", df)
    mo.md("✓ Navigation filters → parquet")
    return


if __name__ == "__main__":
    app.run()
