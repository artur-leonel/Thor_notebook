"""01 — Mass Budget: MBS com margem AIAA S-120 (~30% conceitual)."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import load_state, save_state, save_table
    from thor.io.inputs import items_with_params, num
    from thor.models.vehicle_state import MassBudget, MassItem
    return MassBudget, MassItem, items_with_params, load_state, mo, num, pl, save_state, save_table


@app.cell
def _(mo):
    mo.md("# Camada 0 — Mass Budget (MBS)\n\nInputs: `mass` section in `thor_inputs.csv`")
    return


@app.cell
def _(MassItem, items_with_params, num):
    growth = num("mass", "growth_allowance", "_config")
    items = []
    for row in items_with_params("mass", "dry_kg"):
        m = float(row["dry_kg"])
        items.append(MassItem(name=row["item"], dry_kg=m, growth_kg=m * growth))
    propellant_kg = num("mass", "propellant_kg", "_config")
    return growth, items, propellant_kg


@app.cell
def _(MassBudget, growth, items, mo, pl, propellant_kg):
    df = pl.DataFrame(
        {
            "subsystem": [i.name for i in items],
            "dry_kg": [i.dry_kg for i in items],
            "growth_kg": [i.growth_kg for i in items],
            "total_kg": [i.total_kg for i in items],
        }
    )
    budget = MassBudget(items=items, growth_allowance=growth, propellant_kg=propellant_kg)
    totals = mo.vstack([
        mo.md(f"**Dry mass:** {budget.dry_mass_kg:.0f} kg"),
        mo.md(f"**Propellant:** {budget.propellant_kg:.0f} kg"),
        mo.md(f"**Wet mass:** {budget.wet_mass_kg:.0f} kg"),
        mo.md(f"**Com margem {growth*100:.0f}%:** {budget.total_with_margin_kg:.0f} kg"),
        mo.ui.table(df),
    ])
    totals
    return budget, df, totals


@app.cell
def _(budget, df, load_state, mo, pl, save_state, save_table):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(df["total_kg"].to_list(), labels=df["subsystem"].to_list(), autopct="%1.0f%%", startangle=90)
    ax.set_title("MBS THOR (com growth allowance)")

    state = load_state()
    state.mass = budget
    save_state(state)
    save_table("mass_budget", df)
    mo.vstack([fig, mo.md("✓ Mass budget → vehicle_state (loop com propulsão em 42)")])
    return ax, fig, plt, state


if __name__ == "__main__":
    app.run()
