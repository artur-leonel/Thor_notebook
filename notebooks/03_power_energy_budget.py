"""03 — Power / Energy Budget: Wh por fase."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import load_state, save_state, save_table
    from thor.io.inputs import phase_rows
    from thor.models.vehicle_state import PowerBudget, PowerItem
    return PowerBudget, PowerItem, load_state, mo, phase_rows, pl, save_state, save_table


@app.cell
def _(mo):
    mo.md("# Camada 0 — Power / Energy Budget\n\nInputs: `phase` → `power_wh` in `thor_inputs.csv`")
    return


@app.cell
def _(PowerItem, phase_rows):
    items = []
    for row in phase_rows():
        dur = float(row["duration_s"])
        wh = float(row["power_wh"])
        if dur <= 0 and wh <= 0:
            continue
        avg_pwr = wh * 3600 / dur if dur > 0 else 0.0
        items.append(PowerItem(phase=row["item"], average_power_W=avg_pwr, duration_s=dur))
    return (items,)


@app.cell
def _(PowerBudget, items, mo, pl):
    budget = PowerBudget(items=items)
    df = pl.DataFrame(
        {
            "phase": [i.phase for i in items],
            "power_W": [i.average_power_W for i in items],
            "duration_h": [i.duration_s / 3600 for i in items],
            "energy_Wh": [i.energy_wh for i in items],
        }
    )
    mo.vstack([
        mo.md(f"**Energia total:** {budget.total_wh:.0f} Wh ({budget.total_wh/1000:.1f} kWh)"),
        mo.ui.table(df),
    ])
    return budget, df


@app.cell
def _(budget, df, load_state, mo, save_state, save_table):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(df["phase"].to_list(), df["energy_Wh"].to_list(), color="#16a34a")
    ax.set_ylabel("Energia [Wh]")
    ax.set_title("Orçamento energético por fase")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    state = load_state()
    state.power = budget
    save_state(state)
    save_table("power_budget", df)
    mo.vstack([fig, mo.md("✓ Power budget → vehicle_state")])
    return ax, fig, plt, state


if __name__ == "__main__":
    app.run()
