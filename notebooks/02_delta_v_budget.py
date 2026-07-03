"""02 — ΔV Budget: injeção, phasing, rendezvous, deorbit, RCS."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import load_state, save_state, save_table
    from thor.io.inputs import items_with_params
    from thor.models.vehicle_state import DeltaVBudget, DeltaVItem
    return DeltaVBudget, DeltaVItem, items_with_params, load_state, mo, pl, save_state, save_table


@app.cell
def _(mo):
    mo.md("# Camada 0 — ΔV Budget\n\nInputs: `delta_v` section in `thor_inputs.csv`")
    return


@app.cell
def _(DeltaVItem, items_with_params):
    dv_items = [
        DeltaVItem(
            phase=row["item"],
            delta_v_m_s=float(row["delta_v_m_s"]),
            margin_frac=float(row["margin_frac"]),
        )
        for row in items_with_params("delta_v", "delta_v_m_s", "margin_frac")
    ]
    return (dv_items,)


@app.cell
def _(DeltaVBudget, dv_items, mo, pl):
    budget = DeltaVBudget(items=dv_items)
    df = pl.DataFrame(
        {
            "phase": [i.phase for i in dv_items],
            "delta_v_m_s": [i.delta_v_m_s for i in dv_items],
            "margin_frac": [i.margin_frac for i in dv_items],
            "with_margin_m_s": [i.delta_v_m_s * (1 + i.margin_frac) for i in dv_items],
        }
    )
    onboard = sum(i.delta_v_m_s * (1 + i.margin_frac) for i in dv_items if "launcher" not in i.phase)
    mo.vstack([
        mo.md(f"**ΔV onboard (pós-injeção):** {onboard:.0f} m/s"),
        mo.md(f"**ΔV total missão:** {budget.total_m_s:.0f} m/s"),
        mo.ui.table(df),
    ])
    return budget, df, onboard


@app.cell
def _(budget, df, load_state, mo, save_state, save_table):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(df["phase"].to_list(), df["with_margin_m_s"].to_list(), color="#dc2626")
    ax.set_ylabel("ΔV [m/s]")
    ax.set_title("Orçamento ΔV THOR")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    state = load_state()
    state.delta_v = budget
    save_state(state)
    save_table("delta_v_budget", df)
    mo.vstack([fig, mo.md("✓ ΔV budget → vehicle_state")])
    return ax, fig, plt, state


if __name__ == "__main__":
    app.run()
