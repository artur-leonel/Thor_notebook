"""00 — Mission CONOPS: fases, eventos, requisitos top-level → MissionSpec."""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import load_state, save_state, save_table
    from thor.io.inputs import num, phase_rows, txt
    from thor.models.mission_spec import MissionEvent, MissionPhase, MissionPhaseSpec, MissionSpec
    return (
        MissionEvent,
        MissionPhase,
        MissionPhaseSpec,
        MissionSpec,
        load_state,
        mo,
        num,
        phase_rows,
        pl,
        save_state,
        save_table,
        txt,
    )


@app.cell
def _(mo):
    mo.md("""
    # Camada 0 — Mission CONOPS

    Inputs: `data/inputs/thor_inputs.csv` → sections `mission`, `phase`.
    """)
    return


@app.cell
def _(MissionPhase, MissionPhaseSpec, phase_rows):
    phases = []
    for row in phase_rows():
        phase_name = row["item"]
        phases.append(
            MissionPhaseSpec(
                phase=MissionPhase(phase_name),
                duration_s=float(row["duration_s"]),
                delta_v_m_s=float(row["delta_v_m_s"]),
                power_wh=float(row["power_wh"]),
                requirements={"EI_alt_m": 122_000} if phase_name == "entry_interface" else {},
            )
        )
    return (phases,)


@app.cell
def _(MissionEvent, MissionPhase, MissionSpec, num, phases, txt):
    events = [
        MissionEvent(name="Liftoff", phase=MissionPhase.ASCENT, t_offset_s=0, duration_s=600),
        MissionEvent(name="Orbit insertion", phase=MissionPhase.INJECTION, t_offset_s=600),
        MissionEvent(
            name="Deorbit burn cutoff",
            phase=MissionPhase.DEORBIT,
            t_offset_s=sum(p.duration_s for p in phases[:-3]),
        ),
        MissionEvent(name="Entry Interface", phase=MissionPhase.ENTRY_INTERFACE, t_offset_s=0, notes="~122 km"),
        MissionEvent(name="Touchdown", phase=MissionPhase.LANDING, t_offset_s=0),
    ]
    mission = MissionSpec(
        name=txt("mission", "name"),
        vehicle_class=txt("mission", "vehicle_class"),
        phases=phases,
        events=events,
        top_level={
            "max_g_load": num("mission", "max_g_load"),
            "peak_heat_flux_W_cm2": num("mission", "peak_heat_flux_W_cm2"),
            "downmass_kg": num("mission", "downmass_kg"),
            "crossrange_km": num("mission", "crossrange_km"),
            "EI_alt_m": num("mission", "EI_alt_m"),
        },
    )
    return (mission,)


@app.cell
def _(mission, mo, pl):
    df_phases = pl.DataFrame(
        {
            "phase": [p.phase.value for p in mission.phases],
            "duration_h": [p.duration_s / 3600 for p in mission.phases],
            "delta_v_m_s": [p.delta_v_m_s for p in mission.phases],
            "power_wh": [p.power_wh for p in mission.phases],
        }
    )
    summary = mo.vstack([
        mo.md(f"**Duração total:** {mission.total_duration_s/86400:.1f} dias"),
        mo.md(f"**ΔV missão:** {mission.total_delta_v_m_s:.0f} m/s (excl. ascent)"),
        mo.ui.table(df_phases),
    ])
    summary
    return (df_phases,)


@app.cell
def _(df_phases, load_state, mission, mo, save_state, save_table):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(df_phases["phase"].to_list(), df_phases["duration_h"].to_list(), color="#2563eb")
    ax.set_xlabel("Duração [h]")
    ax.set_title("Timeline de fases THOR")
    plt.tight_layout()

    state = load_state()
    state.mission = mission
    save_state(state)
    save_table("mission_phases", df_phases)
    mo.vstack([fig, mo.md("✓ `MissionSpec` → `data/vehicle_state.json` + `mission_phases.parquet`")])
    return


if __name__ == "__main__":
    app.run()
