"""10 — Orbit design: órbita-alvo, elementos, janelas."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import load_state, save_state, save_table
    from thor.io.inputs import num
    from thor.models.vehicle_state import OrbitState
    from thor.physics.astro import circular_velocity, mean_motion
    return OrbitState, circular_velocity, load_state, mean_motion, mo, num, pl, save_state, save_table


@app.cell
def _(mo):
    mo.md("# Camada 1 — Orbit Design\n\nInputs: `orbit` section in `thor_inputs.csv`")
    return


@app.cell
def _(OrbitState, num):
    orbit = OrbitState(
        altitude_km=num("orbit", "altitude_km"),
        inclination_deg=num("orbit", "inclination_deg"),
        eccentricity=num("orbit", "eccentricity"),
        raan_deg=num("orbit", "raan_deg"),
        arg_perigee_deg=num("orbit", "arg_perigee_deg"),
        true_anomaly_deg=num("orbit", "true_anomaly_deg"),
    )
    return (orbit,)


@app.cell
def _(circular_velocity, mean_motion, mo, orbit, pl):
    alt_m = orbit.altitude_km * 1000
    vc = circular_velocity(alt_m)
    n = mean_motion(alt_m)
    period_s = 2 * 3.14159265 / n
    df = pl.DataFrame(
        {
            "parameter": ["a [km]", "i [deg]", "V_circ [m/s]", "T [min]", "n [rad/s]"],
            "value": [
                orbit.altitude_km + 6371,
                orbit.inclination_deg,
                round(vc, 1),
                round(period_s / 60, 1),
                n,
            ],
        }
    )
    mo.ui.table(df)
    return alt_m, df, n, period_s, vc


@app.cell
def _(load_state, mo, orbit, save_state):
    state = load_state()
    state.orbit = orbit
    save_state(state)
    mo.md("✓ OrbitState → vehicle_state")
    return state


if __name__ == "__main__":
    app.run()
