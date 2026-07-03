"""51 — Attitude dynamics 6DOF (quaternions)."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl

    from thor.io.handoff import save_table
    from thor.io.inputs import num
    return mo, np, num, pl, save_table


@app.cell
def _(mo):
    mo.md("# Camada 5 — Attitude Dynamics 6DOF\n\nInputs: `gnc` in `thor_inputs.csv`")
    return


@app.cell
def _(mo, np, num):
    def omega_matrix(w):
        wx, wy, wz = w
        return 0.5 * np.array(
            [
                [0, -wx, -wy, -wz],
                [wx, 0, wz, -wy],
                [wy, -wz, 0, wx],
                [wz, wy, -wx, 0],
            ]
        )

    q = np.array([1.0, 0, 0, 0])
    w = np.array([num("gnc", "gyro_x_rad_s"), num("gnc", "gyro_y_rad_s"), num("gnc", "gyro_z_rad_s")])
    dt = num("gnc", "dt_s")
    q_new = q + omega_matrix(w) @ q * dt
    q_new /= np.linalg.norm(q_new)
    mo.md(f"Quaternion após {dt:.0f} s: `{q_new.round(4)}`")
    return dt, omega_matrix, q, q_new, w


@app.cell
def _(dt, mo, pl, q_new, save_table, w):
    df = pl.DataFrame(
        {
            "metric": ["q0", "q1", "q2", "q3", "omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s", "dt_s"],
            "value": [float(q_new[0]), float(q_new[1]), float(q_new[2]), float(q_new[3]), float(w[0]), float(w[1]), float(w[2]), dt],
        }
    )
    save_table("attitude_dynamics_6dof", df)
    mo.ui.table(df)
    return df


if __name__ == "__main__":
    app.run()
