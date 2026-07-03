"""32 — Control surfaces: MRV mesh-derived surfaces."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from astreia_mrv.analysis.control_authority import control_surface_table, estimate_control_authority
    from astreia_mrv.config import load_design
    from astreia_mrv.geometry.lifting_body import generate_lifting_body
    from thor.io.handoff import load_state, save_state, save_table
    from thor.io.inputs import num, txt
    return control_surface_table, estimate_control_authority, generate_lifting_body, load_design, load_state, mo, num, pl, save_state, save_table, txt


@app.cell
def _(mo):
    mo.md("# Camada 3 — MRV Control Surfaces\n\nReads the same MRV YAML used by the OpenSCAD handoff and reports the actual generated mesh surfaces.")
    return


@app.cell
def _(control_surface_table, estimate_control_authority, generate_lifting_body, load_design, num, pl, txt):
    config_path = txt("mrv", "config_path", default="configs/mrv3.yaml")
    design = load_design(config_path)
    geometry = generate_lifting_body(design)
    length = geometry.reference_length_m
    rows = []
    for row in control_surface_table(geometry):
        rows.append(
            {
                **row,
                "x_min_frac": float(row["x_min_m"] / length),
                "x_max_frac": float(row["x_max_m"] / length),
                "integrated_gap_m": 0.0,
            }
        )
    surfaces_df = pl.DataFrame(rows)
    authority_df = pl.DataFrame(
        estimate_control_authority(
            geometry,
            cm_required=num("control", "cm_required"),
            cl_required=num("control", "cl_required", default=0.03),
            cn_required=num("control", "cn_required", default=0.02),
            yaw_rcs_assist_coeff=num("control", "yaw_rcs_assist_coeff", default=0.016),
        )
    )
    return authority_df, config_path, design, geometry, surfaces_df


@app.cell
def _(authority_df, config_path, geometry, load_state, mo, pl, save_state, save_table, surfaces_df):
    mo.vstack([mo.ui.table(surfaces_df), mo.ui.table(authority_df)])
    save_table("control_surfaces", surfaces_df)
    save_table("control_authority", authority_df)
    _state = load_state()
    _state.aero.reference_area_m2 = geometry.reference_area_m2
    failed = ", ".join(authority_df.filter(~pl.col("passes"))["axis"].to_list())
    _state.notes["control_surfaces"] = f"Mesh-derived MRV control surfaces from {config_path}"
    _state.notes["control_authority"] = (
        f"Conceptual authority needs revision for: {failed}" if failed else "Conceptual authority passes baseline thresholds"
    )
    save_state(_state)
    mo.md("✓ Control-surface and authority tables generated from the same MRV mesh used by OpenSCAD")
    return


if __name__ == "__main__":
    app.run()
