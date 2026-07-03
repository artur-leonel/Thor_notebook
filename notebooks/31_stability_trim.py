"""31 — Stability & trim: CG vs CP, margem estática."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from astreia_mrv.analysis.aerodynamics import aerodynamic_coefficients
    from astreia_mrv.analysis.control_authority import estimate_control_authority
    from astreia_mrv.config import load_design
    from astreia_mrv.geometry.lifting_body import generate_lifting_body
    from thor.io.handoff import load_state, save_state, save_table
    from thor.io.inputs import float_list, num, txt
    return (
        aerodynamic_coefficients,
        estimate_control_authority,
        float_list,
        generate_lifting_body,
        load_design,
        load_state,
        mo,
        num,
        pl,
        save_state,
        save_table,
        txt,
    )


@app.cell
def _(mo):
    mo.md("# Camada 3 — Stability & Trim\n\nInputs: `stability` section in `thor_inputs.csv`")
    return


@app.cell
def _(aerodynamic_coefficients, estimate_control_authority, float_list, generate_lifting_body, load_design, num, txt):
    config_path = txt("mrv", "config_path", default="configs/mrv3.yaml")
    design = load_design(config_path)
    geometry = generate_lifting_body(design)
    alpha_trim = num("stability", "alpha_trim_deg")
    mach = max(float_list("aero_db", "mach_list"))
    aero_trim = aerodynamic_coefficients(geometry, mach=mach, alpha_deg=alpha_trim)
    cg_x_m = geometry.metadata["parameters"]["x_cg_frac"] * geometry.reference_length_m
    cp_x_m = aero_trim["x_cp_m"]
    static_margin = (cp_x_m - cg_x_m) / geometry.reference_length_m
    authority = estimate_control_authority(
        geometry,
        cm_required=num("control", "cm_required"),
        cl_required=num("control", "cl_required", default=0.03),
        cn_required=num("control", "cn_required", default=0.02),
        yaw_rcs_assist_coeff=num("control", "yaw_rcs_assist_coeff", default=0.016),
    )
    return aero_trim, alpha_trim, authority, cg_x_m, config_path, cp_x_m, design, geometry, mach, static_margin


@app.cell
def _(aero_trim, alpha_trim, authority, cg_x_m, cp_x_m, geometry, mach, mo, pl, static_margin):
    authority_rows = [
        {
            "parameter": f"{row['axis']}_authority_margin",
            "value": row["margin"],
            "unit": "coeff",
            "status": row["status"],
            "note": row["note"],
        }
        for row in authority
    ]
    df = pl.DataFrame(
        [
            {"parameter": "x_CG_m", "value": cg_x_m, "unit": "m", "status": "", "note": ""},
            {"parameter": "x_CP_m", "value": cp_x_m, "unit": "m", "status": "", "note": "from MRV mesh modified-Newtonian estimate"},
            {"parameter": "static_margin", "value": static_margin, "unit": "x/L", "status": "", "note": ""},
            {"parameter": "alpha_trim_deg", "value": alpha_trim, "unit": "deg", "status": "", "note": ""},
            {"parameter": "mach", "value": mach, "unit": "", "status": "", "note": ""},
            {"parameter": "Cm_at_trim", "value": aero_trim["Cm"], "unit": "coeff", "status": "", "note": ""},
            {"parameter": "reference_area_m2", "value": geometry.reference_area_m2, "unit": "m2", "status": "", "note": ""},
            *authority_rows,
        ]
    )
    status = "needs revision" if any(not row["passes"] for row in authority) else "conceptual pass"
    mo.vstack([mo.md(f"**Status:** {status}"), mo.ui.table(df)])
    return df, status


@app.cell
def _(config_path, df, load_state, mo, save_state, save_table, status):
    state = load_state()
    state.notes["stability_trim"] = f"MRV mesh trim/control authority from {config_path}: {status}"
    save_state(state)
    save_table("stability_trim", df)
    mo.md("✓ MRV trim/control authority → parquet")
    return


if __name__ == "__main__":
    app.run()
