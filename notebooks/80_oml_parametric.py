"""80 — Astreia-MRV OML parametric geometry."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from thor.io.handoff import load_state, save_state, save_table
    from thor.io.inputs import txt
    from thor.io.openscad import export_mrv_to_openscad

    return export_mrv_to_openscad, load_state, mo, pl, save_state, save_table, txt


@app.cell
def _(mo):
    mo.md("# Layer 8 — Astreia-MRV OML\n\nThis notebook uses the MRV YAML config selected in `data/inputs/thor_inputs.csv`, generates the parametric reentry vehicle mesh, and writes OpenSCAD/OBJ/STL outputs.")
    return


@app.cell
def _(export_mrv_to_openscad, txt):
    config_path = txt("mrv", "config_path", default="configs/mrv3.yaml")
    out_dir = txt("mrv", "output_dir", default="outputs/mrv3_notebook")
    design, geometry, export = export_mrv_to_openscad(config_path, out_dir)
    return config_path, design, export, geometry, out_dir


@app.cell
def _(design, export, geometry, mo, pl, save_table):
    mesh_quality = geometry.metadata.get("mesh_quality", {})
    summary = pl.DataFrame(
        {
            "vehicle": [design.scale.name],
            "family": [design.geometry_family],
            "recovery": [design.recovery_mode],
            "volume_m3": [geometry.volume_m3],
            "wetted_area_m2": [geometry.wetted_area_m2],
            "reference_area_m2": [geometry.reference_area_m2],
            "out_dir": [str(export.out_dir)],
            "scad_path": [str(export.scad_path)],
            "obj_path": [str(export.obj_path)],
            "stl_path": [str(export.stl_path)],
            "geometry_json_path": [str(export.geometry_json_path)],
            "metrics_json_path": [str(export.metrics_json_path)],
            "three_view_path": [str(export.three_view_path)],
            "shaded_render_path": [str(export.shaded_render_path)],
            "engineering_views_path": [str(export.engineering_views_path)],
            "control_inspection_path": [str(export.control_inspection_path)],
            "openscad_available": [export.openscad_available],
            "watertight": [bool(mesh_quality.get("watertight", False))],
            "positive_volume": [bool(mesh_quality.get("positive_volume", False))],
        }
    )
    save_table("mrv_oml", summary)
    mo.vstack([
        mo.md(f"**OpenSCAD:** `{export.scad_path}`"),
        mo.ui.table(summary),
    ])
    return mesh_quality, summary


@app.cell
def _(design, geometry, load_state, mo, save_state):
    state = load_state()
    state.aero.reference_area_m2 = geometry.reference_area_m2
    state.aero.nose_radius_m = geometry.metadata["parameters"].get("R_N", state.aero.nose_radius_m)
    state.notes["mrv_geometry"] = (
        f"{design.scale.name} {design.geometry_family}; OpenSCAD export generated from parametric MRV mesh."
    )
    save_state(state)
    mo.md("Saved geometry handoff to `vehicle_state.json` and table `mrv_oml`.")
    return (state,)


if __name__ == "__main__":
    app.run()
