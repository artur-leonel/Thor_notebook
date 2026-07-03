"""82 — OpenSCAD handoff preview for Astreia-MRV."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo

    from thor.io.handoff import load_table
    from thor.mrv.handoff import artifact_markdown_table, validate_mrv_oml_row
    from thor.io.inputs import txt
    from thor.io.openscad import openscad_command, render_scad_to_stl

    return (
        Path,
        artifact_markdown_table,
        load_table,
        mo,
        openscad_command,
        render_scad_to_stl,
        txt,
        validate_mrv_oml_row,
    )


@app.cell
def _(mo):
    mo.md("# Layer 8.2 — OpenSCAD Output\n\nRun `80_oml_parametric.py` first. This notebook points OpenSCAD at the generated `geometry.scad` file and optionally renders with the local `openscad` binary if installed.")
    return


@app.cell
def _(Path, load_table, txt):
    table = load_table("mrv_oml")
    if table is not None and table.height:
        scad_path = Path(table["scad_path"][0])
        row = {c: str(table[c][0]) for c in table.columns}
    else:
        scad_path = Path(txt("mrv", "output_dir", default="outputs/mrv3_notebook")) / txt(
            "mrv", "open_scad_file", default="geometry.scad"
        )
        out_dir = scad_path.parent
        row = {
            "out_dir": str(out_dir),
            "scad_path": str(scad_path),
            "obj_path": str(out_dir / "geometry.obj"),
            "stl_path": str(out_dir / "geometry.stl"),
            "geometry_json_path": str(out_dir / "geometry.json"),
            "metrics_json_path": str(out_dir / "metrics.json"),
            "three_view_path": str(out_dir / "three_view.png"),
            "shaded_render_path": str(out_dir / "shaded_render.png"),
            "engineering_views_path": str(out_dir / "engineering_views.png"),
            "control_inspection_path": str(out_dir / "control_surface_inspection.png"),
        }
    return scad_path, row


@app.cell
def _(artifact_markdown_table, mo, openscad_command, validate_mrv_oml_row, scad_path, row):
    exists = scad_path.exists()
    preview = scad_path.read_text(encoding="utf-8").splitlines()[:16] if exists else []
    artifact_table = artifact_markdown_table(row)
    row_ok, missing_fields, missing_files = validate_mrv_oml_row(row, require_files=True)
    warning = ""
    if not row_ok:
        lines = []
        if missing_fields:
            lines.append("missing fields: " + ", ".join(missing_fields))
        if missing_files:
            lines.append("missing files: " + ", ".join(missing_files))
        warning = "\n\n".join(lines)
    mo.vstack([
        mo.md(f"**SCAD file:** `{scad_path}`"),
        mo.md(f"**Open command:** `{openscad_command(scad_path)}`"),
        mo.md("```scad\n" + "\n".join(preview) + "\n```" if preview else "SCAD file has not been generated yet."),
        mo.md(artifact_table),
        mo.md("### Artifact status"),
        mo.md(warning if warning else "✅ All required artifacts are present in handoff row."),
    ])
    return exists, preview


@app.cell
def _(exists, mo, render_scad_to_stl, scad_path):
    rendered = False
    render_path = scad_path.with_name("openscad_rendered.stl")
    if exists:
        rendered = render_scad_to_stl(scad_path, render_path)
    mo.md(
        f"OpenSCAD render attempted: **{rendered}**. "
        f"{'Rendered STL: `' + str(render_path) + '`' if rendered else 'Install OpenSCAD locally to render from this notebook.'}"
    )
    return render_path, rendered


if __name__ == "__main__":
    app.run()
