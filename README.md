# Astreia-MRV THOR Branch — Reentry Notebooks + OpenSCAD

This worktree is a feature branch based on [`Astreia-space/Thor_notebook`](https://github.com/Astreia-space/Thor_notebook), with Astreia-MRV configs, parametric geometry, conceptual aero/heating, constraints, and an OpenSCAD handoff layer added on top.

Marimo notebooks share state via Pydantic (`data/vehicle_state.json`) and Polars tables (`data/parquet/`). The MRV geometry notebooks generate OBJ/STL/JSON plus `geometry.scad` so the vehicle can be opened in OpenSCAD.

## Setup

```bash
cd /home/jetson-nano-drone/astreia-mrv-thor
python3 -m venv .venv
source .venv/bin/activate
pip install -U -r requirements.txt
pip install -e .
```

## Inputs

Edit all sizing numbers in one spreadsheet:

```text
data/inputs/thor_inputs.csv
```

Original THOR sections remain available: `mission`, `phase`, `mass`, `delta_v`, `link`, `orbit`, `entry`, `aero`, `geometry`, `propulsion`, `tps`, `docking`, `nav`, `loads`, `thermal`, `power`, `mdao`, `trade`.

MRV additions:

- `mrv`: config path, recovery mode, OpenSCAD output directory.
- `mrv_payload`: payload box and mass.

## Run

```bash
./scripts/validate_pipeline.py
./scripts/smoke_check_mrv_oml.py
./scripts/audit_mrv_output_provenance.py
PYTHONPATH=$PWD python -m marimo export session notebooks --force-overwrite --continue-on-error
./scripts/marimo.sh notebooks/80_oml_parametric.py
./scripts/marimo.sh notebooks/81_packaging_payload.py
./scripts/marimo.sh notebooks/82_openscad_export.py
```

Suggested order:

```text
00_inputs → 00-04 → 10,12-14 → 20-24 → 30-33 → 40-42
→ 50-53 → 60-62 → 70-71 → 80-82 → 90-91
```

The default OpenSCAD output is:

```text
outputs/mrv3_notebook/geometry.scad
```

For Onshape, use STEP rather than STL/OBJ. STL/OBJ imports as mesh only; the
STEP handoff imports as solid bodies:

```bash
PYTHONPATH=.cad-venv/lib/python3.10/site-packages \
  python scripts/export_onshape_step.py \
  --spec outputs/mrv3_notebook/onshape_brep_spec.json \
  --output-dir outputs/mrv3_notebook/onshape_step
```

Upload this file to Onshape:

```text
outputs/mrv3_notebook/onshape_step/geometry_onshape.step
```

The STEP handoff is a faceted BREP solid generated directly from the same
watertight MRV OML used by the plots. It should match the current geometry in
Onshape; it is not the older simplified analytic loft. STEP does not preserve
the Python/THOR feature history, so the export also includes editable
reconstruction aids:

```text
outputs/mrv3_notebook/onshape_parameters.csv
outputs/mrv3_notebook/onshape_loft_sections.csv
outputs/mrv3_notebook/onshape_variables.fs
outputs/mrv3_notebook/onshape_brep_spec.json
```

Use the CSV files as the dimensional source for Onshape sketches/lofts, and
copy `onshape_variables.fs` into a Feature Studio if you want named MRV
variables inside the document. The mesh exports remain useful for visual
comparison only.

This branch keeps the original safety framing: landing accuracy is represented as terminal recovery into an authorized recovery zone, not arbitrary precision impact targeting.

The current mass model is conceptual. If dry mass appears unusually high relative to MRV volume or payload target, the pipeline reports a mass sanity warning rather than treating the design as mature.

Notebook 82 now renders a clickable `MRV OML Artifact Handoff` table, and the handoff schema lives in shared `thor/mrv/handoff.py` helpers (`validate_mrv_oml_row`, `artifact_markdown_table`).

The generated `geometry.json` and `metrics.json` include THOR provenance:
the selected `thor_inputs.csv`, selected MRV YAML config, hashes for both,
the notebook generator, and the `rx` values used for the current geometry.
Run `./scripts/audit_mrv_output_provenance.py` to verify the current artifacts
still match the live THOR CSV/YAML inputs.
