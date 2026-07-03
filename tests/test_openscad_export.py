import json

from astreia_mrv.config import load_design
from astreia_mrv.geometry.cad_export import export_geometry, export_openscad_polyhedron
from astreia_mrv.geometry.lifting_body import generate_lifting_body
from thor.io.inputs import INPUTS_CSV
from thor.io.openscad import export_mrv_to_openscad


def test_openscad_polyhedron_export(tmp_path):
    design = load_design("configs/mrv3.yaml")
    geom = generate_lifting_body(design)
    path = tmp_path / "geometry.scad"
    export_openscad_polyhedron(geom, path)
    text = path.read_text(encoding="utf-8")
    assert "module astreia_vehicle()" in text
    assert "polyhedron(" in text
    assert "points=[" in text
    assert "faces=[" in text


def test_cad_handoff_exports_editable_and_selectable_parts(tmp_path):
    design = load_design("configs/mrv3.yaml")
    geom = generate_lifting_body(design)

    export_geometry(geom, tmp_path)

    assert (tmp_path / "geometry.scad").exists()
    assert (tmp_path / "geometry.stl").exists()
    assert (tmp_path / "geometry.obj").exists()
    assert (tmp_path / "geometry_editable.scad").exists()
    assert (tmp_path / "geometry_grouped.obj").exists()
    assert (tmp_path / "geometry_grouped.mtl").exists()
    assert (tmp_path / "geometry_parts.scad").exists()
    assert (tmp_path / "onshape_brep_spec.json").exists()
    assert (tmp_path / "onshape_parameters.csv").exists()
    assert (tmp_path / "onshape_loft_sections.csv").exists()
    assert (tmp_path / "onshape_variables.fs").exists()
    assert (tmp_path / "ONSHAPE_README.md").exists()
    assert (tmp_path / "parts" / "manifest.json").exists()

    editable = (tmp_path / "geometry_editable.scad").read_text(encoding="utf-8")
    assert 'mode = "solid"' in editable
    assert "module user_additions()" in editable
    assert "module user_cuts()" in editable
    assert "module top_projection()" in editable

    grouped = (tmp_path / "geometry_grouped.obj").read_text(encoding="utf-8")
    for part in (
        "nose_forebody",
        "mid_body",
        "aft_body",
        "body_flap",
        "left_elevon",
        "right_elevon",
        "dorsal_tail_fin",
        "ventral_tail_fin",
    ):
        assert f"o {part}" in grouped
        assert f"usemtl {part}" in grouped

    parts_scad = (tmp_path / "geometry_parts.scad").read_text(encoding="utf-8")
    assert "show_body = true;" in parts_scad
    assert "show_controls = true;" in parts_scad
    assert 'import("parts/body_flap.stl")' in parts_scad

    manifest = json.loads((tmp_path / "parts" / "manifest.json").read_text(encoding="utf-8"))
    manifest_parts = {entry["part"] for entry in manifest}
    assert {
        "nose_forebody",
        "mid_body",
        "aft_body",
        "body_flap",
        "left_elevon",
        "right_elevon",
        "dorsal_tail_fin",
        "ventral_tail_fin",
    } <= manifest_parts

    brep_spec = json.loads((tmp_path / "onshape_brep_spec.json").read_text(encoding="utf-8"))
    assert brep_spec["schema"] == "astreia_mrv_onshape_brep_v1"
    assert brep_spec["source_parameters"] == geom.metadata["parameters"]
    assert brep_spec["normalized_rx"] == geom.metadata["normalized_rx"]
    assert brep_spec["measurements"]["core_body_width_m"] == 2.0 * geom.metadata["parameters"]["body_half_width"]
    assert brep_spec["body"]["name"] == "astreia_mrv_body"
    assert len(brep_spec["body"]["sections"]) >= 8
    assert {part["name"] for part in brep_spec["parts"]} == {
        "left_canted_aft_fin",
        "right_canted_aft_fin",
        "dorsal_tail_fin",
        "ventral_tail_fin",
        "center_body_flap",
    }

    parameter_csv = (tmp_path / "onshape_parameters.csv").read_text(encoding="utf-8")
    assert "mrv_body_half_width" in parameter_csv
    assert "rx_y_3_2" in parameter_csv
    assert "measure_total_span_m" in parameter_csv

    loft_csv = (tmp_path / "onshape_loft_sections.csv").read_text(encoding="utf-8")
    assert "section,point_index,x_m,y_m,z_m" in loft_csv
    assert "mid_body" in loft_csv

    variables_fs = (tmp_path / "onshape_variables.fs").read_text(encoding="utf-8")
    assert "setVariable" in variables_fs
    assert "mrv_body_half_width" in variables_fs

    geometry_json = json.loads((tmp_path / "geometry.json").read_text(encoding="utf-8"))
    assert geometry_json["onshape_parameters_csv"] == str(tmp_path / "onshape_parameters.csv")
    assert geometry_json["onshape_loft_sections_csv"] == str(tmp_path / "onshape_loft_sections.csv")
    assert geometry_json["onshape_variables_featurescript"] == str(tmp_path / "onshape_variables.fs")


def test_thor_openscad_export_stamps_input_provenance(tmp_path):
    design, geometry, export = export_mrv_to_openscad("configs/mrv3.yaml", tmp_path)

    geometry_json = json.loads(export.geometry_json_path.read_text(encoding="utf-8"))
    metrics_json = json.loads(export.metrics_json_path.read_text(encoding="utf-8"))

    for payload in (geometry_json, metrics_json):
        provenance = payload["provenance"]
        assert provenance["schema"] == "astreia_mrv_thor_provenance_v1"
        assert provenance["generator"] == "thor.io.openscad.export_mrv_to_openscad"
        assert provenance["notebook"] == "notebooks/80_oml_parametric.py"
        assert provenance["thor_inputs_csv"] == str(INPUTS_CSV)
        assert provenance["config_path"] == "configs/mrv3.yaml"
        assert provenance["output_dir"] == str(tmp_path)
        assert provenance["vehicle"]["name"] == design.scale.name
        assert provenance["payload"]["name"] == design.payload.name
        assert provenance["rx"] == dict(sorted(design.rx.items()))

    assert geometry_json["parameters"] == geometry.metadata["parameters"]
    assert metrics_json["geometry"]["reference_length_m"] == geometry.reference_length_m
