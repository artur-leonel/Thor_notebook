import pytest

from astreia_mrv.config import load_design
from astreia_mrv.geometry.lifting_body import generate_lifting_body
from thor.io.inputs import num, txt


def test_thor_mrv_inputs_match_generated_baseline_geometry():
    config_path = txt("mrv", "config_path")
    design = load_design(config_path)
    geometry = generate_lifting_body(design)
    bounds = geometry.mesh.bounds
    span = bounds[1] - bounds[0]

    assert txt("mrv", "geometry_family") == design.geometry_family
    assert num("geometry", "length_m") == pytest.approx(geometry.reference_length_m)
    assert num("geometry", "width_m") == pytest.approx(float(span[1]))
    assert num("aero", "reference_area_m2") == pytest.approx(geometry.reference_area_m2)
    assert num("aero", "nose_radius_m") == pytest.approx(geometry.metadata["parameters"]["R_N"])
    assert num("tps", "shield_area_m2") == pytest.approx(geometry.wetted_area_m2)
    assert num("mass_props", "height_m") == pytest.approx(float(span[2]))


def test_thor_mrv_payload_rows_match_yaml_payload():
    design = load_design(txt("mrv", "config_path"))

    assert txt("mrv_payload", "name") == design.payload.name
    assert num("mrv_payload", "mass_kg") == pytest.approx(design.payload.mass_kg)
    assert num("mrv_payload", "length_m") == pytest.approx(design.payload.length_m)
    assert num("mrv_payload", "width_m") == pytest.approx(design.payload.width_m)
    assert num("mrv_payload", "height_m") == pytest.approx(design.payload.height_m)
