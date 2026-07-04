from astreia_mrv.config import load_design
from astreia_mrv.geometry.lifting_body import _build_fuselage, _ring_vertex_groups, generate_lifting_body
from astreia_mrv.parameters import map_lifting_body_parameters
import numpy as np
import pytest
import trimesh


def test_lifting_body_positive_and_symmetric():
    design = load_design("configs/mrv3.yaml")
    geom = generate_lifting_body(design)
    assert geom.volume_m3 > 0.0
    assert geom.wetted_area_m2 > 0.0
    assert geom.metadata["mesh_quality"]["symmetry_error_m"] < 1e-6
    assert {
        "body_flap",
        "right_elevon",
        "left_elevon",
        "right_strake",
        "left_strake",
        "dorsal_fin",
        "ventral_fin",
    }.issubset(geom.control_surfaces)
    assert all(len(surface.faces) > 0 for surface in geom.control_surfaces.values())


def test_lifting_body_controls_are_integrated_with_body_mesh():
    geom = generate_lifting_body(load_design("configs/mrv3.yaml"))
    mesh = trimesh.Trimesh(vertices=geom.mesh.vertices, faces=geom.mesh.faces, process=False)
    assert len(mesh.split(only_watertight=False)) == 1
    assert {"body_flap", "elevon", "strake", "dorsal_fin", "ventral_fin"}.issubset(set(geom.mesh.face_zone))
    assert len(geom.control_surfaces["body_flap"].faces) >= 12
    assert len(geom.control_surfaces["right_elevon"].faces) >= 15
    assert len(geom.control_surfaces["left_elevon"].faces) >= 15
    assert len(geom.control_surfaces["right_strake"].faces) >= 100
    assert len(geom.control_surfaces["left_strake"].faces) >= 100
    assert len(geom.control_surfaces["dorsal_fin"].faces) >= 20
    assert len(geom.control_surfaces["ventral_fin"].faces) >= 12
    fin_geometry = geom.metadata["control_surfaces"]["geometry"]
    assert fin_geometry["topology"].startswith("single continuous")
    assert 0.035 <= fin_geometry["per_side_fin_extension_m"] <= 0.09


def test_mrv3_lifting_body_has_hypersonic_slender_proportions():
    geom = generate_lifting_body(load_design("configs/mrv3.yaml"))
    mesh = trimesh.Trimesh(vertices=geom.mesh.vertices, faces=geom.mesh.faces, process=False)
    span = mesh.bounds[1] - mesh.bounds[0]
    core_width = geom.metadata["control_surfaces"]["geometry"]["core_body_width_m"]
    assert 3.0 <= span[0] / core_width <= 3.8
    assert 2.45 <= span[0] / span[1] <= 3.2
    assert 1.20 <= span[1] / span[2] <= 1.65
    assert 0.60 <= span[1] <= 0.82


def test_mrv3_payload_volume_has_conceptual_packaging_margin():
    design = load_design("configs/mrv3.yaml")
    geom = generate_lifting_body(design)
    payload_volume = design.payload.length_m * design.payload.width_m * design.payload.height_m
    assert payload_volume / geom.volume_m3 <= 0.60


def test_lifting_body_respects_configured_length_bounds():
    geom = generate_lifting_body(load_design("configs/mrv3.yaml"))
    length = geom.reference_length_m
    x = geom.mesh.vertices[:, 0]
    assert float(x.min()) == pytest.approx(0.0)
    assert float(x.max()) == pytest.approx(length)
    assert not np.any(x > length + 1e-9)


def test_lifting_body_aft_closeout_has_finite_truncated_section():
    geom = generate_lifting_body(load_design("configs/mrv3.yaml"))
    length = geom.reference_length_m
    verts = geom.mesh.vertices
    faces = geom.mesh.faces
    centroids = verts[faces].mean(axis=1)

    aft_zones = np.array([zone == "aft_closeout" for zone in geom.mesh.face_zone])
    assert aft_zones.any()
    assert float(centroids[aft_zones, 0].max()) == pytest.approx(length)

    near_tail = verts[verts[:, 0] >= 0.995 * length]
    assert len(near_tail) >= 12
    assert float(np.ptp(near_tail[:, 1])) > 0.050
    assert float(np.ptp(near_tail[:, 2])) > 0.030


def test_lifting_body_aft_taper_has_no_abrupt_body_step():
    design = load_design("configs/mrv3.yaml")
    params = map_lifting_body_parameters(design)
    fuselage = _build_fuselage(params)
    rings = [
        fuselage.vertices[group]
        for group in _ring_vertex_groups(fuselage)
        if len(group) >= 40 and float(fuselage.vertices[group[0], 0]) >= 0.60 * params.values["L_body"]
    ]
    assert len(rings) >= 4

    x = np.array([float(ring[:, 0].mean()) for ring in rings])
    y_span = np.array([float(np.ptp(ring[:, 1])) for ring in rings])
    span_drop_rate = np.maximum(0.0, y_span[:-1] - y_span[1:]) / np.maximum(np.diff(x), 1e-9)

    assert float(span_drop_rate.max()) < 1.20


def test_mrv3_nose_profile_variants_are_distinct():
    conic = generate_lifting_body(load_design("configs/mrv3.yaml"))
    triangular = generate_lifting_body(load_design("configs/mrv3_triangular.yaml"))

    assert conic.metadata["nose_profile"] == "conic"
    assert triangular.metadata["nose_profile"] == "triangular"
    assert conic.metadata["stations"]["x_nose_match_m"] > 0.10 * conic.reference_length_m
    assert triangular.metadata["stations"]["x_nose_match_m"] > 0.10 * triangular.reference_length_m

    sample_x = 0.12 * conic.reference_length_m
    conic_forebody = conic.mesh.vertices[np.abs(conic.mesh.vertices[:, 0] - sample_x).argmin()]
    triangular_forebody = triangular.mesh.vertices[np.abs(triangular.mesh.vertices[:, 0] - sample_x).argmin()]
    assert not np.allclose(conic_forebody[1:], triangular_forebody[1:], atol=1e-4)
