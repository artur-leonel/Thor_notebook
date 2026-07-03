from astreia_mrv.config import load_design
from astreia_mrv.geometry.capsule import generate_capsule


def test_capsule_positive_geometry():
    design = load_design("configs/mrv1.yaml")
    geom = generate_capsule(design)
    assert geom.volume_m3 > 0.0
    assert geom.wetted_area_m2 > 0.0
    assert geom.metadata["mesh_quality"]["finite_vertices"]
