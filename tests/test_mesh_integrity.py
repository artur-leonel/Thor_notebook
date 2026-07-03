import numpy as np

from astreia_mrv.config import load_design
from astreia_mrv.geometry.lifting_body import generate_lifting_body


def test_mesh_has_no_nan_and_panels_have_area():
    geom = generate_lifting_body(load_design("configs/mrv2.yaml"))
    assert np.isfinite(geom.mesh.vertices).all()
    assert geom.mesh.face_areas.min() > 1e-10
