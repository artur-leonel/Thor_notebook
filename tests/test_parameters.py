from astreia_mrv.config import load_design
from astreia_mrv.parameters import interp_from_rx, map_design_parameters


def test_interp_from_rx():
    assert interp_from_rx(0.0, 2.0, 4.0) == 2.0
    assert interp_from_rx(1.0, 2.0, 4.0) == 4.0
    assert interp_from_rx(0.5, 2.0, 4.0) == 3.0


def test_mrv_configs_load_and_map():
    for name in ["mrv1", "mrv2", "mrv3", "mrv3_triangular"]:
        design = load_design(f"configs/{name}.yaml")
        params = map_design_parameters(design)
        assert params.values["L_body"] > 0.0
        assert all(0.0 <= v <= 1.0 for v in params.normalized.values())
