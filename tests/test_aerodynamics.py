import math

from astreia_mrv.analysis.aerodynamics import aerodynamic_coefficients
from astreia_mrv.analysis.control_authority import estimate_control_authority
from astreia_mrv.config import load_design
from astreia_mrv.geometry.lifting_body import generate_lifting_body


def test_positive_alpha_produces_positive_lift_for_lifting_body():
    geom = generate_lifting_body(load_design("configs/mrv3.yaml"))
    coeff = aerodynamic_coefficients(geom, mach=20.0, alpha_deg=20.0)
    assert coeff["CL"] > 0.0
    assert coeff["CD"] > 0.0
    assert math.isfinite(coeff["x_cp_m"])


def test_control_authority_reports_pitch_roll_yaw():
    geom = generate_lifting_body(load_design("configs/mrv3.yaml"))
    rows = estimate_control_authority(geom)
    axes = {row["axis"] for row in rows}
    assert axes == {"pitch", "roll", "yaw"}
    assert all("available_coeff" in row for row in rows)
