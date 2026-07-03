from dataclasses import replace

from astreia_mrv.analysis.constraints import launch_envelope_fit, payload_fit
from astreia_mrv.config import PayloadSpec, load_design
from astreia_mrv.geometry.lifting_body import generate_lifting_body


def test_baseline_launch_and_payload_fit():
    design = load_design("configs/mrv3.yaml")
    geom = generate_lifting_body(design)
    assert launch_envelope_fit(design, geom).passed
    assert payload_fit(design, geom).passed


def test_payload_violation_detected():
    design = load_design("configs/mrv3.yaml")
    oversized = replace(design.payload, length_m=3.0)
    design = replace(design, payload=oversized)
    geom = generate_lifting_body(design)
    assert not payload_fit(design, geom).passed
