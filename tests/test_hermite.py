import numpy as np

from astreia_mrv.math.hermite import fix_segment_convexity, hermite_point, hermite_to_bezier


def test_hermite_interpolates_endpoints():
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])
    m = np.array([1.0, 0.0])
    assert np.allclose(hermite_point(p0, m, m, p1, 0.0), p0)
    assert np.allclose(hermite_point(p0, m, m, p1, 1.0), p1)


def test_hermite_to_bezier_endpoints():
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])
    m = np.array([0.3, 0.2])
    b = hermite_to_bezier(p0, m, m, p1)
    assert np.allclose(b[0], p0)
    assert np.allclose(b[-1], p1)


def test_convexity_fixer_shrinks_extreme_handles():
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    m0 = np.array([10.0, 10.0])
    m1 = np.array([-10.0, -10.0])
    f0, f1 = fix_segment_convexity(p0, m0, m1, p1)
    assert np.linalg.norm(f0) < np.linalg.norm(m0)
    assert np.linalg.norm(f1) < np.linalg.norm(m1)
