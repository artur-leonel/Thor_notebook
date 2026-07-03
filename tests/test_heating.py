from astreia_mrv.analysis.heating import leading_edge_heat_flux_w_m2, stagnation_heat_flux_w_m2


def test_stagnation_heating_decreases_with_nose_radius():
    q_small = stagnation_heat_flux_w_m2(1e-4, 7800.0, 0.2)
    q_large = stagnation_heat_flux_w_m2(1e-4, 7800.0, 0.8)
    assert q_large < q_small


def test_leading_edge_heating_decreases_with_radius():
    q_small = leading_edge_heat_flux_w_m2(1e-4, 7800.0, 0.02, 65.0)
    q_large = leading_edge_heat_flux_w_m2(1e-4, 7800.0, 0.08, 65.0)
    assert q_large < q_small
