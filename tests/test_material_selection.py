from astreia_mrv.analysis.materials import load_materials, recommended_zone_material


def test_material_selection_loads_zone():
    materials = load_materials("configs/materials.yaml")
    assert "C/C" in recommended_zone_material("nose_cap", materials)
