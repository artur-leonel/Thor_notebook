from pathlib import Path

import pytest

pytest.importorskip("OCP")

from astreia_mrv.config import load_design
from astreia_mrv.geometry.cad_export import export_onshape_brep_spec
from astreia_mrv.geometry.lifting_body import generate_lifting_body
from scripts.export_onshape_step import export_step


def test_onshape_step_export_produces_valid_solid_manifest(tmp_path: Path) -> None:
    geometry = generate_lifting_body(load_design("configs/mrv3.yaml"))
    spec_path = tmp_path / "onshape_brep_spec.json"
    export_onshape_brep_spec(geometry, spec_path)

    manifest = export_step(spec_path, tmp_path / "onshape_step")

    assert manifest["valid_body"] is True
    assert manifest["valid_assembly"] is True
    assert Path(str(manifest["assembly_step"])).exists()
    assert Path(str(manifest["part_steps"]["body"])).exists()
