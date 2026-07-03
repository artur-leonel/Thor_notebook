from pathlib import Path

from thor.mrv.handoff import (
    artifact_markdown_table,
    evaluate_mass_sanity,
    validate_mrv_oml_row,
)


def test_validate_mrv_oml_row_checks_required_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "case"
    out_dir.mkdir()
    row = {
        "out_dir": str(out_dir),
        "scad_path": str(out_dir / "geometry.scad"),
        "obj_path": str(out_dir / "geometry.obj"),
        "stl_path": str(out_dir / "geometry.stl"),
        "geometry_json_path": str(out_dir / "geometry.json"),
        "metrics_json_path": str(out_dir / "metrics.json"),
        "three_view_path": str(out_dir / "three_view.png"),
        "shaded_render_path": str(out_dir / "shaded_render.png"),
        "engineering_views_path": str(out_dir / "engineering_views.png"),
    }

    ok, missing_fields, missing_files = validate_mrv_oml_row(row, require_files=True)
    assert not ok
    assert not missing_fields
    assert set(missing_files) == set(value for key, value in row.items() if key != "out_dir")

    for key, value in row.items():
        if key == "out_dir":
            continue
        Path(value).write_text(f"{key} test\n", encoding="utf-8")

    ok, missing_fields, missing_files = validate_mrv_oml_row(row, require_files=True)
    assert ok
    assert not missing_fields
    assert not missing_files


def test_validate_mrv_oml_row_requires_text_fields() -> None:
    row: dict[str, object] = {
        "out_dir": 123,
        "scad_path": "",
        "obj_path": None,
        "stl_path": "",
        "geometry_json_path": "",
        "metrics_json_path": "",
        "three_view_path": "",
        "shaded_render_path": "",
        "engineering_views_path": "",
    }

    ok, missing_fields, _ = validate_mrv_oml_row(row, require_files=False)
    assert not ok
    assert "out_dir" in missing_fields
    assert "scad_path" in missing_fields


def test_artifact_markdown_table_contains_required_artifacts(tmp_path: Path) -> None:
    row = {
        "out_dir": str(tmp_path),
        "scad_path": "outputs/mrv3_notebook/geometry.scad",
        "obj_path": "outputs/mrv3_notebook/geometry.obj",
        "stl_path": "outputs/mrv3_notebook/geometry.stl",
        "geometry_json_path": "outputs/mrv3_notebook/geometry.json",
        "metrics_json_path": "outputs/mrv3_notebook/metrics.json",
        "three_view_path": "outputs/mrv3_notebook/three_view.png",
        "shaded_render_path": "outputs/mrv3_notebook/shaded_render.png",
        "engineering_views_path": "outputs/mrv3_notebook/engineering_views.png",
    }
    table = artifact_markdown_table(row)
    assert "| Artifact | Exists | Path |" in table
    assert "OpenSCAD model" in table
    assert "Engineering views" in table


def test_mass_sanity_flagging() -> None:
    result = evaluate_mass_sanity(dry_mass_kg=4329, payload_mass_kg=300, volume_m3=1.909)
    assert result["status"] == "warning"
    assert result["bulk_density_kg_m3"] > 1200
    assert result["message"]
